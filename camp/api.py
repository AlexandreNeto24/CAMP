"""Public facade for training and copy-insert-merge generation.

``image_size`` is a square, 512 by default. Extract, recommend, and blend
share that square, and box coordinates are in that space. ``generate`` and
``generate_batch`` return ``Sample``. Blend applies the winning scale,
horizontal flip, and ``rot90`` to the context crop and its mask, then aligns
the mask center to the box center.
"""

import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from camp.blend import SeamlessBlendingPipeline
from camp.dataset import PairedMaskDataset, attention_collate
from camp.infer.engine import RecommendationEngine
from camp.io import class_dirs, find_weights
from camp.models.recommender import build_attention_recommender
from camp.models.segmentor import PretrainedUNet
from camp.sample import Sample

logger = logging.getLogger(__name__)


def _epochs(count):
    return "1 epoch" if count == 1 else "%s epochs" % count


def _log_run(title, rows):
    """Print one title and aligned ``label  value`` rows."""
    logger.info(title)
    for label, value in rows:
        logger.info("  %-8s %s", label, value)


def format_sample(index, total, sample):
    """One generate line: score and the geometry that was pasted."""
    flip = "on" if sample.flip else "off"
    score = float("nan") if sample.score is None else sample.score
    return "%d/%d  score %.4f  scale %.1f  rot %d  flip %s" % (
        index,
        total,
        score,
        sample.scale,
        int(sample.angle),
        flip,
    )


def _sample_from_recommendation(image, mask, recommendation):
    return Sample(
        image=image,
        mask=mask,
        score=recommendation.get("score"),
        scale=float(recommendation.get("scale", 1.0)),
        angle=float(recommendation.get("angle", 0.0)),
        flip=bool(recommendation.get("flip", False)),
    )


def _resolve_device(device):
    """Use CPU when CUDA was requested but this machine has no CUDA device."""
    if isinstance(device, torch.device):
        device = str(device)
    else:
        device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def apply_winning_geometry(image, mask, scale=1.0, angle=0.0, flip=False):
    """Apply one winning placement to a context crop and its mask.

    The order matches ``RecommendationEngine``: scale, horizontal flip, then
    ``rot90`` with ``k = angle // 90``. The image is resized bilinearly and
    the mask with nearest neighbours, at ``floor(size * scale)`` so the
    spatial size matches ``interpolate``. Scale 1, angle 0, and no flip
    return the original arrays, still contiguous.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    scale = float(scale)
    turns = (int(angle) // 90) % 4

    if scale != 1.0:
        height, width = image.shape[:2]
        new_w = max(1, int(math.floor(width * scale)))
        new_h = max(1, int(math.floor(height * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if flip:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if turns != 0:
        image = np.rot90(image, k=turns)
        mask = np.rot90(mask, k=turns)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _as_pair(size):
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


def _debug_side(size):
    """One side of a square size, for the debug writers."""
    return _as_pair(size)[0]


def _save_training_debug(camp, kind, run_dir, fields):
    """Write the same validation figures as ``python -m camp train --debug``.

    Eight figures, beside ``best.pth``. ``kind`` is ``segmentor`` or
    ``recommender``. The count matches ``camp.cli._DEBUG_TRAIN_VIZ``.
    """
    from types import SimpleNamespace

    view = SimpleNamespace(**fields)
    if kind == "segmentor":
        from camp.cli.viz import _save_segmentor_viz

        _save_segmentor_viz(camp, view, Path(run_dir), n=8)
    elif kind == "recommender":
        from camp.cli.viz import _save_recommender_viz

        _save_recommender_viz(camp, view, Path(run_dir), n=8)
    else:
        raise ValueError("kind must be 'segmentor' or 'recommender', got %r" % kind)


def _load_segmentor_weights(model, path):
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    except Exception:
        model.load_adapted_state_dict(str(path), strict=False)
    model.eval()
    return model


class CAMP:
    """Train a segmentor and recommender, then copy, place, and blend instances.

    One ``load`` builds the segmentor and the recommender on the same encoder.
    Inference calls ``eval`` before scoring. The segmentor returns logits.
    """

    def __init__(
        self,
        root=".",
        device="cuda",
        verbose=True,
        image_size=(512, 512),
        core_strength=0.58,
        edge_strength=0.88,
        l_weight=0.72,
    ):
        self.root = Path(root)
        self.verbose = verbose
        self.image_size = _as_pair(image_size)
        self.core_strength = core_strength
        self.edge_strength = edge_strength
        self.l_weight = l_weight
        requested = device
        self.device = _resolve_device(device)
        if str(requested).startswith("cuda") and self.device.type == "cpu":
            logger.warning("CUDA is not available. Using CPU.")
        self.segmentor = None
        self.recommender = None
        self.engine = None
        self.blender = None
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def train_segmentor(
        self,
        class_name,
        encoder="efficientnet-b3",
        decoder_dropout=0.4,
        image_size=512,
        patch_size=128,
        epochs=80,
        batch_size=8,
        lr=3e-4,
        freeze_epochs=30,
        finetune_lr=1e-4,
        accumulation_steps=2,
        patience=25,
        split_ratio=0.2,
        device=None,
        output=None,
        debug=False,
    ):
        """Train ``PretrainedUNet`` and return the ``best.pth`` path.

        ``debug=True`` writes the same eight validation figures as the CLI
        ``--debug`` flag, in ``debug/`` beside that checkpoint.
        """
        from camp.train.segmentor import SegmentorTrainer

        device = _resolve_device(self.device if device is None else device)
        paired, train_ds, val_ds = self._paired_datasets(
            class_name, image_size, patch_size, split_ratio,
            for_segmentor=True,
        )
        run_dir = self._run_dir(class_name, "segmentor", output)
        if self.verbose:
            rows = [
                ("data", "%s train / %s val  %s" % (len(train_ds), len(val_ds), paired)),
                ("setup", "%s  %s px  batch %s" % (encoder, image_size, batch_size)),
            ]
            if freeze_epochs > 0:
                rows.append((
                    "phase 1",
                    "%s  decoder  lr %.1e" % (_epochs(min(freeze_epochs, epochs)), lr),
                ))
            if epochs > freeze_epochs:
                rows.append((
                    "phase 2",
                    "%s  encoder  lr %.1e" % (_epochs(epochs - freeze_epochs), finetune_lr),
                ))
            rows.append(("output", run_dir))
            if debug:
                rows.append(("debug", "8 validation figures"))
            _log_run("segmentor  %s" % class_name, rows)

        model = PretrainedUNet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_dropout=decoder_dropout,
        )
        config = {
            "class_name": class_name,
            "encoder": encoder,
            "decoder_dropout": decoder_dropout,
            "image_size": image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "freeze_epochs": freeze_epochs,
            "finetune_lr": finetune_lr,
            "accumulation_steps": accumulation_steps,
            "patience": patience,
            "split_ratio": split_ratio,
            "debug": bool(debug),
            "timestamp": datetime.now().isoformat(),
        }
        self._write_config(run_dir, config)
        save_path = SegmentorTrainer(model).fit(
            train_ds,
            val_ds,
            str(run_dir),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=str(device),
            accumulation_steps=accumulation_steps,
            freeze_epochs=freeze_epochs,
            full_finetune_lr=finetune_lr,
            early_stopping_patience=patience,
            verbose=self.verbose,
        )
        del model
        if device.type == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        best = self._copy_best(save_path, run_dir / "best.pth")
        if debug:
            _save_training_debug(self, "segmentor", run_dir, {
                "class_name": class_name,
                "split_ratio": split_ratio,
                "image_size": _debug_side(image_size),
                "patch_size": _debug_side(patch_size),
                "encoder": encoder,
                "decoder_dropout": decoder_dropout,
                "batch_size": batch_size,
                "device": device,
            })
        return best

    def train_recommender(
        self,
        class_name,
        encoder="efficientnet-b3",
        embedding_dim=2048,
        num_heads=16,
        dropout=0.10,
        epochs=60,
        finetune_epochs=20,
        lr=3e-4,
        finetune_lr=5e-5,
        batch_size=4,
        accumulation_steps=4,
        patience=30,
        image_size=512,
        patch_size=128,
        split_ratio=0.2,
        device=None,
        output=None,
        segmentor_weights=None,
        debug=False,
    ):
        """Train ``AttentionRecommender`` and return the ``best.pth`` path.

        The frozen segmentor is loaded with ``find_weights`` unless
        ``segmentor_weights`` points at a checkpoint. A missing file raises
        ``FileNotFoundError``. ``debug=True`` writes the same eight
        validation figures as the CLI ``--debug`` flag, in ``debug/`` beside
        that checkpoint.
        """
        from camp.train.recommender import AttentionRecommenderTrainer

        device = _resolve_device(self.device if device is None else device)
        paired, train_ds, val_ds = self._paired_datasets(
            class_name, image_size, patch_size, split_ratio
        )
        if segmentor_weights:
            seg_path = Path(segmentor_weights)
            if not seg_path.is_absolute():
                seg_path = self.root / seg_path
            if not seg_path.is_file():
                raise FileNotFoundError(f"Segmentor weights not found: {seg_path}")
        else:
            seg_path = find_weights(self.root, class_name, "segmentor")

        # Stay on CPU through AttentionRecommender.__init__: that constructor
        # runs a dummy forward before the module is moved.
        segmentor = PretrainedUNet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        _load_segmentor_weights(segmentor, seg_path)
        model = build_attention_recommender(
            segmentor,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout,
        ).to(device)

        run_dir = self._run_dir(class_name, "recommender", output)
        if self.verbose:
            rows = [
                ("data", "%s train / %s val  %s" % (len(train_ds), len(val_ds), paired)),
                ("setup", "%s px  batch %s" % (image_size, batch_size)),
                ("backbone", seg_path),
            ]
            if epochs > 0:
                rows.append(("phase 1", "%s  lr %.1e" % (_epochs(epochs), lr)))
            if finetune_epochs > 0:
                rows.append((
                    "phase 2",
                    "%s  lr %.1e" % (_epochs(finetune_epochs), finetune_lr),
                ))
            rows.append(("output", run_dir))
            if debug:
                rows.append(("debug", "8 validation figures"))
            _log_run("recommender  %s" % class_name, rows)

        config = {
            "class_name": class_name,
            "encoder": encoder,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "patience": patience,
            "split_ratio": split_ratio,
            "segmentor_weights": str(seg_path),
            "debug": bool(debug),
            "timestamp": datetime.now().isoformat(),
        }
        self._write_config(run_dir, config)

        train_workers = 0 if os.name == "nt" else 4
        val_workers = 0 if os.name == "nt" else 2
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=True,
            collate_fn=attention_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            collate_fn=attention_collate,
        )

        trainer = AttentionRecommenderTrainer(model, device=str(device))
        trainer.patience = patience
        trainer.verbose = self.verbose
        trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            accumulation_steps=accumulation_steps,
            run_dir=str(run_dir),
            phase=1,
        )

        if finetune_epochs > 0:
            best_path = run_dir / "best_model.pt"
            if best_path.exists():
                model.load_model(str(best_path))
            trainer2 = AttentionRecommenderTrainer(model, device=str(device))
            trainer2.train_losses = trainer.train_losses.copy()
            trainer2.val_losses = trainer.val_losses.copy()
            trainer2.best_val_loss = trainer.best_val_loss
            trainer2.patience = patience
            trainer2.verbose = self.verbose
            trainer2.fit(
                train_loader,
                val_loader,
                epochs=finetune_epochs,
                lr=finetune_lr,
                accumulation_steps=accumulation_steps,
                run_dir=str(run_dir),
                phase=2,
            )

        del model, trainer, segmentor
        if finetune_epochs > 0:
            del trainer2
        if device.type == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        best = self._copy_best(run_dir / "best_model.pt", run_dir / "best.pth")
        if debug:
            _save_training_debug(self, "recommender", run_dir, {
                "class_name": class_name,
                "split_ratio": split_ratio,
                "image_size": _debug_side(image_size),
                "patch_size": _debug_side(patch_size),
                "encoder": encoder,
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "dropout": dropout,
                "batch_size": batch_size,
                "segmentor_weights": str(seg_path),
                "device": device,
            })
        return best

    def load(self, class_name, encoder="efficientnet-b3"):
        """Load one class. Missing checkpoints raise ``FileNotFoundError``."""
        segmentor_path = find_weights(self.root, class_name, "segmentor")
        recommender_path = find_weights(self.root, class_name, "recommender")

        segmentor = PretrainedUNet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        _load_segmentor_weights(segmentor, segmentor_path)
        recommender = build_attention_recommender(
            segmentor,
            embedding_dim=2048,
            num_heads=16,
        )
        recommender.load_model(str(recommender_path))
        recommender.to(self.device)
        recommender.eval()

        self.segmentor = segmentor
        self.recommender = recommender
        self.engine = RecommendationEngine(self.recommender)
        self.blender = SeamlessBlendingPipeline(
            core_strength=self.core_strength,
            edge_strength=self.edge_strength,
            l_weight=self.l_weight,
        )
        if self.verbose:
            logger.info("loaded  %s  %s", class_name, self.device)
        return self

    def extract(self, source, source_mask=None, min_area=200, max_patches=10, with_source=False):
        """Extract instance patches. Requires ``load()``.

        ``with_source=True`` keeps the resized source, the mask that was
        used to cut instances, and the segmentor probability. A given mask
        is what gets cut. The probability is still computed so the debug
        figure can show the prediction beside that mask.
        """
        self._require_loaded()
        mask_origin = "given" if source_mask is not None else "segmentor"
        img_resized = source.resize(self.image_size, Image.BILINEAR)
        probability = self._segmentor_probability(img_resized) if (
            with_source or source_mask is None
        ) else None

        if source_mask is not None:
            mask_resized = source_mask.resize(self.image_size, Image.NEAREST)
            mask_np = (np.array(mask_resized.convert("L")) > 127).astype(np.uint8)
        elif probability is not None:
            mask_np = (probability > 0.5).astype(np.uint8)
        else:
            raise ValueError(
                "No mask was given and no segmentor is loaded."
            )

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask_np.astype(np.uint8), connectivity=8
        )

        img_np = np.array(img_resized)
        patches = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            instance_mask = (labels == i).astype(np.uint8) * 255
            patch_crop = img_np[y:y + h, x:x + w]
            mask_crop = instance_mask[y:y + h, x:x + w]

            margin = int(max(w, h) * 0.3)
            cx1 = max(0, x - margin)
            cy1 = max(0, y - margin)
            cx2 = min(self.image_size[1], x + w + margin)
            cy2 = min(self.image_size[0], y + h + margin)

            context_crop = img_np[cy1:cy2, cx1:cx2]
            context_mask_np = np.zeros_like(context_crop[:, :, 0], dtype=np.float32)
            obj_y1 = y - cy1
            obj_x1 = x - cx1
            context_mask_np[obj_y1:obj_y1 + h, obj_x1:obj_x1 + w] = mask_crop.astype(np.float32) / 255.0

            patch_pil = Image.fromarray(patch_crop)
            mask_pil = Image.fromarray(mask_crop)
            context_pil = Image.fromarray(context_crop)

            patches.append({
                "patch": patch_pil,
                "mask": mask_pil,
                "patch_tensor": self.to_tensor(patch_crop),
                "mask_tensor": torch.from_numpy(mask_crop).unsqueeze(0).float() / 255.0,
                "bbox": (x, y, x + w, y + h),
                "area": area,
                "patch_with_context": context_pil,
                "context_mask": torch.from_numpy(context_mask_np).unsqueeze(0),
            })

        patches.sort(key=lambda p: p["area"], reverse=True)
        patches = patches[:max_patches]
        if with_source:
            mask_u8 = (mask_np > 0).astype(np.uint8) * 255
            for patch in patches:
                patch["source_image"] = img_np
                patch["source_mask"] = mask_u8
                patch["mask_origin"] = mask_origin
                patch["segmentor_probability"] = probability
        return patches

    def _segmentor_probability(self, image):
        """Sigmoid map at ``image_size``, or None when no segmentor is loaded."""
        if self.segmentor is None:
            return None
        tensor = self.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.segmentor(tensor)
        return torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)

    def recommend(
        self,
        target,
        patch_info,
        top_k=5,
        use_tto=True,
        target_resized=None,
        cache=None,
    ):
        """Score placements with the engine built in ``load()``.

        ``target_resized`` is the target already resized with PIL bilinear.
        ``cache`` reuses that tensor, its encoder features, and the 128x128
        patch tensors across calls in one batch.
        """
        self._require_loaded()
        if self.recommender is None:
            raise ValueError("Recommender is not loaded.")

        _resized, target_t, image_features = self._prepare_target(
            target, target_resized, cache
        )
        patch_t, ctx_t = self._prepare_patch(patch_info, cache)
        return self.engine.recommend_placements(
            target_t,
            patch_t,
            patch_with_context=ctx_t,
            patch_box=list(patch_info["bbox"]),
            top_k=top_k,
            use_tto=use_tto,
            precomputed_image_features=image_features,
        )

    def blend(self, target, patch_info, recommendation, target_resized=None, return_stages=False):
        """Blend one recommendation on CPU and return PIL images.

        The context crop and its mask take the winning scale, horizontal
        flip, and ``rot90`` together. The mask center is then aligned to the
        box center. The composite is not left on the GPU. Missing geometry
        fields mean identity: scale 1, angle 0, no flip.
        """
        self._require_loaded()
        if target_resized is None:
            target_resized = target.resize(self.image_size, Image.BILINEAR)
        target_np = np.array(target_resized)

        box = recommendation["box"]
        box_center_x = int((box[0] + box[2]) / 2)
        box_center_y = int((box[1] + box[3]) / 2)

        ctx_pil = patch_info["patch_with_context"]
        ctx_mask_t = patch_info["context_mask"]
        ctx_np = np.array(ctx_pil)
        ctx_mask_np = (ctx_mask_t.squeeze(0).numpy() * 255).astype(np.uint8)
        ctx_np, ctx_mask_np = apply_winning_geometry(
            ctx_np,
            ctx_mask_np,
            scale=recommendation.get("scale", 1.0),
            angle=recommendation.get("angle", 0.0),
            flip=recommendation.get("flip", False),
        )
        ctx_h, ctx_w = ctx_np.shape[:2]

        coords = cv2.findNonZero(ctx_mask_np)
        if coords is not None:
            mx, my, mw, mh = cv2.boundingRect(coords)
            cx_internal = mx + (mw // 2)
            cy_internal = my + (mh // 2)
        else:
            cx_internal = ctx_w // 2
            cy_internal = ctx_h // 2

        target_ctx_x1 = int(box_center_x - cx_internal)
        target_ctx_y1 = int(box_center_y - cy_internal)
        target_ctx_x2 = target_ctx_x1 + ctx_w
        target_ctx_y2 = target_ctx_y1 + ctx_h

        height, width = target_np.shape[:2]
        img_x1 = max(0, target_ctx_x1)
        img_y1 = max(0, target_ctx_y1)
        img_x2 = min(width, target_ctx_x2)
        img_y2 = min(height, target_ctx_y2)

        if img_x2 <= img_x1 or img_y2 <= img_y1:
            image = Image.fromarray(target_np)
            mask = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
            if not return_stages:
                return image, mask
            return image, mask, {
                "foreground": None,
                "alpha": None,
                "harmonized": None,
                "composite": target_np,
                "pasted_mask": np.zeros((height, width), dtype=np.uint8),
            }

        crop_x1 = img_x1 - target_ctx_x1
        crop_y1 = img_y1 - target_ctx_y1
        crop_x2 = crop_x1 + (img_x2 - img_x1)
        crop_y2 = crop_y1 + (img_y2 - img_y1)

        ctx_np_cropped = ctx_np[crop_y1:crop_y2, crop_x1:crop_x2]
        ctx_mask_cropped = ctx_mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
        bg_crop = target_np[img_y1:img_y2, img_x1:img_x2]

        blend_results = self.blender.run(
            lesion_image=ctx_np_cropped,
            lesion_mask=ctx_mask_cropped,
            background=bg_crop,
            return_intermediate=True,
        )
        blended = np.clip(np.asarray(blend_results["final"], dtype=np.float32), 0.0, 1.0)
        if blended.ndim == 2:
            blended = np.repeat(blended[..., None], 3, axis=2)
        alpha = blend_results.get("alpha")
        if alpha is not None:
            alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
            if alpha.ndim == 3:
                alpha = alpha[..., 0]
        dest_h = img_y2 - img_y1
        dest_w = img_x2 - img_x1
        if blended.shape[0] != dest_h or blended.shape[1] != dest_w:
            blended = cv2.resize(blended, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
        if alpha is not None and (alpha.shape[0] != dest_h or alpha.shape[1] != dest_w):
            alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
        blended_pixels = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
        # Where the instance does not contribute, keep the target. A resized
        # crop would otherwise leave a hard rectangle even at alpha ~ 0.
        if alpha is not None:
            outside = alpha <= 0.01
            blended_pixels[outside] = bg_crop[outside]

        final_composite = target_np.copy()
        final_composite[img_y1:img_y2, img_x1:img_x2] = blended_pixels
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[img_y1:img_y2, img_x1:img_x2] = (ctx_mask_cropped > 127).astype(np.uint8) * 255
        image = Image.fromarray(final_composite)
        mask = Image.fromarray(full_mask)
        if not return_stages:
            return image, mask
        return image, mask, {
            "foreground": blend_results.get("fg_extracted"),
            "alpha": blend_results.get("alpha"),
            "harmonized": blend_results.get("fg_harmonized"),
            "composite": final_composite,
            "pasted_mask": full_mask,
        }

    def generate(
        self,
        target,
        source,
        source_mask=None,
        patch_index=0,
        recommendation_index=0,
        top_k=5,
        use_tto=True,
        debug=False,
    ):
        """Extract, recommend, and blend one sample. Returns a ``Sample``.

        ``debug=True`` fills ``Sample.debug`` with the segmentation mask,
        the winning heatmap, and the blend stages.
        """
        self._require_loaded()
        patches = self.extract(source, source_mask, with_source=debug)
        if not patches:
            raise ValueError("No patch was extracted from the source image.")
        patch_info = patches[min(patch_index, len(patches) - 1)]

        target_resized = target.resize(self.image_size, Image.BILINEAR)
        recommendations = self.recommend(
            target,
            patch_info,
            top_k=top_k,
            use_tto=use_tto,
            target_resized=target_resized,
        )
        if not recommendations:
            raise ValueError("No recommendation was produced.")
        chosen_index = min(recommendation_index, len(recommendations) - 1)
        rec = recommendations[chosen_index]

        if debug:
            image, mask, stages = self.blend(
                target, patch_info, rec, target_resized=target_resized, return_stages=True
            )
        else:
            image, mask = self.blend(
                target, patch_info, rec, target_resized=target_resized
            )
            stages = None
        sample = _sample_from_recommendation(image, mask, rec)
        if debug:
            sample.debug = self._placement_debug(
                patch_info, target_resized, recommendations, chosen_index, stages
            )
        return sample

    def generate_batch(
        self,
        targets,
        sources,
        masks=None,
        n=10,
        out=None,
        use_tto=False,
        seed=None,
        debug_n=0,
    ):
        """Draw ``n`` samples. ``use_tto`` is forwarded to ``recommend``.

        The first ``debug_n`` successful samples carry ``Sample.debug``:
        the mask used to cut the instance, the winning heatmap, the
        blend stages, and ``stem``, the ``aug_XXXX`` name under ``out``. Extraction of every source keeps that mask when
        ``debug_n`` is positive, and the samples stay in this same loop.
        """
        self._require_loaded()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        out_path = None
        if out:
            out_path = Path(out)
            (out_path / "images").mkdir(parents=True, exist_ok=True)
            (out_path / "masks").mkdir(parents=True, exist_ok=True)

        all_patches = []
        for i, src in enumerate(sources):
            src_mask = masks[i] if masks else None
            try:
                patches = self.extract(src, src_mask, with_source=debug_n > 0)
                for patch in patches:
                    patch["source_idx"] = i
                all_patches.extend(patches)
            except Exception as exc:
                logger.warning("Source %s failed during extract: %s", i, exc, exc_info=True)

        if not all_patches:
            raise ValueError("No patch was extracted from any source.")

        if self.verbose:
            logger.info("patches  %s", len(all_patches))

        results = []
        cache = {}
        debug_left = max(0, int(debug_n))

        for i in range(n):
            try:
                target_idx = random.randint(0, len(targets) - 1)
                patch_info = random.choice(all_patches)
                target = targets[target_idx]
                target_resized, _tensor, _features = self._prepare_target(
                    target, None, cache
                )
                recommendations = self.recommend(
                    target,
                    patch_info,
                    top_k=3,
                    use_tto=use_tto,
                    target_resized=target_resized,
                    cache=cache,
                )
                if not recommendations:
                    continue
                rec = recommendations[0]
                want_debug = debug_left > 0
                if want_debug:
                    image, mask, stages = self.blend(
                        target,
                        patch_info,
                        rec,
                        target_resized=target_resized,
                        return_stages=True,
                    )
                else:
                    image, mask = self.blend(
                        target, patch_info, rec, target_resized=target_resized
                    )
                    stages = None
                sample = _sample_from_recommendation(image, mask, rec)
                stem = f"aug_{i:04d}"
                if want_debug:
                    sample.debug = self._placement_debug(
                        patch_info, target_resized, recommendations, 0, stages
                    )
                    sample.debug["stem"] = stem
                    debug_left -= 1
                results.append(sample)
                if out_path is not None:
                    image.save(out_path / "images" / f"{stem}.png")
                    mask.save(out_path / "masks" / f"{stem}.png")
                if self.verbose:
                    logger.info(format_sample(i + 1, n, sample))
            except Exception as exc:
                logger.warning("sample %s failed: %s", i + 1, exc)
                continue

        return results

    def _placement_debug(self, patch_info, target_resized, recommendations, chosen_index, stages):
        """Arrays for one generate strip. Copies so the next sample cannot reuse them."""
        chosen_index = min(chosen_index, len(recommendations) - 1)
        heatmap = recommendations[chosen_index].get("heatmap")
        context_mask = patch_info["context_mask"]
        if torch.is_tensor(context_mask):
            context_mask = context_mask.detach().cpu().numpy()
        context_mask = np.squeeze(np.asarray(context_mask))
        slim = []
        for rec in recommendations:
            slim.append({
                "box": [float(value) for value in rec["box"]],
                "score": float(rec.get("score", 0.0)),
                "scale": float(rec.get("scale", 1.0)),
                "angle": float(rec.get("angle", 0.0)),
                "flip": bool(rec.get("flip", False)),
                "rank": int(rec.get("rank", 0)),
            })

        def stage(name):
            if not stages or stages.get(name) is None:
                return None
            return np.array(stages[name], copy=True)

        return {
            "source_image": np.array(patch_info["source_image"], copy=True),
            "source_mask": np.array(patch_info["source_mask"], copy=True),
            "mask_origin": patch_info.get("mask_origin", "segmentor"),
            "segmentor_probability": (
                None if patch_info.get("segmentor_probability") is None
                else np.array(patch_info["segmentor_probability"], copy=True)
            ),
            "bbox": tuple(int(value) for value in patch_info["bbox"]),
            "area": int(patch_info.get("area") or 0),
            "instance_mask": np.array(patch_info["mask"].convert("L")),
            "input_patch": np.array(patch_info["patch"].convert("RGB")),
            "context": np.array(patch_info["patch_with_context"].convert("RGB")),
            "context_mask": context_mask,
            "target": np.array(target_resized.convert("RGB")),
            "recommendations": slim,
            "chosen_index": chosen_index,
            "heatmap": None if heatmap is None else np.array(heatmap, copy=True),
            "stages": {
                "foreground": stage("foreground"),
                "alpha": stage("alpha"),
                "harmonized": stage("harmonized"),
                "composite": stage("composite"),
                "pasted_mask": stage("pasted_mask"),
            },
        }

    def _prepare_target(self, target, target_resized, cache):
        """Resize the target once and, in eval, encode it once.

        The cache key includes the square ``image_size`` and PIL bilinear,
        which is the filter ``recommend`` and ``blend`` already used.
        """
        key = (
            "target",
            id(target),
            tuple(self.image_size),
            "PIL.BILINEAR",
            "ToTensor",
        )
        if cache is not None and key in cache:
            return cache[key]
        if target_resized is None:
            target_resized = target.resize(self.image_size, Image.BILINEAR)
        tensor = self.to_tensor(target_resized).unsqueeze(0).to(self.device)
        features = None
        if self.recommender is not None and not self.recommender.training:
            with torch.no_grad():
                features = [
                    feature.detach()
                    for feature in self.recommender.feature_extractor(tensor)
                ]
        prepared = (target_resized, tensor, features)
        if cache is not None:
            cache[key] = prepared
        return prepared

    def _prepare_patch(self, patch_info, cache):
        """Build the 128x128 patch and context tensors, optionally from cache.

        The patch uses ``F.interpolate`` bilinear with ``align_corners=False``.
        The context crop uses PIL bilinear. The two filters are not mixed.
        """
        tensor = patch_info["patch_tensor"]
        patch_key = (
            "patch",
            id(tensor),
            tuple(tensor.shape),
            (128, 128),
            "F.interpolate",
            "bilinear",
            False,
        )
        if cache is not None and patch_key in cache:
            patch_t = cache[patch_key]
        else:
            patch_t = F.interpolate(
                tensor.unsqueeze(0),
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).to(self.device)
            if cache is not None:
                cache[patch_key] = patch_t

        ctx_t = None
        if "patch_with_context" in patch_info:
            ctx_pil = patch_info["patch_with_context"]
            context_key = (
                "context",
                id(ctx_pil),
                tuple(ctx_pil.size),
                (128, 128),
                "PIL.BILINEAR",
                "ToTensor",
            )
            if cache is not None and context_key in cache:
                ctx_t = cache[context_key]
            else:
                ctx_t = self.to_tensor(ctx_pil.resize((128, 128), Image.BILINEAR))
                ctx_t = ctx_t.unsqueeze(0).to(self.device)
                if cache is not None:
                    cache[context_key] = ctx_t
        return patch_t, ctx_t

    def _require_loaded(self):
        if self.segmentor is None or self.recommender is None or self.engine is None or self.blender is None:
            raise RuntimeError("Call load() before inference.")

    def _paired_datasets(
        self, class_name, image_size, patch_size, split_ratio, for_segmentor=False
    ):
        paired = class_dirs(self.root, class_name)[0]
        image_size = _as_pair(image_size)
        patch_size = _as_pair(patch_size)
        train_ds = PairedMaskDataset(
            paired,
            split="train",
            split_ratio=split_ratio,
            image_size=image_size,
            patch_size=patch_size,
            apply_aug=True,
            for_segmentor=for_segmentor,
        )
        val_ds = PairedMaskDataset(
            paired,
            split="val",
            split_ratio=split_ratio,
            image_size=image_size,
            patch_size=patch_size,
            apply_aug=False,
            for_segmentor=for_segmentor,
        )
        return paired, train_ds, val_ds

    def _run_dir(self, class_name, kind, output):
        """Run directory under the project root, beside ``data`` and ``weights``.

        A relative ``output`` is resolved from ``self.root``, not from the
        process working directory.
        """
        if output:
            run_dir = Path(output)
            if not run_dir.is_absolute():
                run_dir = self.root / run_dir
        else:
            run_dir = self.root / "weights" / class_name / kind
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_config(self, run_dir, config):
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.dump(config, handle, default_flow_style=False)

    def _copy_best(self, source, destination):
        source = Path(source)
        destination = Path(destination)
        if source.exists() and source != destination:
            shutil.copy2(source, destination)
            if self.verbose:
                logger.info("saved  %s", destination)
        return destination
