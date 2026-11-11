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
        p_remove=0.0,
        device=None,
        output=None,
    ):
        """Train ``PretrainedUNet`` and return the ``best.pth`` path."""
        from camp.train.segmentor import SegmentorTrainer

        device = _resolve_device(self.device if device is None else device)
        paired, train_ds, val_ds = self._paired_datasets(
            class_name, image_size, patch_size, split_ratio, p_remove,
            for_segmentor=True,
        )
        run_dir = self._run_dir(class_name, "segmentor", output)
        if self.verbose:
            logger.info("Segmentor — %s", class_name)
            logger.info("Dataset: %s", paired)
            logger.info("Train: %s | Val: %s", len(train_ds), len(val_ds))
            logger.info("Encoder: %s", encoder)
            logger.info("Epochs: %s (freeze=%s)", epochs, freeze_epochs)
            logger.info("Output: %s", run_dir)

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
            "p_remove": p_remove,
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
        return self._copy_best(save_path, run_dir / "best.pth")

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
        p_remove=0.0,
        device=None,
        output=None,
        segmentor_weights=None,
    ):
        """Train ``AttentionRecommender`` and return the ``best.pth`` path.

        The frozen segmentor is loaded with ``find_weights`` unless
        ``segmentor_weights`` points at a checkpoint. A missing file raises
        ``FileNotFoundError``.
        """
        from camp.train.recommender import AttentionRecommenderTrainer

        device = _resolve_device(self.device if device is None else device)
        paired, train_ds, val_ds = self._paired_datasets(
            class_name, image_size, patch_size, split_ratio, p_remove
        )
        if segmentor_weights:
            seg_path = Path(segmentor_weights)
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
            logger.info("Recommender — %s", class_name)
            logger.info("Dataset: %s", paired)
            logger.info("Train: %s | Val: %s", len(train_ds), len(val_ds))
            logger.info("Segmentor weights: %s", seg_path)
            logger.info("Phase 1: %s epochs, lr=%s", epochs, lr)
            logger.info("Phase 2: %s epochs, lr=%s", finetune_epochs, finetune_lr)
            logger.info("Output: %s", run_dir)

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
            "p_remove": p_remove,
            "segmentor_weights": str(seg_path),
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

        if self.verbose:
            logger.info("Phase 1: %s epochs, lr=%s", epochs, lr)
        trainer = AttentionRecommenderTrainer(model, device=str(device))
        trainer.patience = patience
        trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            accumulation_steps=accumulation_steps,
            run_dir=str(run_dir),
        )

        if finetune_epochs > 0:
            if self.verbose:
                logger.info("Phase 2: %s epochs, lr=%s", finetune_epochs, finetune_lr)
            best_path = run_dir / "best_model.pt"
            if best_path.exists():
                model.load_model(str(best_path))
            trainer2 = AttentionRecommenderTrainer(model, device=str(device))
            trainer2.train_losses = trainer.train_losses.copy()
            trainer2.val_losses = trainer.val_losses.copy()
            trainer2.best_val_loss = trainer.best_val_loss
            trainer2.patience = patience
            trainer2.fit(
                train_loader,
                val_loader,
                epochs=finetune_epochs,
                lr=finetune_lr,
                accumulation_steps=accumulation_steps,
                run_dir=str(run_dir),
            )

        return self._copy_best(run_dir / "best_model.pt", run_dir / "best.pth")

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
            logger.info("Loaded %s on %s", class_name, self.device)
            logger.info("Segmentor weights: %s", segmentor_path)
            logger.info("Recommender weights: %s", recommender_path)
        return self

    def extract(self, source, source_mask=None, min_area=200, max_patches=10):
        """Extract instance patches. Requires ``load()``."""
        self._require_loaded()
        img_resized = source.resize(self.image_size, Image.BILINEAR)

        if source_mask is not None:
            mask_resized = source_mask.resize(self.image_size, Image.NEAREST)
            mask_np = (np.array(mask_resized.convert("L")) > 127).astype(np.uint8)
        elif self.segmentor is not None:
            img_t = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.segmentor(img_t)
                mask_np = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            raise ValueError("Sem máscara fornecida e sem segmentador carregado.")

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
        return patches

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
            raise ValueError("Recommender não carregado.")

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

    def blend(self, target, patch_info, recommendation, target_resized=None):
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
            return Image.fromarray(target_np), Image.fromarray(np.zeros((height, width), dtype=np.uint8))

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
            return_intermediate=False,
        )
        blended_pixels = (blend_results["final"] * 255).astype(np.uint8)

        final_composite = target_np.copy()
        final_composite[img_y1:img_y2, img_x1:img_x2] = blended_pixels
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[img_y1:img_y2, img_x1:img_x2] = (ctx_mask_cropped > 127).astype(np.uint8) * 255
        return Image.fromarray(final_composite), Image.fromarray(full_mask)

    def generate(
        self,
        target,
        source,
        source_mask=None,
        patch_index=0,
        recommendation_index=0,
        top_k=5,
        use_tto=True,
    ):
        """Extract, recommend, and blend one sample. Returns a ``Sample``."""
        self._require_loaded()
        patches = self.extract(source, source_mask)
        if not patches:
            raise ValueError("Nenhum patch extraído da source image.")
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
            raise ValueError("Nenhuma recomendação gerada.")
        rec = recommendations[min(recommendation_index, len(recommendations) - 1)]

        image, mask = self.blend(
            target, patch_info, rec, target_resized=target_resized
        )
        return Sample(image=image, mask=mask, score=rec["score"])

    def generate_batch(
        self,
        targets,
        sources,
        masks=None,
        n=10,
        out=None,
        use_tto=False,
        seed=None,
    ):
        """Draw ``n`` samples. ``use_tto`` is forwarded to ``recommend``."""
        self._require_loaded()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        out_path = None
        if out:
            out_path = Path(out)
            (out_path / "images").mkdir(parents=True, exist_ok=True)
            (out_path / "masks").mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info("Extracting patches from %s sources", len(sources))
        all_patches = []
        for i, src in enumerate(sources):
            src_mask = masks[i] if masks else None
            try:
                patches = self.extract(src, src_mask)
                for patch in patches:
                    patch["source_idx"] = i
                all_patches.extend(patches)
            except Exception as exc:
                logger.warning("Source %s failed during extract: %s", i, exc, exc_info=True)

        if not all_patches:
            raise ValueError("Nenhum patch extraído de nenhuma source.")

        if self.verbose:
            logger.info("Available patches: %s", len(all_patches))

        results = []
        old_verbose = self.verbose
        self.verbose = False
        cache = {}

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
                image, mask = self.blend(
                    target, patch_info, rec, target_resized=target_resized
                )
                results.append(Sample(image=image, mask=mask, score=rec["score"]))
                if out_path is not None:
                    image.save(out_path / "images" / f"aug_{i:04d}.png")
                    mask.save(out_path / "masks" / f"aug_{i:04d}.png")
                if old_verbose and (i + 1) % 10 == 0:
                    logger.info("Generated %s/%s images", i + 1, n)
            except Exception as exc:
                logger.warning(
                    "generate_batch sample %s failed: %s", i, exc, exc_info=True
                )
                continue

        self.verbose = old_verbose
        if self.verbose:
            logger.info("Batch finished: %s/%s images", len(results), n)
        return results

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
        self, class_name, image_size, patch_size, split_ratio, p_remove, for_segmentor=False
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
            p_remove=p_remove,
            for_segmentor=for_segmentor,
        )
        val_ds = PairedMaskDataset(
            paired,
            split="val",
            split_ratio=split_ratio,
            image_size=image_size,
            patch_size=patch_size,
            apply_aug=False,
            p_remove=0.0,
            for_segmentor=for_segmentor,
        )
        return paired, train_ds, val_ds

    def _run_dir(self, class_name, kind, output):
        if output:
            run_dir = Path(output)
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
                logger.info("Checkpoint copied to %s", destination)
        return destination
