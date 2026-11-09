"""Public facade for training and copy-insert-merge generation."""

import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from camp.blend import SeamlessBlendingPipeline
from camp.dataset import PairedMaskDataset, attention_collate
from camp.infer.engine import RecommendationEngine
from camp.io import class_dirs, find_weights
from camp.models.recommender import AttentionRecommender
from camp.models.segmentor import PretrainedUNet
from camp.sample import Sample
from camp.train.recommender import AttentionRecommenderTrainer
from camp.train.segmentor import SegmentorTrainer


def _resolve_device(device):
    """Use CPU when CUDA was requested but this machine has no CUDA device."""
    if isinstance(device, torch.device):
        device = str(device)
    else:
        device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


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
    """Train a segmentor and recommender, then copy, place, and blend instances."""

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
        if self.verbose and str(requested).startswith("cuda") and self.device.type == "cpu":
            print("CUDA is not available. Using CPU.")
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
        device = _resolve_device(self.device if device is None else device)
        paired, train_ds, val_ds = self._paired_datasets(
            class_name, image_size, patch_size, split_ratio, p_remove
        )
        run_dir = self._run_dir(class_name, "segmentor", output)
        if self.verbose:
            print(f"Segmentor — {class_name}")
            print(f"  Dataset: {paired}")
            print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
            print(f"  Encoder: {encoder}")
            print(f"  Epochs: {epochs} (freeze={freeze_epochs})")
            print(f"  Output: {run_dir}")

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
        model = AttentionRecommender(
            backbone_model=segmentor,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout,
        ).to(device)

        run_dir = self._run_dir(class_name, "recommender", output)
        if self.verbose:
            print(f"Recommender — {class_name}")
            print(f"  Dataset: {paired}")
            print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
            print(f"  Segmentor: {seg_path}")
            print(f"  Phase 1: {epochs} epochs, lr={lr}")
            print(f"  Phase 2: {finetune_epochs} epochs, lr={finetune_lr}")
            print(f"  Output: {run_dir}")

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
            print(f"Phase 1: {epochs} epochs, lr={lr}")
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
                print(f"Phase 2: {finetune_epochs} epochs, lr={finetune_lr}")
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
        recommender = AttentionRecommender(
            backbone_model=segmentor,
            embedding_dim=2048,
            num_heads=16,
            dropout_rate=0.3,
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
            print(f"Loaded {class_name} on {self.device}")
            print(f"  Segmentor: {segmentor_path}")
            print(f"  Recommender: {recommender_path}")
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

            if self.verbose:
                print(
                    f"  Patch {i}: area={area}, bbox=({x},{y},{x + w},{y + h}), "
                    f"mask sum={mask_crop.sum()}, shape={mask_crop.shape}"
                )
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
        if self.verbose:
            print(f"  Extraídos {len(patches)} patches (áreas: {[p['area'] for p in patches]})")
        return patches

    def recommend(self, target, patch_info, top_k=5, use_tto=True):
        """Score placements with the engine built in ``load()``."""
        self._require_loaded()
        if self.recommender is None:
            raise ValueError("Recommender não carregado.")

        target_resized = target.resize(self.image_size, Image.BILINEAR)
        target_t = self.to_tensor(target_resized).unsqueeze(0).to(self.device)

        patch_t = patch_info["patch_tensor"]
        patch_t = torch.nn.functional.interpolate(
            patch_t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).to(self.device)

        ctx_t = None
        if "patch_with_context" in patch_info:
            ctx_pil = patch_info["patch_with_context"]
            ctx_t = self.to_tensor(ctx_pil.resize((128, 128), Image.BILINEAR))
            ctx_t = ctx_t.unsqueeze(0).to(self.device)

        recs = self.engine.recommend_placements(
            target_t,
            patch_t,
            patch_with_context=ctx_t,
            patch_box=list(patch_info["bbox"]),
            top_k=top_k,
            use_tto=use_tto,
        )
        if self.verbose and recs:
            print(f"  Top-{len(recs)} recomendações (melhor score: {recs[0]['score']:.4f})")
        return recs

    def blend(self, target, patch_info, recommendation):
        """Blend one recommendation with the pipeline built in ``load()``."""
        self._require_loaded()
        target_resized = target.resize(self.image_size, Image.BILINEAR)
        target_np = np.array(target_resized)

        box = recommendation["box"]
        box_center_x = int((box[0] + box[2]) / 2)
        box_center_y = int((box[1] + box[3]) / 2)

        ctx_pil = patch_info["patch_with_context"]
        ctx_mask_t = patch_info["context_mask"]
        ctx_np = np.array(ctx_pil)
        ctx_mask_np = (ctx_mask_t.squeeze(0).numpy() * 255).astype(np.uint8)
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
        """Extract, recommend, and blend one sample."""
        self._require_loaded()
        if self.verbose:
            print("\n─── CAMP.generate() ───")
        if self.verbose:
            print("[1/3] Extracting patches...")
        patches = self.extract(source, source_mask)
        if not patches:
            raise ValueError("Nenhum patch extraído da source image.")
        patch_info = patches[min(patch_index, len(patches) - 1)]

        if self.verbose:
            print("[2/3] Recommending location...")
        recommendations = self.recommend(target, patch_info, top_k=top_k, use_tto=use_tto)
        if not recommendations:
            raise ValueError("Nenhuma recomendação gerada.")
        rec = recommendations[min(recommendation_index, len(recommendations) - 1)]

        if self.verbose:
            print("[3/3] Paste and Blend...")
        image, mask = self.blend(target, patch_info, rec)
        if self.verbose:
            print("─── Pipeline concluído ───\n")
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
            print(f"Pré-extraindo patches de {len(sources)} sources...")
        all_patches = []
        for i, src in enumerate(sources):
            src_mask = masks[i] if masks else None
            try:
                patches = self.extract(src, src_mask)
                for patch in patches:
                    patch["source_idx"] = i
                all_patches.extend(patches)
            except Exception as exc:
                if self.verbose:
                    print(f"  Aviso: source {i} falhou — {exc}")

        if not all_patches:
            raise ValueError("Nenhum patch extraído de nenhuma source.")

        if self.verbose:
            print(f"Total: {len(all_patches)} patches disponíveis")

        results = []
        old_verbose = self.verbose
        self.verbose = False

        for i in range(n):
            try:
                target_idx = random.randint(0, len(targets) - 1)
                patch_info = random.choice(all_patches)
                target = targets[target_idx]
                recommendations = self.recommend(target, patch_info, top_k=3, use_tto=use_tto)
                if not recommendations:
                    continue
                rec = recommendations[0]
                image, mask = self.blend(target, patch_info, rec)
                results.append(Sample(image=image, mask=mask, score=rec["score"]))
                if out_path is not None:
                    image.save(out_path / "images" / f"aug_{i:04d}.png")
                    mask.save(out_path / "masks" / f"aug_{i:04d}.png")
                if old_verbose and (i + 1) % 10 == 0:
                    print(f"  Geradas {i + 1}/{n} imagens...")
            except Exception as exc:
                if old_verbose:
                    print(f"  Erro na imagem {i}: {exc}")
                continue

        self.verbose = old_verbose
        if self.verbose:
            print(f"\nBatch concluído: {len(results)}/{n} imagens geradas com sucesso")
        return results

    def _require_loaded(self):
        if self.segmentor is None or self.recommender is None or self.engine is None or self.blender is None:
            raise RuntimeError("Call load() before inference.")

    def _paired_datasets(self, class_name, image_size, patch_size, split_ratio, p_remove):
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
        )
        val_ds = PairedMaskDataset(
            paired,
            split="val",
            split_ratio=split_ratio,
            image_size=image_size,
            patch_size=patch_size,
            apply_aug=False,
            p_remove=0.0,
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
                print(f"Checkpoint copied to {destination}")
        return destination
