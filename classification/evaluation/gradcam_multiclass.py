"""Multiclass Grad-CAM utilities for CNN classifiers.

This module mirrors the binary Grad-CAM helpers but keeps the original
implementation untouched so binary workflows remain stable.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from classification.evaluation.gradcam import (
    GradCAM,
    calculate_cam_dice,
    get_resnet_target_layer,
    save_cam_panel,
)


def _load_mask_for_image(image_path: Path):
    dataset_root = image_path.parent.parent
    mask_path = dataset_root / 'masks' / image_path.with_suffix('.png').name
    if not mask_path.exists():
        mask_path = dataset_root / 'masks' / image_path.with_suffix('.jpg').name
    if not mask_path.exists():
        return None

    return Image.open(mask_path).convert('L')


def export_gradcam_examples(
    model,
    dataset,
    output_dir: Path,
    samples_per_class: Optional[dict] = None,
    device: str = 'cpu',
    return_per_class_metrics: bool = False,
):
    if samples_per_class is None:
        samples_per_class = {0: 3, 1: 3}

    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_resnet_target_layer(model)
    cam = GradCAM(model, target_layer)

    saved_counts = {cls: 0 for cls in samples_per_class.keys()}
    sample_dice_scores = []
    dice_scores_by_class = {}

    try:
        for index in range(len(dataset)):
            sample = dataset[index]

            if sample.get('source') not in {'local', None}:
                continue

            # --- CORREÇÃO DAS SOFT LABELS ---
            raw_label = sample['label']
            if isinstance(raw_label, torch.Tensor) and raw_label.numel() > 1:
                label = int(raw_label.argmax().item())
            else:
                label = int(raw_label.item() if isinstance(raw_label, torch.Tensor) else raw_label)
            # --------------------------------

            needs_visual = label in samples_per_class and saved_counts[label] < samples_per_class[label]
            needs_metric = label > 0

            if not needs_visual and not needs_metric:
                continue

            image_path = Path(sample['path'])
            image = Image.open(image_path).convert('RGB')
            image_tensor = sample['image'].unsqueeze(0).to(device)

            raw_heatmap, pred_class = cam(image_tensor)
            heatmap_pil = Image.fromarray(np.uint8(255 * raw_heatmap)).resize((image.width, image.height), Image.BILINEAR)
            heatmap = np.array(heatmap_pil) / 255.0

            title = f'idx={index} label={label} pred={pred_class}'
            mask_np = None

            if needs_metric:
                mask_pil = _load_mask_for_image(image_path)
                if mask_pil is not None:
                    mask_img = mask_pil.resize((image.width, image.height), Image.NEAREST)
                    mask_np = np.array(mask_img) / 255.0
                    dice_score = calculate_cam_dice(heatmap, mask_np, threshold=0.3)
                    title += f' | Dice={dice_score:.3f}'
                    sample_dice_scores.append(dice_score)
                    dice_scores_by_class.setdefault(label, []).append(dice_score)

            if needs_visual:
                filename = f'gradcam_class{label}_{saved_counts[label]:03d}.png'
                save_cam_panel(image, heatmap, output_dir / filename, title=title, mask=mask_np)
                saved_counts[label] += 1

    finally:
        cam.close()

    metrics = {}
    if sample_dice_scores:
        metrics['cam_dice'] = float(np.mean(sample_dice_scores))

    if return_per_class_metrics:
        class_means = []
        for class_label in sorted(dice_scores_by_class.keys()):
            class_scores = dice_scores_by_class[class_label]
            if not class_scores:
                continue
            class_mean = float(np.mean(class_scores))
            metrics[f'cam_dice_class_{class_label}'] = class_mean
            class_means.append(class_mean)

        if class_means:
            metrics['cam_dice_macro'] = float(np.mean(class_means))

    return metrics