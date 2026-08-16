"""Multiclass Grad-CAM utilities for torchvision ViT classifiers."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from classification.evaluation.vit_gradcam import calculate_cam_dice


def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    num_tokens = result.size(1)
    grid_size = int(math.sqrt(num_tokens))
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_vit_target_layer(model):
    return [model.backbone.encoder.layers[-1].ln_1]


def _load_mask_for_image(image_path: Path):
    dataset_root = image_path.parent.parent
    mask_path = dataset_root / 'masks' / image_path.with_suffix('.png').name
    if not mask_path.exists():
        mask_path = dataset_root / 'masks' / image_path.with_suffix('.jpg').name
    if not mask_path.exists():
        return None
    return Image.open(mask_path).convert('L')


def save_vit_cam_panel(image: Image.Image, cam_heatmap: np.ndarray, output_path: Path, title: str = '', mask: Optional[np.ndarray] = None):
    heatmap_h, heatmap_w = cam_heatmap.shape
    image_resized = image.resize((heatmap_w, heatmap_h), Image.BILINEAR)

    img_float = np.float32(image_resized.convert('RGB')) / 255.0
    overlay = show_cam_on_image(img_float, cam_heatmap, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_resized.convert('RGB'))
    axes[0].set_title('Original')

    axes[1].imshow(cam_heatmap, cmap='jet')
    axes[1].set_title('Heatmap')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')

    if mask is not None:
        for ax in axes:
            ax.contour(mask, levels=[0.5], colors=['lime'], linewidths=2, alpha=0.8)

    if title:
        fig.suptitle(title)

    for axis in axes:
        axis.axis('off')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def export_vit_gradcam_examples(
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

    target_layers = get_vit_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

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

            grayscale_cam = cam(input_tensor=image_tensor)[0, :]

            with torch.no_grad():
                logits = model(image_tensor)
                pred_class = int(logits.argmax(dim=1).item())

            title = f'idx={index} label={label} pred={pred_class}'
            mask_np = None

            if needs_metric:
                mask_pil = _load_mask_for_image(image_path)
                if mask_pil is not None:
                    mask_img = mask_pil.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]), Image.NEAREST)
                    mask_np = np.array(mask_img) / 255.0
                    dice_score = calculate_cam_dice(grayscale_cam, mask_np, threshold=0.3)
                    title += f' | Dice={dice_score:.3f}'
                    sample_dice_scores.append(dice_score)
                    dice_scores_by_class.setdefault(label, []).append(dice_score)

            if needs_visual:
                filename = f'vit_gradcam_class{label}_{saved_counts[label]:03d}.png'
                save_path = output_dir / filename
                save_vit_cam_panel(image, grayscale_cam, save_path, title=title, mask=mask_np)
                saved_counts[label] += 1

    finally:
        pass

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