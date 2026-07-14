"""Adaptive Grad-CAM utilities for torchvision ViT classifiers."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classification.evaluation.gradcam import save_cam_panel


class AdaptiveViTGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_index: Optional[int] = None):
        if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
            raise ValueError('AdaptiveViTGradCAM expects a single image tensor with shape [1, C, H, W].')

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[:, class_index].sum()
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients
        if activations is None or gradients is None:
            raise RuntimeError('ViT Grad-CAM hooks did not capture activations/gradients.')

        if activations.dim() != 3:
            raise RuntimeError(f'Unexpected ViT activation shape: {tuple(activations.shape)}')

        token_scores = (activations * gradients).sum(dim=-1)
        token_scores = token_scores[:, 1:]
        token_scores = F.relu(token_scores)

        num_tokens = token_scores.size(1)
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            grid_size = int(round(math.sqrt(num_tokens)))

        cam = token_scores.reshape(1, 1, grid_size, grid_size)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_index


def get_vit_target_layer(model):
    encoder_layers = getattr(model.backbone.encoder, 'layers', None)
    if encoder_layers is None:
        raise AttributeError('Could not locate ViT encoder layers.')
    return list(encoder_layers.children())[-1]


def export_vit_gradcam_examples(model, dataset, output_dir: Path, max_samples: int = 6, device: str = 'cpu'):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_vit_target_layer(model)
    cam = AdaptiveViTGradCAM(model, target_layer)

    try:
        saved = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample.get('source') not in {'local', None}:
                continue

            image_path = Path(sample['path'])
            image = Image.open(image_path).convert('RGB')
            image_tensor = sample['image'].unsqueeze(0).to(device)
            heatmap, pred_class = cam(image_tensor)

            label = int(sample['label'].item())
            title = f'idx={index} label={label} pred={pred_class}'
            save_cam_panel(image, heatmap, output_dir / f'vit_gradcam_{saved:03d}.png', title=title)
            saved += 1
            if saved >= max_samples:
                break
    finally:
        cam.close()
