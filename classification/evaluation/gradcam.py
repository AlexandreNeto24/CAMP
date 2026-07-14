"""Standard Grad-CAM utilities for CNN classifiers."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
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
            raise ValueError('GradCAM expects a single image tensor with shape [1, C, H, W].')

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
            raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_index


def get_resnet_target_layer(model):
    return model.backbone.layer4[-1]


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm

    image_rgb = image.convert('RGB')
    if heatmap.shape[:2] != (image_rgb.size[1], image_rgb.size[0]):
        heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image_rgb.size, Image.BILINEAR)
        heatmap = np.asarray(heatmap_image).astype(np.uint8) / 255.0

    heatmap = np.uint8(255 * heatmap)
    color_map = cm.get_cmap('jet')
    heatmap_rgb = color_map(heatmap / 255.0)[:, :, :3]
    heatmap_rgb = np.uint8(255 * heatmap_rgb)

    overlay = np.array(image_rgb).astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_cam_panel(image: Image.Image, heatmap: np.ndarray, output_path: Path, title: str = ''):
    import matplotlib.pyplot as plt

    overlay = overlay_heatmap(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.convert('RGB'))
    axes[0].set_title('Original')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')

    if title:
        fig.suptitle(title)

    for axis in axes:
        axis.axis('off')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def export_gradcam_examples(model, dataset, output_dir: Path, max_samples: int = 6, device: str = 'cpu'):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_resnet_target_layer(model)
    cam = GradCAM(model, target_layer)

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
            save_cam_panel(image, heatmap, output_dir / f'gradcam_{saved:03d}.png', title=title)
            saved += 1
            if saved >= max_samples:
                break
    finally:
        cam.close()
