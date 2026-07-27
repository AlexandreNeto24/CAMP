# """Standard Grad-CAM utilities for CNN classifiers."""

# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image


# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None
#         self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
#         self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

#     def _forward_hook(self, module, inputs, output):
#         self.activations = output

#     def _backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def close(self):
#         self.forward_handle.remove()
#         self.backward_handle.remove()

#     def __call__(self, image_tensor: torch.Tensor, class_index: Optional[int] = None):
#         if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
#             raise ValueError('GradCAM expects a single image tensor with shape [1, C, H, W].')

#         self.model.eval()
#         self.model.zero_grad(set_to_none=True)
#         logits = self.model(image_tensor)
#         if class_index is None:
#             class_index = int(logits.argmax(dim=1).item())

#         score = logits[:, class_index].sum()
#         score.backward(retain_graph=True)

#         activations = self.activations
#         gradients = self.gradients
#         if activations is None or gradients is None:
#             raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

#         weights = gradients.mean(dim=(2, 3), keepdim=True)
#         cam = (weights * activations).sum(dim=1, keepdim=True)
#         cam = F.relu(cam)
#         cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
#         cam = cam.squeeze().detach().cpu().numpy()
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
#         return cam, class_index


# def get_resnet_target_layer(model):
#     return model.backbone.layer4[-1]


# def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
#     import matplotlib.cm as cm

#     image_rgb = image.convert('RGB')
#     if heatmap.shape[:2] != (image_rgb.size[1], image_rgb.size[0]):
#         heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image_rgb.size, Image.BILINEAR)
#         heatmap = np.asarray(heatmap_image).astype(np.uint8) / 255.0

#     heatmap = np.uint8(255 * heatmap)
#     color_map = cm.get_cmap('jet')
#     heatmap_rgb = color_map(heatmap / 255.0)[:, :, :3]
#     heatmap_rgb = np.uint8(255 * heatmap_rgb)

#     overlay = np.array(image_rgb).astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     return Image.fromarray(overlay)


# def save_cam_panel(image: Image.Image, heatmap: np.ndarray, output_path: Path, title: str = ''):
#     import matplotlib.pyplot as plt

#     overlay = overlay_heatmap(image, heatmap)

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     axes[0].imshow(image.convert('RGB'))
#     axes[0].set_title('Original')
#     axes[1].imshow(heatmap, cmap='jet')
#     axes[1].set_title('Heatmap')
#     axes[2].imshow(overlay)
#     axes[2].set_title('Overlay')

#     if title:
#         fig.suptitle(title)

#     for axis in axes:
#         axis.axis('off')

#     fig.tight_layout()
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=180, bbox_inches='tight')
#     plt.close(fig)


# def export_gradcam_examples(
#     model, 
#     dataset, 
#     output_dir: Path, 
#     samples_per_class: dict = {0: 3, 1: 3},  # Ex: 3 normais, 3 pólipos
#     device: str = 'cpu'
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     target_layer = get_resnet_target_layer(model)
#     cam = GradCAM(model, target_layer)

#     # Inicializar contadores para cada classe pedida
#     saved_counts = {cls: 0 for cls in samples_per_class.keys()}
#     total_to_save = sum(samples_per_class.values())
#     total_saved = 0

#     try:
#         for index in range(len(dataset)):
#             sample = dataset[index]
            
#             # Filtro original de source
#             if sample.get('source') not in {'local', None}:
#                 continue

#             label = int(sample['label'].item())

#             # Se a classe atual não nos interessa ou se já atingimos a quota para ela, avança
#             if label not in samples_per_class or saved_counts[label] >= samples_per_class[label]:
#                 continue

#             image_path = Path(sample['path'])
#             image = Image.open(image_path).convert('RGB')
#             image_tensor = sample['image'].unsqueeze(0).to(device)
#             heatmap, pred_class = cam(image_tensor)

#             title = f'idx={index} label={label} pred={pred_class}'
            
#             # Nome do ficheiro mais descritivo: ex -> gradcam_class1_000.png
#             filename = f'gradcam_class{label}_{saved_counts[label]:03d}.png'
#             save_cam_panel(image, heatmap, output_dir / filename, title=title)
            
#             # Atualizar contadores
#             saved_counts[label] += 1
#             total_saved += 1

#             # Terminar se já tivermos todas as imagens de todas as classes
#             if total_saved >= total_to_save:
#                 break
                
#     finally:
#         cam.close()

"""Standard Grad-CAM utilities for CNN classifiers."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def calculate_cam_dice(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.3) -> float:
    cam_bin = (cam > threshold).astype(bool)
    mask_bin = (mask > 0).astype(bool)
    
    intersection = np.logical_and(cam_bin, mask_bin).sum()
    total = cam_bin.sum() + mask_bin.sum()
    
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
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


def save_cam_panel(image: Image.Image, heatmap: np.ndarray, output_path: Path, title: str = '', mask: Optional[np.ndarray] = None):
    overlay = overlay_heatmap(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.convert('RGB'))
    axes[0].set_title('Original')
    
    axes[1].imshow(heatmap, cmap='jet')
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


def export_gradcam_examples(
    model, 
    dataset, 
    output_dir: Path, 
    samples_per_class: dict = {0: 3, 1: 3}, 
    device: str = 'cpu'
):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_resnet_target_layer(model)
    cam = GradCAM(model, target_layer)

    saved_counts = {cls: 0 for cls in samples_per_class.keys()}
    dice_scores = []

    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            
            if sample.get('source') not in {'local', None}:
                continue

            label = int(sample['label'].item())

            # VERIFICAÇÃO DUPLA:
            # Precisamos de gravar a imagem?
            needs_visual = (label in samples_per_class and saved_counts[label] < samples_per_class[label])
            # Precisamos da métrica? (Todas as lesões reais)
            needs_metric = (label == 1)

            # Se não precisamos nem da imagem visual nem da métrica, saltamos à frente
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

            # ─── 1. CÁLCULO DA MÉTRICA PARA TODAS AS IMAGENS COM LABEL 1 ───
            if label == 1:
                dataset_root = image_path.parent.parent
                # mask path that can be in png or jpg format
                mask_path = dataset_root / 'masks' / image_path.with_suffix('.png').name
                if not mask_path.exists():
                    mask_path = dataset_root / 'masks' / image_path.with_suffix('.jpg').name
                # mask_path = dataset_root / 'masks' / image_path.name
                
                if mask_path.exists():
                    mask_pil = Image.open(mask_path).convert('L')
                    mask_img = mask_pil.resize((image.width, image.height), Image.NEAREST)
                    mask_np = np.array(mask_img) / 255.0
                    
                    dice_score = calculate_cam_dice(heatmap, mask_np, threshold=0.3)
                    title += f' | Dice={dice_score:.3f}'
                    dice_scores.append(dice_score)

            # ─── 2. EXPORTAÇÃO VISUAL APENAS PARA A QUOTA ───
            if needs_visual:
                filename = f'gradcam_class{label}_{saved_counts[label]:03d}.png'
                save_cam_panel(image, heatmap, output_dir / filename, title=title, mask=mask_np)
                saved_counts[label] += 1
                
        # Removido o break prematuro para garantir que iteramos por todo o dataset de teste!
                
    finally:
        cam.close()
        
    if dice_scores:
        return {'cam_dice': np.mean(dice_scores)}
    return {}