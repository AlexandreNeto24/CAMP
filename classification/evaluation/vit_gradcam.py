# """Adaptive Grad-CAM utilities for torchvision ViT classifiers."""

# import math
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image

# from classification.evaluation.gradcam import save_cam_panel


# class AdaptiveViTGradCAM:
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
#             raise ValueError('AdaptiveViTGradCAM expects a single image tensor with shape [1, C, H, W].')

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
#             raise RuntimeError('ViT Grad-CAM hooks did not capture activations/gradients.')

#         if activations.dim() != 3:
#             raise RuntimeError(f'Unexpected ViT activation shape: {tuple(activations.shape)}')

#         token_scores = (activations * gradients).sum(dim=-1)
#         token_scores = token_scores[:, 1:]
#         token_scores = F.relu(token_scores)

#         num_tokens = token_scores.size(1)
#         grid_size = int(math.sqrt(num_tokens))
#         if grid_size * grid_size != num_tokens:
#             grid_size = int(round(math.sqrt(num_tokens)))

#         cam = token_scores.reshape(1, 1, grid_size, grid_size)
#         cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
#         cam = cam.squeeze().detach().cpu().numpy()
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
#         return cam, class_index


# def get_vit_target_layer(model):
#     encoder_layers = getattr(model.backbone.encoder, 'layers', None)
#     if encoder_layers is None:
#         raise AttributeError('Could not locate ViT encoder layers.')
#     return list(encoder_layers.children())[-1]


# def export_vit_gradcam_examples(
#     model, 
#     dataset, 
#     output_dir: Path, 
#     samples_per_class: dict = {0: 3, 1: 3}, 
#     device: str = 'cpu'
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     target_layer = get_vit_target_layer(model)
#     cam = AdaptiveViTGradCAM(model, target_layer)

#     # Inicializar contadores para cada classe
#     saved_counts = {cls: 0 for cls in samples_per_class.keys()}
#     total_to_save = sum(samples_per_class.values())
#     total_saved = 0

#     try:
#         for index in range(len(dataset)):
#             sample = dataset[index]
#             if sample.get('source') not in {'local', None}:
#                 continue
                
#             label = int(sample['label'].item())

#             # Se a classe não está no dicionário ou já atingimos o limite dela, salta para a próxima
#             if label not in samples_per_class or saved_counts[label] >= samples_per_class[label]:
#                 continue

#             image_path = Path(sample['path'])
#             image = Image.open(image_path).convert('RGB')
#             image_tensor = sample['image'].unsqueeze(0).to(device)
#             heatmap, pred_class = cam(image_tensor)

#             title = f'idx={index} label={label} pred={pred_class}'
            
#             # Formatar o nome do ficheiro: vit_gradcam_class0_000.png
#             filename = f'vit_gradcam_class{label}_{saved_counts[label]:03d}.png'
#             save_cam_panel(image, heatmap, output_dir / filename, title=title)
            
#             # Atualizar os contadores
#             saved_counts[label] += 1
#             total_saved += 1
            
#             # Quebrar o loop quando as quotas todas estiverem cheias
#             if total_saved >= total_to_save:
#                 break
#     finally:
#         cam.close()

# """Grad-CAM utilities for torchvision ViT classifiers using pytorch-grad-cam."""

# import math
# from pathlib import Path

# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt

# # Importações da biblioteca oficial
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image


# def reshape_transform(tensor):
#     """
#     Transforma a sequência de tokens do ViT (B, N, C) numa grelha 2D (B, C, H, W).
#     """
#     # 1. Descartar o CLS token (o primeiro elemento)
#     result = tensor[:, 1:, :]
    
#     # 2. Calcular o tamanho da grelha (ex: 196 tokens -> 14x14)
#     num_tokens = result.size(1)
#     grid_size = int(math.sqrt(num_tokens))
    
#     # 3. Fazer o reshape para a grelha e trocar as dimensões para [Batch, Channels, H, W]
#     result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
#     result = result.transpose(2, 3).transpose(1, 2)
    
#     return result


# def get_vit_target_layer(model):
#     """
#     No torchvision, a melhor camada para intercetar num ViT é a LayerNorm final
#     do último bloco do encoder.
#     """
#     return [model.backbone.encoder.layers[-1].ln_1]


# def save_vit_cam_panel(image: Image.Image, cam_heatmap: np.ndarray, output_path: Path, title: str = ''):
#     """
#     Gera o painel com as 3 vistas (Original, Heatmap, Overlay).
#     """
#     # 1. Obter as dimensões do heatmap gerado pelo modelo (ex: 224x224)
#     heatmap_h, heatmap_w = cam_heatmap.shape
    
#     # 2. Redimensionar a imagem original PIL para coincidir perfeitamente com o heatmap
#     # NOTA: Usar Image.Resampling.BILINEAR ou Image.BILINEAR dependendo da tua versão do Pillow
#     image_resized = image.resize((heatmap_w, heatmap_h), Image.BILINEAR)
    
#     # 3. Converter para float e gerar o Overlay
#     img_float = np.float32(image_resized.convert('RGB')) / 255.0
#     overlay = show_cam_on_image(img_float, cam_heatmap, use_rgb=True)

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
#     # Usar a imagem já redimensionada no painel para ficar coerente ao lado do heatmap
#     axes[0].imshow(image_resized.convert('RGB'))
#     axes[0].set_title('Original')
    
#     # Mostrar apenas o mapa de calor isolado (usando o colormap 'jet' por padrão)
#     axes[1].imshow(cam_heatmap, cmap='jet')
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

# def export_vit_gradcam_examples(
#     model, 
#     dataset, 
#     output_dir: Path, 
#     samples_per_class: dict = {0: 3, 1: 3}, 
#     device: str = 'cpu'
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     target_layers = get_vit_target_layer(model)
    
#     # Inicializar o objeto GradCAM da biblioteca oficial
#     cam = GradCAM(
#         model=model, 
#         target_layers=target_layers, 
#         reshape_transform=reshape_transform
#     )

#     saved_counts = {cls: 0 for cls in samples_per_class.keys()}
#     total_to_save = sum(samples_per_class.values())
#     total_saved = 0

#     try:
#         for index in range(len(dataset)):
#             sample = dataset[index]
            
#             if sample.get('source') not in {'local', None}:
#                 continue

#             label = int(sample['label'].item())

#             if label not in samples_per_class or saved_counts[label] >= samples_per_class[label]:
#                 continue

#             image_path = Path(sample['path'])
#             image = Image.open(image_path).convert('RGB')
#             image_tensor = sample['image'].unsqueeze(0).to(device)
            
#             # Com a biblioteca, não precisas de calcular o argmax manualmente, 
#             # se não passares targets ele calcula a explicação para a classe de maior score!
#             grayscale_cam = cam(input_tensor=image_tensor)[0, :]
            
#             # Para obtermos a previsão real, fazemos apenas uma inferência rápida
#             with torch.no_grad():
#                 logits = model(image_tensor)
#                 pred_class = int(logits.argmax(dim=1).item())

#             title = f'idx={index} label={label} pred={pred_class}'
#             filename = f'vit_gradcam_class{label}_{saved_counts[label]:03d}.png'
#             save_path = output_dir / filename
            
#             save_vit_cam_panel(image, grayscale_cam, save_path, title=title)
            
#             saved_counts[label] += 1
#             total_saved += 1

#             if total_saved >= total_to_save:
#                 break
#     finally:
#         # Libertar memória no final do processo
#         pass

"""Grad-CAM utilities for torchvision ViT classifiers using pytorch-grad-cam."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    num_tokens = result.size(1)
    grid_size = int(math.sqrt(num_tokens))
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_vit_target_layer(model):
    return [model.backbone.encoder.layers[-1].ln_1]


def calculate_cam_dice(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.3) -> float:
    cam_bin = (cam > threshold).astype(bool)
    mask_bin = (mask > 0).astype(bool)
    
    intersection = np.logical_and(cam_bin, mask_bin).sum()
    total = cam_bin.sum() + mask_bin.sum()
    
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


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
    samples_per_class: dict = {0: 3, 1: 3}, 
    device: str = 'cpu'
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_layers = get_vit_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    saved_counts = {cls: 0 for cls in samples_per_class.keys()}
    dice_scores = []

    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            
            if sample.get('source') not in {'local', None}:
                continue

            label = int(sample['label'].item())

            needs_visual = (label in samples_per_class and saved_counts[label] < samples_per_class[label])
            needs_metric = (label == 1)

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
                    mask_img = mask_pil.resize(
                        (grayscale_cam.shape[1], grayscale_cam.shape[0]), 
                        Image.NEAREST
                    )
                    mask_np = np.array(mask_img) / 255.0
                    
                    dice_score = calculate_cam_dice(grayscale_cam, mask_np, threshold=0.3)
                    title += f' | Dice={dice_score:.3f}'
                    dice_scores.append(dice_score)

            # ─── 2. EXPORTAÇÃO VISUAL APENAS PARA A QUOTA ───
            if needs_visual:
                filename = f'vit_gradcam_class{label}_{saved_counts[label]:03d}.png'
                save_path = output_dir / filename
                save_vit_cam_panel(image, grayscale_cam, save_path, title=title, mask=mask_np)
                saved_counts[label] += 1

    finally:
        pass
        
    if dice_scores:
        return {'cam_dice': np.mean(dice_scores)}
    return {}