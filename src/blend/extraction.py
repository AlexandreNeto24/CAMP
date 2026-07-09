"""
FASE 1: EXTRACTION (Alpha Matting)

Extrai a instância com alpha matting baseado em trimap.
Input: lesion_image (raw) + mask
Output: foreground (unmultiplied) + alpha
"""

import numpy as np
import cv2
import torch
from typing import Tuple


# def trimap_based_matting(
#     lesion_image: torch.Tensor,
#     mask: torch.Tensor,
#     erosion_kernel: int = 5,
#     dilation_kernel: int = 15,
#     blur_sigma: float = 8.0,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Extrai instância com alpha matting baseado em trimap.
    
#     O trimap divide a imagem em: foreground (erosão), background (dilatação), e transição.
#     O alpha é suavizado com Gaussian blur para transições naturais.
    
#     Args:
#         lesion_image: [3, H, W] tensor, valores em [0, 1] ou [0, 255]
#         mask: [1, H, W] tensor, máscara binária
#         erosion_kernel: tamanho do kernel de erosão (foreground puro)
#         dilation_kernel: tamanho do kernel de dilatação (background puro)
#         blur_sigma: sigma do Gaussian blur para suavidade
    
#     Returns:
#         alpha: [1, H, W] tensor, alpha map suavizado
#         foreground: [3, H, W] tensor, instância extraída
#     """
    
#     # Converte para numpy
#     print(type(lesion_image), lesion_image.shape, lesion_image.dtype)
#     img_np = lesion_image.permute(1, 2, 0).numpy()
#     if img_np.dtype in [np.uint8] or np.max(img_np) > 2.0:
#         img_np = (img_np.astype(np.float32) / 255.0) if np.max(img_np) > 1.0 else img_np.astype(np.float32)
#     else:
#         img_np = img_np.astype(np.float32)
    
#     mask_np = mask.squeeze(0).numpy()
#     if mask_np.dtype == np.uint8 or np.max(mask_np) > 1.0:
#         mask_np = (mask_np.astype(np.float32) / 255.0) if np.max(mask_np) > 1.0 else mask_np.astype(np.float32)
#     mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    
#     # Trimap: foreground puro (erosão) + background puro (dilatação)
#     kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel))
#     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    
#     fg_region = cv2.erode(mask_np, kernel_erode, iterations=1)
#     bg_region = cv2.dilate(mask_np, kernel_dilate, iterations=1)
    
#     # Alpha suavizado com Gaussian blur
#     alpha_smooth = cv2.GaussianBlur(
#         mask_np.astype(np.float32) / 255.0,
#         (15, 15),
#         blur_sigma
#     )
#     alpha = np.clip(alpha_smooth, 0.0, 1.0)
    
#     # Composita: fg com alpha aplicado
#     alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
#     foreground = img_np * alpha_3ch
    
#     # Converte para tensores
#     alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).float()
#     fg_tensor = torch.from_numpy(foreground).permute(2, 0, 1).float()
    
#     return alpha_tensor, fg_tensor

def trimap_based_matting(
    lesion_image: torch.Tensor,
    mask: torch.Tensor,
    erosion_kernel: int = 5,
    dilation_kernel: int = 15,
    blur_sigma: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrai instância com alpha matting baseado em trimap.
    Aceita tanto Tensores PyTorch como Arrays NumPy.
    """
    
    # ─── 1. TRATAR A IMAGEM DA LESÃO ───
    if isinstance(lesion_image, torch.Tensor):
        # Se for tensor (CHW), move para CPU, tira o gradiente, permuta para HWC e converte para numpy
        img_np = lesion_image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        # Se já for numpy, assumimos que já vem em HWC (ex: do OpenCV)
        img_np = lesion_image.copy()

    # Normalizar para [0.0, 1.0]
    if img_np.dtype == np.uint8 or np.max(img_np) > 2.0:
        img_np = (img_np.astype(np.float32) / 255.0) if np.max(img_np) > 1.0 else img_np.astype(np.float32)
    else:
        img_np = img_np.astype(np.float32)
    
    # ─── 2. TRATAR A MÁSCARA ───
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().squeeze().numpy()
    else:
        mask_np = mask.copy().squeeze()

    # Converter máscara para binário [0, 255]
    if mask_np.dtype == np.uint8 or np.max(mask_np) > 1.0:
        mask_np = (mask_np.astype(np.float32) / 255.0) if np.max(mask_np) > 1.0 else mask_np.astype(np.float32)
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    
    # ─── 3. LÓGICA DE MATTING E BLUR (Inalterado) ───
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    
    fg_region = cv2.erode(mask_np, kernel_erode, iterations=1)
    bg_region = cv2.dilate(mask_np, kernel_dilate, iterations=1)
    
    # Alpha suavizado com Gaussian blur
    alpha_smooth = cv2.GaussianBlur(
        mask_np.astype(np.float32) / 255.0,
        (15, 15),
        blur_sigma
    )
    alpha = np.clip(alpha_smooth, 0.0, 1.0)
    
    # Composita: fg com alpha aplicado
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    foreground = img_np * alpha_3ch
    
    # ─── 4. DEVOLVER COMO TENSORES ───
    alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).float()
    fg_tensor = torch.from_numpy(foreground).permute(2, 0, 1).float()
    
    return alpha_tensor, fg_tensor
