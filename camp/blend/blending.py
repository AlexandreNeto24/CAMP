"""
FASE 4: BLENDING (Pyramid-based multiscale blending)

Blend smoothly using Laplacian pyramid.
Alpha é aplicado aqui para obter o resultado final.
"""

import numpy as np
import cv2
import torch
from typing import Union, List, Tuple


def build_gaussian_pyramid(
    image: np.ndarray,
    levels: int = 5,
) -> List[np.ndarray]:
    """
    Constrói Gaussian pyramid (downsampling progressivo).
    
    Args:
        image: [H, W, 3] ou [H, W] array
        levels: número de níveis
    
    Returns:
        pyramid: lista de imagens em diferentes escalas
    """
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_laplacian_pyramid(
    image: np.ndarray,
    levels: int = 5,
) -> List[np.ndarray]:
    """
    Constrói Laplacian pyramid (diferenças entre níveis Gaussianos).
    
    Captura detalhes em cada escala: L[i] = G[i] - expand(G[i+1])
    
    Args:
        image: [H, W, 3] ou [H, W] array
        levels: número de níveis
    
    Returns:
        laplacian: lista de piramides Laplacianas
    """
    gaussian = build_gaussian_pyramid(image, levels)
    laplacian = []
    
    for i in range(levels - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacian.append(gaussian[i] - expanded)
    
    laplacian.append(gaussian[-1])
    return laplacian


def pyramid_mask_blend(
    foreground: Union[torch.Tensor, np.ndarray],
    alpha: Union[torch.Tensor, np.ndarray],
    background: Union[torch.Tensor, np.ndarray],
    feather_px: int = 1,
    levels: int = 4,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Blending multiscala com pyramid Laplaciana.
    
    Algoritmo:
    1. Constrói Laplacian pyramids para foreground, background
    2. Constrói Gaussian pyramid para alpha
    3. Blenda em cada nível: L_blend[i] = alpha[i] * L_fg[i] + (1 - alpha[i]) * L_bg[i]
    4. Reconstrói a partir da pyramid blendada
    
    Vantagem: blending suave em múltiplas escalas, evita seams.
    
    Args:
        foreground: [3, H, W] ou [H, W, 3] tensor/array (NÃO pré-multiplicado!)
        alpha: [1, H, W] ou [H, W] mask
        background: [3, H, W] ou [H, W, 3] tensor/array
        feather_px: pixels para feathering (não usado agora, mantido para compatibilidade)
        levels: níveis do pyramid
    
    Returns:
        blended: [3, H, W] tensor/array resultado final
    """
    
    # Converte para numpy HWC
    if isinstance(foreground, torch.Tensor):
        if foreground.ndim == 3 and foreground.shape[0] == 3:
            fg_np = foreground.permute(1, 2, 0).numpy().astype(np.float32)
        else:
            fg_np = foreground.numpy().astype(np.float32)
        is_torch = True
    else:
        fg_np = foreground.astype(np.float32)
        is_torch = False
    
    if isinstance(background, torch.Tensor):
        if background.ndim == 3 and background.shape[0] == 3:
            bg_np = background.permute(1, 2, 0).numpy().astype(np.float32)
        else:
            bg_np = background.numpy().astype(np.float32)
    else:
        bg_np = background.astype(np.float32)
    
    if isinstance(alpha, torch.Tensor):
        if alpha.ndim == 3:
            alpha_np = alpha.squeeze(0).numpy().astype(np.float32)
        else:
            alpha_np = alpha.numpy().astype(np.float32)
    else:
        alpha_np = alpha.astype(np.float32)
    
    # Clipa para [0, 1]
    fg_np = np.clip(fg_np, 0, 1)
    bg_np = np.clip(bg_np, 0, 1)
    alpha_np = np.clip(alpha_np, 0, 1)
    
    # Se alpha 2D, expande para 3 canais
    if alpha_np.ndim == 2:
        alpha_3ch = np.stack([alpha_np] * 3, axis=2)
    else:
        alpha_3ch = alpha_np
    
    # Constrói pyramids
    fg_lap = build_laplacian_pyramid(fg_np, levels)
    bg_lap = build_laplacian_pyramid(bg_np, levels)
    alpha_gauss = build_gaussian_pyramid(alpha_3ch if alpha_3ch.ndim == 3 else np.stack([alpha_np]*3, axis=2), levels)
    
    # Blenda em cada nível
    blended_lap = []
    for i in range(len(fg_lap)):
        alpha_i = alpha_gauss[i]
        if alpha_i.ndim == 2:
            alpha_i = np.stack([alpha_i] * 3, axis=2)
        
        # Garante shapes compatíveis
        alpha_i = np.clip(alpha_i, 0, 1)
        blended_level = fg_lap[i] * alpha_i + bg_lap[i] * (1 - alpha_i)
        blended_lap.append(blended_level)
    
    # Reconstrói a partir da pyramid blendada
    result = blended_lap[-1]
    for i in range(len(blended_lap) - 2, -1, -1):
        size = (blended_lap[i].shape[1], blended_lap[i].shape[0])
        expanded = cv2.pyrUp(result, dstsize=size)
        result = expanded + blended_lap[i]
    
    result = np.clip(result, 0, 1)
    
    # Converte de volta se era tensor
    if is_torch:
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).float()
        return result_tensor
    
    return result
