"""
FASE 2 & 3: PREPARATION + COLOR HARMONIZATION

Recupera cores unmultiplied e aplica harmonização local com Reinhard.
"""

import numpy as np
import cv2
from skimage import color
from typing import Tuple


def recover_unmultiplied(
    foreground: np.ndarray,
    alpha: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Recupera cores unmultiplied a partir de composita alpha-premultiplicada.
    
    Fórmula: fg_unmult = fg / clip(alpha, eps)
    
    Isto é necessário porque o alpha matting retorna fg * alpha.
    Para operações de cor, precisamos das cores "puras".
    
    Args:
        foreground: [H, W, 3] array com cores premultiplicadas
        alpha: [H, W] alpha map
        eps: valor mínimo para evitar divisão por zero
    
    Returns:
        fg_unmult: [H, W, 3] cores unmultiplied
    """
    alpha_clipped = np.clip(alpha, eps, 1.0)
    alpha_3ch = np.stack([alpha_clipped] * 3, axis=2)
    fg_unmult = np.clip(foreground / alpha_3ch, 0.0, 1.0)
    return fg_unmult


def bbox_from_mask(
    mask_hw: np.ndarray,
    pad: int = 24,
) -> Tuple[int, int, int, int]:
    """
    Extrai bounding box da máscara com padding.
    
    Args:
        mask_hw: [H, W] máscara
        pad: padding em pixels
    
    Returns:
        (y0, y1, x0, x1): bounding box coordinates
    """
    ys, xs = np.where(mask_hw > 0.01)
    if len(xs) == 0:
        h, w = mask_hw.shape
        return 0, h, 0, w
    
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = mask_hw.shape
    
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad)
    x1 = min(w, x1 + pad)
    
    return y0, y1, x0, x1


def soft_band_weight_map(
    alpha_hw: np.ndarray,
    band_px: int = 2,
    core_strength: float = 0.60,
    edge_strength: float = 0.90,
) -> np.ndarray:
    """
    Cria mapa de pesos espaciais suave: mais forte nas bordas, mais fraco no núcleo.
    
    Usa distance transform para saber distância de cada pixel à borda.
    - Pixels profundos no interior → core_strength
    - Pixels perto da borda → edge_strength
    - Transição suave (smoothstep)
    
    Args:
        alpha_hw: [H, W] alpha map
        band_px: largura da zona de transição
        core_strength: força na zona interior (0.0-1.0)
        edge_strength: força na zona de borda (0.0-1.0)
    
    Returns:
        weight_map: [H, W] mapa de pesos
    """
    mask_bin = (alpha_hw > 0.01).astype(np.uint8)
    if mask_bin.max() == 0:
        return np.zeros_like(alpha_hw, dtype=np.float32)
    
    # Distance transform: distância de cada pixel à borda
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5).astype(np.float32)
    maxd = float(dist.max() + 1e-6)
    dist_norm = dist / maxd
    
    # Smoothstep interpolation
    t = np.clip(dist_norm / max(band_px / maxd, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    
    # Mix core_strength e edge_strength
    weight = core_strength + (edge_strength - core_strength) * smooth
    weight[mask_bin == 0] = 0.0
    
    return weight.astype(np.float32)


def local_reinhard_soft(
    fg_unmult: np.ndarray,
    bg: np.ndarray,
    alpha_hw: np.ndarray,
    band_px: int = 2,
    core_strength: float = 0.60,
    edge_strength: float = 0.90,
    l_weight: float = 0.70,
) -> np.ndarray:
    """
    Harmonização local com Reinhard em LAB color space.
    
    Algoritmo:
    1. Extrai bounding box local da inserção
    2. Computa estatísticas (mean, std) no espaço LAB
    3. Aplica Reinhard transfer: (fg - mean_fg) / std_fg * std_bg + mean_bg
    4. Preserva L* (luminância) para manter perceção de profundidade
    5. Aplica mapa de pesos suave para manter núcleo intacto
    
    Args:
        fg_unmult: [H, W, 3] foreground unmultiplied
        bg: [H, W, 3] background
        alpha_hw: [H, W] alpha map
        band_px: largura da banda de harmonização
        core_strength: força no núcleo (preserva mais)
        edge_strength: força na borda (adapta mais ao background)
        l_weight: peso da preservação de L* (luminância)
    
    Returns:
        fg_corr: [H, W, 3] foreground harmonizado
    """
    fg_unmult = np.clip(fg_unmult, 0.0, 1.0)
    bg = np.clip(bg, 0.0, 1.0)
    alpha_hw = np.clip(alpha_hw, 0.0, 1.0)
    
    # Extrai região local (crop bbox)
    y0, y1, x0, x1 = bbox_from_mask(alpha_hw, pad=max(16, band_px * 4))
    fg_crop = fg_unmult[y0:y1, x0:x1]
    bg_crop = bg[y0:y1, x0:x1]
    alpha_crop = alpha_hw[y0:y1, x0:x1]
    
    # Máscara da região
    mask_crop = alpha_crop > 0.01
    if mask_crop.sum() < 20:
        mask_crop = alpha_crop > 0.0
    
    # Converte para LAB
    fg_lab = color.rgb2lab(fg_crop)
    bg_lab = color.rgb2lab(bg_crop)
    
    # Estatísticas
    src = fg_lab[mask_crop]
    tgt = bg_lab[mask_crop]
    if src.shape[0] < 5 or tgt.shape[0] < 5:
        src = fg_lab.reshape(-1, 3)
        tgt = bg_lab.reshape(-1, 3)
    
    mean_src = src.mean(axis=0)
    std_src = src.std(axis=0) + 1e-6
    mean_tgt = tgt.mean(axis=0)
    std_tgt = tgt.std(axis=0) + 1e-6
    
    # Reinhard transfer
    transferred = (fg_lab - mean_src) / std_src * std_tgt + mean_tgt
    
    # Preserva L* (luminância)
    transferred[:, :, 0] = l_weight * transferred[:, :, 0] + (1.0 - l_weight) * fg_lab[:, :, 0]
    transferred_rgb = np.clip(color.lab2rgb(transferred), 0.0, 1.0)
    
    # Mapa de pesos suave
    mix = soft_band_weight_map(alpha_crop, band_px=band_px, core_strength=core_strength, edge_strength=edge_strength)
    mix3 = np.stack([mix, mix, mix], axis=2)
    fg_corr_crop = fg_crop * (1.0 - mix3) + transferred_rgb * mix3
    
    # Aplica na imagem completa
    fg_corr = fg_unmult.copy()
    fg_corr[y0:y1, x0:x1] = fg_corr_crop
    
    return np.clip(fg_corr, 0.0, 1.0)
