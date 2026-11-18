"""Unmultiply the foreground and harmonize its colour locally.

Colour transfer is Reinhard in CIELAB, limited to the paste and mixed
with a spatial weight so the boundary and the interior can differ.
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
    """Recover unmultiplied colours from an alpha-premultiplied composite.

    ``fg_unmult = fg / clip(alpha, eps)``. Alpha matting returns
    ``foreground * alpha``. Colour transfer needs the colours themselves.

    Args:
        foreground: ``[H, W, 3]`` premultiplied colours.
        alpha: ``[H, W]`` alpha map.
        eps: Lower clamp on alpha, to avoid division by zero.

    Returns:
        Unmultiplied foreground, ``[H, W, 3]``, clipped to ``[0, 1]``.
    """
    alpha_clipped = np.clip(alpha, eps, 1.0)
    alpha_3ch = np.stack([alpha_clipped] * 3, axis=2)
    fg_unmult = np.clip(foreground / alpha_3ch, 0.0, 1.0)
    return fg_unmult


def bbox_from_mask(
    mask_hw: np.ndarray,
    pad: int = 24,
) -> Tuple[int, int, int, int]:
    """Bounding box of a mask, expanded by ``pad`` pixels and clipped.

    Args:
        mask_hw: ``[H, W]`` mask.
        pad: Pixels added on each side.

    Returns:
        ``(y0, y1, x0, x1)``. The full frame when the mask is empty.
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
    """Spatial mix of ``core_strength`` and ``edge_strength``.

    Distance to the mask boundary is 0 on the edge and largest in the
    interior. The weight is ``core_strength`` on that edge and moves to
    ``edge_strength`` across ``band_px``. The transition is a smoothstep.
    Pixels outside the mask stay at 0.

    Args:
        alpha_hw: ``[H, W]`` alpha map.
        band_px: Width of the transition, in pixels.
        core_strength: Weight on the mask boundary, in ``[0, 1]``.
        edge_strength: Weight inside the mask, in ``[0, 1]``.

    Returns:
        ``[H, W]`` weight map.
    """
    mask_bin = (alpha_hw > 0.01).astype(np.uint8)
    if mask_bin.max() == 0:
        return np.zeros_like(alpha_hw, dtype=np.float32)
    
    # Distance from each interior pixel to the mask boundary.
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5).astype(np.float32)
    maxd = float(dist.max() + 1e-6)
    dist_norm = dist / maxd
    
    # Smoothstep interpolation
    t = np.clip(dist_norm / max(band_px / maxd, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    
    # Mix core_strength on the boundary with edge_strength inside.
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
    """Local Reinhard colour transfer in CIELAB.

    Steps:
    1. Crop to the mask bounding box.
    2. Match mean and standard deviation in CIELAB.
    3. ``(fg - mean_fg) / std_fg * std_bg + mean_bg``.
    4. Keep a share of the original L* so luminance does not jump.
    5. Mix the transferred colour with the original using
       ``soft_band_weight_map``.

    Args:
        fg_unmult: ``[H, W, 3]`` unmultiplied foreground.
        bg: ``[H, W, 3]`` background.
        alpha_hw: ``[H, W]`` alpha map.
        band_px: Width of the spatial mix, in pixels.
        core_strength: Harmonization weight on the mask boundary.
        edge_strength: Harmonization weight inside the mask.
        l_weight: Share of the transferred L* that replaces the original.

    Returns:
        Harmonized foreground, ``[H, W, 3]``.
    """
    fg_unmult = np.clip(fg_unmult, 0.0, 1.0)
    bg = np.clip(bg, 0.0, 1.0)
    alpha_hw = np.clip(alpha_hw, 0.0, 1.0)
    
    # Local crop around the paste.
    y0, y1, x0, x1 = bbox_from_mask(alpha_hw, pad=max(16, band_px * 4))
    fg_crop = fg_unmult[y0:y1, x0:x1]
    bg_crop = bg[y0:y1, x0:x1]
    alpha_crop = alpha_hw[y0:y1, x0:x1]
    
    # Pixels that belong to the paste.
    mask_crop = alpha_crop > 0.01
    if mask_crop.sum() < 20:
        mask_crop = alpha_crop > 0.0
    
    # CIELAB.
    fg_lab = color.rgb2lab(fg_crop)
    bg_lab = color.rgb2lab(bg_crop)
    
    # Mean and standard deviation inside the paste.
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
    
    # Keep a share of the original luminance.
    transferred[:, :, 0] = l_weight * transferred[:, :, 0] + (1.0 - l_weight) * fg_lab[:, :, 0]
    transferred_rgb = np.clip(color.lab2rgb(transferred), 0.0, 1.0)
    
    # Spatial mix of the original crop and the transferred colour.
    mix = soft_band_weight_map(alpha_crop, band_px=band_px, core_strength=core_strength, edge_strength=edge_strength)
    mix3 = np.stack([mix, mix, mix], axis=2)
    fg_corr_crop = fg_crop * (1.0 - mix3) + transferred_rgb * mix3
    
    # Write the crop back into the full frame.
    fg_corr = fg_unmult.copy()
    fg_corr[y0:y1, x0:x1] = fg_corr_crop
    
    return np.clip(fg_corr, 0.0, 1.0)
