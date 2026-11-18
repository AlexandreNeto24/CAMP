"""Laplacian pyramid blend.

The alpha mask is applied here. It produces the final composite.
"""

import numpy as np
import cv2
import torch
from typing import Union, List, Tuple


def build_gaussian_pyramid(
    image: np.ndarray,
    levels: int = 5,
) -> List[np.ndarray]:
    """Build a Gaussian pyramid by repeated downsampling.

    Args:
        image: ``[H, W, 3]`` or ``[H, W]``.
        levels: Number of levels, including the original image.

    Returns:
        Images from the finest scale to the coarsest.
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
    """Build a Laplacian pyramid from differences between Gaussian levels.

    Each level stores the detail at that scale:
    ``L[i] = G[i] - expand(G[i + 1])``. The coarsest level is kept as is.

    Args:
        image: ``[H, W, 3]`` or ``[H, W]``.
        levels: Number of levels.

    Returns:
        Laplacian levels, finest first.
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
    """Blend a foreground onto a background with a Laplacian pyramid.

    Steps:
    1. Laplacian pyramids of the foreground and the background.
    2. Gaussian pyramid of the alpha mask.
    3. At each level, ``L = alpha * L_fg + (1 - alpha) * L_bg``.
    4. Reconstruct the image from the blended pyramid.

    Blending at several scales avoids a hard seam.

    Args:
        foreground: ``[3, H, W]`` or ``[H, W, 3]``. Must not be premultiplied.
        alpha: ``[1, H, W]`` or ``[H, W]``.
        background: ``[3, H, W]`` or ``[H, W, 3]``.
        feather_px: Kept for callers. The pyramid does not use it.
        levels: Number of pyramid levels.

    Returns:
        The composite as ``[3, H, W]``, a tensor when ``foreground`` was one.
    """
    
    # NumPy images in HWC.
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
    
    # Clip to [0, 1].
    fg_np = np.clip(fg_np, 0, 1)
    bg_np = np.clip(bg_np, 0, 1)
    alpha_np = np.clip(alpha_np, 0, 1)
    
    # A 2D alpha is repeated across the three channels.
    if alpha_np.ndim == 2:
        alpha_3ch = np.stack([alpha_np] * 3, axis=2)
    else:
        alpha_3ch = alpha_np
    
    # Build the pyramids.
    fg_lap = build_laplacian_pyramid(fg_np, levels)
    bg_lap = build_laplacian_pyramid(bg_np, levels)
    alpha_gauss = build_gaussian_pyramid(alpha_3ch if alpha_3ch.ndim == 3 else np.stack([alpha_np]*3, axis=2), levels)
    
    # Blend each level.
    blended_lap = []
    for i in range(len(fg_lap)):
        alpha_i = alpha_gauss[i]
        if alpha_i.ndim == 2:
            alpha_i = np.stack([alpha_i] * 3, axis=2)
        
        # Keep the alpha inside [0, 1] at this level.
        alpha_i = np.clip(alpha_i, 0, 1)
        blended_level = fg_lap[i] * alpha_i + bg_lap[i] * (1 - alpha_i)
        blended_lap.append(blended_level)
    
    # Reconstruct from the coarsest level upward.
    result = blended_lap[-1]
    for i in range(len(blended_lap) - 2, -1, -1):
        size = (blended_lap[i].shape[1], blended_lap[i].shape[0])
        expanded = cv2.pyrUp(result, dstsize=size)
        result = expanded + blended_lap[i]
    
    result = np.clip(result, 0, 1)
    
    # Return a tensor when the foreground was a tensor.
    if is_torch:
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).float()
        return result_tensor
    
    return result
