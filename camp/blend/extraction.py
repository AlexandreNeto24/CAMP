"""Alpha extraction for the paste.

The returned alpha is a Gaussian blur of the binary mask. ``erosion_kernel``
and ``dilation_kernel`` stay in the signature so existing callers do not
break; the blur does not use them.
"""

import cv2
import numpy as np
import torch
from typing import Tuple


def trimap_based_matting(
    lesion_image: torch.Tensor,
    mask: torch.Tensor,
    erosion_kernel: int = 5,
    dilation_kernel: int = 15,
    blur_sigma: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(alpha, foreground)`` for a lesion crop and a binary mask.

    Accepts a CHW tensor or an HWC numpy array. Alpha is
    ``GaussianBlur`` of the binary mask. Foreground is the image multiplied
    by that alpha.
    """
    del erosion_kernel, dilation_kernel

    if isinstance(lesion_image, torch.Tensor):
        img_np = lesion_image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img_np = lesion_image.copy()

    if img_np.dtype == np.uint8 or np.max(img_np) > 2.0:
        img_np = (img_np.astype(np.float32) / 255.0) if np.max(img_np) > 1.0 else img_np.astype(np.float32)
    else:
        img_np = img_np.astype(np.float32)

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().squeeze().numpy()
    else:
        mask_np = mask.copy().squeeze()

    if mask_np.dtype == np.uint8 or np.max(mask_np) > 1.0:
        mask_np = (mask_np.astype(np.float32) / 255.0) if np.max(mask_np) > 1.0 else mask_np.astype(np.float32)
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

    alpha_smooth = cv2.GaussianBlur(
        mask_np.astype(np.float32) / 255.0,
        (15, 15),
        blur_sigma
    )
    alpha = np.clip(alpha_smooth, 0.0, 1.0)

    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    foreground = img_np * alpha_3ch

    alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).float()
    fg_tensor = torch.from_numpy(foreground).permute(2, 0, 1).float()

    return alpha_tensor, fg_tensor
