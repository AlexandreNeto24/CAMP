"""Seamless paste: extract, unmultiply, harmonize, then blend.

Flow:
  Raw lesion → alpha matting
    → unmultiplied foreground
    → local Reinhard harmonization
    → Laplacian pyramid blend
    → final image
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union

from .extraction import trimap_based_matting
from .harmonization import recover_unmultiplied, local_reinhard_soft
from .blending import pyramid_mask_blend


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return a NumPy array, moving a tensor to CPU first."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_hwc(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return ``(H, W, C)``. A ``(3, H, W)`` array is transposed."""
    a = to_numpy(x).copy()
    if a.ndim == 3 and a.shape[0] == 3:  # CHW
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:  # H,W
        a = a[..., None]
    return a


def ensure_alpha_hw(alpha: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return alpha as ``(H, W)``."""
    a = to_numpy(alpha)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a.squeeze(0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        return a
    raise ValueError(f'Unhandled alpha shape: {a.shape}')


def norm01(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Map an image to float in ``[0, 1]``.

    Integer arrays are divided by 255. Float arrays whose peak is above
    2 are treated as 0–255.
    """
    x = to_numpy(x)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32) / 255.0 if (np.nanmax(x) > 2.0) else x.astype(np.float32)


class SeamlessBlendingPipeline:
    """Four-step seamless paste of one lesion onto a background.

    Example:
        pipeline = SeamlessBlendingPipeline(
            band_px=2,
            core_strength=0.58,
            edge_strength=0.88,
            l_weight=0.72,
        )
        result = pipeline.run(
            lesion_image=lesion_tensor,
            lesion_mask=mask_tensor,
            background=bg_tensor,
        )
    """
    
    def __init__(
        self,
        band_px: int = 2,
        core_strength: float = 0.58,
        edge_strength: float = 0.88,
        l_weight: float = 0.72,
        pyramid_levels: int = 4,
    ):
        """
        Args:
            band_px: Width, in pixels, of the harmonization transition.
            core_strength: Harmonization weight on the mask boundary.
            edge_strength: Harmonization weight inside the mask.
            l_weight: How much of the transferred L* channel is kept.
            pyramid_levels: Levels of the Laplacian blend.
        """
        self.band_px = band_px
        self.core_strength = core_strength
        self.edge_strength = edge_strength
        self.l_weight = l_weight
        self.pyramid_levels = pyramid_levels
        self.results = {}
    
    def run(
        self,
        lesion_image: Union[torch.Tensor, np.ndarray],
        lesion_mask: Union[torch.Tensor, np.ndarray],
        background: Union[torch.Tensor, np.ndarray],
        return_intermediate: bool = True,
    ) -> Dict:
        """Run extraction, harmonization, and the pyramid blend.

        Args:
            lesion_image: Raw lesion, ``[3, H, W]``.
            lesion_mask: Binary mask, ``[1, H, W]`` or ``[H, W]``.
            background: Image the lesion is pasted onto, ``[3, H, W]``.
            return_intermediate: Include the stage arrays in the result.

        Returns:
            A dict with ``final``. When ``return_intermediate`` is set it
            also holds ``fg_extracted``, ``alpha``, ``fg_unmult``, and
            ``fg_harmonized``.
        """
        # The previous call stored ``final`` on this object. Building the
        # return value with ``update(self.results)`` would copy that image
        # over the blend just computed.
        self.results = {}

        # ===== 1. Extraction =====
        # print("[1/4] Extraction (Alpha Matting)")
        alpha_t, fg_t = trimap_based_matting(
            to_numpy(lesion_image) if isinstance(lesion_image, (torch.Tensor, np.ndarray)) else lesion_image,
            to_numpy(lesion_mask) if isinstance(lesion_mask, (torch.Tensor, np.ndarray)) else lesion_mask,
        )
        
        # HWC arrays in [0, 1].
        fg_hwc = norm01(ensure_hwc(fg_t))
        bg_hwc = norm01(ensure_hwc(background))
        alpha_hw = np.clip(norm01(ensure_alpha_hw(alpha_t)), 0.0, 1.0)
        
        # print(f"   → FG: {fg_hwc.shape}, Alpha: {alpha_hw.shape}, BG: {bg_hwc.shape}")
        
        if return_intermediate:
            self.results['fg_extracted'] = fg_hwc
            self.results['alpha'] = alpha_hw
        
        # ===== 2. Unmultiply =====
        # print("[2/4] Preparation (Recover Unmultiplied)")
        eps = 1e-6
        alpha_3 = alpha_hw[..., None]
        fg_unmult = recover_unmultiplied(fg_hwc, alpha_hw, eps=eps)
        # print(f"   → FG unmultiplied: {fg_unmult.shape}")
        
        if return_intermediate:
            self.results['fg_unmult'] = fg_unmult
        
        # ===== 3. Harmonization =====
        # print("[3/4] Harmonization (Local Reinhard)")
        fg_corr = local_reinhard_soft(
            fg_unmult, bg_hwc, alpha_hw,
            band_px=self.band_px,
            core_strength=self.core_strength,
            edge_strength=self.edge_strength,
            l_weight=self.l_weight,
        )
        # print(f"   → FG harmonized: {fg_corr.shape}")
        # print(f"      band_px={self.band_px}, core={self.core_strength}, edge={self.edge_strength}, l_weight={self.l_weight}")
        
        if return_intermediate:
            self.results['fg_harmonized'] = fg_corr
        
        # ===== 4. Blending =====
        # print("[4/4] Blending (Pyramid Multiscale)")
        
        # Tensors for the pyramid blend.
        try:
            fg_corr_t = torch.from_numpy(fg_corr.astype(np.float32)).permute(2, 0, 1).float()
            alpha_t_in = torch.from_numpy(alpha_hw).unsqueeze(0).float()
            bg_t_in = torch.from_numpy(bg_hwc.astype(np.float32)).permute(2, 0, 1).float()
        except Exception:
            fg_corr_t = fg_corr
            alpha_t_in = alpha_hw
            bg_t_in = bg_hwc
        
        # Pyramid blend.
        pyr_out = pyramid_mask_blend(
            fg_corr_t,
            alpha_t_in,
            bg_t_in,
            feather_px=self.band_px,
            levels=self.pyramid_levels,
        )
        
        # Back to a NumPy image.
        pyr_np = to_numpy(pyr_out)
        if pyr_np.dtype == np.uint8:
            pyr_np = pyr_np.astype(np.float32) / 255.0
        if pyr_np.ndim == 3 and pyr_np.shape[0] == 3:
            pyr_vis = pyr_np.transpose(1, 2, 0)
        else:
            pyr_vis = pyr_np
        
        # print(f"   → Final blended: {pyr_vis.shape}")
        
        # ===== Final =====
        result = {
            'final': pyr_vis,
            'status': 'success'
        }
        
        if return_intermediate:
            result.update(self.results)
        # ``update`` must not keep a stale ``final`` from an earlier call.
        result['final'] = pyr_vis
        self.results['final'] = pyr_vis
        return result
