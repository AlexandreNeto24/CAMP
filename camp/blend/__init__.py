"""
Blend V2: Seamless Instance Blending with Alpha Matting & Pyramid Blending

Módulo organizado em 4 fases:
1. Extraction (trimap_based_matting) → Alpha map + Foreground
2. Harmonization (local_reinhard_soft) → Color adaptation
3. Blending (pyramid_mask_blend) → Multiscale seamless blending
4. Pipeline (SeamlessBlendingPipeline) → Tudo integrado

Uso rápido:
    from src.blend import SeamlessBlendingPipeline
    
    pipeline = SeamlessBlendingPipeline(
        core_strength=0.58,
        edge_strength=0.88
    )
    result = pipeline.run(lesion_tensor, mask_tensor, background_tensor)
    final_image = result['final']
"""

from .extraction import trimap_based_matting
from .harmonization import (
    recover_unmultiplied,
    local_reinhard_soft,
    bbox_from_mask,
    soft_band_weight_map,
)
from .blending import (
    pyramid_mask_blend,
    pyramid_poisson_blend,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
)
from .blend_pipeline import SeamlessBlendingPipeline

__all__ = [
    # Phase 1: Extraction
    'trimap_based_matting',
    
    # Phase 2 & 3: Preparation & Harmonization
    'recover_unmultiplied',
    'local_reinhard_soft',
    'bbox_from_mask',
    'soft_band_weight_map',
    
    # Phase 4: Blending
    'pyramid_mask_blend',
    'pyramid_poisson_blend',
    'build_gaussian_pyramid',
    'build_laplacian_pyramid',
    
    # Full Pipelines
    'SeamlessBlendingPipeline'
]
