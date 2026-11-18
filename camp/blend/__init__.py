"""Seamless instance blending: matting, colour transfer, pyramid composite.

Four steps:
1. Extraction (``trimap_based_matting``) returns an alpha map and a foreground.
2. Harmonization (``local_reinhard_soft``) adapts colour in CIELAB.
3. Blending (``pyramid_mask_blend``) composites the frequencies.
4. ``SeamlessBlendingPipeline`` runs the three steps in order.

Quick use:
    from camp.blend import SeamlessBlendingPipeline

    pipeline = SeamlessBlendingPipeline(
        core_strength=0.58,
        edge_strength=0.88,
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
    'build_gaussian_pyramid',
    'build_laplacian_pyramid',
    
    # Full Pipelines
    'SeamlessBlendingPipeline'
]
