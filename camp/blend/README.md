# Seamless instance blending

The paste is four steps. Each step lives in its own module. `SeamlessBlendingPipeline` runs them in order.

```
Extraction          trimap_based_matting
  lesion and mask → foreground and alpha
        ↓
Unmultiply          recover_unmultiplied
  foreground / alpha → colours without the alpha
        ↓
Harmonization       local_reinhard_soft
  CIELAB Reinhard on the paste, mixed with a spatial weight
        ↓
Blending            pyramid_mask_blend
  Laplacian pyramid composite → final image
```

## Modules

### `extraction.py`

```python
from camp.blend import trimap_based_matting

alpha, foreground = trimap_based_matting(
    lesion_image,  # [3, H, W]
    mask,          # [1, H, W]
)
# alpha [1, H, W], foreground [3, H, W]
```

Alpha is a Gaussian blur of the binary mask. `erosion_kernel` and `dilation_kernel` stay in the signature so older callers still import. The blur does not use them.

### `harmonization.py`

```python
from camp.blend import recover_unmultiplied, local_reinhard_soft

fg_unmult = recover_unmultiplied(foreground, alpha)

fg_corr = local_reinhard_soft(
    fg_unmult,
    background,
    alpha,
    band_px=2,           # width of the spatial mix, in pixels
    core_strength=0.58,  # harmonization weight on the mask boundary
    edge_strength=0.88,  # harmonization weight inside the mask
    l_weight=0.72,       # share of the transferred L* channel
)
```

`core_strength` is the weight where the mask meets the background. `edge_strength` is the weight once the pixel is `band_px` inside the mask. A higher weight copies more of the Reinhard colour. With the defaults, the interior is harmonized more than the boundary.

### `blending.py`

```python
from camp.blend import pyramid_mask_blend

result = pyramid_mask_blend(
    fg_corr,      # harmonized foreground, not premultiplied
    alpha,
    background,
    feather_px=1,  # accepted, unused by the pyramid
    levels=4,
)
# [3, H, W]
```

### `blend_pipeline.py`

```python
from camp.blend import SeamlessBlendingPipeline

pipeline = SeamlessBlendingPipeline(
    band_px=2,
    core_strength=0.58,
    edge_strength=0.88,
    l_weight=0.72,
    pyramid_levels=4,
)

result = pipeline.run(
    lesion_image=lesion_tensor,
    lesion_mask=mask_tensor,
    background=bg_tensor,
    return_intermediate=True,
)

final = result["final"]
fg_extracted = result["fg_extracted"]
fg_harmonized = result["fg_harmonized"]
```

## Quick use

```python
import torch
from camp.blend import SeamlessBlendingPipeline

pipeline = SeamlessBlendingPipeline()

lesion = torch.rand(3, 512, 512)
mask = torch.rand(1, 512, 512) > 0.5
background = torch.rand(3, 512, 512)

result = pipeline.run(lesion, mask, background)
final_image = result["final"]  # [512, 512, 3] numpy array
```

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `band_px` | 2 | Width of the spatial mix, in pixels |
| `core_strength` | 0.58 | Harmonization weight on the mask boundary. 0 keeps the original colour, 1 takes the full transfer |
| `edge_strength` | 0.88 | Harmonization weight inside the mask, past `band_px` |
| `l_weight` | 0.72 | Share of the transferred L* that replaces the original luminance |
| `pyramid_levels` | 4 | Laplacian levels. More levels keep finer frequencies in the seam |

To keep more of the original colour inside the mask, lower `edge_strength`. To keep more of it on the boundary, lower `core_strength`. Raising `l_weight` follows the transferred luminance more closely. Raising `pyramid_levels` blends the seam at a coarser scale as well.

## Debugging

```python
result = pipeline.run(lesion, mask, background, return_intermediate=True)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].imshow(result["fg_extracted"])
axs[0].set_title("Extracted")

axs[1].imshow(result["fg_unmult"])
axs[1].set_title("Unmultiplied")

axs[2].imshow(result["fg_harmonized"])
axs[2].set_title("Harmonized")

axs[3].imshow(result["final"])
axs[3].set_title("Final")

plt.tight_layout()
plt.show()
```

## Notes

- Alpha is a Gaussian blur of the binary mask.
- Harmonization is Reinhard transfer in CIELAB, mixed with the spatial weight.
- The composite is a Laplacian pyramid, which avoids a hard seam.
- Images are float32 in `[0, 1]`.
