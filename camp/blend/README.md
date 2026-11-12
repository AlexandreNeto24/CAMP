# Blend V2: Seamless Instance Blending Pipeline

Módulo organizado em **4 fases** para síntese medical imaging seamless.

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────┐
│ FASE 1: EXTRACTION (Alpha Matting)              │
│ Input: lesion_raw (3,H,W) + lesion_mask (1,H,W)│
│ Output: fg (H,W,3), alpha (H,W)                │
│ Arquivo: extraction.py                          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ FASE 2: PREPARATION (Recover Unmultiplied)      │
│ Input: foreground + alpha                       │
│ Output: fg_unmult (cores puras, sem alpha)      │
│ Arquivo: harmonization.py                       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ FASE 3: HARMONIZATION (Local Reinhard)          │
│ Input: fg_unmult + bg + alpha                   │
│ Output: fg_corr (cores harmonizadas)            │
│ ✓ Núcleo preservado (core_strength)             │
│ ✓ Bordas adaptadas (edge_strength)              │
│ Arquivo: harmonization.py                       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ FASE 4: BLENDING (Pyramid Multiscale)           │
│ Input: fg_corr + alpha + bg                     │
│ Output: final_blended                           │
│ ✓ Resultado: fg_corr × alpha + bg × (1-alpha)  │
│ ✓ Multiscale com Laplacian pyramid              │
│ Arquivo: blending.py                            │
└─────────────────────────────────────────────────┘
```

## 📦 Módulos

### `extraction.py` - FASE 1

**Python**

```python
from camp.blend import trimap_based_matting

alpha, foreground = trimap_based_matting(
    lesion_image,           # [3, H, W] tensor
    mask,                   # [1, H, W] tensor
    erosion_kernel=5,       # Foreground puro
    dilation_kernel=15,     # Background puro
)
# Output: alpha [1,H,W], foreground [3,H,W]
```

### `harmonization.py` - FASES 2 & 3

**Python**

```python
from camp.blend import (
    recover_unmultiplied,
    local_reinhard_soft,
)

# Recupera cores puras
fg_unmult = recover_unmultiplied(foreground, alpha)

# Harmoniza com Reinhard local
fg_corr = local_reinhard_soft(
    fg_unmult,
    background,
    alpha,
    band_px=2,              # Largura banda
    core_strength=0.58,     # Força no núcleo (baixa = preserva)
    edge_strength=0.88,     # Força na borda (alta = adapta)
    l_weight=0.72,          # Preservação de luminância
)
```

### `blending.py` - FASE 4

**Python**

```python
from camp.blend import pyramid_mask_blend

result = pyramid_mask_blend(
    fg_corr,                # Foreground harmonizado
    alpha,                  # Alpha map
    background,             # Background
    feather_px=1,           # Feathering
    levels=4,               # Níveis pyramid
)
# Output: final_blended [3,H,W]
```

### `pipeline.py` - INTEGRADO

**Python**

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
    return_intermediate=True,  # Retorna intermediários
)

final = result['final']              # Resultado final
fg_extracted = result['fg_extracted']   # Fase 1
fg_harmonized = result['fg_harmonized'] # Fase 3
```

## 🎯 Uso Rápido

**Python**

```python
import torch
from camp.blend import SeamlessBlendingPipeline

# Setup
pipeline = SeamlessBlendingPipeline()

# Dados
lesion = torch.rand(3, 512, 512)
mask = torch.rand(1, 512, 512) > 0.5
background = torch.rand(3, 512, 512)

# Run
result = pipeline.run(lesion, mask, background)
final_image = result['final']  # [512, 512, 3] numpy array
```

## 📊 Parâmetros Chave

| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `band_px` | 2 | Largura da zona de transição em pixels |
| `core_strength` | 0.58 | Força de harmonização no núcleo (0=sem mudança, 1=total) |
| `edge_strength` | 0.88 | Força de harmonização na borda |
| `l_weight` | 0.72 | Preservação de luminância na harmonização (0=sem, 1=total) |
| `pyramid_levels` | 4 | Níveis multiscala (mais = mais detalhes suave) |

### Ajuste de Parâmetros

**Para preservar mais o núcleo:**
- Diminui `core_strength` (e.g., 0.50)
- Aumenta `l_weight` (e.g., 0.80)

**Para melhor blending nas bordas:**
- Aumenta `edge_strength` (e.g., 0.95)
- Aumenta `pyramid_levels` (e.g., 5)

## 🔍 Debugging

**Python**

```python
result = pipeline.run(lesion, mask, background, return_intermediate=True)

# Visualiza cada fase
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].imshow(result['fg_extracted'])
axs[0].set_title('Fase 1: Extracted')

axs[1].imshow(result['fg_unmult'])
axs[1].set_title('Fase 2: Unmultiplied')

axs[2].imshow(result['fg_harmonized'])
axs[2].set_title('Fase 3: Harmonized')

axs[3].imshow(result['final'])
axs[3].set_title('Fase 4: Final')

plt.tight_layout()
plt.show()
```

## 📝 Notas

- **Alpha matting**: Usa trimap com erosão/dilatação + Gaussian blur
- **Harmonização**: Reinhard transfer em LAB + spatial soft weighting
- **Blending**: Laplacian pyramid multiscala para evitar seams
- **Todos os valores**: Normalizados para [0, 1] float32

## 🚀 Performance

- **Tempo típico**: ~100-200ms por imagem (512×512) em CPU
- **Memória**: ~200MB para 512×512 multi-scale
- **GPU**: Suportado com torch tensors

---

**Autor**: Seamless Blending Pipeline v2  
**Data**: 2024  
**Status**: Production-ready
