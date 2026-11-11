"""Shared helpers for the training figures.

The Agg backend is selected here, once, before pyplot is imported.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union


# ─────────────────────────── Conversões ──────────────────────────

def tensor_to_numpy(t, squeeze_batch: bool = True) -> np.ndarray:
    """
    Converte tensor PyTorch para numpy uint8 (H, W, C) ou (H, W).

    Aceita:
        - (B, C, H, W) → remove batch se squeeze_batch
        - (C, H, W) → transpõe para (H, W, C)
        - (1, H, W) ou (H, W) → grayscale
        - PIL.Image → np.array
    """
    if hasattr(t, 'convert'):  # PIL.Image
        return np.array(t)

    if hasattr(t, 'detach'):
        t = t.detach().cpu()
        if t.dim() == 4 and squeeze_batch:
            t = t[0]
        arr = t.numpy()
    else:
        arr = np.asarray(t)

    # (C, H, W) → (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Squeeze canal único
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # Normalizar para [0, 255] uint8
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return arr


# ─────────────────────────── Save / Grid ──────────────────────────

def save_figure(fig, path: Union[str, Path], dpi: int = 150, close: bool = True):
    """Grava figura matplotlib e opcionalmente fecha-a."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white')
    if close:
        plt.close(fig)
    return str(path)


def make_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    ncols: Optional[int] = None,
    figsize_per_img: float = 5.0,
    title_fontsize: int = 11,
    suptitle: Optional[str] = None,
    title_colors: Optional[List[str]] = None,
    cmaps: Optional[List[Optional[str]]] = None,
    colorbars: Optional[List[bool]] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Cria grid de comparação com N imagens.

    Args:
        images: lista de numpy arrays (H,W,C) ou (H,W)
        titles: lista de títulos
        ncols: colunas (default: len(images))
        cmaps: lista de colormaps (None = auto RGB)
        colorbars: lista de bools — mostrar colorbar para cada imagem
        save_path: se fornecido, grava a figura
    """
    n = len(images)
    if ncols is None:
        ncols = n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_img * ncols, figsize_per_img * nrows),
                             squeeze=False)

    if cmaps is None:
        cmaps = [None] * n
    if colorbars is None:
        colorbars = [False] * n
    if title_colors is None:
        title_colors = [None] * n

    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        img = images[idx]

        cmap = cmaps[idx]
        if cmap is None and img.ndim == 2:
            cmap = 'gray'

        im = ax.imshow(img, cmap=cmap)
        if title_colors[idx] is not None:
            ax.set_title(titles[idx], fontsize=title_fontsize, fontweight='bold', color=title_colors[idx])
        else:
            ax.set_title(titles[idx], fontsize=title_fontsize, fontweight='bold')
        ax.axis('off')

        if colorbars[idx]:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Esconder eixos vazios
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ─────────────────────────── Overlay ──────────────────────────

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Sobrepõe máscara colorida sobre imagem.

    Args:
        image: (H, W, 3) uint8
        mask: (H, W) binária ou float [0,1]
        color: RGB tuple
        alpha: opacidade da máscara
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    if mask.dtype in (np.float32, np.float64):
        mask_bin = (mask > 0.5)
    else:
        mask_bin = (mask > 127) if mask.max() > 1 else (mask > 0.5)

    overlay = img.copy()
    overlay[mask_bin] = color
    result = (img.astype(float) * (1 - alpha) + overlay.astype(float) * alpha)
    return result.clip(0, 255).astype(np.uint8)
