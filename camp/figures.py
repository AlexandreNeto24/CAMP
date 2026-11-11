"""
Visualizações para o módulo Paste: Segmentador + Recommender.

Funções standalone que recebem arrays/tensores — sem dependência dos modelos.
Usadas pelo comando de treino com --save-viz.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union

from camp.viz_utils import (
    tensor_to_numpy, save_figure, make_comparison_grid,
    overlay_mask, plot_training_curves, crop_around_mask,
)


# ═══════════════════════════════════════════════════════════════════
# SEGMENTADOR
# ═══════════════════════════════════════════════════════════════════

def plot_seg_examples(
    images: list,
    masks_gt: list,
    masks_pred: list,
    save_path: Optional[str] = None,
    max_samples: int = 8,
    title: str = 'Segmentation Examples',
    dpi: int = 150,
) -> plt.Figure:
    """
    Grid N×3: imagem original | máscara GT | máscara predita.

    Args:
        images: lista de tensores/arrays (C,H,W) ou (H,W,C)
        masks_gt: lista de tensores/arrays de máscara GT
        masks_pred: lista de tensores/arrays de máscara predita
        max_samples: máximo de linhas
    """
    n = min(len(images), max_samples)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)

    for i in range(n):
        img_np = tensor_to_numpy(images[i])
        gt_np = tensor_to_numpy(masks_gt[i])
        pred_np = tensor_to_numpy(masks_pred[i])

        axes[i][0].imshow(img_np)
        axes[i][0].set_title('Input', fontsize=11, fontweight='bold')
        axes[i][0].axis('off')

        axes[i][1].imshow(gt_np, cmap='gray')
        axes[i][1].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes[i][1].axis('off')

        axes[i][2].imshow(pred_np, cmap='gray')
        axes[i][2].set_title('Prediction', fontsize=11, fontweight='bold')
        axes[i][2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


def plot_seg_overlay(
    image,
    mask_gt,
    mask_pred,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    4-panel: imagem | GT overlay (verde) | Pred overlay (azul) | Diferença (vermelho).
    """
    img_np = tensor_to_numpy(image)
    gt_np = tensor_to_numpy(mask_gt)
    pred_np = tensor_to_numpy(mask_pred)

    gt_overlay = overlay_mask(img_np, gt_np, color=(0, 255, 0), alpha=0.4)
    pred_overlay = overlay_mask(img_np, pred_np, color=(0, 120, 255), alpha=0.4)

    # Diferença: falso positivo (vermelho) + falso negativo (amarelo)
    gt_bin = gt_np > 127 if gt_np.max() > 1 else gt_np > 0.5
    pred_bin = pred_np > 127 if pred_np.max() > 1 else pred_np > 0.5

    diff_img = img_np.copy()
    fp = pred_bin & ~gt_bin  # Falso positivo
    fn = ~pred_bin & gt_bin  # Falso negativo

    diff_overlay = diff_img.copy()
    if diff_overlay.ndim == 2:
        diff_overlay = np.stack([diff_overlay] * 3, axis=-1)
    diff_overlay[fp] = [255, 0, 0]      # Vermelho = FP
    diff_overlay[fn] = [255, 255, 0]    # Amarelo = FN
    diff_result = (diff_img.astype(float) * 0.6 + diff_overlay.astype(float) * 0.4)
    diff_result = diff_result.clip(0, 255).astype(np.uint8)

    titles = ['Input', 'GT (verde)', 'Pred (azul)', 'Erro (verm=FP, amar=FN)']
    imgs = [img_np, gt_overlay, pred_overlay, diff_result]

    fig = make_comparison_grid(
        imgs, titles, ncols=4,
        figsize_per_img=5.0,
        suptitle='Segmentation Overlay',
        save_path=save_path, dpi=dpi,
    )
    return fig


def plot_seg_worst(
    images: list,
    masks_gt: list,
    masks_pred: list,
    dice_scores: list,
    save_path: Optional[str] = None,
    n_worst: int = 4,
    dpi: int = 150,
) -> plt.Figure:
    """
    Mostra as N piores predições (menor Dice) com overlay de erro.
    """
    indices = np.argsort(dice_scores)[:n_worst]

    fig, axes = plt.subplots(n_worst, 3, figsize=(15, 5 * n_worst), squeeze=False)

    for row, idx in enumerate(indices):
        img_np = tensor_to_numpy(images[idx])
        gt_np = tensor_to_numpy(masks_gt[idx])
        pred_np = tensor_to_numpy(masks_pred[idx])

        gt_overlay = overlay_mask(img_np, gt_np, color=(0, 255, 0), alpha=0.4)
        pred_overlay = overlay_mask(img_np, pred_np, color=(0, 120, 255), alpha=0.4)

        axes[row][0].imshow(img_np)
        axes[row][0].set_title(f'Input (#{idx})', fontsize=11, fontweight='bold')
        axes[row][0].axis('off')

        axes[row][1].imshow(gt_overlay)
        axes[row][1].set_title(f'GT (verde)', fontsize=11)
        axes[row][1].axis('off')

        axes[row][2].imshow(pred_overlay)
        axes[row][2].set_title(f'Pred (azul) — Dice={dice_scores[idx]:.3f}',
                               fontsize=11, color='red', fontweight='bold')
        axes[row][2].axis('off')

    fig.suptitle(f'Worst {n_worst} Predictions (by Dice)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDER
# ═══════════════════════════════════════════════════════════════════

# Palette de cores para bboxes ranked
RANK_COLORS = [
    '#00FF00',  # #1 lime
    '#00FFFF',  # #2 cyan
    '#FFD700',  # #3 gold
    '#FF00FF',  # #4 magenta
    '#FFFFFF',  # #5 branco
]


def plot_recommendations(
    image,
    patch,
    recommendations: list,
    heatmap: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_scores: bool = True,
    max_recs: int = 5,
    dpi: int = 150,
) -> plt.Figure:
    """
    3-panel: imagem com bboxes rankeadas | heatmap de compatibilidade | patch.

    Baseado em recommender_viz.py do attention_recommender_v2.ipynb.

    Args:
        image: tensor (C,H,W) ou numpy (H,W,C)
        patch: tensor (C,h,w) ou numpy
        recommendations: lista de dicts com 'box' [x1,y1,x2,y2] e 'score'
        heatmap: (H,W) float — mapa de compatibilidade (opcional)
    """
    img_np = tensor_to_numpy(image)
    patch_np = tensor_to_numpy(patch)

    ncols = 2 if heatmap is None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))

    # Panel 1: Imagem com bboxes
    ax = axes[0]
    ax.imshow(img_np)

    recs = recommendations[:max_recs]
    for i, rec in enumerate(recs):
        box = rec['box']  # [x1, y1, x2, y2]
        score = rec.get('score', 0)
        color = RANK_COLORS[i % len(RANK_COLORS)]

        rect = mpatches.FancyBboxPatch(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2.5, edgecolor=color, facecolor='none',
            boxstyle='round,pad=2',
        )
        ax.add_patch(rect)

        if show_scores:
            ax.text(box[0], box[1] - 5, f'#{i+1} {score:.3f}',
                    fontsize=10, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # Inset do patch (canto inferior direito)
    h_img, w_img = img_np.shape[:2]
    inset_size = max(60, int(min(h_img, w_img) * 0.20))
    ax_inset = fig.add_axes([
        ax.get_position().x1 - 0.12,
        ax.get_position().y0 + 0.02,
        0.10, 0.10
    ])
    ax_inset.imshow(patch_np)
    ax_inset.set_title('Patch', fontsize=8, color='lime')
    ax_inset.axis('off')
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('lime')
        spine.set_linewidth(2)
        spine.set_visible(True)

    ax.set_title(f'Recommendations (top {len(recs)})', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 2: Heatmap (se fornecido)
    if heatmap is not None:
        ax2 = axes[1]
        im = ax2.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax2.set_title('Compatibility Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        patch_ax_idx = 2
    else:
        patch_ax_idx = 1

    # Panel final: Patch ampliado
    ax_p = axes[patch_ax_idx]
    ax_p.imshow(patch_np)
    ax_p.set_title('Source Patch', fontsize=12, fontweight='bold')
    ax_p.axis('off')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


def plot_batch_debug(
    images: list,
    masks: list,
    patches: list,
    heatmaps: list,
    save_path: Optional[str] = None,
    max_samples: int = 4,
    dpi: int = 150,
) -> plt.Figure:
    """
    Debug grid N×4: imagem | segmentor mask | primeiro patch | target heatmap.

    Baseado em debug_visualize_inputs() do attention_recommender.py.
    """
    n = min(len(images), max_samples)

    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)

    for i in range(n):
        img_np = tensor_to_numpy(images[i])
        mask_np = tensor_to_numpy(masks[i])
        patch_np = tensor_to_numpy(patches[i]) if i < len(patches) else np.zeros((64, 64, 3), dtype=np.uint8)
        heat_np = tensor_to_numpy(heatmaps[i]) if i < len(heatmaps) else np.zeros((64, 64), dtype=np.float32)

        axes[i][0].imshow(img_np)
        axes[i][0].set_title('Image', fontsize=11, fontweight='bold')
        axes[i][0].axis('off')

        axes[i][1].imshow(mask_np, cmap='gray')
        axes[i][1].set_title('Segmentor Mask', fontsize=11, fontweight='bold')
        axes[i][1].axis('off')

        axes[i][2].imshow(patch_np)
        axes[i][2].set_title('Patch', fontsize=11, fontweight='bold')
        axes[i][2].axis('off')

        im = axes[i][3].imshow(heat_np if heat_np.ndim == 2 else heat_np.mean(axis=2),
                               cmap='hot')
        axes[i][3].set_title('Target Heatmap', fontsize=11, fontweight='bold')
        axes[i][3].axis('off')
        plt.colorbar(im, ax=axes[i][3], fraction=0.046, pad=0.04)

    fig.suptitle('Batch Debug Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


def plot_score_distribution(
    scores: Dict[str, List[float]],
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Distribuição de scores (KDE ou histograma).

    Baseado em train_paste_two_stage.py.

    Args:
        scores: dict com label → lista de scores
            Ex: {'combined': [...], 'similarity': [...], 'informativity': [...]}
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    try:
        import seaborn as sns
        for label, values in scores.items():
            if len(values) > 1:
                sns.kdeplot(values, label=label, ax=ax)
    except ImportError:
        # Fallback sem seaborn: histogramas
        for label, values in scores.items():
            if len(values) > 1:
                ax.hist(values, bins=30, alpha=0.5, label=label, density=True)

    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


def plot_spatial_density(
    centers: List[Tuple[float, float]],
    image_size: Tuple[int, int] = (512, 512),
    grid_size: int = 16,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Heatmap de densidade espacial das recomendações.

    Baseado em train_paste_two_stage.py visualize_domain_distribution().

    Args:
        centers: lista de (cy, cx) normalizados ou absolutos
        image_size: (H, W)
        grid_size: resolução do grid
    """
    H, W = image_size
    density = np.zeros((grid_size, grid_size))

    for cy, cx in centers:
        # Normalizar se em coordenadas absolutas
        if cy > 1 or cx > 1:
            cy, cx = cy / H, cx / W
        gy = min(int(cy * grid_size), grid_size - 1)
        gx = min(int(cx * grid_size), grid_size - 1)
        density[gy, gx] += 1

    # Normalizar
    if density.max() > 0:
        density = density / density.max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(density, cmap='viridis', interpolation='bilinear',
                   extent=[0, W, H, 0])
    ax.set_title('Spatial Density of Recommendations', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Density')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig
