"""
Visualizações para o módulo Blend: DoveNet + Harmonizer + Pipeline.

Funções standalone — recebem PIL/numpy, sem dependência dos modelos.
Baseadas nas visualizações de blend_final.ipynb.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
from PIL import Image

from src.viz_utils import (
    tensor_to_numpy, save_figure, make_comparison_grid,
    overlay_mask, plot_training_curves, crop_around_mask,
    compute_diff_map, pil_to_numpy,
)


# ═══════════════════════════════════════════════════════════════════
# AMOSTRAS DO DATASET
# ═══════════════════════════════════════════════════════════════════

def plot_blend_sample(
    composite,
    real,
    mask,
    boundary_mask=None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Mostra amostra do HarmonizationDataset: Composite | Real | Mask [| Boundary].

    Baseado em blend_final.ipynb célula 4 (show_sample).
    """
    comp_np = tensor_to_numpy(composite)
    real_np = tensor_to_numpy(real)
    mask_np = tensor_to_numpy(mask)

    images = [comp_np, real_np, mask_np]
    titles = ['Input (Composite)', 'Target (Real)', 'Mask']
    cmaps = [None, None, 'gray']
    cbars = [False, False, False]

    if boundary_mask is not None:
        bound_np = tensor_to_numpy(boundary_mask)
        if bound_np.dtype == np.uint8:
            bound_np = bound_np.astype(np.float32) / 255.0
        images.append(bound_np)
        titles.append('Soft Boundary Mask')
        cmaps.append('inferno')
        cbars.append(True)

    fig = make_comparison_grid(
        images, titles, ncols=len(images),
        figsize_per_img=5.0,
        cmaps=cmaps, colorbars=cbars,
        suptitle='Harmonization Dataset Sample',
        save_path=save_path, dpi=dpi,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# DOVENET: INPUT vs OUTPUT vs TARGET
# ═══════════════════════════════════════════════════════════════════

def plot_dovenet_examples(
    composites: list,
    outputs: list,
    targets: list,
    save_path: Optional[str] = None,
    max_samples: int = 4,
    dpi: int = 150,
) -> plt.Figure:
    """
    Grid N×3: Input | DoveNet Output | Target.

    Baseado em blend_final.ipynb célula 13 (test_baseline).
    """
    n = min(len(composites), max_samples)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)

    for i in range(n):
        comp_np = tensor_to_numpy(composites[i])
        out_np = tensor_to_numpy(outputs[i])
        tgt_np = tensor_to_numpy(targets[i])

        axes[i][0].imshow(comp_np)
        axes[i][0].set_title('Input (Composite)', fontsize=11, fontweight='bold')
        axes[i][0].axis('off')

        axes[i][1].imshow(out_np)
        axes[i][1].set_title('DoveNet Output', fontsize=11, fontweight='bold',
                             color='#e67e22')
        axes[i][1].axis('off')

        axes[i][2].imshow(tgt_np)
        axes[i][2].set_title('Target (Real)', fontsize=11, fontweight='bold')
        axes[i][2].axis('off')

    fig.suptitle('DoveNet Harmonization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# BOUNDARY MASK — Visualização com Colorbar
# ═══════════════════════════════════════════════════════════════════

def plot_boundary_mask(
    composite,
    mask,
    boundary_mask,
    harmonized=None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    4-panel: Input | Object Mask | Soft Boundary (inferno+colorbar) | Harmonized.

    Baseado em blend_final.ipynb célula 14.
    """
    comp_np = tensor_to_numpy(composite)
    mask_np = tensor_to_numpy(mask)
    bound_np = tensor_to_numpy(boundary_mask)

    if bound_np.dtype == np.uint8:
        bound_np = bound_np.astype(np.float32) / 255.0

    ncols = 4 if harmonized is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    axes[0].imshow(comp_np)
    axes[0].set_title('Input Composite', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Object Mask', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    im = axes[2].imshow(bound_np, cmap='inferno')
    axes[2].set_title('Soft Boundary Mask\n(Gradiente suave)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if harmonized is not None:
        harm_np = tensor_to_numpy(harmonized)
        axes[3].imshow(harm_np)
        axes[3].set_title('Harmonized Output', fontsize=11, fontweight='bold',
                          color='#2980b9')
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# PIPELINE STEPS — A visualização principal ⭐
# ═══════════════════════════════════════════════════════════════════

def plot_pipeline_steps(
    composite: Image.Image,
    dovenet_out: Image.Image,
    blend_only: Image.Image,
    ring_mask: Image.Image,
    final_result: Image.Image,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    5 colunas: Input → DoveNet → Hybrid Blend → Ring Mask → Final (SD).

    Baseado em blend_final.ipynb célula 15 (Pipeline v3).
    """
    fig, ax = plt.subplots(1, 5, figsize=(30, 6))

    panels = [
        (composite,   '1. Input\n(Colagem crua)',                           None),
        (dovenet_out, '2. DoveNet\n(Harmonização de cor)',                  None),
        (blend_only,  '3. Hybrid Blend\n(Alpha coseno + Poisson mixed)',    None),
        (ring_mask,   'Ring Mask\n(Zona do SD)',                            'gray'),
        (final_result,'4. Final (+ SD)\n(Fronteira fundida)',               None),
    ]

    for i, (img, title, cmap) in enumerate(panels):
        img_np = pil_to_numpy(img) if hasattr(img, 'convert') else tensor_to_numpy(img)
        ax[i].imshow(img_np, cmap=cmap)
        ax[i].set_title(title, fontsize=12, fontweight='bold')
        ax[i].axis('off')

    # Setas entre painéis
    for i in range(4):
        fig.text((i * 0.2) + 0.17, 0.02, '→', fontsize=28, ha='center',
                 fontweight='bold', color='#555')

    fig.suptitle('Pipeline: DoveNet → Hybrid Blend → SD Diffusion',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# ZOOM NA FRONTEIRA
# ═══════════════════════════════════════════════════════════════════

def plot_pipeline_zoom(
    composite: Image.Image,
    dovenet_out: Image.Image,
    blend_only: Image.Image,
    final_result: Image.Image,
    mask: Image.Image,
    padding: int = 170,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Zoom comparativo na zona de junção: Input | DoveNet | Blend | Final.

    Baseado em blend_final.ipynb célula 15 (zoom section).
    """
    mask_np = pil_to_numpy(mask)
    images_np = [pil_to_numpy(img) for img in [composite, dovenet_out, blend_only, final_result]]

    crops = crop_around_mask(images_np, mask_np, padding=padding)

    titles = [
        'Input (zoom)',
        'DoveNet (zoom)\nCor corrigida',
        'Hybrid Blend (zoom)\nSeam eliminada',
        'Final SD (zoom)\nFronteira fundida',
    ]
    colors = ['#888', '#e67e22', '#27ae60', '#2980b9']

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for i, (crop, title, color) in enumerate(zip(crops, titles, colors)):
        axes[i].imshow(crop)
        axes[i].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[i].axis('off')
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    fig.suptitle('ZOOM: Evolução da zona de junção', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# MAPAS DE DIFERENÇA — o que cada etapa altera ⭐
# ═══════════════════════════════════════════════════════════════════

def plot_difference_maps(
    composite: Image.Image,
    dovenet_out: Image.Image,
    blend_only: Image.Image,
    final_result: Image.Image,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    4 mapas hot: DoveNet vs Input | Blend vs DoveNet | SD vs Blend | Total.
    + estatísticas de contribuição percentual.

    Baseado em blend_final.ipynb célula 15 (difference maps).

    Returns:
        (fig, stats_dict) — stats com 'dovenet_mean', 'blend_mean', 'sd_mean', 'total_mean'
    """
    comp_np = pil_to_numpy(composite).astype(np.float32)
    dove_np = pil_to_numpy(dovenet_out).astype(np.float32)
    blend_np = pil_to_numpy(blend_only).astype(np.float32)
    final_np = pil_to_numpy(final_result).astype(np.float32)

    diff_dovenet = compute_diff_map(dove_np, comp_np)
    diff_blend = compute_diff_map(blend_np, dove_np)
    diff_sd = compute_diff_map(final_np, blend_np)
    diff_total = compute_diff_map(final_np, comp_np)

    fig, axes = plt.subplots(1, 4, figsize=(26, 5))

    vmax_main = max(diff_dovenet.max(), diff_total.max(), 1)
    vmax_detail = max(diff_blend.max(), diff_sd.max(), 1)

    panels = [
        (diff_dovenet, 'Etapa 1: DoveNet vs Input\n(Correcção de COR)',
         '#e67e22', vmax_main * 0.6),
        (diff_blend,   'Etapa 2: Blend vs DoveNet\n(Suavização da BORDA)',
         '#27ae60', vmax_detail * 0.8),
        (diff_sd,      'Etapa 3: SD vs Blend\n(Fusão da FRONTEIRA)',
         '#2980b9', vmax_detail * 0.8),
        (diff_total,   'TOTAL: Final vs Input\n(Todas as mudanças)',
         '#8e44ad', vmax_main * 0.6),
    ]

    for i, (diff, title, color, vmax) in enumerate(panels):
        im = axes[i].imshow(diff, cmap='hot', vmin=0, vmax=max(vmax, 1))
        axes[i].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle('Mapas de Diferença: O que cada etapa altera',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Estatísticas
    mean_dove = float(np.mean(diff_dovenet))
    mean_blend = float(np.mean(diff_blend))
    mean_sd = float(np.mean(diff_sd))
    mean_total = float(np.mean(diff_total))

    stats = {
        'dovenet_mean': mean_dove,
        'blend_mean': mean_blend,
        'sd_mean': mean_sd,
        'total_mean': mean_total,
        'dovenet_pct': mean_dove / max(mean_total, 1e-6) * 100,
        'blend_pct': mean_blend / max(mean_total, 1e-6) * 100,
        'sd_pct': mean_sd / max(mean_total, 1e-6) * 100,
    }

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig, stats


# ═══════════════════════════════════════════════════════════════════
# COMPARAÇÃO DE MÉTODOS DE BLEND
# ═══════════════════════════════════════════════════════════════════

def plot_blend_comparison(
    dovenet_pil: Image.Image,
    results: Dict[str, Image.Image],
    mask_pil: Image.Image,
    save_path: Optional[str] = None,
    zoom_padding: int = 140,
    dpi: int = 150,
) -> plt.Figure:
    """
    2×4 grid: Row 1 = DoveNet + 3 métodos | Row 2 = zooms + barras.

    Baseado em blend_final.ipynb célula 16 (blend_method_comparison).

    Args:
        dovenet_pil: output do DoveNet (PIL)
        results: dict {'alpha': PIL, 'poisson': PIL, 'hybrid': PIL, ...}
        mask_pil: máscara do objecto
    """
    method_info = [
        ('alpha',   'Alpha Coseno\n(Preserva cor DoveNet)',     '#27ae60'),
        ('poisson', 'Poisson Mixed\n(Funde textura na borda)',  '#e67e22'),
        ('hybrid',  'Híbrido\n(Alpha miolo + Poisson borda)',   '#2980b9'),
    ]

    # Filtrar métodos disponíveis
    method_info = [(m, t, c) for m, t, c in method_info if m in results]
    ncols = len(method_info) + 1  # +1 para DoveNet / barras

    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 12))

    # Row 0: Resultados completos
    axes[0, 0].imshow(pil_to_numpy(dovenet_pil))
    axes[0, 0].set_title('DoveNet Output\n(Input ao Harmonizer)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    for i, (method, title, color) in enumerate(method_info):
        axes[0, i + 1].imshow(pil_to_numpy(results[method]))
        axes[0, i + 1].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[0, i + 1].axis('off')

    # Row 1: Zooms + barras
    mask_np = pil_to_numpy(mask_pil)
    dove_np = pil_to_numpy(dovenet_pil).astype(np.float32)

    all_images = [pil_to_numpy(results[m]) for m, _, _ in method_info]
    crops = crop_around_mask(all_images, mask_np, padding=zoom_padding)

    for i, ((method, _, color), crop) in enumerate(zip(method_info, crops)):
        axes[1, i].imshow(crop)
        axes[1, i].set_title(f'{method} (zoom)', fontsize=10, fontweight='bold', color=color)
        axes[1, i].axis('off')
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Barras de mudança vs DoveNet
    diffs = {}
    for key in results:
        result_np = pil_to_numpy(results[key]).astype(np.float32)
        diffs[key] = float(np.mean(np.abs(result_np - dove_np)))

    colors_bar = ['#27ae60', '#e67e22', '#2980b9', '#9b59b6']
    ax_bar = axes[1, -1]
    keys = list(diffs.keys())
    ax_bar.bar(range(len(keys)), [diffs[k] for k in keys],
               color=colors_bar[:len(keys)])
    ax_bar.set_xticks(range(len(keys)))
    ax_bar.set_xticklabels(keys, rotation=15, fontsize=9)
    ax_bar.set_title('Mudança vs DoveNet\n(menor = mais preservação)',
                     fontsize=10, fontweight='bold')
    ax_bar.set_ylabel('Pixels')

    fig.suptitle('Comparação de Métodos de Fusão', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# SD IMPACT: sem SD vs com SD
# ═══════════════════════════════════════════════════════════════════

def plot_sd_impact(
    dovenet_pil: Image.Image,
    composite_pil: Image.Image,
    results_no_sd: Dict[str, Image.Image],
    results_sd: Dict[str, Image.Image],
    ring_mask_pil: Image.Image,
    mask_pil: Image.Image,
    save_path: Optional[str] = None,
    zoom_padding: int = 150,
    dpi: int = 150,
) -> plt.Figure:
    """
    3×4 grid: Row 0 = sem SD | Row 1 = com SD | Row 2 = zoom com SD.
    + plot de diferença SD.

    Baseado em blend_final.ipynb célula 17 (Blend-Only vs Blend+SD).

    Args:
        results_no_sd: dict {'alpha': PIL, 'hybrid': PIL, ...} sem SD
        results_sd: dict {'alpha': PIL, 'hybrid': PIL, ...} com SD
    """
    methods = ['alpha', 'poisson', 'hybrid']
    methods = [m for m in methods if m in results_sd]
    labels = {'alpha': 'Alpha Coseno', 'poisson': 'Poisson Mixed', 'hybrid': 'Híbrido'}
    colors_m = {'alpha': '#27ae60', 'poisson': '#e67e22', 'hybrid': '#2980b9'}

    ncols = len(methods) + 1
    fig, axes = plt.subplots(3, ncols, figsize=(7 * ncols, 18))

    # Col 0: referências
    axes[0, 0].imshow(pil_to_numpy(dovenet_pil))
    axes[0, 0].set_title('DoveNet Output', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(pil_to_numpy(composite_pil))
    axes[1, 0].set_title('Input Original', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[2, 0].imshow(pil_to_numpy(ring_mask_pil), cmap='gray')
    axes[2, 0].set_title('Ring Mask', fontsize=11, fontweight='bold')
    axes[2, 0].axis('off')

    # Cols 1+: métodos
    mask_np = pil_to_numpy(mask_pil)

    for j, m in enumerate(methods):
        col = j + 1
        c = colors_m.get(m, '#333')

        # Row 0: sem SD
        axes[0, col].imshow(pil_to_numpy(results_no_sd[m]))
        axes[0, col].set_title(f'{labels.get(m, m)}\n(Só Blend)', fontsize=11,
                               fontweight='bold', color=c)
        axes[0, col].axis('off')

        # Row 1: com SD
        axes[1, col].imshow(pil_to_numpy(results_sd[m]))
        axes[1, col].set_title(f'{labels.get(m, m)}\n(FINAL — com SD)', fontsize=11,
                               fontweight='bold', color=c)
        axes[1, col].axis('off')
        for spine in axes[1, col].spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Row 2: zoom
        sd_np = pil_to_numpy(results_sd[m])
        crops = crop_around_mask([sd_np], mask_np, padding=zoom_padding)
        axes[2, col].imshow(crops[0])
        axes[2, col].set_title(f'{labels.get(m, m)} (zoom)', fontsize=10,
                               fontweight='bold', color=c)
        axes[2, col].axis('off')
        for spine in axes[2, col].spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig.suptitle('Blend-Only vs Blend + SD Diffusion',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# RING MASK — Visualização dedicada
# ═══════════════════════════════════════════════════════════════════

def plot_ring_mask(
    mask,
    ring_mask,
    image=None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Visualiza a ring mask: Máscara original | Ring mask | Sobreposição.
    """
    mask_np = tensor_to_numpy(mask)
    ring_np = tensor_to_numpy(ring_mask)

    if ring_np.dtype == np.uint8:
        ring_float = ring_np.astype(np.float32) / 255.0
    else:
        ring_float = ring_np.astype(np.float32)

    ncols = 3 if image is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))

    if image is not None:
        img_np = tensor_to_numpy(image)
        ring_overlay = overlay_mask(img_np, ring_float, color=(255, 100, 0), alpha=0.5)
        axes[0].imshow(img_np)
        axes[0].set_title('Image', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        idx = 1
    else:
        idx = 0

    axes[idx].imshow(mask_np, cmap='gray')
    axes[idx].set_title('Object Mask', fontsize=11, fontweight='bold')
    axes[idx].axis('off')

    im = axes[idx + 1].imshow(ring_float, cmap='inferno')
    axes[idx + 1].set_title('Ring Mask\n(Zona SD Inpainting)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# MÉTRICAS DE HARMONIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

def plot_harmonization_metrics(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    phase_boundary: Optional[int] = None,
    title: str = 'DoveNet Training Curves',
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Curvas de treino DoveNet com suporte a 2 fases (phase boundary).

    Baseado em blend_final.ipynb células 7/8 e scripts/train_blend.py.
    """
    metrics = {'Loss/train': train_losses}
    if val_losses:
        metrics['Loss/val'] = val_losses

    fig = plot_training_curves(
        metrics=metrics,
        save_path=save_path,
        title=title,
        phase_boundary=phase_boundary,
        dpi=dpi,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# SD THRESHOLD TEST — min_sd_area
# ═══════════════════════════════════════════════════════════════════

def plot_sd_threshold_test(
    test_cases: List[Dict],
    min_sd_area: int = 2500,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Comparação SD forçado vs adaptativo para diferentes escalas de objecto.

    Baseado em blend_final.ipynb célula 18 (min_sd_area test).

    Args:
        test_cases: lista de dicts com:
            - 'label': str (ex: 'Grande (~50%)')
            - 'area': int (área em px²)
            - 'composite': PIL
            - 'dovenet': PIL
            - 'result_forced': PIL (SD sempre)
            - 'result_smart': PIL (SD adaptativo)
    """
    n = len(test_cases)
    fig, axes = plt.subplots(n, 5, figsize=(30, 6 * n), squeeze=False)

    for row, case in enumerate(test_cases):
        area = case['area']
        sd_applied = area >= min_sd_area

        comp_np = pil_to_numpy(case['composite'])
        dove_np = pil_to_numpy(case['dovenet'])
        forced_np = pil_to_numpy(case['result_forced'])
        smart_np = pil_to_numpy(case['result_smart'])

        axes[row][0].imshow(comp_np)
        axes[row][0].set_title(f"Input\n{case['label']}", fontsize=11, fontweight='bold')
        axes[row][0].axis('off')

        axes[row][1].imshow(dove_np)
        axes[row][1].set_title(f'DoveNet\n(área={area:,}px²)', fontsize=10)
        axes[row][1].axis('off')

        color_forced = '#e74c3c' if not sd_applied else '#27ae60'
        axes[row][2].imshow(forced_np)
        axes[row][2].set_title('SD Forçado', fontsize=11, fontweight='bold', color=color_forced)
        axes[row][2].axis('off')
        if not sd_applied:
            for spine in axes[row][2].spines.values():
                spine.set_edgecolor('#e74c3c')
                spine.set_linewidth(3)
                spine.set_visible(True)

        status = 'SD' if sd_applied else 'Só Blend'
        axes[row][3].imshow(smart_np)
        axes[row][3].set_title(f'Adaptativo\n({status})', fontsize=11,
                               fontweight='bold', color='#2980b9')
        axes[row][3].axis('off')
        for spine in axes[row][3].spines.values():
            spine.set_edgecolor('#2980b9')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Diferença
        diff = compute_diff_map(forced_np.astype(np.float32), smart_np.astype(np.float32))
        if diff.max() > 0:
            axes[row][4].imshow(diff, cmap='hot', vmin=0, vmax=max(diff.max() * 0.7, 1))
            axes[row][4].set_title(f'Diferença\nΔ={np.mean(diff):.1f}px', fontsize=10)
        else:
            axes[row][4].imshow(np.zeros_like(diff), cmap='gray')
            axes[row][4].set_title('Sem diferença', fontsize=10, color='gray')
        axes[row][4].axis('off')

    fig.suptitle(f'Threshold Adaptativo: min_sd_area = {min_sd_area}px²\n'
                 'Objectos pequenos → só Hybrid Blend (sem SD)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)

    return fig
