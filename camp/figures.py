"""Debug figures for the segmentor, the recommender, and generate.

Each figure is a strip of stages. Panels keep continuous maps as float
so a probability is not flattened to gray before it is drawn. These
functions do not load models.
"""

from pathlib import Path

import cv2
import numpy as np

from camp.viz_utils import as_float01, as_uint8_image, save_figure

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURE_DPI = 160
RANK_COLORS = ["#39ff14", "#00d5ff", "#ffb000", "#ff4fd8", "#ffffff"]
TARGET_RGB = (0, 200, 60)
PREDICTION_RGB = (0, 140, 255)
OBJECT_RGB = (255, 60, 60)
INSTANCE_EDGE = "#ff3c3c"


def mask_metrics(target, prediction, threshold=0.5):
    """Dice, IoU, precision, and recall at ``threshold``.

    ``target`` and ``prediction`` are probability maps. A target above 0.5
    counts as positive, which matches the context-ring labels (0 or 0.9)
    and a 0.5 decision on the segmentor output. Both empty is a perfect
    score.
    """
    gt_bin = as_float01(target) > 0.5
    pred_bin = as_float01(prediction) > threshold
    tp = int(np.logical_and(gt_bin, pred_bin).sum())
    fp = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    dice = (2 * tp) / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def plot_seg_overview(samples, save_path=None, dpi=FIGURE_DPI):
    """One row per sample: source, probability, contours, error. Worst at the top."""
    n = len(samples)
    fig, axes = plt.subplots(
        n, 4, figsize=(17.0, 4.1 * n), squeeze=False, constrained_layout=True
    )
    titles = (
        "Source",
        "Probability",
        "Target green, prediction blue",
        "Error  ·  TP green, FP red, FN yellow",
    )
    for row, sample in enumerate(samples):
        metrics = _metrics_of(sample)
        _show(axes[row, 0], as_uint8_image(sample["image"]), "")
        _show(
            axes[row, 1],
            as_float01(sample["probability"]),
            "",
            cmap="inferno",
            vmin=0,
            vmax=1,
            colorbar=True,
        )
        _show(
            axes[row, 2],
            _contour_pair(sample["image"], sample["target"], sample["probability"]),
            "",
            interpolation="nearest",
        )
        _show(axes[row, 3], _error_overlay(sample), "", interpolation="nearest")
        axes[row, 0].set_ylabel(
            "%s\nDice %.3f   IoU %.3f\n%s"
            % (sample.get("band", ""), metrics["dice"], metrics["iou"], _short(sample.get("name", ""), 26)),
            fontsize=10,
            fontweight="bold",
        )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")
    fig.suptitle(
        "Segmentor  ·  validation spread by Dice, worst at the top  ·  threshold 0.5",
        fontsize=14,
        fontweight="bold",
    )
    return _finish(fig, save_path, dpi)


def plot_seg_detail(sample, save_path=None, dpi=FIGURE_DPI):
    """Inputs and probability on the first row, target, prediction, and error on the second."""
    metrics = _metrics_of(sample)
    image = as_uint8_image(sample["image"])
    probability = as_float01(sample["probability"])
    target = as_float01(sample["target"])
    fig, axes = plt.subplots(
        2, 3, figsize=(15.5, 10.6), squeeze=False, constrained_layout=True
    )
    _show(axes[0, 0], image, "Source")
    _show(
        axes[0, 1],
        probability,
        "Probability  ·  max %.2f" % float(probability.max()),
        cmap="inferno",
        vmin=0,
        vmax=1,
        colorbar=True,
    )
    _show(
        axes[0, 2],
        (probability > 0.5).astype(np.float32),
        "Prediction > 0.5  ·  %s of the image" % _coverage(probability),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _show(
        axes[1, 0],
        _draw_contour(image, target, TARGET_RGB, thickness=2),
        "Target  ·  %s of the image" % _coverage(target),
        interpolation="nearest",
    )
    _show(
        axes[1, 1],
        _draw_contour(image, probability, PREDICTION_RGB, thickness=2),
        "Prediction contour",
        interpolation="nearest",
    )
    _show(
        axes[1, 2],
        _error_overlay(sample),
        "Error  ·  TP green, FP red, FN yellow",
        interpolation="nearest",
    )
    fig.suptitle(
        "Segmentor  ·  %s  ·  %s\nDice %.3f    IoU %.3f    precision %.3f    recall %.3f"
        % (
            sample.get("band", "sample"),
            _short(sample.get("name", ""), 48),
            metrics["dice"],
            metrics["iou"],
            metrics["precision"],
            metrics["recall"],
        ),
        fontsize=13,
        fontweight="bold",
    )
    return _finish(fig, save_path, dpi)


def plot_recommender_trace(sample, save_path=None, dpi=FIGURE_DPI):
    """Inputs on the first row, target against prediction on the second."""
    fig, axes = plt.subplots(
        2, 4, figsize=(17.5, 10.0), squeeze=False, constrained_layout=True
    )
    image = as_uint8_image(sample["image"])
    _show(axes[0, 0], image, "Target image  ·  instance box in red")
    patch_box = sample.get("patch_box")
    if patch_box is not None:
        _draw_rect(axes[0, 0], patch_box, INSTANCE_EDGE, linewidth=2.0)
    _show(
        axes[0, 1],
        _mask_overlay(image, sample["target_mask"], TARGET_RGB),
        "Training target  ·  context ring",
    )
    context = as_uint8_image(sample["context"])
    _show(
        axes[0, 2],
        _draw_contour(context, sample["context_mask"], OBJECT_RGB, thickness=2),
        "Context crop  ·  object outlined",
    )
    _show(
        axes[0, 3],
        _mask_overlay(context, sample["context_mask"], OBJECT_RGB, alpha=0.55),
        "Excluded object over the crop",
    )

    target_hm = as_float01(sample["target_heatmap"])
    pred_hm = as_float01(sample["predicted_heatmap"])
    if pred_hm.shape != target_hm.shape:
        pred_hm = cv2.resize(pred_hm, (target_hm.shape[1], target_hm.shape[0]))
    error = np.abs(pred_hm - target_hm)
    _show(
        axes[1, 0],
        target_hm,
        "Target heatmap  ·  %s" % _range_phrase(target_hm),
        cmap="inferno",
        vmin=0,
        vmax=1,
        colorbar=True,
    )
    _show(
        axes[1, 1],
        pred_hm,
        "Predicted heatmap  ·  %s" % _range_phrase(pred_hm),
        cmap="inferno",
        vmin=0,
        vmax=1,
        colorbar=True,
    )
    _show(
        axes[1, 2],
        error,
        "Absolute error  ·  max %.2f  mean %.2f" % (float(error.max()), float(error.mean())),
        cmap="magma",
        vmin=0,
        vmax=1,
        colorbar=True,
    )
    boxes = sample.get("boxes") or []
    _show(axes[1, 3], image, "Placement boxes")
    _draw_boxes(axes[1, 3], boxes, image.shape[:2], image.shape[:2])

    axes[0, 0].set_ylabel("Input", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Prediction", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Recommender  ·  %s  ·  %s  ·  %s\nMSE %.4f    peak inside target %.3f    peak outside %.3f"
        % (
            sample.get("band", "sample"),
            _short(sample.get("name", ""), 48),
            sample.get("patch_label", ""),
            float(sample.get("mse", 0.0)),
            float(sample.get("peak_in", 0.0)),
            float(sample.get("peak_out", 0.0)),
        ),
        fontsize=13,
        fontweight="bold",
    )
    return _finish(fig, save_path, dpi)


def plot_generate_trace(debug, save_path=None, dpi=FIGURE_DPI):
    """Full generate path for one sample.

    Row 1 is the segmentation: the source, the segmentor probability,
    the mask that was actually cut, and the chosen instance.
    Row 2 is the recommender: the target, the winning heatmap, the boxes
    it proposed, and the context crop it was given.
    Row 3 is the paste: extraction, alpha, harmonization, and the result.
    """
    fig, axes = plt.subplots(
        3, 4, figsize=(17.5, 14.6), squeeze=False, constrained_layout=True
    )
    source = as_uint8_image(debug["source_image"])
    target = as_uint8_image(debug["target"])
    chosen = _chosen(debug)
    boxes = debug.get("recommendations") or []
    instance = _instance_full(debug, source.shape[:2])

    _show(axes[0, 0], source, "Source")
    _show_segmentor_prediction(axes[0, 1], debug)
    if debug.get("mask_origin") == "given":
        mask_title = "Given mask  ·  this is what was cut"
    else:
        mask_title = "Segmentor mask  ·  threshold 0.5, this is what was cut"
    _show(axes[0, 2], _mask_overlay(source, debug["source_mask"], TARGET_RGB), mask_title)
    _show(
        axes[0, 3],
        _draw_contour(source, instance, TARGET_RGB),
        "Chosen instance  ·  area %s px" % int(debug.get("area") or 0),
    )

    _show(axes[1, 0], target, "Target")
    heatmap = debug.get("heatmap")
    if heatmap is None:
        _show(axes[1, 1], np.zeros(target.shape[:2], dtype=np.float32), "Heatmap unavailable")
    else:
        heat = as_float01(heatmap)
        _show(
            axes[1, 1],
            heat,
            "Recommender heatmap  ·  %s" % _range_phrase(heat),
            cmap="inferno",
            vmin=0,
            vmax=1,
            colorbar=True,
        )
        if chosen is not None:
            _draw_boxes(axes[1, 1], [chosen], target.shape[:2], heat.shape[:2])
    _show(axes[1, 2], target, "Recommendations")
    _draw_boxes(axes[1, 2], boxes, target.shape[:2], target.shape[:2])
    context = as_uint8_image(debug["context"])
    _show(
        axes[1, 3],
        _draw_contour(context, debug["context_mask"], OBJECT_RGB),
        "Context sent to the recommender",
    )

    stages = debug.get("stages") or {}
    foreground = stages.get("foreground")
    if foreground is None:
        _show(axes[2, 0], np.zeros((32, 32), dtype=np.float32), "Foreground unavailable")
    else:
        _show(axes[2, 0], as_uint8_image(foreground), "Extracted foreground  ·  crop")
    alpha = stages.get("alpha")
    if alpha is None:
        _show(axes[2, 1], np.zeros((32, 32), dtype=np.float32), "Alpha unavailable")
    else:
        _show(
            axes[2, 1],
            as_float01(alpha),
            "Blend alpha  ·  crop",
            cmap="gray",
            vmin=0,
            vmax=1,
            colorbar=True,
        )
    _show(axes[2, 2], _stage_rgb(stages.get("harmonized"), target), "Harmonized foreground  ·  crop")
    composite = stages.get("composite")
    pasted = target if composite is None else as_uint8_image(composite)
    _show(axes[2, 3], pasted, "Pasted and blended")

    axes[0, 0].set_ylabel("Segmentation", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Recommender", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("Paste and blend", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Generate  ·  %s\n%s" % (debug.get("stem", "sample"), _geometry_phrase(chosen)),
        fontsize=13,
        fontweight="bold",
    )
    return _finish(fig, save_path, dpi)


def _metrics_of(sample):
    if "dice" in sample:
        return sample
    return mask_metrics(sample["target"], sample["probability"])


def _short(name, limit):
    stem = Path(str(name)).stem if name else ""
    if len(stem) <= limit:
        return stem
    return stem[: limit - 1] + "…"


def _coverage(mask):
    return "%.1f%%" % (100.0 * float(np.mean(as_float01(mask) > 0.5)))


def _range_phrase(values):
    return "range %.2f–%.2f" % (float(np.min(values)), float(np.max(values)))


def _error_overlay(sample):
    image = as_uint8_image(sample["image"])
    target = as_float01(sample["target"]) > 0.5
    prediction = as_float01(sample["probability"]) > 0.5
    if target.shape != image.shape[:2]:
        target = cv2.resize(target.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if prediction.shape != image.shape[:2]:
        prediction = cv2.resize(
            prediction.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        ) > 0
    base = image.astype(np.float32) * 0.45
    painted = base.copy()
    painted[np.logical_and(target, prediction)] = (0, 170, 60)
    painted[np.logical_and(prediction, np.logical_not(target))] = (220, 40, 40)
    painted[np.logical_and(target, np.logical_not(prediction))] = (235, 200, 20)
    return painted.clip(0, 255).astype(np.uint8)


def _contour_pair(image, target, prediction):
    # Target is drawn last and wider so the prediction frame cannot hide it.
    outlined = _draw_contour(image, prediction, PREDICTION_RGB, thickness=1)
    return _draw_contour(outlined, target, TARGET_RGB, thickness=2)


def _mask_overlay(image, mask, color, alpha=0.45):
    base = as_uint8_image(image)
    mask_bin = as_float01(mask) > 0.5
    if mask_bin.shape != base.shape[:2]:
        mask_bin = cv2.resize(
            mask_bin.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
        ) > 0
    overlay = base.copy()
    overlay[mask_bin] = color
    mixed = base.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    mixed[np.logical_not(mask_bin)] = base[np.logical_not(mask_bin)]
    return mixed.clip(0, 255).astype(np.uint8)


def _draw_contour(image, mask, color, thickness=2):
    canvas = as_uint8_image(image)
    binary = (as_float01(mask) > 0.5).astype(np.uint8)
    if binary.shape != canvas.shape[:2]:
        binary = cv2.resize(binary, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = canvas.copy()
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _instance_full(debug, image_hw):
    canvas = np.zeros(image_hw, dtype=np.uint8)
    crop = as_float01(debug["instance_mask"])
    x1, y1, x2, y2 = [int(v) for v in debug["bbox"]]
    height, width = crop.shape[:2]
    y1 = max(0, min(y1, image_hw[0] - 1))
    x1 = max(0, min(x1, image_hw[1] - 1))
    y2 = min(image_hw[0], y1 + height)
    x2 = min(image_hw[1], x1 + width)
    if y2 <= y1 or x2 <= x1:
        return canvas
    canvas[y1:y2, x1:x2] = (crop[: y2 - y1, : x2 - x1] > 0.5).astype(np.uint8) * 255
    return canvas


def _draw_rect(ax, box, color, linewidth=2.0, scale=(1.0, 1.0)):
    x1, y1, x2, y2 = [float(v) for v in box]
    sy, sx = scale
    ax.add_patch(
        mpatches.Rectangle(
            (x1 * sx, y1 * sy),
            (x2 - x1) * sx,
            (y2 - y1) * sy,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )


def _draw_boxes(ax, boxes, image_hw, panel_hw):
    """Colored rectangles with the rank and score written on each box."""
    if not boxes:
        ax.set_title(ax.get_title() + "  ·  none", fontsize=10, fontweight="bold")
        return
    scale = (panel_hw[0] / max(image_hw[0], 1), panel_hw[1] / max(image_hw[1], 1))
    shown = list(enumerate(boxes[: len(RANK_COLORS)]))
    # Worst first, so the best box is the one left on top.
    for index, rec in reversed(shown):
        _draw_rect(
            ax,
            rec["box"],
            RANK_COLORS[index],
            linewidth=2.6 if index == 0 else 1.6,
            scale=scale,
        )
        _score_label(ax, rec, index, scale, panel_hw)


def _score_label(ax, rec, index, scale, panel_hw):
    x1, y1, _x2, _y2 = [float(value) for value in rec["box"]]
    sy, sx = scale
    text_x = x1 * sx
    # Step labels by rank so boxes that share a corner stay readable.
    text_y = y1 * sy - 4 - index * 14
    if text_y < 10:
        text_y = min(y1 * sy + 12 + index * 14, panel_hw[0] - 4)
    ax.text(
        text_x,
        text_y,
        "#%d  %.3f" % (index + 1, float(rec.get("score", 0.0))),
        color=RANK_COLORS[index],
        fontsize=8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )


def _show_segmentor_prediction(ax, debug):
    probability = debug.get("segmentor_probability")
    if probability is None:
        _show(ax, np.zeros((32, 32), dtype=np.float32), "Segmentor prediction unavailable")
        return
    values = as_float01(probability)
    if debug.get("mask_origin") == "given":
        role = "not used to cut"
    else:
        role = "used to cut"
    _show(
        ax,
        values,
        "Segmentor prediction  ·  %s  ·  %s" % (_range_phrase(values), role),
        cmap="inferno",
        vmin=0,
        vmax=1,
        colorbar=True,
    )


def _geometry_phrase(chosen):
    if not chosen:
        return "no recommendation"
    flip = "on" if chosen.get("flip") else "off"
    return "score %.3f    scale %.2f    rot %d    flip %s" % (
        float(chosen.get("score", 0.0)),
        float(chosen.get("scale", 1.0)),
        int(chosen.get("angle", 0)),
        flip,
    )


def _chosen(debug):
    boxes = debug.get("recommendations") or []
    index = int(debug.get("chosen_index") or 0)
    if not boxes:
        return None
    return boxes[min(index, len(boxes) - 1)]


def _stage_rgb(value, fallback):
    if value is None:
        return as_uint8_image(fallback)
    return as_uint8_image(value)


def _show(ax, image, title, cmap=None, vmin=None, vmax=None, colorbar=False, interpolation="bilinear"):
    shown = image
    if shown.ndim == 2 and cmap is None:
        cmap = "gray"
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    _quiet(ax)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def _quiet(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _finish(fig, save_path, dpi):
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    return fig
