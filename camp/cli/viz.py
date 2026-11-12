"""Debug figures written by ``--debug`` after train or generate."""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from camp.api import _load_segmentor_weights
from camp.dataset import PairedMaskDataset, attention_collate, segmentor_collate
from camp.figures import (
    FIGURE_DPI,
    mask_metrics,
    plot_generate_trace,
    plot_recommender_trace,
    plot_seg_detail,
    plot_seg_overview,
)
from camp.infer.extractors import extract_recommendations_from_heatmap
from camp.io import class_dirs, find_weights
from camp.models.recommender import build_attention_recommender
from camp.models.segmentor import PretrainedUNet
from camp.viz_utils import save_figure

logger = logging.getLogger(__name__)


def _run_device(camp, args):
    """Device the training call already resolved, or the one on ``camp``."""
    device = getattr(args, "device", None)
    if device is None:
        return camp.device
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _save_segmentor_viz(camp, args, run_dir, n=8):
    """Score the validation split once, then draw a Dice spread of it."""
    source, _target = class_dirs(camp.root, args.class_name)
    val_ds = PairedMaskDataset(
        source,
        split="val",
        split_ratio=args.split_ratio,
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        apply_aug=False,
        for_segmentor=True,
    )
    if len(val_ds) == 0:
        logger.warning("Segmentor debug skipped: validation split is empty")
        return

    model = PretrainedUNet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_dropout=args.decoder_dropout,
    )
    device = _run_device(camp, args)
    viz_dir = run_dir / "debug"
    viz_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(model, run_dir / "best.pth", device)
    model.to(device)
    model.eval()

    scored = _score_segmentor(model, val_ds, device, args.batch_size)
    chosen = _spread(scored, min(n, len(scored)), "dice", higher_is_better=True)
    samples = _segmentor_samples(model, val_ds, chosen, device)
    _write_segmentor_figures(viz_dir, samples, checkpoint, len(scored))


def _save_recommender_viz(camp, args, run_dir, n=8):
    """Score validation heatmaps once, then draw an error spread of them."""
    source, _target = class_dirs(camp.root, args.class_name)
    val_ds = PairedMaskDataset(
        source,
        split="val",
        split_ratio=args.split_ratio,
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        apply_aug=False,
    )
    if len(val_ds) == 0:
        logger.warning("Recommender debug skipped: validation split is empty")
        return

    if args.segmentor_weights:
        seg_path = Path(args.segmentor_weights)
        if not seg_path.is_absolute():
            seg_path = Path(camp.root) / seg_path
    else:
        seg_path = find_weights(camp.root, args.class_name, "segmentor")
    segmentor = PretrainedUNet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    _load_segmentor_weights(segmentor, seg_path)
    model = build_attention_recommender(
        segmentor,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    device = _run_device(camp, args)
    viz_dir = run_dir / "debug"
    viz_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_recommender_checkpoint(model, run_dir / "best.pth")
    model.to(device)
    model.eval()

    scored = _score_recommender(model, val_ds, device, args.batch_size)
    if not scored:
        logger.warning("Recommender debug skipped: no validation patch produced a heatmap")
        return
    chosen = _spread(scored, min(n, len(scored)), "mse", higher_is_better=False)
    samples = _recommender_samples(model, val_ds, chosen, device, args.patch_size)
    _write_recommender_figures(viz_dir, samples, checkpoint, len(scored))


def save_generate_debug(debug, path):
    """Write one generate strip: segmentation, recommendation, paste."""
    fig = plot_generate_trace(debug)
    save_figure(fig, path, dpi=FIGURE_DPI)
    logger.info("Wrote %s", path)


def write_generate_debugs(samples, output_dir, limit=None):
    """Write ``output_dir/debug/<stem>.png`` and ``index.txt`` for samples with debug.

    ``stem`` matches the image and mask saved under ``output_dir/images``
    and ``output_dir/masks``.
    """
    debug_dir = Path(output_dir) / "debug"
    lines = [
        "generate",
        "one figure per sample, named like the image in images/ and masks/",
        "rows  segmentor prediction and the mask that was cut, recommender, paste and blend",
        "",
    ]
    written = 0
    for position, sample in enumerate(samples):
        debug = sample.debug
        if not debug:
            continue
        stem = debug.get("stem") or ("sample_%04d" % position)
        save_generate_debug(debug, debug_dir / ("%s.png" % stem))
        recs = debug.get("recommendations") or []
        chosen = recs[min(int(debug.get("chosen_index") or 0), len(recs) - 1)] if recs else None
        if chosen:
            geometry = "score %.3f  scale %.2f  rot %d  flip %s" % (
                float(chosen.get("score", 0.0)),
                float(chosen.get("scale", 1.0)),
                int(chosen.get("angle", 0)),
                "on" if chosen.get("flip") else "off",
            )
        else:
            geometry = "no recommendation"
        lines.append(
            "%s  %s  mask %s  area %s  boxes %s"
            % (
                stem,
                geometry,
                debug.get("mask_origin", "segmentor"),
                int(debug.get("area") or 0),
                len(recs),
            )
        )
        written += 1
        if limit is not None and written >= limit:
            break
    if written:
        _write_index(debug_dir / "index.txt", lines)
        logger.info("generate debug  %s figures  %s", written, debug_dir)
    return written


def _score_segmentor(model, dataset, device, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        collate_fn=segmentor_collate,
    )
    rows = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            probabilities = torch.sigmoid(model(images)).cpu().numpy()
            targets = batch["segmentor_target_mask"].cpu().numpy()
            paths = batch.get("image_paths") or [""] * images.shape[0]
            for index in range(images.shape[0]):
                target = targets[index, 0] if targets.ndim == 4 else targets[index]
                metrics = mask_metrics(target, probabilities[index, 0])
                rows.append({
                    "index": seen + index,
                    "name": _file_name(paths[index], seen + index),
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                })
            seen += images.shape[0]
    return rows


def _segmentor_samples(model, dataset, chosen, device):
    items = [dataset[row["index"]] for row in chosen]
    batch = segmentor_collate(items)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(batch["images"].to(device))).cpu().numpy()
    samples = []
    for position, row in enumerate(chosen):
        item = items[position]
        image = item["image"].numpy().transpose(1, 2, 0)
        target = item["segmentor_target_mask"][0].numpy()
        probability = probabilities[position, 0]
        metrics = mask_metrics(target, probability)
        samples.append({
            "name": row["name"],
            "band": "",
            "image": image,
            "target": target,
            "probability": probability,
            **metrics,
        })
    samples.sort(key=lambda sample: sample["dice"])
    return _label_worst_first(samples)


def _write_segmentor_figures(viz_dir, samples, checkpoint, total):
    overview = plot_seg_overview(samples)
    save_figure(overview, viz_dir / "seg_overview.png", dpi=FIGURE_DPI)
    logger.info("Wrote %s (%s of %s)", viz_dir / "seg_overview.png", len(samples), total)
    lines = [
        "segmentor",
        "checkpoint  %s" % checkpoint,
        "selection  %s of %s validation images, spread by Dice, worst at the top" % (len(samples), total),
        "",
    ]
    for position, sample in enumerate(samples):
        fig = plot_seg_detail(sample)
        path = viz_dir / ("seg_%02d_%s.png" % (position, sample["band"]))
        save_figure(fig, path, dpi=FIGURE_DPI)
        logger.info(
            "Wrote %s  Dice %.3f  IoU %.3f",
            path,
            sample["dice"],
            sample["iou"],
        )
        lines.append(
            "%02d  %-7s  dice %.3f  iou %.3f  precision %.3f  recall %.3f  %s"
            % (
                position,
                sample["band"],
                sample["dice"],
                sample["iou"],
                sample["precision"],
                sample["recall"],
                sample["name"],
            )
        )
    _write_index(viz_dir / "index.txt", lines)


def _score_recommender(model, dataset, device, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
        collate_fn=attention_collate,
    )
    rows = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            count = batch["images"].shape[0]
            paths = batch.get("image_paths") or [""] * count
            prediction = _predict_heatmaps(model, batch, device)
            if prediction is None:
                seen += count
                continue
            heatmaps = batch["heatmaps"]
            indices = batch["sample_indices"]
            for local in range(count):
                patch_ids = (indices == local).nonzero(as_tuple=True)[0]
                if patch_ids.numel() == 0:
                    continue
                target = heatmaps[patch_ids]
                pred = prediction[patch_ids]
                if pred.dim() == 4:
                    pred = pred[:, 0]
                if pred.shape[-2:] != target.shape[-2:]:
                    pred = F.interpolate(
                        pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
                    )[:, 0]
                mse = float(torch.mean((pred - target) ** 2).item())
                rows.append({
                    "index": seen + local,
                    "name": _file_name(paths[local], seen + local),
                    "mse": mse,
                })
            seen += count
    return rows


def _recommender_samples(model, dataset, chosen, device, patch_size):
    items = [dataset[row["index"]] for row in chosen]
    batch = attention_collate(items)
    with torch.no_grad():
        prediction = _predict_heatmaps(model, batch, device)
    samples = []
    indices = batch["sample_indices"]
    for position, row in enumerate(chosen):
        patch_ids = (indices == position).nonzero(as_tuple=True)[0]
        if prediction is None or patch_ids.numel() == 0:
            continue
        heatmaps = batch["heatmaps"][patch_ids]
        strength = heatmaps.amax(dim=(1, 2))
        pick = int(torch.argmax(strength).item())
        patch_index = patch_ids[pick]
        target = heatmaps[pick].numpy()
        pred = prediction[patch_index, 0].numpy()
        if pred.shape != target.shape:
            pred_t = torch.from_numpy(pred)[None, None]
            pred = F.interpolate(pred_t, size=target.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        image = batch["images"][position].numpy().transpose(1, 2, 0)
        context = batch["patches_with_context"][patch_index].numpy().transpose(1, 2, 0)
        context_mask = batch["patches_context_masks"][patch_index, 0].numpy()
        train_mask = batch["segmentor_target_mask"][position, 0].numpy()
        patch_box = [float(value) for value in batch["boxes"][patch_index].tolist()]
        inside = target > 0.5
        peak_in = float(pred[inside].max()) if inside.any() else 0.0
        peak_out = float(pred[~inside].max()) if (~inside).any() else 0.0
        boxes = extract_recommendations_from_heatmap(
            pred,
            image_hw(image),
            _patch_window(context, patch_size),
            top_k=5,
        )
        samples.append({
            "name": row["name"],
            "band": "",
            "image": image,
            "target_mask": train_mask,
            "patch_box": patch_box,
            "context": context,
            "context_mask": context_mask,
            "target_heatmap": target,
            "predicted_heatmap": pred,
            "boxes": boxes,
            "mse": float(np.mean((pred - target) ** 2)),
            "peak_in": peak_in,
            "peak_out": peak_out,
            "patch_label": "patch %s/%s" % (pick + 1, int(patch_ids.numel())),
        })
    samples.sort(key=lambda sample: sample["mse"], reverse=True)
    return _label_worst_first(samples)


def _write_recommender_figures(viz_dir, samples, checkpoint, total):
    lines = [
        "recommender",
        "checkpoint  %s" % checkpoint,
        "selection  %s of %s validation images, spread by heatmap MSE, worst at the top" % (len(samples), total),
        "boxes  window mean, stride patch/4, NMS IoU 0.3",
        "",
    ]
    for position, sample in enumerate(samples):
        fig = plot_recommender_trace(sample)
        path = viz_dir / ("rec_%02d_%s.png" % (position, sample["band"]))
        save_figure(fig, path, dpi=FIGURE_DPI)
        logger.info("Wrote %s  MSE %.4f", path, sample["mse"])
        lines.append(
            "%02d  %-7s  mse %.4f  peak_in %.3f  peak_out %.3f  %s  %s"
            % (
                position,
                sample["band"],
                sample["mse"],
                sample["peak_in"],
                sample["peak_out"],
                sample["patch_label"],
                sample["name"],
            )
        )
    _write_index(viz_dir / "index.txt", lines)


def _predict_heatmaps(model, batch, device):
    patches = batch["patches"]
    if patches.numel() == 0:
        return None
    images = batch["images"].to(device)
    sample_indices = batch["sample_indices"].to(device)
    context = batch.get("patches_with_context")
    context_mask = batch.get("patches_context_masks")
    if context is not None:
        context = context.to(device)
    if context_mask is not None:
        context_mask = context_mask.to(device)
    heatmap = model(
        images[sample_indices],
        patch=patches.to(device),
        patch_with_context=context,
        context_mask=context_mask,
    )
    if isinstance(heatmap, tuple):
        heatmap = heatmap[0]
    return heatmap.detach().cpu()


def _spread(rows, count, key, higher_is_better):
    """Pick ``count`` rows evenly from the best score to the worst."""
    ordered = sorted(rows, key=lambda row: row[key], reverse=higher_is_better)
    if len(ordered) <= count:
        return list(ordered)
    positions = [int(round(pos)) for pos in np.linspace(0, len(ordered) - 1, count)]
    seen = set()
    chosen = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            chosen.append(ordered[pos])
    return chosen


def _label_worst_first(samples):
    """Label a list that is already ordered worst to best."""
    if not samples:
        return samples
    if len(samples) == 1:
        samples[0]["band"] = "only"
        return samples
    samples[0]["band"] = "worst"
    samples[-1]["band"] = "best"
    middle = (len(samples) - 1) / 2
    for index in range(1, len(samples) - 1):
        samples[index]["band"] = "low" if index < middle else "high"
    return samples


def _load_checkpoint(model, path, device):
    if not path.exists():
        logger.warning("No checkpoint at %s; figures use the current weights", path)
        return "missing  %s" % path
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return "loaded  %s" % path


def _load_recommender_checkpoint(model, path):
    if not path.exists():
        logger.warning("No checkpoint at %s; figures use the current weights", path)
        return "missing  %s" % path
    model.load_model(str(path))
    return "loaded  %s" % path


def _file_name(path, index):
    name = Path(path).name if path else ""
    return name or ("sample_%s" % index)


def image_hw(image):
    array = np.asarray(image)
    return array.shape[:2]


def _patch_window(context, patch_size):
    array = np.asarray(context)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        height, width = array.shape[1], array.shape[2]
    else:
        height, width = array.shape[0], array.shape[1]
    if height < 2 or width < 2:
        return (int(patch_size), int(patch_size))
    return (int(height), int(width))


def _write_index(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
