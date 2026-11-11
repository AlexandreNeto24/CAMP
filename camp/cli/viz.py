"""Debug figures written by ``--save-viz``."""

import logging
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from camp.api import _load_segmentor_weights
from camp.dataset import PairedMaskDataset, attention_collate
from camp.figures import plot_batch_debug, plot_recommendations, plot_score_distribution, plot_seg_examples, plot_seg_overlay, plot_seg_worst
from camp.io import class_dirs, find_weights
from camp.models.recommender import build_attention_recommender
from camp.models.segmentor import PretrainedUNet
from camp.viz_utils import save_figure

logger = logging.getLogger(__name__)


def _save_segmentor_viz(camp, args, run_dir):
    """Write segmentor debug figures for one training run."""
    source, _target = class_dirs(camp.root, args.class_name)
    val_ds = PairedMaskDataset(
        source,
        split="val",
        split_ratio=args.split_ratio,
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        apply_aug=False,
        p_remove=0.0,
    )
    model = PretrainedUNet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_dropout=args.decoder_dropout,
    )
    device = camp.device
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.save_viz, len(val_ds))
    logger.info("Writing %s segmentor debug figures", n)

    best_path = run_dir / "best.pth"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, n), shuffle=False,
        num_workers=2, collate_fn=attention_collate,
    )
    images_all, masks_gt_all, masks_pred_all, dice_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            masks = batch["segmentor_target_mask"].to(device)
            preds = torch.sigmoid(model(imgs))
            for i in range(imgs.shape[0]):
                if len(images_all) >= n:
                    break
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                gt_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i, 0].cpu().numpy()
                pred_bin = (pred_np > 0.5).astype(np.float32)
                intersection = (pred_bin * gt_np).sum()
                dice = (2 * intersection) / (pred_bin.sum() + gt_np.sum() + 1e-8)
                images_all.append(img_np)
                masks_gt_all.append(gt_np)
                masks_pred_all.append(pred_np)
                dice_all.append(dice)
            if len(images_all) >= n:
                break

    fig = plot_seg_examples(images_all, masks_gt_all, masks_pred_all)
    save_figure(fig, viz_dir / "seg_examples.png")
    logger.info("Wrote %s (%s samples)", viz_dir / "seg_examples.png", len(images_all))

    if len(dice_all) >= 2:
        best_idx = int(np.argmax(dice_all))
        worst_idx = int(np.argmin(dice_all))
        for label, idx in (("best", best_idx), ("worst", worst_idx)):
            fig = plot_seg_overlay(images_all[idx], masks_gt_all[idx], masks_pred_all[idx])
            save_figure(fig, viz_dir / f"seg_overlay_{label}.png")
            logger.info(
                "Wrote %s (Dice=%.3f)",
                viz_dir / f"seg_overlay_{label}.png",
                dice_all[idx],
            )

    if len(dice_all) >= 4:
        fig = plot_seg_worst(images_all, masks_gt_all, masks_pred_all, dice_all)
        save_figure(fig, viz_dir / "seg_worst.png")
        logger.info("Wrote %s", viz_dir / "seg_worst.png")


def _save_recommender_viz(camp, args, run_dir):
    """Write recommender debug figures for one training run."""
    source, _target = class_dirs(camp.root, args.class_name)
    val_ds = PairedMaskDataset(
        source,
        split="val",
        split_ratio=args.split_ratio,
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        apply_aug=False,
        p_remove=0.0,
    )
    if args.segmentor_weights:
        seg_path = Path(args.segmentor_weights)
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

    device = camp.device
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.save_viz, len(val_ds))
    logger.info("Writing %s recommender debug figures", n)

    best_path = run_dir / "best.pth"
    if best_path.exists():
        model.load_model(str(best_path))
    model.to(device)
    model.eval()

    loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, n), shuffle=False,
        num_workers=2, collate_fn=attention_collate,
    )
    images_all, masks_all, patches_all, heatmaps_all, scores_all = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            masks = batch["segmentor_target_mask"].to(device)
            patches = batch["patches"].to(device)
            sample_indices = batch["sample_indices"].to(device)
            patches_ctx = batch.get("patches_with_context", None)
            if patches_ctx is not None:
                patches_ctx = patches_ctx.to(device)
            if patches.numel() == 0:
                continue

            batch_images = imgs[sample_indices]
            result = model(batch_images, patch=patches, patch_with_context=patches_ctx)
            heatmap = result[0] if isinstance(result, tuple) else result

            unique_indices = sample_indices.unique().cpu().tolist()
            for ui in unique_indices:
                if len(images_all) >= n:
                    break
                i = int(ui)
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                mask_np = masks[i, 0].cpu().numpy() if masks.dim() == 4 else masks[i].cpu().numpy()
                patch_mask = (sample_indices == i)
                patch_idx = patch_mask.nonzero(as_tuple=True)[0][0].item()
                patch_np = patches[patch_idx, :3].cpu().numpy().transpose(1, 2, 0)
                patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
                hm_np = heatmap[patch_idx, 0].cpu().numpy() if heatmap is not None and heatmap.dim() >= 3 else None
                images_all.append(img_np)
                masks_all.append(mask_np)
                patches_all.append(patch_np)
                heatmaps_all.append(hm_np)
                if hm_np is not None:
                    scores_all.append(float(hm_np.max()))
            if len(images_all) >= n:
                break

    try:
        fig = plot_batch_debug(images_all, masks_all, patches_all, heatmaps_all)
        save_figure(fig, viz_dir / "recommender_batch.png")
        logger.info("Wrote %s (%s samples)", viz_dir / "recommender_batch.png", len(images_all))
    except Exception as exc:
        logger.warning("recommender_batch.png failed: %s", exc)

    for idx in range(min(3, len(images_all))):
        try:
            recs = []
            img_h, img_w = images_all[idx].shape[:2]
            if heatmaps_all[idx] is not None:
                hm = heatmaps_all[idx]
                labeled, num_features = ndimage.label(hm > hm.max() * 0.5)
                for j in range(1, min(num_features + 1, 6)):
                    ys, xs = np.where(labeled == j)
                    if len(ys) > 0:
                        cy, cx = int(ys.mean()), int(xs.mean())
                        sy = img_h / hm.shape[0]
                        sx = img_w / hm.shape[1]
                        cy_img, cx_img = int(cy * sy), int(cx * sx)
                        ps = args.patch_size // 2
                        recs.append({
                            "box": [
                                max(0, cx_img - ps), max(0, cy_img - ps),
                                min(img_w, cx_img + ps), min(img_h, cy_img + ps),
                            ],
                            "score": float(hm[cy, cx]),
                        })
                recs.sort(key=lambda rec: rec["score"], reverse=True)
            fig = plot_recommendations(images_all[idx], patches_all[idx], recs, heatmaps_all[idx])
            save_figure(fig, viz_dir / f"recommendation_{idx}.png")
            logger.info("Wrote %s", viz_dir / f"recommendation_{idx}.png")
        except Exception as exc:
            logger.warning("recommendation_%s.png failed: %s", idx, exc)

    if scores_all:
        try:
            fig = plot_score_distribution({"score": scores_all})
            save_figure(fig, viz_dir / "score_distribution.png")
            logger.info("Wrote %s", viz_dir / "score_distribution.png")
        except Exception as exc:
            logger.warning("score_distribution.png failed: %s", exc)
