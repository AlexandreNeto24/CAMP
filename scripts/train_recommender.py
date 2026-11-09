#!/usr/bin/env python
"""Train the attention recommender on a frozen segmentor.

Phase 1 trains the head. Phase 2 fine-tunes it when --finetune-epochs is above 0.

    python scripts/train_recommender.py --class metaplasia --epochs 60
    python scripts/train_recommender.py --class polyps --epochs 60 --finetune-epochs 20
    python scripts/train_recommender.py --class metaplasia --segmentor-weights weights/metaplasia/segmentor/best.pth
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from camp import CAMP
from camp.api import _load_segmentor_weights
from camp.dataset import PairedMaskDataset, attention_collate
from camp.io import class_dirs, find_weights
from camp.models.recommender import AttentionRecommender
from camp.models.segmentor import PretrainedUNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAMP — Treino do Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--class-name", "--class", dest="class_name", required=True,
                        help="Classe alvo (ex: metaplasia, polyps, airplane)")
    parser.add_argument("--data-dir", default="data",
                        help="Directório base dos datasets (default: data)")
    parser.add_argument("--encoder", default="efficientnet-b3",
                        help="Backbone encoder do segmentador (default: efficientnet-b3)")
    parser.add_argument("--embedding-dim", type=int, default=2048,
                        help="Dimensão do embedding (default: 2048)")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Número de attention heads (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="Dropout rate (default: 0.10)")
    parser.add_argument("--segmentor-weights", default=None,
                        help="Caminho para pesos do segmentador (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Épocas fase 1 (default: 60)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate fase 1 (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 30)")
    parser.add_argument("--finetune-epochs", type=int, default=20,
                        help="Épocas de fine-tuning (default: 20, 0=skip)")
    parser.add_argument("--finetune-lr", type=float, default=5e-5,
                        help="Learning rate fine-tuning (default: 5e-5)")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Tamanho da imagem (default: 512)")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Tamanho dos patches (default: 128)")
    parser.add_argument("--split-ratio", type=float, default=0.2,
                        help="Fracção para validação (default: 0.2)")
    parser.add_argument("--p-remove", type=float, default=0.0,
                        help="Probabilidade de blindfold (default: 0.0)")
    parser.add_argument("--output", default=None,
                        help="Directório de output (default: weights/<class>/recommender/)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--debug", action="store_true",
                        help="Activar visualizações de debug (equivale a --save-viz 8)")
    parser.add_argument("--save-viz", type=int, default=0, metavar="N",
                        help="Guardar N exemplos visuais no final do treino (default: 0)")
    parser.add_argument("--viz-every", type=int, default=5, metavar="K",
                        help="Guardar vizualizações a cada K épocas (requer --save-viz; default: 5)")
    args = parser.parse_args()
    if args.debug and args.save_viz == 0:
        args.save_viz = 8
    return args


def _root(data_dir):
    path = Path(data_dir)
    if not path.is_absolute():
        path = project_root / path
    if path.name != "data":
        print(f"Erro: --data-dir tem de ser a pasta 'data' (recebido: {data_dir})")
        sys.exit(1)
    return path.parent


def main():
    args = parse_args()
    camp = CAMP(
        root=_root(args.data_dir),
        device=args.device,
        verbose=True,
        image_size=(args.image_size, args.image_size),
    )
    try:
        best = camp.train_recommender(
            args.class_name,
            encoder=args.encoder,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            epochs=args.epochs,
            finetune_epochs=args.finetune_epochs,
            lr=args.lr,
            finetune_lr=args.finetune_lr,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            patience=args.patience,
            image_size=args.image_size,
            patch_size=args.patch_size,
            split_ratio=args.split_ratio,
            p_remove=args.p_remove,
            device=args.device,
            output=args.output,
            segmentor_weights=args.segmentor_weights,
        )
    except FileNotFoundError as exc:
        print(f"Erro: {exc}")
        sys.exit(1)

    run_dir = Path(best).parent
    if args.save_viz > 0:
        _save_recommender_viz(camp, args, run_dir)

    print(f"Pesos: {best}")
    print(f"Config: {run_dir / 'config.yaml'}")


def _save_recommender_viz(camp, args, run_dir):
    """Write debug figures with the helpers in src.paste.viz."""
    import numpy as np
    import torch
    from scipy import ndimage
    from torch.utils.data import DataLoader

    from src.paste.viz import plot_batch_debug, plot_recommendations, plot_score_distribution
    from src.viz_utils import save_figure

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
    model = AttentionRecommender(
        backbone_model=segmentor,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )

    device = camp.device
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.save_viz, len(val_ds))
    print(f"\n  Gerando {n} visualizações de debug...")

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
        print(f"    recommender_batch.png ({len(images_all)} amostras)")
    except Exception as exc:
        print(f"    recommender_batch.png falhou: {exc}")

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
                            "bbox": [
                                max(0, cx_img - ps), max(0, cy_img - ps),
                                min(img_w, cx_img + ps), min(img_h, cy_img + ps),
                            ],
                            "score": float(hm[cy, cx]),
                        })
                recs.sort(key=lambda rec: rec["score"], reverse=True)
            fig = plot_recommendations(images_all[idx], patches_all[idx], recs, heatmaps_all[idx])
            save_figure(fig, viz_dir / f"recommendation_{idx}.png")
            print(f"    recommendation_{idx}.png")
        except Exception as exc:
            print(f"    recommendation_{idx}.png falhou: {exc}")

    if scores_all:
        try:
            fig = plot_score_distribution(scores_all)
            save_figure(fig, viz_dir / "score_distribution.png")
            print("    score_distribution.png")
        except Exception as exc:
            print(f"    score_distribution.png falhou: {exc}")


if __name__ == "__main__":
    main()
