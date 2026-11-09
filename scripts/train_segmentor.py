#!/usr/bin/env python
"""Train the segmentor (PretrainedUNet).

Phase 1 trains the decoder with the encoder frozen.
Phase 2 unfreezes the encoder for the remaining epochs.

    python scripts/train_segmentor.py --class metaplasia --epochs 80
    python scripts/train_segmentor.py --class polyps --epochs 100 --batch-size 4 --output weights/polyps/segmentor/
    python scripts/train_segmentor.py --class airplane --encoder efficientnet-b3 --freeze-epochs 30
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from camp import CAMP
from camp.dataset import PairedMaskDataset, attention_collate
from camp.io import class_dirs
from camp.models.segmentor import PretrainedUNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAMP — Treino do Segmentador",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--class-name", "--class", dest="class_name", required=True,
                        help="Classe alvo (ex: metaplasia, polyps, airplane)")
    parser.add_argument("--data-dir", default="data",
                        help="Directório base dos datasets (default: data)")
    parser.add_argument("--encoder", default="efficientnet-b3",
                        help="Backbone encoder (default: efficientnet-b3)")
    parser.add_argument("--decoder-dropout", type=float, default=0.4,
                        help="Dropout do decoder (default: 0.4)")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Tamanho da imagem (default: 512)")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Tamanho dos patches (default: 128)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Total de épocas (default: 80)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate fase 1 (default: 3e-4)")
    parser.add_argument("--freeze-epochs", type=int, default=30,
                        help="Épocas com encoder congelado (default: 30)")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate fase 2 full finetune (default: 1e-4)")
    parser.add_argument("--accumulation-steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--split-ratio", type=float, default=0.2,
                        help="Fracção para validação (default: 0.2)")
    parser.add_argument("--p-remove", type=float, default=0.0,
                        help="Probabilidade de blindfold (default: 0.0)")
    parser.add_argument("--output", default=None,
                        help="Directório de output (default: weights/<class>/segmentor/)")
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
        best = camp.train_segmentor(
            args.class_name,
            encoder=args.encoder,
            decoder_dropout=args.decoder_dropout,
            image_size=args.image_size,
            patch_size=args.patch_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            finetune_lr=args.finetune_lr,
            accumulation_steps=args.accumulation_steps,
            patience=args.patience,
            split_ratio=args.split_ratio,
            p_remove=args.p_remove,
            device=args.device,
            output=args.output,
        )
    except FileNotFoundError as exc:
        print(f"Erro: {exc}")
        sys.exit(1)

    run_dir = Path(best).parent
    if args.save_viz > 0:
        _save_segmentor_viz(camp, args, run_dir)

    print(f"Pesos: {best}")
    print(f"Config: {run_dir / 'config.yaml'}")


def _save_segmentor_viz(camp, args, run_dir):
    """Write debug figures with the helpers in src.paste.viz."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.paste.viz import plot_seg_examples, plot_seg_overlay, plot_seg_worst
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
    print(f"\n  Gerando {n} visualizações de debug...")

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
    print(f"    seg_examples.png ({len(images_all)} amostras)")

    if len(dice_all) >= 2:
        best_idx = int(np.argmax(dice_all))
        worst_idx = int(np.argmin(dice_all))
        for label, idx in (("best", best_idx), ("worst", worst_idx)):
            fig = plot_seg_overlay(images_all[idx], masks_gt_all[idx], masks_pred_all[idx])
            save_figure(fig, viz_dir / f"seg_overlay_{label}.png")
            print(f"    seg_overlay_{label}.png (Dice={dice_all[idx]:.3f})")

    if len(dice_all) >= 4:
        fig = plot_seg_worst(images_all, masks_gt_all, masks_pred_all, dice_all)
        save_figure(fig, viz_dir / "seg_worst.png")
        print("    seg_worst.png")


if __name__ == "__main__":
    main()
