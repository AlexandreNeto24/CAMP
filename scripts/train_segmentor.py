#!/usr/bin/env python
"""
Treino do Segmentador (PretrainedUNet).

Estratégia de 2 fases:
  Fase 1 — Decoder only (encoder congelado), OneCycleLR
  Fase 2 — Gradual unfreeze do encoder, CosineAnnealing + warmup

Uso:
    python scripts/train_segmentor.py --class metaplasia --epochs 80
    python scripts/train_segmentor.py --class polyps --epochs 100 --batch-size 4 --output weights/polyps/segmentor/
    python scripts/train_segmentor.py --class airplane --encoder efficientnet-b3 --freeze-epochs 30
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser(
        description='CIM — Treino do Segmentador',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Dados
    p.add_argument('--class-name', '--class', dest='class_name', required=True,
                   help='Classe alvo (ex: metaplasia, polyps, airplane)')
    p.add_argument('--data-dir', default='data',
                   help='Directório base dos datasets (default: data)')

    # Modelo
    p.add_argument('--encoder', default='efficientnet-b3',
                   help='Backbone encoder (default: efficientnet-b3)')
    p.add_argument('--decoder-dropout', type=float, default=0.4,
                   help='Dropout do decoder (default: 0.4)')
    p.add_argument('--image-size', type=int, default=512,
                   help='Tamanho da imagem (default: 512)')
    p.add_argument('--patch-size', type=int, default=128,
                   help='Tamanho dos patches (default: 128)')

    # Treino
    p.add_argument('--epochs', type=int, default=80,
                   help='Total de épocas (default: 80)')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Batch size (default: 8)')
    p.add_argument('--lr', type=float, default=3e-4,
                   help='Learning rate fase 1 (default: 3e-4)')
    p.add_argument('--freeze-epochs', type=int, default=30,
                   help='Épocas com encoder congelado (default: 30)')
    p.add_argument('--finetune-lr', type=float, default=1e-4,
                   help='Learning rate fase 2 full finetune (default: 1e-4)')
    p.add_argument('--accumulation-steps', type=int, default=2,
                   help='Gradient accumulation steps (default: 2)')
    p.add_argument('--patience', type=int, default=25,
                   help='Early stopping patience (default: 25)')
    p.add_argument('--split-ratio', type=float, default=0.2,
                   help='Fracção para validação (default: 0.2)')
    p.add_argument('--p-remove', type=float, default=0.0,
                   help='Probabilidade de blindfold (default: 0.0)')

    # Output
    p.add_argument('--output', default=None,
                   help='Directório de output (default: weights/<class>/segmentor/)')
    p.add_argument('--device', default='cuda', help='Device (default: cuda)')

    # Debug / Visualização
    p.add_argument('--debug', action='store_true',
                   help='Activar visualizações de debug (equivale a --save-viz 8)')
    p.add_argument('--save-viz', type=int, default=0, metavar='N',
                   help='Guardar N exemplos visuais no final do treino (default: 0)')
    p.add_argument('--viz-every', type=int, default=5, metavar='K',
                   help='Guardar vizualizações a cada K épocas (requer --save-viz; default: 5)')

    args = p.parse_args()
    # --debug implica --save-viz 8 se não especificado
    if args.debug and args.save_viz == 0:
        args.save_viz = 8
    return args


def main():
    args = parse_args()

    import torch
    from src.paste.attention_dataset import AttentionDataset
    from src.paste.segmentation_model import PretrainedUNet, train_segmentor

    # ── Dataset ──
    data_path = project_root / args.data_dir / args.class_name / args.class_name
    if not (data_path / 'images').is_dir():
        data_path = project_root / args.data_dir / args.class_name
    if not (data_path / 'images').is_dir():
        print(f"Erro: Directório de imagens não encontrado em {data_path}/images")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Treino Segmentador — {args.class_name}")
    print(f"{'='*60}")
    print(f"  Dataset: {data_path}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Épocas: {args.epochs} (freeze={args.freeze_epochs})")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, FT-LR: {args.finetune_lr}")

    img_size = (args.image_size, args.image_size)
    patch_size = (args.patch_size, args.patch_size)

    train_ds = AttentionDataset(
        root_path=str(data_path),
        image_size=img_size,
        patch_size=patch_size,
        split_mode='ratio',
        split_ratio=args.split_ratio,
        apply_aug=True,
        p_remove=args.p_remove
    )
    val_ds = AttentionDataset(
        root_path=str(data_path),
        image_size=img_size,
        patch_size=patch_size,
        split_mode='ratio',
        split_ratio=args.split_ratio,
        apply_aug=False,
        p_remove=0.0
    )

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Modelo ──
    model = PretrainedUNet(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        decoder_dropout=args.decoder_dropout
    )

    # ── Output dir ──
    if args.output:
        run_dir = Path(args.output)
    else:
        run_dir = project_root / 'weights' / args.class_name / 'segmentor'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {run_dir}")
    print(f"{'='*60}\n")

    # ── Salvar config ──
    import yaml
    config = {
        'class_name': args.class_name,
        'encoder': args.encoder,
        'decoder_dropout': args.decoder_dropout,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'freeze_epochs': args.freeze_epochs,
        'finetune_lr': args.finetune_lr,
        'accumulation_steps': args.accumulation_steps,
        'patience': args.patience,
        'split_ratio': args.split_ratio,
        'p_remove': args.p_remove,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Treino ──
    save_path = train_segmentor(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        run_dir=str(run_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        accumulation_steps=args.accumulation_steps,
        freeze_epochs=args.freeze_epochs,
        full_finetune_lr=args.finetune_lr,
        early_stopping_patience=args.patience
    )

    # ── Copiar melhor peso com nome standard ──
    best_src = Path(save_path)
    best_dst = run_dir / 'best.pth'
    if best_src.exists() and best_src != best_dst:
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"\n✓ Peso final copiado para {best_dst}")

    # ── Visualizações de debug ──
    if args.save_viz > 0:
        _save_segmentor_viz(model, val_ds, run_dir, args, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Treino concluído!")
    print(f"  Pesos: {run_dir / 'best.pth'}")
    print(f"  Config: {run_dir / 'config.yaml'}")
    print(f"  Métricas: {run_dir / 'segmentor_metrics_subplot.png'}")
    if args.save_viz > 0:
        print(f"  Viz: {run_dir / 'viz'}")
    if torch.cuda.is_available():
        print(f"  VRAM pico: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"{'='*60}")


def _save_segmentor_viz(model, val_ds, run_dir, args, device='cuda'):
    """Gera visualizações pós-treino do segmentador."""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from src.paste.attention_dataset import attention_collate
    from src.paste.viz import plot_seg_examples, plot_seg_overlay, plot_seg_worst
    from src.viz_utils import save_figure

    viz_dir = run_dir / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.save_viz, len(val_ds))

    print(f"\n  Gerando {n} visualizações de debug...")

    # Carregar melhor modelo
    best_path = run_dir / 'best.pth'
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Coletar predições
    loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, n), shuffle=False,
        num_workers=2, collate_fn=attention_collate
    )

    images_all, masks_gt_all, masks_pred_all, dice_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['images'].to(device)
            masks = batch['segmentor_target_mask'].to(device)
            preds = torch.sigmoid(model(imgs))

            for i in range(imgs.shape[0]):
                if len(images_all) >= n:
                    break
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                gt_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i, 0].cpu().numpy()

                # Dice
                pred_bin = (pred_np > 0.5).astype(np.float32)
                intersection = (pred_bin * gt_np).sum()
                dice = (2 * intersection) / (pred_bin.sum() + gt_np.sum() + 1e-8)

                images_all.append(img_np)
                masks_gt_all.append(gt_np)
                masks_pred_all.append(pred_np)
                dice_all.append(dice)

            if len(images_all) >= n:
                break

    # 1) Grid de exemplos (N×3)
    fig = plot_seg_examples(images_all, masks_gt_all, masks_pred_all)
    save_figure(fig, viz_dir / 'seg_examples.png')
    print(f"    ✓ seg_examples.png ({len(images_all)} amostras)")

    # 2) Overlay detalhado do melhor e pior
    if len(dice_all) >= 2:
        best_idx = int(np.argmax(dice_all))
        worst_idx = int(np.argmin(dice_all))
        for label, idx in [('best', best_idx), ('worst', worst_idx)]:
            fig = plot_seg_overlay(
                images_all[idx], masks_gt_all[idx], masks_pred_all[idx]
            )
            save_figure(fig, viz_dir / f'seg_overlay_{label}.png')
            print(f"    ✓ seg_overlay_{label}.png (Dice={dice_all[idx]:.3f})")

    # 3) Worst-N
    if len(dice_all) >= 4:
        fig = plot_seg_worst(images_all, masks_gt_all, masks_pred_all, dice_all)
        save_figure(fig, viz_dir / 'seg_worst.png')
        print(f"    ✓ seg_worst.png")


if __name__ == '__main__':
    main()
