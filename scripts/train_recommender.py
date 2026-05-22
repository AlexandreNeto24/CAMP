#!/usr/bin/env python
"""
Treino do Recommender de Atenção (AttentionRecommender).

Estratégia de 2 fases:
  Fase 1 — Treino principal (lr=3e-4, 60 épocas)
  Fase 2 — Fine-tuning (lr=5e-5, 20 épocas)

Backbone (Segmentor) é congelado — apenas treina cross-attention + heatmap decoder.

Uso:
    python scripts/train_recommender.py --class metaplasia --epochs 60
    python scripts/train_recommender.py --class polyps --epochs 60 --finetune-epochs 20
    python scripts/train_recommender.py --class metaplasia --segmentor-weights weights/metaplasia/segmentor/best.pth
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
import shutil
from scipy import ndimage
import torch
from torch.utils.data import DataLoader

from src.paste.attention_dataset import AttentionDataset, attention_collate
from src.paste.attention_dataset import attention_collate
from src.paste.segmentation_model import PretrainedUNet
from src.paste.models.attention_recommender import AttentionRecommender
from src.paste.utils.trainer import AttentionRecommenderTrainer
from src.paste.viz import plot_recommendations, plot_batch_debug, plot_score_distribution
from src.viz_utils import save_figure


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser(
        description='CIM — Treino do Recommender',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Dados
    p.add_argument('--class-name', '--class', dest='class_name', required=True,
                   help='Classe alvo (ex: metaplasia, polyps, airplane)')
    p.add_argument('--data-dir', default='data',
                   help='Directório base dos datasets (default: data)')

    # Modelo
    p.add_argument('--encoder', default='efficientnet-b3',
                   help='Backbone encoder do segmentador (default: efficientnet-b3)')
    p.add_argument('--embedding-dim', type=int, default=2048,
                   help='Dimensão do embedding (default: 2048)')
    p.add_argument('--num-heads', type=int, default=16,
                   help='Número de attention heads (default: 16)')
    p.add_argument('--dropout', type=float, default=0.10,
                   help='Dropout rate (default: 0.10)')
    p.add_argument('--segmentor-weights', default=None,
                   help='Caminho para pesos do segmentador (default: auto-detect)')

    # Treino Fase 1
    p.add_argument('--epochs', type=int, default=60,
                   help='Épocas fase 1 (default: 60)')
    p.add_argument('--lr', type=float, default=3e-4,
                   help='Learning rate fase 1 (default: 3e-4)')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Batch size (default: 4)')
    p.add_argument('--accumulation-steps', type=int, default=4,
                   help='Gradient accumulation steps (default: 4)')
    p.add_argument('--patience', type=int, default=30,
                   help='Early stopping patience (default: 30)')

    # Treino Fase 2 (Fine-tune)
    p.add_argument('--finetune-epochs', type=int, default=20,
                   help='Épocas de fine-tuning (default: 20, 0=skip)')
    p.add_argument('--finetune-lr', type=float, default=5e-5,
                   help='Learning rate fine-tuning (default: 5e-5)')

    # Dataset
    p.add_argument('--image-size', type=int, default=512,
                   help='Tamanho da imagem (default: 512)')
    p.add_argument('--patch-size', type=int, default=128,
                   help='Tamanho dos patches (default: 128)')
    p.add_argument('--split-ratio', type=float, default=0.2,
                   help='Fracção para validação (default: 0.2)')
    p.add_argument('--p-remove', type=float, default=0.0,
                   help='Probabilidade de blindfold (default: 0.0)')

    # Output
    p.add_argument('--output', default=None,
                   help='Directório de output (default: weights/<class>/recommender/)')
    p.add_argument('--device', default='cuda', help='Device (default: cuda)')

    # Debug / Visualização
    p.add_argument('--debug', action='store_true',
                   help='Activar visualizações de debug (equivale a --save-viz 8)')
    p.add_argument('--save-viz', type=int, default=0, metavar='N',
                   help='Guardar N exemplos visuais no final do treino (default: 0)')
    p.add_argument('--viz-every', type=int, default=5, metavar='K',
                   help='Guardar vizualizações a cada K épocas (requer --save-viz; default: 5)')

    args = p.parse_args()
    if args.debug and args.save_viz == 0:
        args.save_viz = 8
    return args


def main():
    args = parse_args()

    # ── Dataset ──
    data_path = project_root / args.data_dir / args.class_name / args.class_name
    if not (data_path / 'images').is_dir():
        data_path = project_root / args.data_dir / args.class_name
    if not (data_path / 'images').is_dir():
        print(f"Erro: Directório de imagens não encontrado em {data_path}/images")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Treino Recommender — {args.class_name}")
    print(f"{'='*60}")
    print(f"  Dataset: {data_path}")
    print(f"  Embedding: {args.embedding_dim}, Heads: {args.num_heads}")
    print(f"  Fase 1: {args.epochs} épocas, lr={args.lr}")
    print(f"  Fase 2: {args.finetune_epochs} épocas, lr={args.finetune_lr}")

    # create the datasets and dataloaders
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=attention_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=attention_collate
    )

    # ── Backbone (Segmentor congelado) ──
    # Construir em CPU — o AttentionRecommender.__init__ faz um forward pass
    # com dummy tensor CPU para inferir dimensões. Só depois movemos para CUDA.
    segmentor = PretrainedUNet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1
    )  # CPU

    # Carregar pesos do segmentador (em CPU)
    seg_weights = args.segmentor_weights
    if seg_weights is None:
        # Auto-detect: weights/<class>/segmentor/best.pth
        candidate = project_root / 'weights' / args.class_name / 'segmentor' / 'best.pth'
        if not candidate.exists():
            candidate = project_root / 'weights' / args.class_name / 'segmentor' / 'best_segmentor.pth'
        if candidate.exists():
            seg_weights = str(candidate)

    if seg_weights and Path(seg_weights).exists():
        try:
            state = torch.load(seg_weights, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            segmentor.load_state_dict(state, strict=False)
            print(f"  Segmentor: {seg_weights}")
        except Exception:
            segmentor.load_adapted_state_dict(seg_weights, strict=False)
            print(f"  Segmentor (adapted): {seg_weights}")
    else:
        segmentor = PretrainedUNet(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )  # CPU
        print(f"  Segmentor: ImageNet features (sem pesos treinados)")

    segmentor.eval()

    # ── Modelo (construído em CPU, depois movido para CUDA) ──
    model = AttentionRecommender(
        backbone_model=segmentor,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    ).to(args.device)  # Agora sim, tudo para CUDA

    # ── Output dir ──
    if args.output:
        run_dir = Path(args.output)
    else:
        run_dir = project_root / 'weights' / args.class_name / 'recommender'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {run_dir}")
    print(f"{'='*60}\n")

    # ── Salvar config ──
    config = {
        'class_name': args.class_name,
        'encoder': args.encoder,
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'patience': args.patience,
        'split_ratio': args.split_ratio,
        'p_remove': args.p_remove,
        'segmentor_weights': seg_weights,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Fase 1: Treino principal ──
    print(f"{'─'*40}")
    print(f"  FASE 1: Treino ({args.epochs} épocas, lr={args.lr})")
    print(f"{'─'*40}")

    trainer = AttentionRecommenderTrainer(model, device=args.device)
    trainer.patience = args.patience

    trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        run_dir=str(run_dir)
    )

    # ── Fase 2: Fine-tuning ──
    if args.finetune_epochs > 0:
        print(f"\n{'─'*40}")
        print(f"  FASE 2: Fine-tuning ({args.finetune_epochs} épocas, lr={args.finetune_lr})")
        print(f"{'─'*40}")

        # Carregar melhor modelo da fase 1
        best_path = run_dir / 'best_model.pt'
        if best_path.exists():
            model.load_model(str(best_path))

        # Novo trainer com loss history herdado
        trainer2 = AttentionRecommenderTrainer(model, device=args.device)
        trainer2.train_losses = trainer.train_losses.copy()
        trainer2.val_losses = trainer.val_losses.copy()
        trainer2.best_val_loss = trainer.best_val_loss
        trainer2.patience = args.patience

        trainer2.fit(
            train_loader, val_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            accumulation_steps=args.accumulation_steps,
            run_dir=str(run_dir)
        )

    # ── Copiar melhor peso com nome standard ──
    best_src = run_dir / 'best_model.pt'
    best_dst = run_dir / 'best.pth'
    if best_src.exists():
        
        shutil.copy2(best_src, best_dst)

    # ── Visualizações de debug ──
    if args.save_viz > 0:
        _save_recommender_viz(model, val_ds, run_dir, args, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Treino concluído!")
    print(f"  Pesos: {run_dir / 'best.pth'}")
    print(f"  Config: {run_dir / 'config.yaml'}")
    print(f"  Curvas: {run_dir / 'losses.png'}")
    if args.save_viz > 0:
        print(f"  Viz: {run_dir / 'viz'}")
    if torch.cuda.is_available():
        print(f"  VRAM pico: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"{'='*60}")


def _save_recommender_viz(model, val_ds, run_dir, args, device='cuda'):
    """Gera visualizações pós-treino do recommender."""

    viz_dir = run_dir / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.save_viz, len(val_ds))

    print(f"\n  Gerando {n} visualizações de debug...")

    # Carregar melhor modelo
    best_path = run_dir / 'best.pth'
    if best_path.exists():
        model.load_model(str(best_path))
    model.to(device)
    model.eval()

    # Coletar predições
    loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, n), shuffle=False,
        num_workers=2, collate_fn=attention_collate
    )

    images_all, masks_all, patches_all, heatmaps_all, scores_all = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['images'].to(device)
            masks = batch['segmentor_target_mask'].to(device)
            patches = batch['patches'].to(device)
            sample_indices = batch['sample_indices'].to(device)
            patches_ctx = batch.get('patches_with_context', None)
            if patches_ctx is not None:
                patches_ctx = patches_ctx.to(device)

            if patches.numel() == 0:
                continue

            # Alinhar imagens com patches via sample_indices
            batch_images = imgs[sample_indices]

            # O modelo retorna (pred_heatmaps, patch_feat, img_feat) com return_intermediate
            result = model(batch_images, patch=patches, patch_with_context=patches_ctx)
            # result é o pred_heatmaps [N_patches, 1, H, W] sem return_intermediate
            if isinstance(result, tuple):
                heatmap = result[0]
            else:
                heatmap = result

            # Iterar por imagens únicas no batch
            unique_indices = sample_indices.unique().cpu().tolist()
            for ui in unique_indices:
                if len(images_all) >= n:
                    break
                i = int(ui)
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                mask_np = masks[i, 0].cpu().numpy() if masks.dim() == 4 else masks[i].cpu().numpy()

                # Pegar o primeiro patch desta imagem
                patch_mask = (sample_indices == i)
                patch_idx = patch_mask.nonzero(as_tuple=True)[0][0].item()
                patch_np = patches[patch_idx, :3].cpu().numpy().transpose(1, 2, 0)
                patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)

                # Heatmap predito para este patch
                hm_np = heatmap[patch_idx, 0].cpu().numpy() if heatmap is not None and heatmap.dim() >= 3 else None

                images_all.append(img_np)
                masks_all.append(mask_np)
                patches_all.append(patch_np)
                heatmaps_all.append(hm_np)

                # Scores: pegar o valor máximo do heatmap como proxy
                if hm_np is not None:
                    scores_all.append(float(hm_np.max()))

            if len(images_all) >= n:
                break

    # 1) Batch debug grid (images, masks, patches, heatmaps)
    try:
        fig = plot_batch_debug(images_all, masks_all, patches_all, heatmaps_all)
        save_figure(fig, viz_dir / 'recommender_batch.png')
        print(f"    ✓ recommender_batch.png ({len(images_all)} amostras)")
    except Exception as e:
        print(f"    ✗ recommender_batch.png falhou: {e}")

    # 2) Recomendações individuais (top 3)
    for idx in range(min(3, len(images_all))):
        try:
            # Build fake recommendations from scores if available
            recs = []
            img_h, img_w = images_all[idx].shape[:2]
            if heatmaps_all[idx] is not None:
                hm = heatmaps_all[idx]
                # Find top-5 peaks in heatmap
                labeled, num_features = ndimage.label(hm > hm.max() * 0.5)
                for j in range(1, min(num_features + 1, 6)):
                    ys, xs = np.where(labeled == j)
                    if len(ys) > 0:
                        cy, cx = int(ys.mean()), int(xs.mean())
                        # Scale to image coords
                        sy = img_h / hm.shape[0]
                        sx = img_w / hm.shape[1]
                        cy_img, cx_img = int(cy * sy), int(cx * sx)
                        ps = args.patch_size // 2
                        recs.append({
                            'bbox': [max(0, cx_img-ps), max(0, cy_img-ps),
                                     min(img_w, cx_img+ps), min(img_h, cy_img+ps)],
                            'score': float(hm[cy, cx])
                        })
                recs.sort(key=lambda r: r['score'], reverse=True)

            fig = plot_recommendations(
                images_all[idx], patches_all[idx], recs, heatmaps_all[idx]
            )
            save_figure(fig, viz_dir / f'recommendation_{idx}.png')
            print(f"    ✓ recommendation_{idx}.png")
        except Exception as e:
            print(f"    ✗ recommendation_{idx}.png falhou: {e}")

    # 3) Distribuição de scores
    if scores_all:
        try:
            fig = plot_score_distribution(scores_all)
            save_figure(fig, viz_dir / 'score_distribution.png')
            print(f"    ✓ score_distribution.png")
        except Exception as e:
            print(f"    ✗ score_distribution.png falhou: {e}")


if __name__ == '__main__':
    main()
