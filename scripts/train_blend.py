#!/usr/bin/env python
"""
Treino do DoveNet (Harmonização de cor/tom).

Estratégia de 2 fases:
  Fase 1 — Treino principal (lr=1e-4, 25 épocas)
  Fase 2 — Fine-tuning (lr=1e-5, 10 épocas)

Loss: MSE (global + local×2 + boundary×5) + GradientLoss ×0.5

Uso:
    python scripts/train_blend.py --class metaplasia --epochs 25
    python scripts/train_blend.py --class polyps --epochs 25 --finetune-epochs 10
    python scripts/train_blend.py --class metaplasia --data-dir data --output weights/metaplasia/dovenet/
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from src.blend.data.dataset import HarmonizationDataset
from src.blend.models.dovenet import DoveNetGenerator
from src.blend.models.dovenet import GradientLoss
from src.blend.viz import plot_dovenet_examples, plot_boundary_mask, plot_blend_sample
from src.viz_utils import save_figure

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser(
        description='CIM — Treino do DoveNet (Harmonização)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Dados
    p.add_argument('--class-name', '--class', dest='class_name', required=True,
                   help='Classe alvo (ex: metaplasia, polyps, airplane)')
    p.add_argument('--data-dir', default='data',
                   help='Directório base dos datasets (default: data)')

    # Treino Fase 1
    p.add_argument('--epochs', type=int, default=25,
                   help='Épocas fase 1 (default: 25)')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate fase 1 (default: 1e-4)')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Batch size (default: 8)')

    # Treino Fase 2 (Fine-tune)
    p.add_argument('--finetune-epochs', type=int, default=10,
                   help='Épocas de fine-tuning (default: 10, 0=skip)')
    p.add_argument('--finetune-lr', type=float, default=1e-5,
                   help='Learning rate fine-tuning (default: 1e-5)')

    # Loss weights
    p.add_argument('--w-pixel', type=float, default=1.0,
                   help='Peso loss pixel (default: 1.0)')
    p.add_argument('--w-grad', type=float, default=0.5,
                   help='Peso loss gradient (default: 0.5)')
    p.add_argument('--w-local', type=float, default=2.0,
                   help='Peso MSE local/objecto (default: 2.0)')
    p.add_argument('--w-boundary', type=float, default=5.0,
                   help='Peso MSE boundary focus (default: 5.0)')

    # Dataset
    p.add_argument('--image-size', type=int, default=512,
                   help='Tamanho da imagem (default: 512)')
    p.add_argument('--split-ratio', type=float, default=0.2,
                   help='Fracção para validação (default: 0.2)')
    p.add_argument('--save-interval', type=int, default=10,
                   help='Intervalo de checkpoint em épocas (default: 10)')

    # Output
    p.add_argument('--output', default=None,
                   help='Directório de output (default: weights/<class>/dovenet/)')
    p.add_argument('--device', default='cuda', help='Device (default: cuda)')

    # Debug / Visualização
    p.add_argument('--debug', action='store_true',
                   help='Activar visualizações de debug (equivale a --save-viz 8)')
    p.add_argument('--save-viz', type=int, default=0, metavar='N',
                   help='Guardar N exemplos visuais por epoch de viz (default: 0)')
    p.add_argument('--viz-every', type=int, default=5, metavar='K',
                   help='Guardar vizualizações a cada K épocas (default: 5)')

    args = p.parse_args()
    if args.debug and args.save_viz == 0:
        args.save_viz = 8
    return args


def train_dovenet(model, train_loader, val_loader, run_dir, epochs, lr,
                  w_pixel, w_grad, w_local, w_boundary, save_interval, device,
                  save_viz=0, viz_every=5):
    """
    Loop de treino do DoveNet com loss híbrida + validação.
    """


    criterion_pixel = nn.MSELoss().to(device)
    criterion_grad = GradientLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_path = run_dir / 'best.pth'

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            composite = batch['pixel_values'].to(device)
            real = batch['ground_truth'].to(device)
            mask = batch['mask'].to(device)
            boundary = batch['boundary_mask'].to(device)

            optimizer.zero_grad()
            output = model(composite, mask, boundary)

            # Loss híbrida
            loss_pixel_global = criterion_pixel(output, real)
            loss_pixel_local = criterion_pixel(output * mask, real * mask)
            loss_boundary_focus = criterion_pixel(output * boundary, real * boundary)
            loss_pixel = loss_pixel_global + (w_local * loss_pixel_local) + (w_boundary * loss_boundary_focus)
            loss_grad = criterion_grad(output, real)
            loss = (w_pixel * loss_pixel) + (w_grad * loss_grad)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'L_Pix': loss_pixel.item(), 'L_Grad': loss_grad.item()})

        avg_train = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                composite = batch['pixel_values'].to(device)
                real = batch['ground_truth'].to(device)
                mask = batch['mask'].to(device)
                boundary = batch['boundary_mask'].to(device)

                output = model(composite, mask, boundary)

                loss_pixel_global = criterion_pixel(output, real)
                loss_pixel_local = criterion_pixel(output * mask, real * mask)
                loss_boundary_focus = criterion_pixel(output * boundary, real * boundary)
                loss_pixel = loss_pixel_global + (w_local * loss_pixel_local) + (w_boundary * loss_boundary_focus)
                loss_grad = criterion_grad(output, real)
                loss = (w_pixel * loss_pixel) + (w_grad * loss_grad)

                val_loss += loss.item()

        avg_val = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val)

        print(f"  Epoch {epoch+1}: train={avg_train:.6f} | val={avg_val:.6f}")

        # Checkpoint por melhor val loss
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Melhor modelo guardado (val_loss={best_val_loss:.6f})")

        # Checkpoint periódico
        if (epoch + 1) % save_interval == 0:
            ckpt_path = run_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        # Visualizações periódicas
        if save_viz > 0 and (epoch + 1) % viz_every == 0:
            _save_epoch_viz(model, val_loader, run_dir, epoch + 1, save_viz, device)

    # ── Curvas de loss ──
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=2)
    plt.plot(val_losses, label='Val Loss', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DoveNet Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Salvar histórico
    np.save(run_dir / 'train_losses.npy', np.array(train_losses))
    np.save(run_dir / 'val_losses.npy', np.array(val_losses))

    return train_losses, val_losses, best_val_loss


def main():
    args = parse_args()

    # ── Dataset ──
    # Tenta data/<class>/<class>/ e data/<class>/
    data_path = project_root / args.data_dir / args.class_name / args.class_name
    if not (data_path / 'images').is_dir():
        data_path = project_root / args.data_dir / args.class_name
    if not (data_path / 'images').is_dir():
        print(f"Erro: Directório de imagens não encontrado em {data_path}/images")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Treino DoveNet — {args.class_name}")
    print(f"{'='*60}")
    print(f"  Dataset: {data_path}")
    print(f"  Fase 1: {args.epochs} épocas, lr={args.lr}")
    print(f"  Fase 2: {args.finetune_epochs} épocas, lr={args.finetune_lr}")
    print(f"  Loss weights: pixel={args.w_pixel}, grad={args.w_grad}, local={args.w_local}, boundary={args.w_boundary}")

    img_size = (args.image_size, args.image_size)

    train_ds = HarmonizationDataset(
        root_path=str(data_path),
        image_size=img_size,
        split='train',
        split_ratio=args.split_ratio
    )
    val_ds = HarmonizationDataset(
        root_path=str(data_path),
        image_size=img_size,
        split='val',
        split_ratio=args.split_ratio
    )

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Modelo ──
    model = DoveNetGenerator()

    # ── Output dir ──
    if args.output:
        run_dir = Path(args.output)
    else:
        run_dir = project_root / 'weights' / args.class_name / 'dovenet'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {run_dir}")
    print(f"{'='*60}\n")

    # ── Salvar config ──

    config = {
        'class_name': args.class_name,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'batch_size': args.batch_size,
        'split_ratio': args.split_ratio,
        'w_pixel': args.w_pixel,
        'w_grad': args.w_grad,
        'w_local': args.w_local,
        'w_boundary': args.w_boundary,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Fase 1: Treino principal ──
    print(f"{'─'*40}")
    print(f"  FASE 1: Treino ({args.epochs} épocas, lr={args.lr})")
    print(f"{'─'*40}")

    t0 = time.time()
    train_losses, val_losses, best_val = train_dovenet(
        model, train_loader, val_loader, run_dir,
        epochs=args.epochs, lr=args.lr,
        w_pixel=args.w_pixel, w_grad=args.w_grad,
        w_local=args.w_local, w_boundary=args.w_boundary,
        save_interval=args.save_interval, device=args.device,
        save_viz=args.save_viz, viz_every=args.viz_every
    )
    t1 = time.time()
    print(f"  Fase 1 concluída em {t1-t0:.0f}s (best val_loss={best_val:.6f})")

    # ── Fase 2: Fine-tuning ──
    if args.finetune_epochs > 0:
        print(f"\n{'─'*40}")
        print(f"  FASE 2: Fine-tuning ({args.finetune_epochs} épocas, lr={args.finetune_lr})")
        print(f"{'─'*40}")

        # Carregar melhor modelo da fase 1
        best_path = run_dir / 'best.pth'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=args.device))
            print(f"  Carregados pesos da Fase 1")

        train_losses2, val_losses2, best_val2 = train_dovenet(
            model, train_loader, val_loader, run_dir,
            epochs=args.finetune_epochs, lr=args.finetune_lr,
            w_pixel=args.w_pixel, w_grad=args.w_grad,
            w_local=args.w_local, w_boundary=args.w_boundary,
            save_interval=args.save_interval, device=args.device,
            save_viz=args.save_viz, viz_every=args.viz_every
        )
        t2 = time.time()
        print(f"  Fase 2 concluída em {t2-t1:.0f}s (best val_loss={best_val2:.6f})")

        # Curva combinada
        all_train = train_losses + train_losses2
        all_val = val_losses + val_losses2
        plt.figure(figsize=(12, 5))
        plt.plot(all_train, label='Train Loss', marker='o', markersize=2)
        plt.plot(all_val, label='Val Loss', marker='s', markersize=2)
        plt.axvline(x=len(train_losses)-0.5, color='red', linestyle='--', alpha=0.5, label='Fine-tune start')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'DoveNet — {args.class_name} (Fase 1 + Fine-tune)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(run_dir / 'loss_curves_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        np.save(run_dir / 'train_losses_combined.npy', np.array(all_train))
        np.save(run_dir / 'val_losses_combined.npy', np.array(all_val))

    print(f"\n{'='*60}")
    print(f"  Treino concluído!")
    print(f"  Pesos: {run_dir / 'best.pth'}")
    print(f"  Config: {run_dir / 'config.yaml'}")
    print(f"  Curvas: {run_dir / 'loss_curves.png'}")
    if args.save_viz > 0:
        print(f"  Viz: {run_dir / 'viz'}")
    if torch.cuda.is_available():
        print(f"  VRAM pico: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"{'='*60}")


def _save_epoch_viz(model, val_loader, run_dir, epoch, n_examples, device):
    """Gera visualizações numa época específica durante o treino."""

    viz_dir = run_dir / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    composites, outputs, targets, masks, boundaries = [], [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            comp = batch['pixel_values'].to(device)
            real = batch['ground_truth'].to(device)
            mask = batch['mask'].to(device)
            boundary = batch['boundary_mask'].to(device)

            out = model(comp, mask, boundary)

            for i in range(comp.shape[0]):
                if len(composites) >= n_examples:
                    break
                composites.append(comp[i].cpu())
                outputs.append(out[i].cpu())
                targets.append(real[i].cpu())
                masks.append(mask[i].cpu())
                boundaries.append(boundary[i].cpu())

            if len(composites) >= n_examples:
                break

    # 1) Grid de exemplos (N×3)
    try:
        fig = plot_dovenet_examples(composites, outputs, targets)
        save_figure(fig, viz_dir / f'epoch_{epoch}.png')
        print(f"    ✓ viz/epoch_{epoch}.png")
    except Exception as e:
        print(f"    ✗ viz/epoch_{epoch}.png falhou: {e}")

    # 2) Boundary mask detail (primeiro exemplo)
    try:
        fig = plot_boundary_mask(
            composites[0], masks[0], boundaries[0], outputs[0]
        )
        save_figure(fig, viz_dir / f'boundary_epoch_{epoch}.png')
    except Exception:
        pass

    model.train()


if __name__ == '__main__':
    main()
