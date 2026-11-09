#!/usr/bin/env python
"""Generate augmented images with CAMP.

    python scripts/generate.py --class metaplasia --num 100
    python scripts/generate.py --class metaplasia --num 50 --no-tto --seed 42
    python scripts/generate.py --class polyps --num 10 --output data/augmented/polyps/
"""

import argparse
import random
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from camp import CAMP
from camp.io import class_dirs, load_image_mask_pairs, load_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAMP — Geração de imagens aumentadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scripts/generate.py --class metaplasia --num 100
  python scripts/generate.py --class polyps --num 50 --no-tto --seed 42
  python scripts/generate.py --class metaplasia --output ./augmented/
        """,
    )
    parser.add_argument("--class-name", "--class", dest="class_name", type=str, required=True,
                        help="Classe alvo (ex: metaplasia, polyps, airplane)")
    parser.add_argument("--num", "-n", type=int, default=10,
                        help="Número de imagens a gerar (default: 10)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Directório de output (default: data/augmented/<class>/)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directório base dos datasets (default: data)")
    parser.add_argument("--target-dir", type=str, default=None,
                        help="Diretório de imagens alvo. Se não definido, usa data/<class>/normal/ quando existir")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Diretório de imagens source. Se não definido, usa data/<class>/<class>/")
    parser.add_argument("--no-tto", action="store_true",
                        help="Desactivar Test-Time Optimization em todas as imagens")
    parser.add_argument("--encoder", type=str, default="efficientnet-b3",
                        help="Encoder do segmentador (default: efficientnet-b3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed para reproduzibilidade")
    parser.add_argument("--max-source", type=int, default=50,
                        help="Máximo de imagens source e target a usar (default: 50)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--save-intermediates", type=int, default=0, metavar="N",
                        help="Gerar as primeiras N imagens uma a uma antes do batch (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Activar debug (equivale a --save-intermediates 5)")
    args = parser.parse_args()
    if args.debug and args.save_intermediates == 0:
        args.save_intermediates = 5
    return args


def _root(data_dir):
    path = Path(data_dir)
    if not path.is_absolute():
        path = project_root / path
    if path.name != "data":
        print(f"Erro: --data-dir tem de ser a pasta 'data' (recebido: {data_dir})")
        sys.exit(1)
    return path.parent


def _save_sample(sample, output_dir, index):
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    sample.image.save(image_dir / f"aug_{index:04d}.png")
    sample.mask.save(mask_dir / f"aug_{index:04d}.png")


def main():
    args = parse_args()
    root = _root(args.data_dir)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = root / "data" / "augmented" / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    source_dir, target_dir = class_dirs(root, args.class_name)
    if args.source_dir:
        source_dir = Path(args.source_dir)
    if args.target_dir:
        target_dir = Path(args.target_dir)
    print(f"  Target dir: {target_dir}")
    print(f"  Source dir: {source_dir}")

    target_images = load_images(target_dir, max_images=args.max_source)
    if len(target_images) < 1:
        print(f"Erro: precisa de pelo menos 1 imagem alvo em {target_dir}")
        sys.exit(1)
    source_images, source_masks = load_image_mask_pairs(source_dir, max_images=args.max_source)
    if len(source_images) < 1:
        print(f"Erro: precisa de pelo menos 1 imagem source em {source_dir}")
        sys.exit(1)

    use_tto = not args.no_tto
    camp = CAMP(root=root, device=args.device, verbose=True)
    try:
        camp.load(args.class_name, encoder=args.encoder)
    except FileNotFoundError as exc:
        print(f"Erro: {exc}")
        sys.exit(1)

    print(f"\nGerando {args.num} imagens (tto={'on' if use_tto else 'off'})...\n")
    t0 = time.time()
    results = []
    n_inter = min(args.save_intermediates, args.num)

    if n_inter > 0:
        rng = random.Random(args.seed)
        print(f"  Gerando {n_inter} imagens individuais (use_tto={use_tto})")
        for idx in range(n_inter):
            src_idx = rng.randint(0, len(source_images) - 1)
            tgt_idx = rng.randint(0, len(target_images) - 1)
            sample = camp.generate(
                target_images[tgt_idx],
                source_images[src_idx],
                source_mask=source_masks[src_idx] if source_masks else None,
                use_tto=use_tto,
            )
            _save_sample(sample, output_dir, idx)
            results.append(sample)

    remaining = args.num - n_inter
    if remaining > 0:
        # The batch writer numbers files from aug_0000. Skip it when the
        # single-image path already occupied those names, and continue the index.
        batch = camp.generate_batch(
            target_images,
            source_images,
            masks=source_masks,
            n=remaining,
            out=None if n_inter else output_dir,
            use_tto=use_tto,
            seed=args.seed,
        )
        if n_inter:
            for offset, sample in enumerate(batch):
                _save_sample(sample, output_dir, n_inter + offset)
        results.extend(batch)

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  Concluído: {len(results)}/{args.num} imagens")
    print(f"  Tempo: {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/imagem)")
    print(f"  Output: {output_dir}")
    print(f"  Imagens: {output_dir / 'images'}")
    print(f"  Máscaras: {output_dir / 'masks'}")
    if torch.cuda.is_available():
        print(f"  VRAM pico: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
