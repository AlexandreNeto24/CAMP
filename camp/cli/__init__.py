"""Command line: train segmentor, train recommender, generate.

``python -m camp`` and the ``camp`` console script both call ``main``.
Logging is configured here, once. Library modules do not call ``basicConfig``.
"""

import argparse
import logging
import random
import time
from pathlib import Path

import torch

from camp import CAMP
from camp.io import class_dirs, load_image_mask_pairs, load_images

logger = logging.getLogger(__name__)

# camp/cli/__init__.py -> repository root (the directory that contains camp/ and data/).
REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["REPO_ROOT", "build_parser", "main"]


def _root(data_dir):
    path = Path(data_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.name != "data":
        logger.error("--data-dir must be the 'data' folder (got %s)", data_dir)
        raise SystemExit(1)
    return path.parent


def _finalize_args(args):
    """``--debug`` fills the viz/intermediate count only when that count is still 0."""
    if not getattr(args, "debug", False):
        return args
    if hasattr(args, "save_viz") and args.save_viz == 0:
        args.save_viz = 8
    if hasattr(args, "save_intermediates") and args.save_intermediates == 0:
        args.save_intermediates = 5
    return args


def _add_class_argument(parser):
    parser.add_argument(
        "--class-name", "--class",
        dest="class_name",
        required=True,
        help="Classe alvo (ex: metaplasia, polyps, airplane)",
    )


def _add_data_dir(parser):
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directório base dos datasets (default: data)",
    )


def _add_device(parser):
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")


def _add_segmentor_arguments(parser):
    _add_class_argument(parser)
    _add_data_dir(parser)
    parser.add_argument(
        "--encoder",
        default="efficientnet-b3",
        help="Backbone encoder (default: efficientnet-b3)",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=0.4,
        help="Dropout do decoder (default: 0.4)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Tamanho da imagem (default: 512)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Tamanho dos patches (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Total de épocas (default: 80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate fase 1 (default: 3e-4)",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=30,
        help="Épocas com encoder congelado (default: 30)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Learning rate fase 2 full finetune (default: 1e-4)",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (default: 25)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="Fracção para validação (default: 0.2)",
    )
    parser.add_argument(
        "--p-remove",
        type=float,
        default=0.0,
        help="Probabilidade de blindfold (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directório de output (default: weights/<class>/segmentor/)",
    )
    _add_device(parser)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar visualizações de debug (equivale a --save-viz 8)",
    )
    parser.add_argument(
        "--save-viz",
        type=int,
        default=0,
        metavar="N",
        help="Guardar N exemplos visuais no final do treino (default: 0)",
    )
    parser.set_defaults(func=_train_segmentor)


def _add_recommender_arguments(parser):
    _add_class_argument(parser)
    _add_data_dir(parser)
    parser.add_argument(
        "--encoder",
        default="efficientnet-b3",
        help="Backbone encoder do segmentador (default: efficientnet-b3)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Dimensão do embedding (default: 2048)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Número de attention heads (default: 16)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout rate (default: 0.10)",
    )
    parser.add_argument(
        "--segmentor-weights",
        default=None,
        help="Caminho para pesos do segmentador (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Épocas fase 1 (default: 60)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate fase 1 (default: 3e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20,
        help="Épocas de fine-tuning (default: 20, 0=skip)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=5e-5,
        help="Learning rate fine-tuning (default: 5e-5)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Tamanho da imagem (default: 512)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Tamanho dos patches (default: 128)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="Fracção para validação (default: 0.2)",
    )
    parser.add_argument(
        "--p-remove",
        type=float,
        default=0.0,
        help="Probabilidade de blindfold (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directório de output (default: weights/<class>/recommender/)",
    )
    _add_device(parser)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar visualizações de debug (equivale a --save-viz 8)",
    )
    parser.add_argument(
        "--save-viz",
        type=int,
        default=0,
        metavar="N",
        help="Guardar N exemplos visuais no final do treino (default: 0)",
    )
    parser.set_defaults(func=_train_recommender)


def _add_generate_arguments(parser):
    _add_class_argument(parser)
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=10,
        help="Número de imagens a gerar (default: 10)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Directório de output (default: data/augmented/<class>/)",
    )
    _add_data_dir(parser)
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Diretório de imagens alvo. Se não definido, usa data/<class>/normal/ quando existir",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Diretório de imagens source. Se não definido, usa data/<class>/<class>/",
    )
    parser.add_argument(
        "--no-tto",
        action="store_true",
        help="Desactivar Test-Time Optimization em todas as imagens",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="efficientnet-b3",
        help="Encoder do segmentador (default: efficientnet-b3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproduzibilidade",
    )
    parser.add_argument(
        "--max-source",
        type=int,
        default=50,
        help="Máximo de imagens source e target a usar (default: 50)",
    )
    _add_device(parser)
    parser.add_argument(
        "--save-intermediates",
        type=int,
        default=0,
        metavar="N",
        help="Gerar as primeiras N imagens uma a uma antes do batch (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar debug (equivale a --save-intermediates 5)",
    )
    parser.set_defaults(func=_generate)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="camp",
        description="CAMP train and generate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train",
        help="Train the segmentor or the recommender",
        description="Train the segmentor or the recommender.",
    )
    stages = train.add_subparsers(dest="stage", required=True)

    segmentor = stages.add_parser(
        "segmentor",
        help="CAMP — Treino do Segmentador",
        description="CAMP — Treino do Segmentador",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_segmentor_arguments(segmentor)

    recommender = stages.add_parser(
        "recommender",
        help="CAMP — Treino do Recommender",
        description="CAMP — Treino do Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_recommender_arguments(recommender)

    generate = commands.add_parser(
        "generate",
        help="CAMP — Geração de imagens aumentadas",
        description="CAMP — Geração de imagens aumentadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python -m camp generate --class <class> --num 100
  python -m camp generate --class <class> --num 50 --no-tto --seed 42
  python -m camp generate --class <class> --output ./augmented/
        """,
    )
    _add_generate_arguments(generate)
    return parser


def _train_segmentor(args):
    root = _root(args.data_dir)
    camp = CAMP(
        root=root,
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
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    run_dir = Path(best).parent
    if args.save_viz > 0:
        from camp.cli.viz import _save_segmentor_viz

        _save_segmentor_viz(camp, args, run_dir)

    logger.info("Weights: %s", best)
    logger.info("Config: %s", run_dir / "config.yaml")


def _train_recommender(args):
    root = _root(args.data_dir)
    camp = CAMP(
        root=root,
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
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    run_dir = Path(best).parent
    if args.save_viz > 0:
        from camp.cli.viz import _save_recommender_viz

        _save_recommender_viz(camp, args, run_dir)

    logger.info("Weights: %s", best)
    logger.info("Config: %s", run_dir / "config.yaml")


def _save_sample(sample, output_dir, index):
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    sample.image.save(image_dir / f"aug_{index:04d}.png")
    sample.mask.save(mask_dir / f"aug_{index:04d}.png")


def _generate(args):
    root = _root(args.data_dir)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = root / "data" / "augmented" / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", output_dir)

    source_dir, target_dir = class_dirs(root, args.class_name)
    if args.source_dir:
        source_dir = Path(args.source_dir)
    if args.target_dir:
        target_dir = Path(args.target_dir)
    logger.info("Target dir: %s", target_dir)
    logger.info("Source dir: %s", source_dir)

    target_images = load_images(target_dir, max_images=args.max_source)
    if len(target_images) < 1:
        logger.error("Need at least one target image in %s", target_dir)
        raise SystemExit(1)
    source_images, source_masks = load_image_mask_pairs(source_dir, max_images=args.max_source)
    if len(source_images) < 1:
        logger.error("Need at least one source image in %s", source_dir)
        raise SystemExit(1)

    use_tto = not args.no_tto
    camp = CAMP(root=root, device=args.device, verbose=True)
    try:
        camp.load(args.class_name, encoder=args.encoder)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("Generating %s images (tto=%s)", args.num, "on" if use_tto else "off")
    t0 = time.time()
    results = []
    n_inter = min(args.save_intermediates, args.num)

    if n_inter > 0:
        rng = random.Random(args.seed)
        logger.info("Generating %s images one by one (use_tto=%s)", n_inter, use_tto)
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
    logger.info("Finished: %s/%s images", len(results), args.num)
    logger.info(
        "Time: %.1fs (%.1fs/image)", elapsed, elapsed / max(len(results), 1)
    )
    logger.info("Output: %s", output_dir)
    logger.info("Images: %s", output_dir / "images")
    logger.info("Masks: %s", output_dir / "masks")
    if torch.cuda.is_available():
        logger.info(
            "Peak VRAM: %.1f GB", torch.cuda.max_memory_allocated() / 1024 ** 3
        )


def _configure_logging():
    """Show INFO for a normal ``camp`` run and leave DEBUG quiet."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("camp").setLevel(logging.INFO)


def main(argv=None):
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    _finalize_args(args)
    args.func(args)
