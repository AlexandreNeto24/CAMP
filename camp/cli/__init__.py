"""Command line: train segmentor, train recommender, generate.

``python -m camp`` and the ``camp`` console script both call ``main``.
Logging is configured here, once. Library modules do not call ``basicConfig``.

Defaults come from ``configs/default.yaml`` (or ``--config PATH``). CLI flags
override the file.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from camp import CAMP
from camp.cli.config import (
    DEFAULT_CONFIG_RELATIVE,
    load_cli_config,
    resolve_config_path,
    section_defaults,
)
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


# How many debug figures ``--debug`` writes after train / generate.
_DEBUG_TRAIN_VIZ = 8
_DEBUG_GENERATE_STRIPS = 5


def _peek_config_path(argv):
    """Read ``--config`` from argv without a full parse. Default: configs/default.yaml."""
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token == "--config" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return str(DEFAULT_CONFIG_RELATIVE)


def _add_config_argument(parser):
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_RELATIVE),
        help="YAML with CLI defaults (default: configs/default.yaml)",
    )


def _add_class_argument(parser, cfg):
    parser.add_argument(
        "--class-name", "--class",
        dest="class_name",
        default=cfg.get("class_name"),
        help="Target class folder under data/ (or set common.class_name in the YAML)",
    )


def _add_data_dir(parser, cfg):
    default = cfg.get("data_dir", "data")
    parser.add_argument(
        "--data-dir",
        default=default,
        help="Dataset root folder named data (default: %s)" % default,
    )


def _add_device(parser, cfg):
    default = cfg.get("device", "cuda")
    parser.add_argument(
        "--device",
        default=default,
        help="Device (default: %s)" % default,
    )


def _add_encoder(parser, cfg, help_text):
    default = cfg.get("encoder", "efficientnet-b3")
    parser.add_argument(
        "--encoder",
        default=default,
        help="%s (default: %s)" % (help_text, default),
    )


def _add_segmentor_arguments(parser, cfg):
    _add_config_argument(parser)
    _add_class_argument(parser, cfg)
    _add_data_dir(parser, cfg)
    _add_encoder(parser, cfg, "Backbone encoder")
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=cfg.get("decoder_dropout", 0.4),
        help="Decoder dropout (default: %s)" % cfg.get("decoder_dropout", 0.4),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=cfg.get("image_size", 512),
        help="Image size (default: %s)" % cfg.get("image_size", 512),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=cfg.get("patch_size", 128),
        help="Patch size (default: %s)" % cfg.get("patch_size", 128),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.get("epochs", 80),
        help="Total epochs (default: %s)" % cfg.get("epochs", 80),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.get("batch_size", 8),
        help="Batch size (default: %s)" % cfg.get("batch_size", 8),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.get("lr", 3e-4),
        help="Phase-1 learning rate (default: %s)" % cfg.get("lr", 3e-4),
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=cfg.get("freeze_epochs", 30),
        help="Epochs with frozen encoder (default: %s)" % cfg.get("freeze_epochs", 30),
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=cfg.get("finetune_lr", 1e-4),
        help="Phase-2 learning rate (default: %s)" % cfg.get("finetune_lr", 1e-4),
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=cfg.get("accumulation_steps", 2),
        help="Gradient accumulation steps (default: %s)" % cfg.get("accumulation_steps", 2),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=cfg.get("patience", 25),
        help="Early-stopping patience (default: %s)" % cfg.get("patience", 25),
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=cfg.get("split_ratio", 0.2),
        help="Validation fraction (default: %s)" % cfg.get("split_ratio", 0.2),
    )
    parser.add_argument(
        "--output",
        default=cfg.get("output"),
        help="Output dir, relative to the project root (default: weights/<class>/segmentor/)",
    )
    _add_device(parser, cfg)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save %s validation debug figures beside best.pth" % _DEBUG_TRAIN_VIZ,
    )
    # After store_true: set_defaults must come last or action defaults win.
    parser.set_defaults(debug=bool(cfg.get("debug", False)), func=_train_segmentor)


def _add_recommender_arguments(parser, cfg):
    _add_config_argument(parser)
    _add_class_argument(parser, cfg)
    _add_data_dir(parser, cfg)
    _add_encoder(parser, cfg, "Segmentor backbone encoder")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=cfg.get("embedding_dim", 2048),
        help="Embedding dimension (default: %s)" % cfg.get("embedding_dim", 2048),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=cfg.get("num_heads", 16),
        help="Attention heads (default: %s)" % cfg.get("num_heads", 16),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=cfg.get("dropout", 0.10),
        help="Dropout rate (default: %s)" % cfg.get("dropout", 0.10),
    )
    parser.add_argument(
        "--segmentor-weights",
        default=cfg.get("segmentor_weights"),
        help="Segmentor checkpoint path (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.get("epochs", 60),
        help="Phase-1 epochs (default: %s)" % cfg.get("epochs", 60),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.get("lr", 3e-4),
        help="Phase-1 learning rate (default: %s)" % cfg.get("lr", 3e-4),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.get("batch_size", 4),
        help="Batch size (default: %s)" % cfg.get("batch_size", 4),
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=cfg.get("accumulation_steps", 4),
        help="Gradient accumulation steps (default: %s)" % cfg.get("accumulation_steps", 4),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=cfg.get("patience", 30),
        help="Early-stopping patience (default: %s)" % cfg.get("patience", 30),
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=cfg.get("finetune_epochs", 20),
        help="Fine-tune epochs (default: %s, 0=skip)" % cfg.get("finetune_epochs", 20),
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=cfg.get("finetune_lr", 5e-5),
        help="Fine-tune learning rate (default: %s)" % cfg.get("finetune_lr", 5e-5),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=cfg.get("image_size", 512),
        help="Image size (default: %s)" % cfg.get("image_size", 512),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=cfg.get("patch_size", 128),
        help="Patch size (default: %s)" % cfg.get("patch_size", 128),
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=cfg.get("split_ratio", 0.2),
        help="Validation fraction (default: %s)" % cfg.get("split_ratio", 0.2),
    )
    parser.add_argument(
        "--output",
        default=cfg.get("output"),
        help="Output dir, relative to the project root (default: weights/<class>/recommender/)",
    )
    _add_device(parser, cfg)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save %s validation debug figures beside best.pth" % _DEBUG_TRAIN_VIZ,
    )
    parser.set_defaults(debug=bool(cfg.get("debug", False)), func=_train_recommender)


def _add_generate_arguments(parser, cfg):
    _add_config_argument(parser)
    _add_class_argument(parser, cfg)
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=cfg.get("num", 10),
        help="Number of images to generate (default: %s)" % cfg.get("num", 10),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=cfg.get("output"),
        help="Output dir, relative to the project root (default: data/augmented/<class>/)",
    )
    _add_data_dir(parser, cfg)
    parser.add_argument(
        "--target-dir",
        type=str,
        default=cfg.get("target_dir"),
        help="Target image directory (default: data/<class>/normal/ when present)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=cfg.get("source_dir"),
        help="Source image directory (default: data/<class>/<class>/)",
    )
    parser.add_argument(
        "--no-tto",
        action="store_true",
        help="Disable test-time optimisation on every image",
    )
    _add_encoder(parser, cfg, "Segmentor encoder")
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.get("seed"),
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--max-source",
        type=int,
        default=cfg.get("max_source", 50),
        help="Max source and target images to load (default: %s)" % cfg.get("max_source", 50),
    )
    _add_device(parser, cfg)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save %s full-pipeline strips under output/debug "
            "(patch, segmented area, heatmap, collage)" % _DEBUG_GENERATE_STRIPS
        ),
    )
    parser.set_defaults(
        no_tto=not bool(cfg.get("use_tto", True)),
        debug=bool(cfg.get("debug", False)),
        func=_generate,
    )


def build_parser(config_data=None):
    """Build the CLI parser. ``config_data`` is the loaded YAML mapping."""
    if config_data is None:
        config_data = {}
    seg_cfg = section_defaults(config_data, "segmentor")
    rec_cfg = section_defaults(config_data, "recommender")
    gen_cfg = section_defaults(config_data, "generate")

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
        help="CAMP — Segmentor training",
        description="CAMP — Segmentor training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_segmentor_arguments(segmentor, seg_cfg)

    recommender = stages.add_parser(
        "recommender",
        help="CAMP — Recommender training",
        description="CAMP — Recommender training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_recommender_arguments(recommender, rec_cfg)

    generate = commands.add_parser(
        "generate",
        help="CAMP — Augmented image generation",
        description="CAMP — Augmented image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m camp generate
  python -m camp generate --class <class> --num 100
  python -m camp generate --class <class> --num 50 --no-tto --seed 42
  python -m camp generate --config configs/default.yaml
        """,
    )
    _add_generate_arguments(generate, gen_cfg)
    return parser


def _require_class_name(args, config_path):
    if args.class_name:
        return
    logger.error(
        "class_name is missing: set common.class_name in %s or pass --class",
        config_path,
    )
    raise SystemExit(1)


def _train_segmentor(args):
    root = _root(args.data_dir)
    camp = CAMP(
        root=root,
        device=args.device,
        verbose=True,
        image_size=(args.image_size, args.image_size),
    )
    try:
        camp.train_segmentor(
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
            device=args.device,
            output=args.output,
            debug=args.debug,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


def _train_recommender(args):
    root = _root(args.data_dir)
    camp = CAMP(
        root=root,
        device=args.device,
        verbose=True,
        image_size=(args.image_size, args.image_size),
    )
    try:
        camp.train_recommender(
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
            device=args.device,
            output=args.output,
            segmentor_weights=args.segmentor_weights,
            debug=args.debug,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


def _generate(args):
    root = _root(args.data_dir)
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
    else:
        output_dir = root / "data" / "augmented" / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir, target_dir = class_dirs(root, args.class_name)
    if args.source_dir:
        source_dir = Path(args.source_dir)
    if args.target_dir:
        target_dir = Path(args.target_dir)

    target_images = load_images(target_dir, max_images=args.max_source)
    if len(target_images) < 1:
        logger.error("Need at least one target image in %s", target_dir)
        raise SystemExit(1)
    source_images, source_masks = load_image_mask_pairs(source_dir, max_images=args.max_source)
    if len(source_images) < 1:
        logger.error("Need at least one source image in %s", source_dir)
        raise SystemExit(1)

    use_tto = not args.no_tto
    logger.info(
        "generate  %s  %s images  tto %s",
        args.class_name,
        args.num,
        "on" if use_tto else "off",
    )
    logger.info("  %-8s %s images  %s", "source", len(source_images), source_dir)
    logger.info("  %-8s %s images  %s", "target", len(target_images), target_dir)
    logger.info("  %-8s %s", "output", output_dir)

    camp = CAMP(root=root, device=args.device, verbose=True)
    try:
        camp.load(args.class_name, encoder=args.encoder)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    t0 = time.time()
    n_debug = min(_DEBUG_GENERATE_STRIPS, args.num) if args.debug else 0
    results = camp.generate_batch(
        target_images,
        source_images,
        masks=source_masks,
        n=args.num,
        out=output_dir,
        use_tto=use_tto,
        seed=args.seed,
        debug_n=n_debug,
    )
    if n_debug:
        from camp.cli.viz import write_generate_debugs

        write_generate_debugs(results, output_dir, limit=n_debug)

    elapsed = time.time() - t0
    peak = ""
    if torch.cuda.is_available():
        peak = "  peak %.1f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3)
    logger.info(
        "done  %s/%s  %.1fs  %.1fs each%s",
        len(results),
        args.num,
        elapsed,
        elapsed / max(len(results), 1),
        peak,
    )


def _configure_logging():
    """Show INFO for a normal ``camp`` run and leave DEBUG quiet."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("camp").setLevel(logging.INFO)


def main(argv=None):
    _configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _peek_config_path(argv)
    resolved = resolve_config_path(config_path, REPO_ROOT)
    try:
        config_data = load_cli_config(config_path, REPO_ROOT)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    if not resolved.is_file() and Path(config_path) != DEFAULT_CONFIG_RELATIVE:
        logger.error("Config file not found: %s", resolved)
        raise SystemExit(1)

    parser = build_parser(config_data)
    args = parser.parse_args(argv)
    _require_class_name(args, resolved)
    args.func(args)
