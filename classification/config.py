"""Configuration helpers for the classification pipeline."""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Main classification script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m classification.main_classification --dataset metaplasia --model-architecture resnet34
  python classification/main_classification.py --dataset metaplasia --multiplier 1x --strategy strategy1
        ''',
    )

    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for classification')
    parser.add_argument(
        '--augmentation-technique',
        type=str,
        default='our_approach',
        choices=['our_approach', 'cutmix', 'saliencymix', 'copypaste'],
        help='Synthetic augmentation technique to merge into train',
    )
    parser.add_argument(
        '--multiplier',
        type=str,
        default='1x',
        choices=['1x', '2x', '3x'],
        help='Synthetic multiplier folder to use',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='strategy1',
        help='Strategy folder name to use for the synthetic tar and metadata',
    )
    parser.add_argument(
        '--model-architecture',
        type=str,
        default='resnet34',
        choices=['resnet34', 'vit_b_16'],
        help='Classification model architecture to use',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--image-size', type=int, default=512, help='Resize images to this size')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Number of warmup epochs with frozen backbone')
    parser.add_argument('--num-epochs', '--finetune-epochs', dest='finetune_epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-3, help='Learning rate for warmup head training')
    parser.add_argument('--lr', '--finetune-lr', dest='finetune_lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--max-synthetic-samples', type=int, default=None, help='Optional cap for synthetic samples loaded from the tar')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save checkpoints and reports')
    parser.add_argument('--save-gradcam-samples', type=int, default=0, help='Number of test samples to export with Grad-CAM overlays')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Disable ImageNet pretrained weights')
    parser.set_defaults(pretrained=True)

    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print('CUDA was requested but is not available. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_name)


def build_paths(args):
    dataset_root = project_root / 'data' / args.dataset
    augmented_root = project_root / 'data' / 'augmented' / args.dataset
    strategy_root = augmented_root / args.augmentation_technique / args.multiplier / args.strategy

    strategy_metadata_path = strategy_root / 'strategy_metadata.json'
    synthetic_tar_path = strategy_root / f'{args.dataset}_{args.multiplier}_{args.augmentation_technique}.tar'
    kfold_splits_path = augmented_root / 'k_fold_splits.json'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = project_root / 'runs' / 'classification' / args.dataset / args.strategy / timestamp

    return {
        'dataset_root': dataset_root,
        'augmented_root': augmented_root,
        'strategy_root': strategy_root,
        'strategy_metadata_path': strategy_metadata_path,
        'synthetic_tar_path': synthetic_tar_path,
        'kfold_splits_path': kfold_splits_path,
        'output_dir': output_dir,
    }


def ensure_required_files(paths: dict):
    required = {
        'strategy metadata': paths['strategy_metadata_path'],
        'synthetic tar': paths['synthetic_tar_path'],
        'k-fold splits': paths['kfold_splits_path'],
        'local dataset root': paths['dataset_root'],
    }
    missing = [f'{name}: {path}' for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Missing required inputs:\n' + '\n'.join(missing))
