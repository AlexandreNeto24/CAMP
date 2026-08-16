#!/usr/bin/env python
"""Main multiclass classification entrypoint for the CIM project."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from classification.config import project_root, resolve_device, set_seed
from classification.data.data_loaders import (
    LocalBinaryClassificationDataset,
    build_image_transform,
    build_strategy_selection,
    classification_collate,
    load_json,
)
from classification.evaluation.visualization_multiclass import export_model_explanations
from classification.models.models import (
    build_classifier,
    freeze_backbone,
    get_classifier_head,
    unfreeze_backbone,
)
from classification.training import tensor_to_device, train_stage


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Main multiclass classification script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m classification.main_multi_class --augmentation-technique our_approach --multiplier 1x
  python classification/main_multi_class.py --classes airplane,boat,dog,sheep --save-gradcam-samples 100
        ''',
    )

    parser.add_argument('--dataset', type=str, default='multiclass', help='Dataset tag used for run naming')
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
        choices=['0x', '1x', '2x', '3x'],
        help='Synthetic multiplier folder to use',
    )
    parser.add_argument(
        '--model-architecture',
        type=str,
        default='resnet34',
        choices=['resnet34', 'vit_b_16'],
        help='Classification model architecture to use',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--image-size', type=int, default=512, help='Resize images to this size')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs with frozen backbone')
    parser.add_argument('--num-epochs', '--finetune-epochs', dest='finetune_epochs', type=int, default=20, help='Number of fine-tuning epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-3, help='Learning rate for warmup head training')
    parser.add_argument('--lr', '--finetune-lr', dest='finetune_lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for AdamW')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker count')
    parser.add_argument('--max-synthetic-samples', type=int, default=None, help='Optional cap for synthetic samples loaded from the tar')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save checkpoints and reports')
    parser.add_argument('--save-gradcam-samples', type=int, default=0, help='Number of test samples to export with Grad-CAM overlays')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated class names to include; defaults to all detected class folders')
    parser.add_argument('--reference-class', type=str, default=None, help='Class used to enumerate strategies; defaults to the first selected class')
    parser.add_argument('--synthetic-classes', type=str, default=None, help='Comma-separated class names allowed to contribute synthetic samples; defaults to all selected classes')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Disable ImageNet pretrained weights')
    parser.set_defaults(pretrained=True)

    return parser


def parse_csv_argument(value):
    if value is None:
        return None
    items = [item.strip() for item in value.split(',') if item.strip()]
    return items or None


def detect_class_names(data_root: Path):
    class_names = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir() or path.name == 'augmented':
            continue
        if (path / 'normal').exists() and (path / path.name / 'images').exists():
            class_names.append(path.name)
    return class_names


class SoftLabelWrapper(Dataset):
    """
    Wraps standard binary/multiclass datasets returning integer hard-labels, 
    and converts them to probability vectors (Soft Labels) for mixed batching.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label_idx = int(item['label'])
        soft_label = torch.zeros(self.num_classes, dtype=torch.float32)
        soft_label[label_idx] = 1.0
        item['label'] = soft_label
        return item


class SyntheticTarMulticlassDataset(Dataset):
    def __init__(
        self,
        tar_path,
        label,
        num_classes,
        image_size=512,
        strategy_id=None,
        train_folds=None,
        transform=None,
        max_samples=None,
    ):
        self.transform = transform or build_image_transform(image_size)
        self.label = label
        self.num_classes = num_classes
        self.samples = []

        import webdataset as wds

        tar_path = Path(tar_path)
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            return

        synthetic_iter = wds.WebDataset(str(tar_path), shardshuffle=False, empty_check=False).decode('pil')
        allowed_folds = set(train_folds or [])

        try:
            for sample in synthetic_iter:
                metadata = sample.get('json', {}) or {}
                sample_strategy = metadata.get('strategy')
                sample_role = metadata.get('strategy_role')
                sample_train_folds = set(metadata.get('train_folds', []))
                technique = metadata.get('technique', '')

                if strategy_id is not None and sample_strategy != strategy_id:
                    continue
                if allowed_folds and not sample_train_folds.intersection(allowed_folds):
                    continue
                if sample_role is not None and sample_role != 'train':
                    continue

                if technique in ['cutmix', 'saliencymix']:
                    confidence = float(metadata.get('confidence', 1.0))
                else:
                    confidence = 1.0

                self.samples.append({
                    'image': sample['jpg'],
                    'path': metadata.get('source_id') or metadata.get('target_id') or sample.get('__key__', 'synthetic'),
                    'source': 'synthetic',
                    'confidence': confidence
                })

                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        except ValueError:
            return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image.convert('RGB'))

        soft_label = torch.zeros(self.num_classes, dtype=torch.float32)
        soft_label[0] = 1.0 - sample['confidence']
        soft_label[self.label] = sample['confidence']

        return {
            'image': image,
            'label': soft_label,
            'path': sample['path'],
            'source': sample['source'],
        }


def build_multiclass_dataset_for_class(
    data_root: Path,
    augmented_root: Path,
    class_name: str,
    class_label: int,
    num_classes: int,
    technique: str,
    multiplier: str,
    strategy_name: str,
    image_size: int,
    max_synthetic_samples=None,
    synthetic_enabled=True,
):
    class_root = data_root / class_name
    class_aug_root = augmented_root / class_name

    # Se o multiplier for '0x' (SOT), usamos os metadados e o tar do '1x'
    tar_multiplier = '1x' if multiplier == '0x' else multiplier

    kfold_splits = load_json(class_aug_root / 'k_fold_splits.json')
    strategy_metadata = load_json(class_aug_root / technique / tar_multiplier / strategy_name / 'strategy_metadata.json')
    fold_selection = build_strategy_selection(class_root, strategy_metadata, kfold_splits)

    train_selection = fold_selection['train']
    val_selection = fold_selection['val']
    test_selection = fold_selection['test']

    # Se for '0x', excluímos os dados reais de treino
    use_real_train = (multiplier != '0x')

    train_components = []
    if use_real_train:
        train_real_positive = SoftLabelWrapper(LocalBinaryClassificationDataset(train_selection['instance_images'], label=class_label, image_size=image_size), num_classes)
        train_real_negative = SoftLabelWrapper(LocalBinaryClassificationDataset(train_selection['normal_images'], label=0, image_size=image_size), num_classes)
        train_components.extend([train_real_negative, train_real_positive])

    val_positive = SoftLabelWrapper(LocalBinaryClassificationDataset(val_selection['instance_images'], label=class_label, image_size=image_size), num_classes)
    val_negative = SoftLabelWrapper(LocalBinaryClassificationDataset(val_selection['normal_images'], label=0, image_size=image_size), num_classes)
    test_positive = SoftLabelWrapper(LocalBinaryClassificationDataset(test_selection['instance_images'], label=class_label, image_size=image_size), num_classes)
    test_negative = SoftLabelWrapper(LocalBinaryClassificationDataset(test_selection['normal_images'], label=0, image_size=image_size), num_classes)

    synthetic_tar_path = class_aug_root / technique / tar_multiplier / strategy_name / f'{class_name}_{tar_multiplier}_{technique}.tar'

    synthetic_dataset = None
    if synthetic_enabled and synthetic_tar_path.exists():
        synthetic_dataset = SyntheticTarMulticlassDataset(
            synthetic_tar_path,
            label=class_label,
            num_classes=num_classes,
            image_size=image_size,
            strategy_id=fold_selection['strategy_id'],
            train_folds=fold_selection['train_folds'],
            max_samples=max_synthetic_samples,
        )
        if len(synthetic_dataset) > 0:
            train_components.append(synthetic_dataset)

    datasets = {
        'strategy_metadata': strategy_metadata,
        'fold_selection': fold_selection,
        'val_positive': val_positive,
        'val_negative': val_negative,
        'test_positive': test_positive,
        'test_negative': test_negative,
        'train_synthetic': synthetic_dataset,
        'train': ConcatDataset(train_components),
        'val': ConcatDataset([val_negative, val_positive]),
        'test': ConcatDataset([test_negative, test_positive]),
    }
    if use_real_train:
        datasets['train_real_positive'] = train_real_positive
        datasets['train_real_negative'] = train_real_negative

    return datasets


def build_multiclass_datasets(
    data_root: Path,
    augmented_root: Path,
    classes,
    reference_class: str,
    technique: str,
    multiplier: str,
    strategy_name: str,
    image_size: int = 512,
    max_synthetic_samples=None,
    synthetic_classes=None,
):
    class_to_label = {class_name: index + 1 for index, class_name in enumerate(classes)}
    datasets_by_class = {}
    train_components = []
    val_components = []
    test_components = []
    class_summaries = {}
    reference_payload = None

    class_count = len(classes) + 1  # Inclui a classe 0 (normal)

    for class_name in classes:
        use_synthetic = synthetic_classes is None or class_name in synthetic_classes
        class_datasets = build_multiclass_dataset_for_class(
            data_root=data_root,
            augmented_root=augmented_root,
            class_name=class_name,
            class_label=class_to_label[class_name],
            num_classes=class_count,
            technique=technique,
            multiplier=multiplier,
            strategy_name=strategy_name,
            image_size=image_size,
            max_synthetic_samples=max_synthetic_samples,
            synthetic_enabled=use_synthetic,
        )

        datasets_by_class[class_name] = class_datasets
        
        if 'train_real_negative' in class_datasets:
            train_components.extend([class_datasets['train_real_negative'], class_datasets['train_real_positive']])
        if class_datasets['train_synthetic'] is not None:
            train_components.append(class_datasets['train_synthetic'])
            
        val_components.extend([class_datasets['val_negative'], class_datasets['val_positive']])
        test_components.extend([class_datasets['test_negative'], class_datasets['test_positive']])

        train_norm_len = len(class_datasets['train_real_negative']) if 'train_real_negative' in class_datasets else 0
        train_inst_len = len(class_datasets['train_real_positive']) if 'train_real_positive' in class_datasets else 0
        syn_len = len(class_datasets['train_synthetic']) if class_datasets['train_synthetic'] is not None else 0

        class_summaries[class_name] = {
            'class_label': class_to_label[class_name],
            'train_counts': {
                'normal_images': train_norm_len,
                'instance_images': train_inst_len,
                'synthetic_images': syn_len,
            },
            'val_counts': {
                'normal_images': len(class_datasets['val_negative']),
                'instance_images': len(class_datasets['val_positive']),
            },
            'test_counts': {
                'normal_images': len(class_datasets['test_negative']),
                'instance_images': len(class_datasets['test_positive']),
            },
        }

        if class_name == reference_class:
            reference_payload = class_datasets

    if reference_payload is None:
        reference_payload = datasets_by_class[classes[0]]

    datasets = {
        'class_to_label': class_to_label,
        'class_summaries': class_summaries,
        'reference_class': reference_class,
        'reference_fold_selection': reference_payload['fold_selection'],
        'reference_strategy_metadata': reference_payload['strategy_metadata'],
        'by_class': datasets_by_class,
        'train': ConcatDataset(train_components),
        'val': ConcatDataset(val_components),
        'test': ConcatDataset(test_components),
    }

    return datasets


def collect_predictions_multiclass(model, loader, criterion, device='cpu', num_classes=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            images, labels = tensor_to_device(batch, device)
            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            if labels.ndim == 2:
                true_labels = labels.argmax(dim=1)
            else:
                true_labels = labels

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            y_true.extend(true_labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            y_prob.extend(probabilities.detach().cpu().tolist())

    return y_true, y_pred, y_prob, total_loss / max(total_samples, 1)


def compute_multiclass_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    if num_classes is None:
        num_classes = len(set(y_true).union(set(y_pred)))

    labels = list(range(num_classes))
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
    }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics.update({
        'precision_macro': float(macro_precision),
        'recall_macro': float(macro_recall),
        'f1_macro': float(macro_f1),
        'precision_weighted': float(weighted_precision),
        'recall_weighted': float(weighted_recall),
        'f1_weighted': float(weighted_f1),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    })

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for class_index, (precision, recall, f1, support) in enumerate(
        zip(per_class_precision, per_class_recall, per_class_f1, per_class_support)
    ):
        metrics[f'precision_class_{class_index}'] = float(precision)
        metrics[f'recall_class_{class_index}'] = float(recall)
        metrics[f'f1_class_{class_index}'] = float(f1)
        metrics[f'support_class_{class_index}'] = int(support)

    if y_prob is not None:
        try:
            y_prob_array = np.asarray(y_prob)
            metrics['roc_auc_ovr_macro'] = float(
                roc_auc_score(y_true, y_prob_array, multi_class='ovr', average='macro')
            )
        except ValueError:
            metrics['roc_auc_ovr_macro'] = None

    return metrics


def evaluate_multiclass_model(model, loader, criterion, device='cpu', num_classes=None):
    y_true, y_pred, y_prob, loss = collect_predictions_multiclass(
        model, loader, criterion, device=device, num_classes=num_classes
    )
    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob=y_prob, num_classes=num_classes)
    metrics['loss'] = float(loss)
    return metrics


def append_strategy_metrics(metric_store: dict, phase: str, metrics: dict):
    for metric_key, value in metrics.items():
        metric_store.setdefault(phase, {}).setdefault(metric_key, [])
        metric_store[phase][metric_key].append(value)


def average_metric_history(all_metrics: dict):
    avg_metrics = {'val': {}, 'test': {}}

    for phase in ['val', 'test']:
        for metric_key, values in all_metrics[phase].items():
            if metric_key == 'confusion_matrix':
                matrices = [np.asarray(matrix, dtype=float) for matrix in values if matrix is not None]
                if matrices:
                    avg_metrics[phase]['confusion_matrix_mean'] = np.mean(np.stack(matrices), axis=0).tolist()
                continue

            numeric_values = [value for value in values if isinstance(value, (int, float, np.integer, np.floating))]
            if numeric_values:
                avg_metrics[phase][metric_key] = float(np.mean(numeric_values))
                avg_metrics[phase][f'{metric_key}_std'] = float(np.std(numeric_values))

    return avg_metrics


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.model_architecture.startswith('vit') and args.image_size != 224:
        print('ViT-B/16 uses 224x224 inputs by default. Overriding image_size to 224.')
        args.image_size = 224

    data_root = project_root / 'data'
    augmented_root = data_root / 'augmented'
    classes = parse_csv_argument(args.classes)
    if classes is None:
        classes = detect_class_names(data_root)
    classes = sorted(classes)
    if not classes:
        raise ValueError(f'No class folders were detected under {data_root}')

    synthetic_classes = parse_csv_argument(args.synthetic_classes)
    if synthetic_classes is not None:
        synthetic_classes = [c for c in classes if c in synthetic_classes]

    reference_class = args.reference_class or classes[0]
    if reference_class not in classes:
        raise ValueError(f'Reference class {reference_class!r} not in classes: {classes}')

    output_root = args.output_dir or (
        project_root / 'runs' / 'classification' / args.model_architecture / 'multiclass' / args.augmentation_technique / args.multiplier
    )
    output_root = Path(output_root)

    print(f'Project root: {project_root}')
    print(f'Data root: {data_root}')
    print(f'Augmented root: {augmented_root}')
    print(f'Classes: {classes}')
    print(f'Reference class: {reference_class}')
    print(f'Device: {device}')

    all_metrics = {'val': {}, 'test': {}}
    class_count = len(classes) + 1

    for n in range(5):
        strategy_name = f"strategy{n+1}"
        print(f'\n\n=== Starting training for {strategy_name} ===')
        output_dir = output_root / strategy_name
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = build_multiclass_datasets(
            data_root=data_root,
            augmented_root=augmented_root,
            classes=classes,
            reference_class=reference_class,
            technique=args.augmentation_technique,
            multiplier=args.multiplier,
            strategy_name=strategy_name,
            image_size=args.image_size,
            max_synthetic_samples=args.max_synthetic_samples,
            synthetic_classes=synthetic_classes,
        )

        loaders = {
            'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=classification_collate),
            'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=classification_collate),
            'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=classification_collate),
        }

        model = build_classifier(
            args.model_architecture,
            pretrained=args.pretrained,
            num_classes=class_count,
            image_size=args.image_size,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

        history = []
        print('\nStage 1: warmup with frozen backbone and classifier head initialisation')
        freeze_backbone(model)
        warmup_optimizer = torch.optim.AdamW(get_classifier_head(model).parameters(), lr=args.warmup_lr, weight_decay=args.weight_decay)
        
        def evaluate_with_class_count(m, l, crit, device='cpu'):
            return evaluate_multiclass_model(m, l, crit, device=device, num_classes=class_count)

        warmup_history, warmup_best_state, _ = train_stage(
            model=model, train_loader=loaders['train'], val_loader=loaders['val'],
            criterion=criterion, optimizer=warmup_optimizer, epochs=args.warmup_epochs,
            device=device, stage_name='warmup', evaluate_fn=evaluate_with_class_count,
        )
        history.extend(warmup_history)

        if warmup_best_state is not None:
            model.load_state_dict(warmup_best_state)

        print('\nStage 2: fine-tuning the whole backbone')
        unfreeze_backbone(model)
        finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
        finetune_history, finetune_best_state, _ = train_stage(
            model=model, train_loader=loaders['train'], val_loader=loaders['val'],
            criterion=criterion, optimizer=finetune_optimizer, epochs=args.finetune_epochs,
            device=device, stage_name='finetune', evaluate_fn=evaluate_with_class_count,
        )
        history.extend(finetune_history)

        if finetune_best_state is not None:
            model.load_state_dict(finetune_best_state)

        val_metrics = evaluate_with_class_count(model, loaders['val'], criterion, device=device)
        test_metrics = evaluate_with_class_count(model, loaders['test'], criterion, device=device)

        append_strategy_metrics(all_metrics, 'val', val_metrics)
        append_strategy_metrics(all_metrics, 'test', test_metrics)

        best_model_path = output_dir / 'best_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': args.model_architecture,
            'class_to_label': datasets['class_to_label'],
            'reference_class': datasets['reference_class'],
            'strategy_name': strategy_name,
            'args': vars(args),
        }, best_model_path)

        if args.save_gradcam_samples > 0:
            explanation_dir = output_dir / 'gradcam' / args.model_architecture
            base_quota, remainder = divmod(args.save_gradcam_samples, class_count)
            quotas = {label: base_quota + (1 if label < remainder else 0) for label in range(class_count)}

            cam_metrics = export_model_explanations(
                model=model, dataset=datasets['test'], output_dir=explanation_dir,
                architecture=args.model_architecture, samples_per_class=quotas,
                device=device, return_per_class_metrics=True,
            )
            for key, value in cam_metrics.items():
                test_metrics[key] = value
                all_metrics['test'].setdefault(key, []).append(value)

        with open(output_dir / 'evaluation_results.json', 'w') as handle:
            json.dump({'val': val_metrics, 'test': test_metrics}, handle, indent=2)

        plt.figure(figsize=(10, 5))
        plt.plot([record['train_loss'] for record in history], label='Train Loss')
        plt.plot([record['val_loss'] for record in history], label='Val Loss')
        best_epoch = min(range(len(history)), key=lambda index: history[index]['val_loss'])
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.title(f'Training and Validation Loss - {strategy_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_dir / 'loss_plot.png')
        plt.close()

        del model, loaders, datasets
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    avg_metrics = average_metric_history(all_metrics)

    print('\nAverage metrics across all strategies:')
    for phase in ['val', 'test']:
        print(f"  {phase} | " + " | ".join(f"{k}={v:.4f}" for k, v in avg_metrics[phase].items() if isinstance(v, float)))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_root / f'{timestamp}_average_evaluation_results.json', 'w') as handle:
        json.dump(avg_metrics, handle, indent=2)

    with pd.ExcelWriter(output_root / f'{timestamp}_average_evaluation_results.xlsx') as writer:
        pd.DataFrame([avg_metrics['val']]).to_excel(writer, sheet_name='val', index=False)
        pd.DataFrame([avg_metrics['test']]).to_excel(writer, sheet_name='test', index=False)


if __name__ == '__main__':
    main()