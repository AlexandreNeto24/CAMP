#!/usr/bin/env python
"""Main classification entrypoint for the CIM project."""

import torch
import torch.nn as nn

from classification.config import (
    build_arg_parser,
    build_paths,
    ensure_required_files,
    project_root,
    resolve_device,
    set_seed,
)
from classification.data.data_loaders import build_classification_dataloaders, load_json
from classification.evaluation.reporting import save_run_artifacts
from classification.evaluation.metrics import evaluate_model
from classification.evaluation.visualization import export_model_explanations
from classification.models.models import build_classifier, freeze_backbone, get_classifier_head, unfreeze_backbone
from classification.training import train_stage


def main():
    # get arguments, set random seed, and resolve device
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.model_architecture.startswith('vit') and args.image_size != 224:
        print('ViT-B/16 uses 224x224 inputs by default. Overriding image_size to 224.')
        args.image_size = 224
    
    # get sure that all required files exist and build paths
    paths = build_paths(args)
    ensure_required_files(paths)

    paths['output_dir'].mkdir(parents=True, exist_ok=True)

    # load strategy metadata and k-fold splits
    strategy_metadata = load_json(paths['strategy_metadata_path'])
    kfold_splits = load_json(paths['kfold_splits_path'])

    print(f'Project root: {project_root}')
    print(f'Dataset root: {paths["dataset_root"]}')
    print(f'Augmented root: {paths["augmented_root"]}')
    print(f'Strategy root: {paths["strategy_root"]}')
    print(f'Synthetic tar: {paths["synthetic_tar_path"]}')
    print(f'Device: {device}')

    # build datasets and dataloaders
    datasets, loaders = build_classification_dataloaders(
        dataset_root=paths['dataset_root'],
        strategy_metadata=strategy_metadata,
        kfold_splits=kfold_splits,
        synthetic_tar_path=paths['synthetic_tar_path'],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_synthetic_samples=args.max_synthetic_samples,
    )

    fold_selection = datasets['fold_selection']
    print('\nFold selection:')
    print(f"  train folds: {fold_selection['train_folds']}")
    print(f"  val fold: {fold_selection['val_fold']}")
    print(f"  test fold: {fold_selection['test_fold']}")
    print('\nDataset sizes:')
    print(f"  train real negative: {len(datasets['train_real_negative'])}")
    print(f"  train real positive: {len(datasets['train_real_positive'])}")
    print(f"  train synthetic: {len(datasets['train_synthetic'])}")
    print(f"  val total: {len(datasets['val'])}")
    print(f"  test total: {len(datasets['test'])}")
    # build model, criterion, and optimizers
    model = build_classifier(
        args.model_architecture,
        pretrained=args.pretrained,
        num_classes=2,
        image_size=args.image_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    history = []

    print('\nStage 1: warmup with frozen backbone and ImageNet head initialisation')
    freeze_backbone(model)
    warmup_optimizer = torch.optim.AdamW(get_classifier_head(model).parameters(), lr=args.warmup_lr, weight_decay=args.weight_decay)
    warmup_history, warmup_best_state, _ = train_stage(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=warmup_optimizer,
        epochs=args.warmup_epochs,
        device=device,
        stage_name='warmup',
        evaluate_fn=evaluate_model,
    )
    history.extend(warmup_history)

    if warmup_best_state is not None:
        model.load_state_dict(warmup_best_state)

    print('\nStage 2: fine-tuning the whole backbone')
    unfreeze_backbone(model)
    finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    finetune_history, finetune_best_state, _ = train_stage(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=finetune_optimizer,
        epochs=args.finetune_epochs,
        device=device,
        stage_name='finetune',
        evaluate_fn=evaluate_model,
    )
    history.extend(finetune_history)

    if finetune_best_state is not None:
        model.load_state_dict(finetune_best_state)

    val_metrics = evaluate_model(model, loaders['val'], criterion, device=device)
    test_metrics = evaluate_model(model, loaders['test'], criterion, device=device)

    print('\nFinal metrics:')
    print(f"  val  | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"  test | loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

    best_model_path = paths['output_dir'] / 'best_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_architecture': args.model_architecture,
            'strategy_metadata': strategy_metadata,
            'fold_selection': fold_selection,
            'args': vars(args),
        },
        best_model_path,
    )

    summary = save_run_artifacts(
        output_dir=paths['output_dir'],
        args=args,
        paths=paths,
        strategy_metadata=strategy_metadata,
        fold_selection=fold_selection,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    print(f"\nSaved checkpoint to: {best_model_path}")
    print(f"Saved run summary to: {paths['output_dir'] / 'training_summary.json'}")

    if args.save_gradcam_samples > 0:
        explanation_dir = paths['output_dir'] / 'gradcam' / args.model_architecture
        export_model_explanations(
            model=model,
            dataset=datasets['test'],
            output_dir=explanation_dir,
            architecture=args.model_architecture,
            max_samples=args.save_gradcam_samples,
            device=device,
        )
        print(f"Saved visual explanations to: {explanation_dir}")

    return summary


if __name__ == '__main__':
    main()
