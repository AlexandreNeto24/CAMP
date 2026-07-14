"""Run artifact persistence for classification experiments."""

import json
from pathlib import Path


def save_run_artifacts(output_dir: Path, args, paths, strategy_metadata, fold_selection, history, val_metrics, test_metrics):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'args': vars(args),
        'paths': {
            'dataset_root': str(paths['dataset_root']),
            'augmented_root': str(paths['augmented_root']),
            'strategy_root': str(paths['strategy_root']),
            'strategy_metadata_path': str(paths['strategy_metadata_path']),
            'synthetic_tar_path': str(paths['synthetic_tar_path']),
            'kfold_splits_path': str(paths['kfold_splits_path']),
        },
        'strategy_metadata': strategy_metadata,
        'fold_selection': {
            'strategy_id': fold_selection['strategy_id'],
            'train_folds': fold_selection['train_folds'],
            'val_fold': fold_selection['val_fold'],
            'test_fold': fold_selection['test_fold'],
            'train_counts': {
                'instance_images': len(fold_selection['train']['instance_images']),
                'normal_images': len(fold_selection['train']['normal_images']),
            },
            'val_counts': {
                'instance_images': len(fold_selection['val']['instance_images']),
                'normal_images': len(fold_selection['val']['normal_images']),
            },
            'test_counts': {
                'instance_images': len(fold_selection['test']['instance_images']),
                'normal_images': len(fold_selection['test']['normal_images']),
            },
        },
        'history': history,
        'final_validation_metrics': val_metrics,
        'final_test_metrics': test_metrics,
    }

    with open(output_dir / 'training_summary.json', 'w') as handle:
        json.dump(summary, handle, indent=2)

    return summary
