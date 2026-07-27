#!/usr/bin/env python
"""Main classification entrypoint for the CIM project."""

import json
import pandas as pd

import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

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
    
    # all metrics dictionaries will be stored in this list for later averaging
    all_metrics = {
        'val': {}, 
        'test': {}
        }
    
    # save_path_root = paths['output_dir'] / args.model_architecture
    
    for n in range(5):
        print(f"\n\n=== Starting training for strategy {n+1} ===")
        strategy = f"strategy{n+1}"
        strategy_root = paths['multiplier_root'] / strategy
        strategy_metadata_path = strategy_root / 'strategy_metadata.json'
        synthetic_tar_path = strategy_root / f'{args.dataset}_{args.multiplier}_{args.augmentation_technique}.tar'
        kfold_splits_path = paths['augmented_root'] / 'k_fold_splits.json'

        # time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = paths['output_dir'] / strategy 
        # output_dir = save_path_root / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # load strategy metadata and k-fold splits
        strategy_metadata = load_json(strategy_metadata_path)
        kfold_splits = load_json(kfold_splits_path)

        print(f'Project root: {project_root}')
        print(f'Dataset root: {paths["dataset_root"]}')
        print(f'Augmented root: {paths["augmented_root"]}')
        print(f'Multiplier root: {paths["multiplier_root"]}')
        print(f'Synthetic tar: {synthetic_tar_path}')
        print(f'Device: {device}')
        
        # build datasets and dataloaders
        datasets, loaders = build_classification_dataloaders(
            dataset_root=paths['dataset_root'],
            strategy_metadata=strategy_metadata,
            kfold_splits=kfold_splits,
            synthetic_tar_path=synthetic_tar_path,
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
        
        # store the metrics for this strategy in the all_metrics dictionary into a list so that we can average them later
        for metric_key in val_metrics.keys():
            if metric_key not in all_metrics['val']:
                all_metrics['val'][metric_key] = []
            all_metrics['val'][metric_key].append(val_metrics[metric_key])
        for metric_key in test_metrics.keys():
            if metric_key not in all_metrics['test']:
                all_metrics['test'][metric_key] = []
            all_metrics['test'][metric_key].append(test_metrics[metric_key])
            
        # print(all_metrics)

        print('\nFinal metrics:')
        print(f"  val  | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
              f" f1={val_metrics['f1']:.4f} auc={val_metrics['roc_auc']:.4f}")
        print(f"  test | loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}"
              f" f1={test_metrics['f1']:.4f} auc={test_metrics['roc_auc']:.4f}")
       
        best_model_path = output_dir / 'best_model.pt'
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
            output_dir=output_dir,
            args=args,
            paths=paths,
            strategy_metadata=strategy_metadata,
            fold_selection=fold_selection,
            history=history,
            val_metrics=all_metrics['val'],
            test_metrics=all_metrics['test'],
        )

        print(f"\nSaved checkpoint to: {best_model_path}")
        print(f"Saved run summary to: {output_dir / 'training_summary.json'}")

        # if args.save_gradcam_samples > 0:
        #     explanation_dir = output_dir / 'gradcam' / args.model_architecture
            
        #     # Dividir metade para a classe 0 e metade para a classe 1
        #     half = args.save_gradcam_samples // 2
        #     quotas = {0: half, 1: args.save_gradcam_samples - half}
            
        #     export_model_explanations(
        #         model=model,
        #         dataset=datasets['test'],
        #         output_dir=explanation_dir,
        #         architecture=args.model_architecture,
        #         samples_per_class=quotas,
        #         device=device,
        #     )
        #     print(f"Saved visual explanations to: {explanation_dir}")
        
        if args.save_gradcam_samples > 0:
            explanation_dir = output_dir / 'gradcam' / args.model_architecture
            
            # Dinamicamente divide o total pedido pelo user pelas duas classes
            half = args.save_gradcam_samples // 2
            quotas = {0: half, 1: args.save_gradcam_samples - half}
            
            # 1. Capturamos as métricas devolvidas pela exportação
            cam_metrics = export_model_explanations(
                model=model,
                dataset=datasets['test'],
                output_dir=explanation_dir,
                architecture=args.model_architecture,
                samples_per_class=quotas,
                device=device,
            )
            print(f"Saved visual explanations to: {explanation_dir}")
            print('\nGrad-CAM metrics:')
            for key, value in cam_metrics.items():
                print(f"  {key} = {value:.4f}")
            
            # 2. Se a função devolveu o Dice, adicionamos ao test_metrics desta estratégia
            if cam_metrics and 'cam_dice' in cam_metrics:
                test_metrics['cam_dice'] = cam_metrics['cam_dice']
                print(f"  test | cam_dice_mean = {test_metrics['cam_dice']:.4f}")
                
            # se nao tiver cam_dice ainda no all_metrics['test'], inicializa com lista vazia
            if 'cam_dice' not in all_metrics['test']:
                all_metrics['test']['cam_dice'] = []
            
        # 3. Adicionar as métricas para esta estratégia ao all_metrics
        # (Como o test_metrics agora tem o 'cam_dice', ele vai ser incluído na média final automaticamente!)
        for metric_key in val_metrics.keys():
            all_metrics['val'][metric_key].append(val_metrics[metric_key])
        for metric_key in test_metrics.keys():
            all_metrics['test'][metric_key].append(test_metrics[metric_key])
                
        # save val and test results from this strategy to a JSON file
        results_path = output_dir / 'evaluation_results.json'
        results = {
            'val': val_metrics,
            'test': test_metrics
        }
        with open(results_path, 'w') as f:
            json.dump(results, f)
            
        # save a plot of the training history for loss of training and validation, highlighting the best epoch to understand if there is overfitting
        plt.figure(figsize=(10, 5))
        plt.plot([h['train_loss'] for h in history], label='Train Loss')
        plt.plot([h['val_loss'] for h in history], label='Val Loss')
        best_epoch = min(range(len(history)), key=lambda i: history[i]['val_loss'])
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_dir / 'loss_plot.png')   

            
            
    avg_metrics = {
        'val': {},
        'test': {},
    }
    
    # print(all_metrics)
    
    for phase in ['val', 'test']:
        for metric_key, values in all_metrics[phase].items():
            # ignore the confusion matrix for averaging, as it is a list of lists
            if metric_key == 'confusion_matrix':
                continue
            
            # 1. Cenário Ideal: É uma lista (como esperado dos 5 folds)
            if isinstance(values, list) and len(values) > 0:
                # Verifica se o primeiro elemento da lista é um número
                if isinstance(values[0], (int, float)):
                    avg_metrics[phase][metric_key] = sum(values) / len(values)
            
            # 2. Cenário de Segurança: Se for um número solto (float/int)
            elif isinstance(values, (int, float)):
                avg_metrics[phase][metric_key] = values
                
    # calculate the average metrics across all strategies
    avg_metrics = {'val': {}, 'test': {}}
    
    for phase in ['val', 'test']:
        for metric_key, values_list in all_metrics[phase].items():
            # Só faz a média (sum / len) se os elementos da lista forem números!
            if isinstance(values_list[0], (int, float)):
                avg_metrics[phase][metric_key] = sum(values_list) / len(values_list)
            else:
                # Se for a matriz de confusão, guarda a lista intacta
                avg_metrics[phase][metric_key] = values_list
                
    # add the standard deviations
    for phase in ['val', 'test']:
        for metric_key, values in all_metrics[phase].items():
            if metric_key == 'confusion_matrix':
                continue
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], (int, float)):
                avg_metrics[phase][f"{metric_key}_std"] = (sum((x - avg_metrics[phase][metric_key]) ** 2 for x in values) / len(values)) ** 0.5
    
    # do for dice, since it only exists in test_metrics if Grad-CAM was run
    if 'cam_dice' in all_metrics['test']:
        values = all_metrics['test']['cam_dice']
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], (int, float)):
            avg_metrics['test']['cam_dice'] = sum(values) / len(values)
            avg_metrics['test']['cam_dice_std'] = (sum((x - avg_metrics['test']['cam_dice']) ** 2 for x in values) / len(values)) ** 0.5
    
    # print the average metrics across all strategies
    print("\nAverage metrics across all strategies:")
    print(f"  val  | acc={avg_metrics['val']['accuracy']:.4f}"
          f" f1={avg_metrics['val']['f1']:.4f}"
          f" recall={avg_metrics['val']['recall']:.4f}"
          f" precision={avg_metrics['val']['precision']:.4f}"
          f" specificity={avg_metrics['val']['specificity']:.4f}"
          f" roc_auc={avg_metrics['val']['roc_auc']:.4f}")
    print(f"  test | acc={avg_metrics['test']['accuracy']:.4f}"
          f" f1={avg_metrics['test']['f1']:.4f}"
          f" recall={avg_metrics['test']['recall']:.4f}"
          f" precision={avg_metrics['test']['precision']:.4f}"
          f" specificity={avg_metrics['test']['specificity']:.4f}"
          f" roc_auc={avg_metrics['test']['roc_auc']:.4f}")
    
    
    
    # get root output directory for saving the average metrics
    output_dir = output_dir.parent  # go up one level to the strategy directory
    timestamp = paths['timestamp']  # get the timestamp from the paths dictionary
    # save the average metrics to a JSON file 
    with open(output_dir / f'{timestamp}_average_evaluation_results.json', 'w') as f:
        json.dump(avg_metrics, f)
        
    # save average and std in excel table
    with pd.ExcelWriter(output_dir / f'{timestamp}_average_evaluation_results.xlsx') as writer:
        pd.DataFrame([avg_metrics['val']]).to_excel(writer, sheet_name='val', index=False)
        pd.DataFrame([avg_metrics['test']]).to_excel(writer, sheet_name='test', index=False)
        
     
    return 

if __name__ == '__main__':
    main()
