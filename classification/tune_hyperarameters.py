#!/usr/bin/env python
"""Hyperparameter tuning entrypoint using Optuna for the CIM project."""

import argparse
import copy
import json
import optuna
from pathlib import Path

import torch
import torch.nn as nn

from classification.config import (
    build_paths,
    project_root,
    resolve_device,
    set_seed,
)
from classification.data.data_loaders import build_classification_dataloaders, load_json
from classification.evaluation.metrics import evaluate_model
from classification.models.models import build_classifier, freeze_backbone, get_classifier_head, unfreeze_backbone


def build_tune_parser():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--augmentation-technique', type=str, default='our_approach')
    parser.add_argument('--multiplier', type=str, default='1x')
    parser.add_argument('--model-architecture', type=str, default='vit_b_16')
    parser.add_argument('--batch-size', type=int, default=16) # Mantido para fallback global se necessário
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-synthetic-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials to run')
    parser.add_argument('--finetune-epochs', type=int, default=15, help='Fixed number of fine-tuning epochs per trial')
    
    # Necessário para a função build_paths do config.py não estoirar
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save outputs')
    
    parser.set_defaults(pretrained=True)
    return parser


def tune_train_stage(model, train_loader, val_loader, criterion, optimizer, epochs, device, trial, global_epoch_offset=0):
    """
    Uma versão dedicada da função de treino apenas para o Optuna, 
    que inclui reportar métricas a meio e cortar treinos maus (Pruning).
    """
    best_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # O loader devolve um dicionário, extraímos as imagens e as labels
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)

        # Avaliação no validation set
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        history.append({'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        # ─── LÓGICA DE PRUNING DO OPTUNA ───
        current_step = global_epoch_offset + epoch
        trial.report(val_loss, current_step)

        if trial.should_prune():
            print(f"Trial pruned at global epoch {current_step} (val_loss: {val_loss:.4f})")
            raise optuna.exceptions.TrialPruned()
            
    return history, best_state, best_loss


def objective(trial, args, paths, device):
    """
    The Optuna objective function. It suggests hyperparameters, runs a training 
    cycle on Fold 1 (Strategy 1), and returns the best validation loss.
    """
    # 1. Sugestão Inteligente de Hiperparâmetros
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    warmup_lr = trial.suggest_float('warmup_lr', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 5)
    
    # Optuna escolhe o batch size. Adicionei o 64 porque temos proteção OOM!
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: lr={lr:.2e}, warmup_lr={warmup_lr:.2e}, wd={weight_decay:.2e}, warmup_epochs={warmup_epochs}, batch={batch_size}")

    # 2. Configurar os caminhos para a Strategy 1 (apenas um fold para afinar)
    strategy = "strategy1"
    strategy_root = paths['multiplier_root'] / strategy
    strategy_metadata_path = strategy_root / 'strategy_metadata.json'
    synthetic_tar_path = strategy_root / f'{args.dataset}_{args.multiplier}_{args.augmentation_technique}.tar'
    kfold_splits_path = paths['augmented_root'] / 'k_fold_splits.json'

    strategy_metadata = load_json(strategy_metadata_path)
    kfold_splits = load_json(kfold_splits_path)

    # 3. Construir os DataLoaders (usando o batch_size do Optuna)
    datasets, loaders = build_classification_dataloaders(
        dataset_root=paths['dataset_root'],
        strategy_metadata=strategy_metadata,
        kfold_splits=kfold_splits,
        synthetic_tar_path=synthetic_tar_path,
        image_size=args.image_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        max_synthetic_samples=args.max_synthetic_samples,
    )

    # 4. Inicializar o Modelo
    model = build_classifier(
        args.model_architecture,
        pretrained=args.pretrained,
        num_classes=2,
        image_size=args.image_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    # Bloco try...except para apanhar faltas de RAM na placa gráfica (OOM)
    try:
        # 5. Estágio 1: Warmup
        freeze_backbone(model)
        warmup_optimizer = torch.optim.AdamW(get_classifier_head(model).parameters(), lr=warmup_lr, weight_decay=weight_decay)
        
        _, warmup_best_state, _ = tune_train_stage(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            criterion=criterion,
            optimizer=warmup_optimizer,
            epochs=warmup_epochs,
            device=device,
            trial=trial,
            global_epoch_offset=0
        )

        if warmup_best_state is not None:
            model.load_state_dict(warmup_best_state)

        # 6. Estágio 2: Fine-Tuning
        unfreeze_backbone(model)
        finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        finetune_history, _, _ = tune_train_stage(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            criterion=criterion,
            optimizer=finetune_optimizer,
            epochs=args.finetune_epochs,
            device=device,
            trial=trial,
            global_epoch_offset=warmup_epochs
        )

        # 7. A métrica a minimizar
        best_val_loss = min([h['val_loss'] for h in finetune_history])
        return best_val_loss

    except RuntimeError as e:
        # Rede de segurança: Se a GPU não aguentar o Batch Size, limpa a memória e corta o Trial sem crashar o script
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number} failed due to CUDA Out Of Memory (Batch Size: {batch_size}). Safely pruning...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            # Qualquer outro erro estrutural (código mal escrito, etc) faz crashar para tu poderes corrigir
            raise e


def main():
    parser = build_tune_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.model_architecture.startswith('vit') and args.image_size != 224:
        print('ViT-B/16 uses 224x224 inputs by default. Overriding image_size to 224.')
        args.image_size = 224
        
    paths = build_paths(args)
    db_path = project_root / 'tuning_results'
    db_path.mkdir(parents=True, exist_ok=True)

    db_name = f"{args.model_architecture}_{args.dataset}_tuning.db"
    db_url = f"sqlite:///tuning_results/{db_name}"
    
    print(f"Starting Hyperparameter Tuning for {args.model_architecture} on {args.dataset}")
    print(f"Running {args.n_trials} trials...")

    # Base de dados local para guardar progresso
    study = optuna.create_study(
        direction="minimize", 
        study_name=f"{args.model_architecture}_tuning",
        storage=db_url,
        load_if_exists=True
    )
    
    func = lambda trial: objective(trial, args, paths, device)
    study.optimize(func, n_trials=args.n_trials)

    print("\n" + "="*50)
    print("Tuning Completed!")
    print("="*50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            print(f"  --{key.replace('_', '-')}: {value:.2e}")
        else:
            print(f"  --{key.replace('_', '-')}: {value}")

    # Guardar os hiperparâmetros campeões
    output_file = project_root / 'tuning_results' / f"best_params_{args.model_architecture}_{args.dataset}.json"
    with open(output_file, 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    print(f"\nSaved best hyperparameters to: {output_file}")


if __name__ == '__main__':
    main()