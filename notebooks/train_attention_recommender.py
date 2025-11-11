import os
import sys
import argparse
import datetime
import yaml
import json
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Configuração de caminhos
project_root = Path(__file__).parent.parent  # notebooks → project_root
sys.path.insert(0, str(project_root))


# Importa as classes dos módulos personalizados
from src.paste.attention_dataset import AttentionDataset, attention_collate
from src.paste.attention_recommender import AttentionRecommender
from src.paste.segmentation_model import PretrainedUNet, train_segmentor # <--- ALTERAR AQUI

# Configurações
data_path = os.path.abspath(os.path.join(project_root, 'data', 'airplane'))
save_weights_path = os.path.abspath(os.path.join(project_root, 'models', 'paste', 'weights'))
hyp_path = os.path.abspath(os.path.join(project_root, "cfg", "default.yaml"))
os.makedirs(save_weights_path, exist_ok=True)

# Carrega hiperparâmetros
with open(hyp_path, 'r') as f:
    hyp_dict = yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser(description='Treina modelo AttentionRecommender com backbone de segmentação')
    parser.add_argument('--run-dir', type=str, default=None, 
                      help='Diretório para salvar resultados (se não especificado, usa timestamp)')
    
    # Argumentos para o backbone de segmentação
    parser.add_argument('--train-segmentor', action='store_true',
                      help='Se deve treinar um novo segmentador U-Net antes do AttentionRecommender')
    parser.add_argument('--segmentor-path', type=str, required=False, 
                      help='Caminho para o segmentador U-Net pré-treinado (ignorado se --train-segmentor for usado)')
    parser.add_argument('--segmentor-epochs', type=int, default=200)
    parser.add_argument('--freeze-epochs', type=int, default=50)
    parser.add_argument('--full-finetune-lr', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--embedding-dim', type=int, default=1536)
    parser.add_argument('--num-heads', type=int, default=32) 
    parser.add_argument('--device', type=str, default='0', 
                      help='Dispositivo para treinamento (0, 1, cpu)')
    return parser.parse_args()

def train_attention_recommender(model, train_dataset, val_dataset, run_dir, 
                              epochs=30, batch_size=2, lr=0.0005, device='cuda', accumulation_steps=4):
    """f
    Treina o modelo AttentionRecommender com acumulação de gradientes.
    """
    print(f"Treinando AttentionRecommender por {epochs} épocas...")
    print(f"Batch size real: {batch_size}, Passos de acumulação: {accumulation_steps}, Batch size efetivo: {batch_size * accumulation_steps}")
    
    if isinstance(device, str) and device.isdigit(): device = f'cuda:{device}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("AVISO: CUDA solicitado mas não disponível. Usando CPU."); device = 'cpu'
    device = torch.device(device)
    model.to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=attention_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=attention_collate, num_workers=2, pin_memory=True)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch+1}/{epochs}")
        
        # --- TREINAMENTO ---
        model.train()
        train_loss = 0.0
        # optimizer.zero_grad() # Limpa os gradientes no início da acumulação da época

        for i, batch in enumerate(tqdm(train_loader, desc="Treinando")):
            images = batch['images'].to(device)
            patches = batch['patches'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            sample_indices = batch['sample_indices'].to(device)
            weight_masks = batch['weight_masks'].to(device)
            
            if patches.shape[0] == 0: continue
            
            batch_images = images[sample_indices]
            # pede features intermédias para perda contrastiva
            pred_out = model(batch_images, patch=patches, weight_mask=weight_masks, return_intermediate=True)
            # pred_out = (heatmap, patch_features, image_features) por contrato
            pred_heatmaps, patch_feats, img_feats = pred_out
            target_heatmaps = F.interpolate(heatmaps.unsqueeze(1), size=pred_heatmaps.shape[-2:], mode='bilinear', align_corners=False)
            loss = model.compute_loss(pred_heatmaps, target_heatmaps, patch_features=patch_feats, image_features=img_feats, contrastive_weight=0.25)
            
            # --- LÓGICA DE ACUMULAÇÃO ---
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            # --- FIM DA LÓGICA ---
            
            train_loss += loss.item() * accumulation_steps
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # === VALIDAÇÃO ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando"):
                images = batch['images'].to(device)
                patches = batch['patches'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                sample_indices = batch['sample_indices'].to(device)
                weight_masks = batch['weight_masks'].to(device)
                
                if patches.shape[0] == 0: continue
                
                batch_images = images[sample_indices]
                pred_out = model(batch_images, patch=patches, weight_mask=weight_masks, return_intermediate=True)
                pred_heatmaps, patch_feats, img_feats = pred_out
                target_heatmaps = F.interpolate(heatmaps.unsqueeze(1), size=pred_heatmaps.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = model.compute_loss(pred_heatmaps, target_heatmaps, patch_features=patch_feats, image_features=img_feats, contrastive_weight=0.25)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Época {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(os.path.join(run_dir, "best_model.pt"))
            print(f"Salvo melhor modelo com loss {val_loss:.4f}")

    plt.figure(figsize=(10, 5)); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss'); plt.legend(); plt.savefig(os.path.join(run_dir, 'training_loss.png')); plt.close()
    model.load_model(os.path.join(run_dir, "best_model.pt"))
    return model

def evaluate_model(model, val_dataset, num_samples=10, save_dir=None):
    """
    Avalia o modelo treinado em amostras do conjunto de validação.
    
    Args:
        model: Modelo AttentionRecommender treinado
        val_dataset: Dataset de validação
        num_samples: Número de amostras para avaliação
        save_dir: Diretório para salvar resultados
        
    Returns:
        Métricas de avaliação
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    device = model.device
    
    # Seleciona índices aleatórios
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    metrics = {
        'visual_quality': [],
        'recommendation_scores': []
    };
    
    for i, idx in enumerate(tqdm(indices, desc="Avaliando modelo")):
        # Carrega amostra do dataset
        sample = val_dataset[idx]
        image = sample['image'].to(device)
        patches = sample['patches']
        patch_boxes = sample['patch_boxes']
        
        # Pula se não houver patches
        if len(patches) == 0:
            continue
        
        # Avalia apenas o primeiro patch
        patch = patches[0].to(device)
        original_box = patch_boxes[0].tolist()
        
        # Gera recomendações
        recommendations = model.recommend_placements(
            image.unsqueeze(0), patch.unsqueeze(0), patch_box=original_box
        )
        
        # Avalia as recomendações em relação à posição original
        eval_metrics = model.evaluate_recommendations(
            image, patch, original_box, recommendations
        )
        
        # Coleta métricas
        if eval_metrics:
            for key, value in eval_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
            
            # Também coleta scores das recomendações
            if recommendations:
                metrics['recommendation_scores'].extend([rec['score'] for rec in recommendations])
                
                # Visualiza recomendações e mapa de calor
                if save_dir:
                    fig = model.visualize_recommendations(
                        image, patch, recommendations,
                        show_heatmap=True
                    )
                    plt.savefig(os.path.join(save_dir, f'eval_{i}.png'))
                    plt.close(fig)
                    
                    # Também visualiza simulação da colagem
                    if len(recommendations) > 0:
                        # Simula colagem na Posição recomendada
                        simulated = model.simulate_paste_result(
                            image, patch, recommendations[0]['box']
                        )
                        
                        plt.figure(figsize=(10, 10))
                        plt.imshow(simulated)
                        plt.title("Simulação de Colagem na Posição #1")
                        plt.axis('off')
                        plt.savefig(os.path.join(save_dir, f'sim_{i}.png'))
                        plt.close()
    
    # Calcula estatísticas
    results = {}
    for key, values in metrics.items():
        if values:
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Salva métricas
    if save_dir:
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    """Função principal para treinamento do AttentionRecommender"""
    
    # Get command-line arguments
    args = get_args()
    # args.train_segmentor = True

    # Directory setup
    if args.run_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join(save_weights_path, now + "_attention_unet")
    
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"Usando diretório de execução: {args.run_dir}")
    
    # save args to a text file
    with open(os.path.join(args.run_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Get train and validation datasets
    print("Loading datasets...")
    # TRAIN
    train_dataset = AttentionDataset(
        root_path=data_path,
        image_size=(512, 512),
        patch_size=(128, 128),
        hyp=hyp_dict,
        apply_aug=True,
        split_mode='ratio',
        split='train',
        split_ratio=0.2
    )
    # VALIDATION
    val_dataset = AttentionDataset(
        root_path=data_path,
        image_size=(512, 512),
        patch_size=(128, 128),
        hyp=None,
        apply_aug=False,
        split_mode='ratio',
        split='val',
        split_ratio=0.2
    )
    print(f"Dataset carregado: {len(train_dataset)} amostras de treino, {len(val_dataset)} amostras de validação")
    
    # DEBUG FLAG: Save some dataset examples
    debug_dir = os.path.join(args.run_dir, "debug_dataset_examples")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Guardando exemplos do dataset em: {debug_dir}")

    for i in range(min(5, len(train_dataset))):
        train_dataset.visualize_sample(i, save_path=os.path.join(debug_dir, f"train_sample_{i}.png"))
    for i in range(min(5, len(val_dataset))):
        val_dataset.visualize_sample(i, save_path=os.path.join(debug_dir, f"val_sample_{i}.png"))

    ############################################
    # PHASE 1: TRAIN OR LOAD SEGMENTOR BACKBONE
    ############################################
    segmentor_path = args.segmentor_path
    
    if args.train_segmentor:
        print("Phase 1: Training U-Net segmentor with efficientnet-b0 backbone...")

        segmentor_model = PretrainedUNet(
            encoder_name='efficientnet-b3', 
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            decoder_dropout=0.4
        )

        
        segmentor_path = train_segmentor(
            model=segmentor_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            run_dir=args.run_dir,
            epochs=args.segmentor_epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            device=args.device,
            accumulation_steps=4,
            freeze_epochs=args.freeze_epochs,
            full_finetune_lr=args.full_finetune_lr
        )
        print(f"Segmentador treinado salvo em: {segmentor_path}")
    elif not args.segmentor_path:
        print("ERRO: Deve fornecer --segmentor-path ou usar --train-segmentor")
        return
    
    ############################################
    # PHASE 2: INITIALIZE ATTENTION RECOMMENDER
    ############################################
    print(f"Phase 2: Initializing AttentionRecommender with segmentation backbone: {segmentor_path}")

    backbone = PretrainedUNet(encoder_name='efficientnet-b3', encoder_weights=None) # weights=None porque vamos carregar os nossos
    backbone.load_state_dict(torch.load(segmentor_path))
    
    model = AttentionRecommender(
        backbone_model=backbone,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dropout_rate=0.4
    )
    
    # Train the AttentionRecommender
    print("Starting training of AttentionRecommender...")
    model = train_attention_recommender(
        model,
        train_dataset,
        val_dataset,
        args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        accumulation_steps=2
    )
    
    # TODO: create evaluation function on separate file
    print("Evaluating model...")
    eval_dir = os.path.join(args.run_dir, 'evaluation')
    metrics = evaluate_model(
        model,
        val_dataset,
        num_samples=20,
        save_dir=eval_dir
    )
    
    print("Métricas de avaliação:")
    print(json.dumps(metrics, indent=2))
    
    # --- INÍCIO DA ALTERAÇÃO ---
    # Avalia com patches exógenos
    print("\nAvaliando com patches exógenos...")
    exogenous_eval_dir = os.path.join(args.run_dir, 'exogenous_evaluation')
    exogenous_metrics = model.evaluate_exogenous_patches(
        val_dataset,
        num_samples=15, # Número de pares (fonte, destino) a avaliar
        save_dir=exogenous_eval_dir
    )
    
    print("Métricas de avaliação exógena:")
    print(json.dumps(exogenous_metrics, indent=2))
    # --- FIM DA ALTERAÇÃO ---
    
    # --- Avaliação em imagens normais ---
    print("\nAvaliando em imagens normais (sem instâncias)...")
    # No final do main ou do treino:
    normal_images_dir = os.path.join(data_path, "normal")
    normal_patch_eval_dir = os.path.join(args.run_dir, "normal_patch_evaluation")
    model.evaluate_patch_on_normal(val_dataset, normal_images_dir, normal_patch_eval_dir, num_samples=10)
    
    # Salva um arquivo de finalização
    with open(os.path.join(args.run_dir, 'training_complete.txt'), 'w') as f:
        f.write(f"Treinamento concluído em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nTreinamento concluído! Todos os resultados foram salvos em:")
    print(f"{args.run_dir}")

if __name__ == "__main__":
    main()