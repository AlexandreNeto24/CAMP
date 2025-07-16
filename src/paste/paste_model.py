import os
import sys
import gc
import numpy as np
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from pathlib import Path
from torchvision import transforms
from dataclasses import asdict
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Configuração do ambiente
os.environ["ULTRALYTICS_LOGGING_LEVEL"] = "WARNING"

# Ajuste para o path do ultralytics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ultralytics")))

# Importações específicas do projeto
from ultralytics import YOLO
from torch.utils.data import DataLoader, random_split

from src.paste.utils.train_cfg import TrainConfig
from src.paste.utils.compute_similarity import ( 
    compute_similarity_enhanced,
    extract_multi_scale_roi_features,
)
from torchvision.ops import box_iou
from src.paste.dataset import custom_collate
from src.paste.utils.context_head import ContextEmbeddingHead
from src.paste.utils.attention import CoordinateAttention
from src.paste.utils.feature_extractors import (
    TextureExtractor, 
    ColorHistogramExtractor, 
    ShapeFeatureExtractor,
    MultiModalFusion
)

class Dict2Obj:
    def __init__(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

class PasteModel:
    def __init__(self, yaml_path, model_name, batch_size=None, epochs=None, learning_rate=None):
        train_cfg = TrainConfig.from_yaml(yaml_path, model_name)
        # Sobrescreve só se o argumento for passado
        if batch_size is not None:
            train_cfg.training.batch_size = batch_size
        if epochs is not None:
            train_cfg.training.epochs = epochs
        if learning_rate is not None:
            train_cfg.training.learning_rate = learning_rate
        self.train_cfg = train_cfg
        self.model_name = self.train_cfg.model_name
        self.device = torch.device(self.train_cfg.device if torch.cuda.is_available() else "cpu")
        self.hyp_path = os.path.join(os.getcwd(), self.train_cfg.hyp_path)

        # Desativar completamente mixed precision
        self.scaler = None
        
        # Peso inicial e configuração adaptativa
        self.sim_loss_weight = 5.0  # Reduzido de 20.0 para 5.0
        self.sim_loss_weight_max = 20.0  # Reduzido de 100.0 para 20.0
        self.sim_loss_weight_step = 1.2  # Menos agressivo, era 1.5
            
    def build_model(self):
        self.model = YOLO(self.train_cfg.model_weights).to(self.device)
        
        # Garante que todo o modelo está no dispositivo correto
        self.model.model = self.model.model.float().to(self.device)
        
        # Extratores de características dedicados
        self.texture_extractor = TextureExtractor(num_filters=8).to(self.device)
        self.color_extractor = ColorHistogramExtractor(bins=8).to(self.device)
        self.shape_extractor = ShapeFeatureExtractor().to(self.device)
    
        # Descobre o número de canais das features multi-escala
        dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
        with torch.no_grad():
            features = self.model.model.model[:6](dummy_input)
            if isinstance(features, (list, tuple)):
                in_dim = sum([f.shape[1] for f in features[-2:]])
                print("Expected output channels should be:", in_dim)
            else:
                in_dim = features.shape[1]
                print("Expected output channels should be:", in_dim)
            
        # Módulo de Context Embedding com Coordinate Attention
        self.context_head = ContextEmbeddingHead(in_dim, out_dim=128).to(self.device)
    
        # Módulo de fusão multimodal
        self.fusion_module = MultiModalFusion(
            embedding_dim=128,
            texture_dim=8,
            color_dim=32,
            shape_dim=16
        ).to(self.device)

        # Access the detection head
        detect_layer = self.model.model.model[-1]
        detect_layer.nc = 1
        detect_layer.no = detect_layer.reg_max * 4 + detect_layer.nc
        print("Expected output channels should be:", detect_layer.no)

        # CORREÇÃO CRÍTICA: Garante que os novos módulos vão para o dispositivo correto
        if hasattr(detect_layer, 'cv3'):
            for i in range(len(detect_layer.cv3)):
                old_cv3 = detect_layer.cv3[i]
                
                if isinstance(old_cv3, nn.Sequential):
                    conv_modules = [m for m in old_cv3.modules() if isinstance(m, nn.Conv2d)]
                    if conv_modules:
                        conv_module = conv_modules[0]
                        out_channels = conv_module.out_channels // 80
                        
                        # Aqui está a correção: .to(self.device)
                        detect_layer.cv3[i] = nn.Sequential(
                            nn.Conv2d(conv_module.in_channels, out_channels, 
                                    kernel_size=conv_module.kernel_size,
                                    stride=conv_module.stride, 
                                    padding=conv_module.padding,
                                    bias=conv_module.bias is not None)
                        ).to(self.device)  # Move para o dispositivo correto
                else:
                    if hasattr(old_cv3, 'weight'):
                        out_channels = old_cv3.weight.shape[0] // 80
                        # Aqui está a correção: .to(self.device)
                        new_cv3 = nn.Conv2d(
                            old_cv3.in_channels,
                            out_channels,
                            kernel_size=old_cv3.kernel_size,
                            stride=old_cv3.stride,
                            padding=old_cv3.padding,
                            bias=old_cv3.bias is not None
                        ).to(self.device)  # Move para o dispositivo correto
                        detect_layer.cv3[i] = new_cv3

        # Garante que tudo esteja no dispositivo correto após as modificações
        self.model.model = self.model.model.to(self.device)

        if hasattr(detect_layer, 'use_dfl'):
            detect_layer.use_dfl = True

        self.model.model.names = {0: 'Similar Region'}

        # Load hyperparameters
        with open(self.hyp_path, "r") as f:
            self.hyp = yaml.safe_load(f)

        self.model.model.criterion = self.model.model.init_criterion()  # Ensure criterion is initialized
        self.model.model.criterion.hyp = Dict2Obj(self.hyp)  # Update hyperparameters
        
        if not os.path.exists(self.train_cfg.logging.checkpoint_path):
            os.makedirs(os.path.join(os.getcwd(), self.train_cfg.logging.checkpoint_path))
        
        print(f'Save path: {os.path.join(os.getcwd(), self.train_cfg.logging.checkpoint_path)}')
        self.blocks = list(self.model.model.model.children())
        
        # Adicione no final do método build_model() - após init_criterion
        if hasattr(self.model.model.criterion, 'proj'):
            self.model.model.criterion.proj = self.model.model.criterion.proj.to(self.device)
        
        return self.model
    
    def save_model(self, model_filename="best_model.pt", hparams_filename="hyp.yaml"):
        hparams_path = os.path.join(self.weights_path, hparams_filename)
        model_path = os.path.join(self.weights_path, model_filename)

        # Salvar hiperparâmetros
        with open(hparams_path, "w") as f:
            yaml.dump(asdict(self.train_cfg), f)
        print(f"Hyperparameters saved in {hparams_path}")

        # Salvar modelo Ultralytics diretamente no ficheiro correto
        self.model.save(model_path)
        print(f"Weights saved in {model_path}")
        return
    
    def get_optimizer(self):
        opt_name = self.hyp.get('optimizer', 'SGD').lower()
        if opt_name == "auto":
            opt_name = "adam"  # ou "sgd" se preferires
        lr = self.hyp.get('lr0', self.train_cfg.training.learning_rate)
        wd = self.hyp.get('weight_decay', 0.0005)
        momentum = self.hyp.get('momentum', 0.937)
        model = self.model.model  # O modelo PyTorch real

        # Separar grupos de parâmetros
        g_decay, g_no_decay = [], []
        bn = tuple(torch.nn.modules.batchnorm._BatchNorm for _ in range(1))

        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                # Bias e batchnorm SEM decay
                if name == "bias" or isinstance(module, bn):
                    g_no_decay.append(param)
                else:
                    g_decay.append(param)

        param_groups = [
            {'params': g_decay, 'weight_decay': wd},
            {'params': g_no_decay, 'weight_decay': 0.0}
        ]

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(param_groups, lr=lr, betas=(0.9, 0.999))
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(param_groups, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported.")

        print(f"Optimizer groups: decay={len(g_decay)}, no_decay={len(g_no_decay)}")
        return optimizer
        
    def get_scheduler(self, optimizer):
        if self.hyp.get('cos_lr', False):
            return CosineAnnealingLR(optimizer, T_max=self.train_cfg.training.epochs)
        else:
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.train_cfg.training.reduce_lr_factor,
                patience=self.train_cfg.training.reduce_lr_patience,
                verbose=True
            )
            
    def batch_train_loop(self, dataloader, back_prop=True):
        # Ativar detecção de anomalias no início da função
        torch.autograd.set_detect_anomaly(True)

        # ADICIONE ESTA VERIFICAÇÃO PARA MOVER self.proj PARA GPU
        if hasattr(self.model.model.criterion, 'proj'):
            if self.model.model.criterion.proj.device != self.device:
                self.model.model.criterion.proj = self.model.model.criterion.proj.to(self.device)
    
        # Verificações de debug (opcional - pode remover depois)
        model_device = next(self.model.model.parameters()).device
        print(f"Modelo está no dispositivo: {model_device}")
    
        det_loss_total = 0.0
        sim_loss_total = 0.0
        box_loss_total = 0.0
        cls_loss_total = 0.0
        dfl_loss_total = 0.0
        num_batches = 0
        
        # Limpa memória no início
        gc.collect()
        torch.cuda.empty_cache()
        
        # Desabilitar mixed precision completamente
        scaler = None

        for i, batch in enumerate(dataloader):
            # Limpa memória a cada 2 batches
            if i % 2 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if batch is None or batch[0] is None or (hasattr(batch[0], 'shape') and batch[0].shape[0] == 0):
                print("Batch vazio, pulando...")
                continue

            # Novo formato de retorno do custom_collate
            global_imgs, global_mask, all_boxes, all_labels, all_img_indices = batch

            # Força todos os tensores para float32
            global_imgs = global_imgs.to(self.device, dtype=torch.float32)
            all_boxes = all_boxes.to(self.device, dtype=torch.float32)
            all_labels = all_labels.to(self.device)
            all_img_indices = all_img_indices.to(self.device)

            # Formato de labels esperado por YOLO: [batch_idx, class_idx, x, y, w, h]
            # Pega apenas boxes com label=1 (lesões) para treinamento YOLO
            lesion_mask = all_labels == 1
            yolo_boxes = all_boxes[lesion_mask]
            img_indices = all_img_indices[lesion_mask]
            
            labels = torch.cat([img_indices.unsqueeze(1).float(), yolo_boxes], dim=1)

            batch_dict = {
                "img": global_imgs,
                "batch_idx": img_indices.unsqueeze(1).float(),
                "cls": labels[:, 1].long(),
                "bboxes": labels[:, 2:],
            }

            # Forward pass
            pred = self.model.model(global_imgs)
            
            # Calcular perda de detecção
            det_loss, loss_items = self.model.model.criterion(pred, batch_dict)
            
            # Calcular similarity loss com todas as boxes (lesão e background)
            sim_loss_batch = compute_similarity_enhanced(
                raw_preds=pred,
                global_imgs=global_imgs,
                all_boxes=all_boxes,
                all_labels=all_labels,
                all_img_indices=all_img_indices,
                model=self.model,
                margin=1.0,
                context_head=self.context_head,
                texture_extractor=self.texture_extractor,
                color_extractor=self.color_extractor,
                shape_extractor=self.shape_extractor,
                fusion_module=self.fusion_module
            )
            
            # Combine losses
            global_loss = det_loss + self.sim_loss_weight * sim_loss_batch

            if back_prop:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Backward pass
                global_loss.backward()
                
                # Clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=10.0)
                
                # Optimizer step
                self.optimizer.step()

            det_loss_total += det_loss.item()
            sim_loss_total += sim_loss_batch.item()
            box_loss_total += loss_items[0].item()
            cls_loss_total += loss_items[1].item()
            dfl_loss_total += loss_items[2].item()
            num_batches += 1

        # Normalize by number of batches
        det_loss_total /= max(num_batches, 1)
        sim_loss_total /= max(num_batches, 1)
        box_loss_total /= max(num_batches, 1)
        cls_loss_total /= max(num_batches, 1)
        dfl_loss_total /= max(num_batches, 1)

        # Apply hyperparameter weights
        box_weight = self.hyp.get('box', 1.0)
        cls_weight = self.hyp.get('cls', 1.0)
        dfl_weight = self.hyp.get('dfl', 1.0)
        box_loss_total *= box_weight
        cls_loss_total *= cls_weight
        dfl_loss_total *= dfl_weight
        
        return det_loss_total, sim_loss_total, box_loss_total, cls_loss_total, dfl_loss_total

    def train(self, dataset, val_split=0.2):
        MAX_UNFREEZE_BLOCKS = 3
        self.history = {
            "train_det_loss": [],
            "train_sim_loss": [],
            "train_box_loss": [],
            "train_cls_loss": [],
            "train_dfl_loss": [],
            "val_det_loss": [],
            "val_sim_loss": [],
            "val_box_loss": [],
            "val_cls_loss": [],
            "val_dfl_loss": [],
            # Novas métricas
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_map50": []
        }
        
        epochs_no_improvement = 0
        unfreeze_step = 0
        val_loss_best = float('inf')
        
        base_checkpoint_path = self.train_cfg.logging.checkpoint_path
    
        # Cria diretoria com timestamp
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.weights_path = os.path.join(base_checkpoint_path, now)
        os.makedirs(self.weights_path, exist_ok=True)
        
        # split the dataset
        n = len(dataset)
        n_train = int(n * (1 - val_split))
        n_val = n - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.train_cfg.training.batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=self.train_cfg.training.batch_size, shuffle=False, collate_fn=custom_collate)
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # create optimizer
        self.optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(self.optimizer)
        for epoch in range(self.train_cfg.training.epochs):
            # Atualiza a época atual no dataset de treino (caso seja Subset)
            train_data = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
            train_data.current_epoch = epoch
            train_data.total_epochs = self.train_cfg.training.epochs

            print(f"Epoch {epoch+1}/{self.train_cfg.training.epochs}")
            # if epoch % 5 == 0:
            #     self.plot_feature_embeddings(val_loader, epoch)
            if epochs_no_improvement > self.train_cfg.training.unfreeze_patience:
                print(f"Chegou a {epochs_no_improvement} épocas sem melhoria. Tentando descongelar mais blocos...")
                unfreeze_step += 1
                
                # Lógica para descongelar blocos
                found_block = False
                for i in range(22, 0, -1):  # Começa do bloco 22 e vai descendo
                    if i - unfreeze_step >= 0 and hasattr(self.blocks[i-unfreeze_step], "parameters"):
                        for param in self.blocks[i-unfreeze_step].parameters():
                            param.requires_grad = True
                        print(f"Descongelado bloco {i-unfreeze_step}")
                        trainable_params = sum(p.numel() for p in self.blocks[i-unfreeze_step].parameters() if p.requires_grad)
                        if trainable_params > 0:
                            print(f"Parâmetros treináveis após descongelar: {trainable_params}")
                            found_block = True
                            break
                
                # Redefinir contador de épocas sem melhoria para dar uma chance às camadas descongeladas
                if found_block:
                    epochs_no_improvement = 0
                    # Ajuste da taxa de aprendizado para as camadas recém-descongeladas
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1  # Reduz LR para camadas descongeladas
        
                # Em vez de early stopping imediato, apenas avisa e continua o treinamento
                # Assim o early_stop_patience será respeitado mesmo se não houver mais camadas para descongelar
                else:
                    print("Nenhum bloco com parâmetros encontrado para descongelar. Continuando o treino...")

            # Train step
            train_det_loss, train_sim_loss, train_box_loss, train_cls_loss, train_dfl_loss = self.batch_train_loop(train_loader)
            print(f"Train Loss: {train_det_loss:.4f}, Sim Loss: {train_sim_loss:.4f}, Box Loss: {train_box_loss:.4f}, Cls Loss: {train_cls_loss:.4f}, DFL Loss: {train_dfl_loss:.4f}")    
                
            # Aumento adaptativo do peso da similarity loss
            if train_sim_loss < 0.1:
                self.sim_loss_weight = min(self.sim_loss_weight * self.sim_loss_weight_step, self.sim_loss_weight_max)
                print(f"Aumentando peso da similarity loss para {self.sim_loss_weight:.2f}")

            # Validation step
            self.model.model.eval()
            with torch.no_grad():
                val_det_loss, val_sim_loss, val_box_loss, val_cls_loss, val_dfl_loss = self.batch_train_loop(val_loader, back_prop=False)
            print(f"Val Loss: {val_det_loss:.4f}, Sim Loss: {val_sim_loss:.4f}, Box Loss: {val_box_loss:.4f}, Cls Loss: {val_cls_loss:.4f}, DFL Loss: {val_dfl_loss:.4f}")
            if val_det_loss < val_loss_best:
                val_loss_best = val_det_loss
                epochs_no_improvement = 0
                print(f"Nova melhor loss de validação: {val_loss_best:.4f}")
                self.save_model(model_filename="best_model.pt")
            else:
                epochs_no_improvement += 1
                print(f"Sem melhora há {epochs_no_improvement} épocas")

            if epochs_no_improvement >= self.train_cfg.training.earlystop_patience:
                print(f"Early stopping após {epochs_no_improvement} épocas sem melhora!")
                break

            # Unfreeze step
            if epochs_no_improvement > self.train_cfg.training.unfreeze_patience:
                if unfreeze_step < MAX_UNFREEZE_BLOCKS and unfreeze_step < len(self.blocks) - 1:
                    # Descongela mais um bloco (de trás para frente)
                    block_idx = len(self.blocks) - 2 - unfreeze_step  # -2 porque último (-1) já está treinável
                    if block_idx >= 0:
                        print(f"Descongelando bloco {block_idx}")
                        for param in self.blocks[block_idx].parameters():
                            param.requires_grad = True
                        unfreeze_step += 1
                        epochs_no_improvement = 0 # Reset counter
                        # Recria otimizador com novos parâmetros treináveis
                        self.optimizer = self.get_optimizer()
                        scheduler = self.get_scheduler(self.optimizer)
                    else:
                        print("Todos os blocos já estão descongelados")

            # Descomentar e ativar as métricas de avaliação
            precision, recall, f1, map50 = self.evaluate_detection_metrics(val_loader)
            print(f"Val Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, mAP@0.5: {map50:.3f}")
            self.history["val_precision"].append(precision)
            self.history["val_recall"].append(recall)
            self.history["val_f1"].append(f1)
            self.history["val_map50"].append(map50)
            
            self.history["train_det_loss"].append(float(train_det_loss))
            self.history["train_sim_loss"].append(float(train_sim_loss))
            self.history["train_box_loss"].append(float(train_box_loss))
            self.history["train_cls_loss"].append(float(train_cls_loss))
            self.history["train_dfl_loss"].append(float(train_dfl_loss))
            self.history["val_det_loss"].append(float(val_det_loss))
            self.history["val_sim_loss"].append(float(val_sim_loss))
            self.history["val_box_loss"].append(float(val_box_loss))
            self.history["val_cls_loss"].append(float(val_cls_loss))
            self.history["val_dfl_loss"].append(float(val_dfl_loss))
            
            # Visualizações periódicas
            viz_frequency = 5  # Gerar visualizações a cada 5 épocas
            if (epoch + 1) % viz_frequency == 0:
                print(f"\nGerando visualizações da época {epoch+1}...")
                
                # Cria subdiretórios para as épocas
                epoch_viz_dir = os.path.join(self.weights_path, f"epoch_{epoch+1}")
                os.makedirs(epoch_viz_dir, exist_ok=True)
                
                # Plot de métricas até a época atual
                metrics_path = os.path.join(epoch_viz_dir, f"metrics_epoch_{epoch+1}.png")
                self.plot_metrics(save_path=metrics_path)
                
                # Plot de exemplos de predições
                val_dataset_viz = val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset
                preds_path = os.path.join(epoch_viz_dir, f"predictions_epoch_{epoch+1}.png")
                self.plot_predictions(val_dataset_viz, num_images=6, conf_thresh=0.25, save_path=preds_path)
                
                # Plot t-SNE dos embeddings (mais pesado computacionalmente)
                try:
                    tsne_path = os.path.join(epoch_viz_dir, f"tsne_epoch_{epoch+1}.png")
                    self.plot_embeddings_tsne(val_loader, epoch=epoch+1, save_path=tsne_path)
                except Exception as e:
                    print(f"Erro ao gerar t-SNE: {e}")
                
                print(f"Visualizações da época {epoch+1} salvas em {epoch_viz_dir}")

        # Salvar histórico em arquivo YAML ao final do treinamento
        hist_file = os.path.join(self.weights_path, "history.yaml")
        with open(hist_file, "w") as f:
            yaml.dump(self.history, f)
        print(f"Training history saved in {hist_file}")
        
        # ADICIONAR AQUI: Plot de métricas e exemplos de predição
        print("Gerando visualizações e gráficos de métricas...")
        
        # Plota gráficos de evolução de métricas
        self.plot_metrics()
        
        # Plota exemplos de predições no conjunto de validação
        val_dataset = val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset
        self.plot_predictions(val_dataset, num_images=6, conf_thresh=0.25)
        
        # Opcionalmente: gerar visualização t-SNE dos embeddings finais
        try:
            self.plot_embeddings_tsne(val_loader, epoch=epoch)
        except Exception as e:
            print(f"Erro ao gerar t-SNE: {e}")
        
        print("Treinamento concluído! Visualizações salvas em:", self.weights_path)
        
        return self.history
    
    # Substituir a função completa
    def evaluate_detection_metrics(self, val_loader):
        """
        Calcula métricas de detecção (precision, recall, F1, mAP@0.5) para avaliar
        a performance do modelo em detectar regiões de metaplasia.
        """
        print("Avaliando métricas de detecção...")
        self.model.model.eval()
        all_preds = []
        all_targets = []
        
        # Silenciador temporário
        original_level = os.environ.get("ULTRALYTICS_LOGGING_LEVEL", "INFO")
        os.environ["ULTRALYTICS_LOGGING_LEVEL"] = "ERROR"
        
        try:
            with torch.no_grad():
                for batch in val_loader:
                    # Novo formato de batch: (images, masks, all_boxes, all_labels, all_img_indices)
                    global_imgs, global_masks, all_boxes, all_labels, all_img_indices = batch
                    
                    # Enviar para o dispositivo correto
                    global_imgs = global_imgs.to(self.device)
                    all_boxes = all_boxes.to(self.device)
                    all_labels = all_labels.to(self.device)
                    all_img_indices = all_img_indices.to(self.device)
                    
                    # Extrair boxes YOLO (apenas as com label=1 são lesões)
                    lesion_mask = all_labels == 1
                    yolo_boxes = all_boxes[lesion_mask]
                    img_indices = all_img_indices[lesion_mask]
                    
                    # Rodar o modelo para detecção
                    results = self.model(global_imgs, conf=0.05, verbose=False)
                    
                    # Processar cada imagem no batch
                    for i in range(len(results)):
                        result = results[i]
                        
                        # Boxes preditas (em formato x1y1x2y2)
                        if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy.numel() > 0:
                            det_boxes = result.boxes.xyxy.cpu().numpy()
                            det_scores = result.boxes.conf.cpu().numpy()
                            
                            # Adicionar o índice da imagem como primeira coluna
                            img_preds = np.column_stack([
                                np.full(det_boxes.shape[0], i),
                                det_boxes,
                                det_scores
                            ])
                            all_preds.append(img_preds)
                        
                        # GT boxes para esta imagem (converter de YOLO para x1y1x2y2)
                        img_mask = img_indices == i
                        if img_mask.sum() > 0:
                            gt_boxes_yolo = yolo_boxes[img_mask, 1:5].cpu().numpy()  # [cx, cy, w, h]
                            gt_boxes = np.zeros_like(gt_boxes_yolo)
                            
                            # Converter de [cx, cy, w, h] para [x1, y1, x2, y2]
                            gt_boxes[:, 0] = (gt_boxes_yolo[:, 0] - gt_boxes_yolo[:, 2] / 2) * 640  # x1
                            gt_boxes[:, 1] = (gt_boxes_yolo[:, 1] - gt_boxes_yolo[:, 3] / 2) * 640  # y1
                            gt_boxes[:, 2] = (gt_boxes_yolo[:, 0] + gt_boxes_yolo[:, 2] / 2) * 640  # x2
                            gt_boxes[:, 3] = (gt_boxes_yolo[:, 1] + gt_boxes_yolo[:, 3] / 2) * 640  # y2
                            
                            # Adicionar o índice da imagem como primeira coluna
                            img_targets = np.column_stack([
                                np.full(gt_boxes.shape[0], i),
                                gt_boxes
                            ])
                            all_targets.append(img_targets)
        
            # Verificar se temos predições e alvos
            if not all_preds or not all_targets:
                print("Nenhuma predição encontrada para avaliação.")
                return 0.0, 0.0, 0.0, 0.0
            
            # Concatenar tudo
            all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 6))
            all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 5))
            
            # Calcula métricas
            metrics = self.calculate_metrics(all_preds, all_targets, iou_threshold=0.5)
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            map50 = metrics['map50']
            
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP@0.5: {map50:.4f}")
            
            return precision, recall, f1, map50
        finally:
            # Restaura o nível original de logging
            os.environ["ULTRALYTICS_LOGGING_LEVEL"] = original_level

    def calculate_metrics(self, predictions, targets, iou_threshold=0.5):
        """
        Calcula precision, recall, F1 e mAP@0.5
        
        Args:
            predictions: numpy array [N, 6] com [img_id, x1, y1, x2, y2, conf]
            targets: numpy array [M, 5] com [img_id, x1, y1, x2, y2]
            iou_threshold: IoU threshold para considerar um match
        """
        # Agrupamento por img_id
        img_ids = np.unique(targets[:, 0])
        tp = 0
        fp = 0
        fn = 0
        aps = []
        
        for img_id in img_ids:
            img_preds = predictions[predictions[:, 0] == img_id]
            img_targets = targets[targets[:, 0] == img_id]
            
            if img_preds.shape[0] == 0:
                fn += img_targets.shape[0]
                continue
            
            # Ordena por confiança
            img_preds = img_preds[(-img_preds[:, 5]).argsort()]
            
            # Para cada predição, verifica se é TP ou FP
            detected_targets = []
            precisions = []
            recalls = []
            for i in range(img_preds.shape[0]):
                pred_box = img_preds[i, 1:5]
                
                best_iou = 0
                best_target_idx = -1
                
                # Encontra o melhor match
                for j in range(img_targets.shape[0]):
                    if j in detected_targets:
                        continue
                    
                    target_box = img_targets[j, 1:5]
                    iou = self.calculate_iou(pred_box, target_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = j
                
                if best_iou >= iou_threshold:
                    tp += 1
                    detected_targets.append(best_target_idx)
                else:
                    fp += 1
                    
                # Precision e recall para esta predição
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn + len(img_targets) - len(detected_targets)) if (tp + fn + len(img_targets) - len(detected_targets)) > 0 else 0
                precisions.append(precision)
                recalls.append(recall)
            
            # Average precision para esta imagem
            ap = 0
            for i in range(11):  # 11 pontos (0, 0.1, ..., 1.0)
                r = i / 10
                p_interp = 0
                
                for j in range(len(recalls)):
                    if recalls[j] >= r:
                        p_interp = max(p_interp, precisions[j])
                
                ap += p_interp / 11
            
            aps.append(ap)
            
            # Atualiza FN
            fn += img_targets.shape[0] - len(detected_targets)
        
        # Métricas finais
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        map50 = np.mean(aps) if aps else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map50': map50
        }

    def calculate_iou(self, box1, box2):
        """
        Calcula IoU entre duas boxes [x1, y1, x2, y2]
        """
        # Coordenadas da interseção
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Verifica se há interseção
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Áreas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def plot_metrics(self, save_path=None):
        history = self.history
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Métricas de Treino e Validação por Época")

        # Métricas de loss
        loss_metrics = [
            ("train_det_loss", "val_det_loss", "Detection Loss"),
            ("train_sim_loss", "val_sim_loss", "Similarity Loss"),
            ("train_box_loss", "val_box_loss", "Box Loss"),
            ("train_cls_loss", "val_cls_loss", "Class Loss"),
            ("train_dfl_loss", "val_dfl_loss", "DFL Loss"),
        ]

        for idx, (train_key, val_key, title) in enumerate(loss_metrics):
            if idx < 5:  # 5 primeiros slots para losses
                row, col = divmod(idx, 3)
                ax = axs[row, col]
                
                if train_key in history and history[train_key]:
                    ax.plot(history[train_key], label="Treino")
                if val_key in history and history[val_key]:
                    ax.plot(history[val_key], label="Validação")
                    
                ax.set_title(title)
                ax.set_xlabel("Época")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
    
        # Métricas de avaliação (última posição)
        if all(key in history and history[key] for key in ["val_precision", "val_recall", "val_f1", "val_map50"]):
            ax = axs[1, 2]
            ax.plot(history["val_precision"], label="Precision", marker='o')
            ax.plot(history["val_recall"], label="Recall", marker='s')
            ax.plot(history["val_f1"], label="F1", marker='^')
            ax.plot(history["val_map50"], label="mAP@0.5", marker='*')
            ax.set_title("Métricas de Avaliação")
            ax.set_xlabel("Época")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)
        else:
            fig.delaxes(axs[1, 2])  # Remove o slot se não tiver métricas

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path is None:
            save_path = os.path.join(self.weights_path, "metrics.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Figura de métricas guardada em {save_path}")

    def plot_embeddings_tsne(self, val_loader, epoch=0, sample_limit=1000, save_path=None):
        """
        Visualiza embeddings usando t-SNE para verificar separação entre classes
        """
        from sklearn.manifold import TSNE
        
        print("Gerando visualização t-SNE dos embeddings...")
        self.model.model.eval()
        embeddings_list = []
        labels_list = []
        
        # Coleta embeddings
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 5:  # Limita a 5 batches para economizar memória
                    break
                
                # Novo formato de batch (5 elementos em vez de 8)
                global_imgs, global_masks, all_boxes, all_labels, all_img_indices = batch
                
                # Extrair boxes YOLO (apenas as com label=1 são lesões)
                lesion_mask = all_labels == 1
                yolo_boxes = all_boxes[lesion_mask]
                img_indices = all_img_indices[lesion_mask]
                
                global_imgs = global_imgs.to(self.device)
                all_boxes = all_boxes.to(self.device)
                all_labels = all_labels.to(self.device)
                all_img_indices = all_img_indices.to(self.device)
                
                # Extrai embeddings utilizando o backbone
                backbone = self.model.model.model[:6]
                features = backbone(global_imgs)
                feature_maps = features if isinstance(features, (list, tuple)) else [features]
                
                # Converte coordenadas YOLO para formato [x1, y1, x2, y2]
                xyxy_boxes = torch.zeros_like(all_boxes[:, 1:])
                xyxy_boxes[:, 0] = (all_boxes[:, 1] - all_boxes[:, 3]/2) * 640  # x1
                xyxy_boxes[:, 1] = (all_boxes[:, 2] - all_boxes[:, 4]/2) * 640  # y1
                xyxy_boxes[:, 2] = (all_boxes[:, 1] + all_boxes[:, 3]/2) * 640  # x2
                xyxy_boxes[:, 3] = (all_boxes[:, 2] + all_boxes[:, 4]/2) * 640  # y2
                
                # Extrair features das boxes
                box_features = extract_multi_scale_roi_features(
                    feature_maps, xyxy_boxes, batch_indices=all_img_indices, 
                    output_size=(7, 7), N=2
                )
                
                # Passa pelos context embeddings (se disponível)
                if hasattr(self, 'context_head') and self.context_head is not None:
                    box_features = self.context_head(box_features)
                
                # Coleta embeddings e labels
                embeddings_list.append(box_features.cpu())
                labels_list.append(all_labels.cpu())
    
        # Concatena todos os embeddings e labels
        if not embeddings_list:
            print("Nenhum embedding encontrado para visualização t-SNE")
            return
            
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        # Limita número de amostras se for muito grande
        if all_embeddings.shape[0] > sample_limit:
            indices = torch.randperm(all_embeddings.shape[0])[:sample_limit]
            all_embeddings = all_embeddings[indices]
            all_labels = all_labels[indices]
        
        # Converte para numpy
        embeddings_np = all_embeddings.numpy()
        labels_np = all_labels.numpy()
        
        # Calcula t-SNE
        print(f"Calculando t-SNE para {embeddings_np.shape[0]} embeddings...")
        tsne = TSNE(
            n_components=2, 
            perplexity=min(30, embeddings_np.shape[0]-1), 
            random_state=42,
            init='pca',  # Novo padrão futuro
            learning_rate='auto'  # Novo padrão futuro
        )
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Plota resultados
        plt.figure(figsize=(10, 8))
        colors = {0: 'red', 1: 'green'}
        markers = {0: 'x', 1: 'o'}
        labels = {0: 'Negativo', 1: 'Positivo'}
        
        for label_value in np.unique(labels_np):
            mask = labels_np == label_value
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1], 
                c=colors[label_value], 
                marker=markers[label_value], 
                label=labels[label_value],
                alpha=0.7
            )
        
        plt.title(f"Visualização t-SNE dos Embeddings (Época {epoch})")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.legend()
        
        # Calcula silhouette score se possível
        if len(np.unique(labels_np)) > 1 and all(np.sum(labels_np == i) >= 2 for i in np.unique(labels_np)):
            from sklearn.metrics import silhouette_score
            try:
                score = silhouette_score(embeddings_np, labels_np)
                plt.suptitle(f"Silhouette Score: {score:.4f}", y=0.95)
                print(f"Silhouette Score: {score:.4f}")
            except:
                print("Não foi possível calcular Silhouette Score")
    
        # Salva figura
        tsne_dir = os.path.join(self.weights_path, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(tsne_dir, f"tsne_epoch_{epoch}.png"))
        plt.close()
        print(f"Visualização t-SNE salva em tsne_epoch_{epoch}.png")
    
    def plot_predictions(self, dataset, num_images=6, conf_thresh=0.25, save_path=None):
        """
        Faz predições em num_images do dataset e guarda as imagens com bounding boxes e confianças.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        import numpy as np
        
        # Carrega os melhores pesos antes de prever
        best_model_path = os.path.join(self.weights_path, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model = YOLO(best_model_path).to(self.device)
            print("Loaded best model weights for prediction plots.")

        self.model.model.eval()
        
        # Silencia logs do YOLO durante predição
        os.environ["ULTRALYTICS_LOGGING_LEVEL"] = "ERROR"

        # Selecionamos mais índices do que precisamos para ter substitutos caso ocorram erros
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        
        # Cria grid para plotar as imagens
        n_cols = min(3, num_images)
        n_rows = (num_images + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        
        # Converte axs para array 2D se for 1D ou escalar
        if num_images == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)
        elif n_cols == 1:
            axs = axs.reshape(-1, 1)
        
        # Contador de imagens processadas com sucesso
        successful_images = 0
        processed_indices = []
        
        # Tenta processar imagens até termos o número desejado ou esgotarmos os índices
        for img_idx in all_indices:
            if successful_images >= num_images:
                break
                
            row, col = divmod(successful_images, n_cols)
            ax = axs[row, col]
            
            try:
                # Carrega amostra - formato atual (3 elementos)
                img, mask, boxes_data = dataset[img_idx]
                img_np = img.permute(1, 2, 0).cpu().numpy()
                
                # Normaliza para visualização
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                img_np = np.clip(img_np, 0, 1)
                
                # Extrai boxes ground truth
                gt_boxes = []
                for box in boxes_data:
                    if box[-1] == 1:  # Label de lesão
                        cls, x, y, w, h, label = box.tolist()
                        # Converte de YOLO [cx, cy, w, h] para [x1, y1, x2, y2]
                        x1 = (x - w/2) * 640
                        y1 = (y - h/2) * 640
                        x2 = (x + w/2) * 640
                        y2 = (y + h/2) * 640
                        gt_boxes.append([x1, y1, x2, y2])
                
                # Faz predição
                img_input = img.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    results = self.model(img_input, verbose=False)[0]
                    pred_boxes = []
                    pred_scores = []
                    
                    if hasattr(results.boxes, 'xyxy') and results.boxes.xyxy.numel() > 0:
                        pred_boxes = results.boxes.xyxy.cpu().numpy()
                        pred_scores = results.boxes.conf.cpu().numpy()

                # Plot imagem
                ax.imshow(img_np)
                ax.set_title(f"Imagem {img_idx}")
                ax.axis('off')

                # Plot ground truth boxes
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
                    )
                    ax.add_patch(rect)
                    
                # Plot predicted boxes
                for box, score in zip(pred_boxes, pred_scores):
                    if score < conf_thresh:
                        continue
                        
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1, y1 - 5, f"{score:.2f}",
                        color='white', fontsize=9, weight='bold',
                        bbox=dict(facecolor='lime', alpha=0.8)
                    )
                    
                # Imagem processada com sucesso
                successful_images += 1
                processed_indices.append(img_idx)
                
            except Exception as e:
                print(f"Erro ao processar imagem {img_idx}: {e}")
                # Continua com a próxima imagem sem incrementar successful_images
        
        # Preenche os slots vazios no grid
        for idx in range(successful_images, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axs[row, col].axis('off')

        plt.tight_layout()
        
        # Salva a figura
        if save_path is None:
            save_path = os.path.join(self.weights_path, "predictions.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Restaura o nível de logging
        os.environ["ULTRALYTICS_LOGGING_LEVEL"] = "WARNING"
        
        print(f"Visualização de predições salva em {save_path} com {successful_images} imagens")
        if successful_images < num_images:
            print(f"Aviso: Apenas {successful_images} de {num_images} imagens puderam ser processadas")

    def infer(self, img_path, conf_thresh=0.25, iou_thresh=0.45, visualize=True, save_path=None):
        """
        Realiza inferência em uma imagem, retornando detecções de metaplasia
        e opcionalmente visualizando os resultados.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        
        # Carrega a imagem
        if isinstance(img_path, str):
            img = Image.open(img_path).convert("RGB")
        else:
            img = img_path  # Assume que é uma imagem PIL
            
        # Redimensiona para 640x640 mantendo proporção
        img = img.resize((640, 640))
        
        # Converte para tensor
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        
        # Inferência
        self.model.model.eval()
        with torch.no_grad():
            # Detecção básica do YOLO
            results = self.model(img_tensor, conf=conf_thresh, iou=iou_thresh)[0]
            
            # Extrai features para análise multimodal
            backbone = self.model.model.model[:6]
            features = backbone(img_tensor)
            feature_maps = features if isinstance(features, (list, tuple)) else [features]
            
            # Extrai detecções
            if hasattr(results.boxes, 'xyxy') and results.boxes.xyxy.numel() > 0:
                boxes = results.boxes.xyxy
                scores = results.boxes.conf
                embeddings = None
                similarity_scores = {}
                patches = []
                
                # Extrai embeddings para cada detecção
                if boxes.shape[0] > 0:
                    batch_indices = torch.zeros(boxes.size(0), dtype=torch.int, device=self.device)
                    embeddings = extract_multi_scale_roi_features(
                        feature_maps, boxes, batch_indices=batch_indices, output_size=(7,7), N=2
                    )
                    
                    # Aplica context head
                    if hasattr(self, 'context_head'):
                        embeddings = self.context_head(embeddings)
                    
                    # Extrai patches para features específicas
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.int().clamp(min=0)
                        if x2 <= x1: x2 = x1 + 1
                        if y2 <= y1: y2 = y1 + 1
                        
                        try:
                            patch = img_tensor[:, :, y1:y2, x1:x2]
                            patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                            patches.append(patch)
                        except:
                            patch = torch.zeros(1, 3, 64, 64, device=self.device)
                            patches.append(patch)
            
                # Extrai características adicionais
                if patches:
                    patches_tensor = torch.cat(patches, dim=0)
                    
                    # Características de textura e cor
                    texture_features = self.texture_extractor(patches_tensor) if hasattr(self, 'texture_extractor') else None
                    color_features = self.color_extractor(patches_tensor) if hasattr(self, 'color_extractor') else None
                    shape_features = self.shape_extractor(boxes) if hasattr(self, 'shape_extractor') else None
                    
                    # Calcula similaridade de features
                    for i in range(boxes.shape[0]):
                        similarity_scores[i] = {
                            'texture': texture_features[i].mean().item() if texture_features is not None else 0,
                            'color': color_features[i].mean().item() if color_features is not None else 0,
                            'shape': shape_features[i].mean().item() if shape_features is not None else 0,
                            'embedding': embeddings[i].mean().item()
                        }
            
            # Visualize
            if visualize or save_path:
                plt.figure(figsize=(10, 10))
                plt.imshow(np.array(img))
                plt.axis('off')
                
                for i, (box, score) in enumerate(zip(boxes.cpu().numpy(), scores.cpu().numpy())):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, 
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    # Adiciona confiança e features
                    info_text = f"Conf: {score:.2f}"
                    if i in similarity_scores:
                        features_text = "\n".join([
                            f"{k}: {v:.2f}" for k, v in similarity_scores[i].items()
                        ])
                        info_text += f"\n{features_text}"
                    
                    plt.text(
                        x1, y1 - 10, info_text,
                        color='yellow', fontsize=8, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1)
                    )
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                    print(f"Resultado salvo em {save_path}")
                
                if visualize:
                    plt.show()
                else:
                    plt.close()
            
                # Retorna detecções
                return {
                    'boxes': boxes.cpu().numpy(),
                    'scores': scores.cpu().numpy(),
                    'embeddings': embeddings.cpu().numpy() if embeddings is not None else None,
                    'similarity_scores': similarity_scores
                }
            else:
                # Caso de nenhuma detecção encontrada
                print("Nenhuma detecção encontrada.")
                if visualize:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(np.array(img))
                    plt.axis('off')
                    plt.title("Nenhuma detecção encontrada")
                    plt.show()
                
                return {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'embeddings': None,
                    'similarity_scores': {}
                }