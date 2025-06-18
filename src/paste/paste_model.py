import os
import sys
import random
import datetime
from dataclasses import asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import gc

# Add the path to the ultralytics directory, which is one level up and then into the 'ultralytics' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ultralytics")))

# Agora o import deve ser direto
from ultralytics import YOLO
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.paste.utils.train_cfg import TrainConfig
from src.paste.utils.compute_similarity import compute_similarity, extract_multi_scale_features
from src.paste.dataset import custom_collate

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
    
            
    def build_model(self):
        self.model = YOLO('yolov8s.pt').to(self.device)
        # Access the detection head from the underlying sequential container.
        detect_layer = self.model.model.model[-1]  # This should be the Detect module

        # Set number of classes to 1 and recompute expected output channels.
        detect_layer.nc = 1
        # For a DFL head with reg_max (typically 16), expected output channels is:
        detect_layer.no = detect_layer.reg_max * 4 + detect_layer.nc  # e.g. 16*4 + 1 = 65
        print("Expected output channels should be:", detect_layer.no)

        # IMPORTANT: Do not change the regression branch (cv2) if you want to use DFL.
        # Update only the classification branch (cv3) so that it outputs 1 channel instead of 80.
        if hasattr(detect_layer, 'cv3'):
            for i in range(len(detect_layer.cv3)):
                seq = detect_layer.cv3[i]
                layers = list(seq.children())
                in_channels = layers[-1].in_channels
                # Reinitialize the last layer to output detect_layer.nc (which is 1) channel.
                layers[-1] = nn.Conv2d(in_channels, detect_layer.nc, kernel_size=1, stride=1).to(self.device)
                detect_layer.cv3[i] = nn.Sequential(*layers).to(self.device)
        else:
            print("cv3 branch not found.")

        # (Optional) If there's a flag for DFL, ensure it is enabled.
        if hasattr(detect_layer, 'use_dfl'):
            detect_layer.use_dfl = True

        # Update model class names.
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
        return self.model
    
    def save_model(self, model_filename="best_model.pt", hparams_filename="hyp.yaml"):
        """
        Guarda os hiperparâmetros em YAML e os pesos do modelo em .pt numa subpasta com timestamp.
        """
        # TODO: actualizar info para o yam
    
        # Caminhos dos ficheiros
        hparams_path = os.path.join(self.weights_path, hparams_filename)
        model_path = os.path.join(self.weights_path, model_filename)
    
        # Salvar hiperparâmetros
        with open(hparams_path, "w") as f:
            yaml.dump(asdict(self.train_cfg), f)
        print(f"Hyperparameters saved in {hparams_path}")
    
        # Salvar pesos do modelo
        torch.save(self.model.state_dict(), model_path)
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
        det_loss_total = 0.0
        sim_loss_total = 0.0
        box_loss_total = 0.0
        cls_loss_total = 0.0
        dfl_loss_total = 0.0
        num_batches = 0

        scaler = GradScaler()
        for batch in dataloader:
            gc.collect()
            torch.cuda.empty_cache()
            if batch is None or batch[0] is None or (hasattr(batch[0], 'shape') and batch[0].shape[0] == 0):
                print("Batch inválido, ignorado.")
                continue
            global_imgs, global_mask, padded_cropped_imgs, cropped_coords, padded_zoomed_imgs, zoomed_coords, yolo_boxes, img_indices = batch

            global_imgs = global_imgs.to(self.device)
            padded_cropped_imgs = padded_cropped_imgs.to(self.device)
            padded_zoomed_imgs = padded_zoomed_imgs.to(self.device)
            yolo_boxes = yolo_boxes.to(self.device)
            img_indices = img_indices.to(self.device)

            # Cada instância tem o seu img_idx (imagem original no batch)
            labels = torch.cat([img_indices.unsqueeze(1).float(), yolo_boxes], dim=1)  # [N_inst, 6]

            batch_dict = {
                "img": global_imgs,                    # Tensor [B, 3, H, W]
                "batch_idx": img_indices.unsqueeze(1).float(),  # [N_inst, 1]
                "cls": labels[:, 1].long(),            # [N_inst]
                "bboxes": labels[:, 2:],               # [N_inst, 4]
            }

            with autocast():
                raw_preds = self.model.model(global_imgs)
                global_loss, loss_items = self.model.model.loss(batch_dict, raw_preds)
                global_loss = global_loss.sum()
                sim_loss_batch = compute_similarity(
                    raw_preds, global_imgs,
                    padded_cropped_imgs, cropped_coords,
                    padded_zoomed_imgs, zoomed_coords,
                    self.model,
                    img_indices
                )

                total_loss = global_loss + sim_loss_batch
            if back_prop:
                self.optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            det_loss_total += global_loss.item()
            sim_loss_total += sim_loss_batch.item()
            box_loss_total += loss_items[0].item()
            cls_loss_total += loss_items[1].item()
            dfl_loss_total += loss_items[2].item()
            num_batches += 1

        # Normalize by number of batches
        det_loss_total /= num_batches
        sim_loss_total /= num_batches
        box_loss_total /= num_batches
        cls_loss_total /= num_batches
        dfl_loss_total /= num_batches

        # ... apply weights if needed ...
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
        # create scheduler
        # self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=1e-4)
        self.optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(self.optimizer)
        for epoch in range(self.train_cfg.training.epochs):
            # Atualiza a época atual no dataset de treino (caso seja Subset)
            train_data = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
            train_data.current_epoch = epoch
            train_data.total_epochs = self.train_cfg.training.epochs

            print(f"Epoch {epoch+1}/{self.train_cfg.training.epochs}")
            if epoch % 5 == 0:
                self.plot_feature_embeddings(val_loader, epoch)
            if epochs_no_improvement > self.train_cfg.training.unfreeze_patience:
                idx = len(self.blocks) - 1 - unfreeze_step
                if idx >= len(self.blocks) - MAX_UNFREEZE_BLOCKS:
                    for param in self.blocks[idx].parameters():
                        param.requires_grad = True
                    print(f"Descongelado bloco {idx}")
                    self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=1e-4)
                    unfreeze_step += 1
                else:
                    print("Limite de blocos descongelados atingido. Early stopping.")
                    self.plot_metrics()
                    self.plot_predictions(val_dataset, num_images=6)
                    break  # Early stop se não há mais blocos para descongelar
                epochs_no_improvement = 0
                self.model.load_state_dict(torch.load(os.path.join(self.weights_path, "best_model.pt")))
                self.model.to(self.device)
                
            # Train step
            train_det_loss, train_sim_loss, train_box_loss, train_cls_loss, train_dfl_loss = self.batch_train_loop(train_loader)
            print(f"Train Loss: {train_det_loss:.4f}, Sim Loss: {train_sim_loss:.4f}, Box Loss: {train_box_loss:.4f}, Cls Loss: {train_cls_loss:.4f}, DFL Loss: {train_dfl_loss:.4f}")    
                
            # Validation step
            self.model.model.eval()
            with torch.no_grad():
                val_det_loss, val_sim_loss, val_box_loss, val_cls_loss, val_dfl_loss = self.batch_train_loop(val_loader, back_prop=False)    
            scheduler.step(val_det_loss)
            print(f"Validation Loss: {val_det_loss:.4f}, Sim Loss: {val_sim_loss:.4f}, Box Loss: {val_box_loss:.4f}, Cls Loss: {val_cls_loss:.4f}, DFL Loss: {val_dfl_loss:.4f}")

            if val_det_loss < val_loss_best:
                val_loss_best = val_det_loss
                epochs_no_improvement = 0
                print(f"Validation loss improved to {val_loss_best:.4f}. Saving model...")
                self.save_model()
            else:
                epochs_no_improvement += 1
                print(f"No improvement in validation loss. Epochs without improvement: {epochs_no_improvement}")
                
            if epochs_no_improvement >= self.train_cfg.training.earlystop_patience:
                print(f"Early stopping triggered after {epochs_no_improvement} epochs without improvement.")
                self.plot_metrics()
                self.plot_predictions(val_dataset, num_images=6)
                break
            
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
        
        self.plot_metrics()
        self.plot_predictions(val_dataset, num_images=6)
            
        return 
    
    def model_summary(self):
        print("===== Modelo Summary =====")
        print(f"Modelo: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Pesos guardados em: {getattr(self, 'weights_path', 'N/A')}")
        print("\n--- Estrutura do modelo ---")
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total de parâmetros: {total_params}")
        print(f"Parâmetros treináveis: {trainable_params}")
        print("\n--- Hiperparâmetros ---")
        print(f"Batch size: {self.train_cfg.training.batch_size}")
        print(f"Epochs: {self.train_cfg.training.epochs}")
        print(f"Learning rate: {self.train_cfg.training.learning_rate}")
        print(f"Earlystop patience: {self.train_cfg.training.earlystop_patience}")
        print(f"Reduce LR patience: {self.train_cfg.training.reduce_lr_patience}")
        print(f"Reduce LR factor: {self.train_cfg.training.reduce_lr_factor}")
        print(f"Imagem size: {getattr(self.train_cfg.training, 'img_size', 'N/A')}")
        print(f"Número de classes: {getattr(self.model.model.model[-1], 'nc', 'N/A')}")
        print("==========================")

    def plot_metrics(self, save_path=None):
        history = self.history
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Métricas de Treino e Validação por Época")
    
        metric_names = [
            ("train_det_loss", "val_det_loss", "Detection Loss"),
            ("train_sim_loss", "val_sim_loss", "Similarity Loss"),
            ("train_box_loss", "val_box_loss", "Box Loss"),
            ("train_cls_loss", "val_cls_loss", "Class Loss"),
            ("train_dfl_loss", "val_dfl_loss", "DFL Loss"),
        ]
    
        for idx, (train_key, val_key, title) in enumerate(metric_names):
            row, col = divmod(idx, 3)
            ax = axs[row, col]
            ax.plot(history[train_key], label="Treino")
            ax.plot(history[val_key], label="Validação")
            ax.set_title(title)
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
    
        if len(metric_names) < 6:
            fig.delaxes(axs[1, 2])
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path is None:
            save_path = os.path.join(self.weights_path, "metrics.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Figura de métricas guardada em {save_path}")
    
    def plot_predictions(self, dataset, num_images=3, conf_thresh=0.25, save_path=None):
        """
        Faz predições em num_images do dataset e guarda as imagens com bounding boxes e confianças.
        """
        # Carrega os melhores pesos antes de prever
        best_model_path = os.path.join(self.weights_path, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.to(self.device)
            print("Loaded best model weights for prediction plots.")

        self.model.model.eval()
        import random

        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
        fig, axs = plt.subplots(1, len(indices), figsize=(6 * len(indices), 6))
        if len(indices) == 1:
            axs = [axs]

        for ax, idx in zip(axs, indices):
            # Carrega sample
            img, _, _, _, _, _, _ = dataset[idx]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_input = img.unsqueeze(0).to(self.device)

            # Faz predição
            with torch.no_grad():
                results = self.model(img_input)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

            # Plot imagem
            ax.imshow(img_np)
            ax.axis('off')
            ax.set_title(f"Sample {idx}")

            # Plot bounding boxes e confianças
            for box, score in zip(boxes, scores):
                if score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{score:.2f}", color='yellow', fontsize=12, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.weights_path, "predictions.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Figura de predições guardada em {save_path}")
        
    def plot_feature_embeddings(self, dataloader, epoch, save_path=None):
        """
        Visualiza embeddings de objeto e background num batch usando t-SNE.
        Também guarda exemplos de feature maps do backbone.
        """
        self.model.model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            global_imgs = batch[0].to(self.device)
            padded_cropped_imgs = batch[2].to(self.device)
            padded_zoomed_imgs = batch[4].to(self.device)

            backbone = self.model.model.model[:6]
            cropped_feats, _ = extract_multi_scale_features(backbone, padded_cropped_imgs, N=3)
            zoomed_feats, _ = extract_multi_scale_features(backbone, padded_zoomed_imgs, N=3)
            cropped_feats = cropped_feats.cpu().numpy()
            zoomed_feats = zoomed_feats.cpu().numpy()

            # Labels: 1 para objeto, 0 para background
            labels = np.concatenate([np.ones(len(cropped_feats)), np.zeros(len(zoomed_feats))])
            all_embeddings = np.concatenate([cropped_feats, zoomed_feats], axis=0)

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings)

            plt.figure(figsize=(8,6))
            plt.scatter(embeddings_2d[labels==1,0], embeddings_2d[labels==1,1], label='Objeto', alpha=0.7)
            plt.scatter(embeddings_2d[labels==0,0], embeddings_2d[labels==0,1], label='Background', alpha=0.7)
            plt.legend()
            plt.title(f"t-SNE dos embeddings extraídos (Epoch {epoch})")
            if save_path is None:
                save_path = f"embeddings_epoch_{epoch}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Plot dos embeddings guardado em {save_path}")

            # --- NOVO: guardar exemplos de feature maps do backbone ---
            fmap_dir = os.path.join(os.path.dirname(save_path), f"feature_maps_epoch_{epoch}")
            self.plot_feature_maps(global_imgs, backbone, epoch, save_dir=fmap_dir, num_imgs=2, num_channels=8)
            print(f"Feature maps guardados em {fmap_dir}")
        
    def plot_feature_maps(self, imgs, backbone, epoch, save_dir="feature_maps", num_imgs=2, num_channels=8):
        """
        Guarda exemplos de feature maps 2D de imgs (tensor [B,3,H,W]) para uma camada do backbone.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            features = backbone(imgs[:num_imgs])  # lista de feature maps
            fmap = features[-2]  # ou outro índice que seja 2D
            print(f"Feature map selecionada: shape={fmap.shape}")
            if fmap.ndim == 4:  # [B, C, H, W]
                for img_idx in range(min(num_imgs, fmap.shape[0])):
                    for ch in range(min(num_channels, fmap.shape[1])):
                        feature = fmap[img_idx, ch].cpu().numpy()
                        if feature.ndim == 2:
                            plt.figure(figsize=(3,3))
                            plt.imshow(feature, cmap='viridis')
                            plt.axis('off')
                            plt.title(f"Epoch {epoch} | Img {img_idx} | Canal {ch}")
                            fname = os.path.join(save_dir, f"epoch{epoch}_img{img_idx}_ch{ch}.png")
                            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
                            plt.close()
                        else:
                            print(f"Canal {ch} não é 2D, shape={feature.shape}, não será guardado.")
            elif fmap.ndim == 3:  # [C, H, W]
                for ch in range(min(num_channels, fmap.shape[0])):
                    feature = fmap[ch].cpu().numpy()
                    if feature.ndim == 2:
                        plt.figure(figsize=(3,3))
                        plt.imshow(feature, cmap='viridis')
                        plt.axis('off')
                        plt.title(f"Epoch {epoch} | Canal {ch}")
                        fname = os.path.join(save_dir, f"epoch{epoch}_ch{ch}.png")
                        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
                        plt.close()
                    else:
                        print(f"Canal {ch} não é 2D, shape={feature.shape}, não será guardado.")
            else:
                print(f"Shape inesperado para feature map: {fmap.shape}")
        
    def evaluate_embeddings(self, dataloader):
        """
        Avalia a qualidade dos embeddings usando o Silhouette Score.
        Um Silhouette Score mais alto indica melhores separações entre as classes.
        """
        self.model.model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            global_imgs = batch[0].to(self.device)
            padded_cropped_imgs = batch[2].to(self.device)
            padded_zoomed_imgs = batch[4].to(self.device)

            backbone = self.model.model.model[:6]
            cropped_feats, _ = extract_multi_scale_features(backbone, padded_cropped_imgs, N=3)
            zoomed_feats, _ = extract_multi_scale_features(backbone, padded_zoomed_imgs, N=3)
            cropped_feats = cropped_feats.cpu().numpy()
            zoomed_feats = zoomed_feats.cpu().numpy()

            labels = np.concatenate([np.ones(len(cropped_feats)), np.zeros(len(zoomed_feats))])
            all_embeddings = np.concatenate([cropped_feats, zoomed_feats], axis=0)
            score = silhouette_score(all_embeddings, labels)
            print(f"Silhouette Score dos embeddings: {score:.3f}")
        
    # TODO: implementar metricas de avaliação 
        
    # TODO: implementar o método de inferência