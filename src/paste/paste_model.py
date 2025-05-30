import os
import sys
import random
import datetime
from dataclasses import asdict
import matplotlib.pyplot as plt

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
from src.paste.utils.compute_similarity import compute_similarity
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
            hyp = yaml.safe_load(f)

        self.model.model.criterion = self.model.model.init_criterion()  # Ensure criterion is initialized
        self.model.model.criterion.hyp = Dict2Obj(hyp)  # Update hyperparameters
        
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
    
    def batch_train_loop(self, dataset, back_prop=True):
        det_loss = 0.0
        sim_loss = 0.0
        box_loss = 0.0
        cls_loss = 0.0
        dfl_loss = 0.0
        
        for batch in dataset:
            # Training code here
            global_imgs, global_mask, padded_cropped_imgs, cropped_coords, padded_zoomed_imgs, zoomed_coords, yolo_boxes = batch
            
            # Mover os tensores para GPU
            global_imgs = global_imgs.to(self.device)
            padded_cropped_imgs = padded_cropped_imgs.to(self.device)
            padded_zoomed_imgs = padded_zoomed_imgs.to(self.device)
            yolo_boxes = yolo_boxes.to(self.device)
            
            # Assuming yolo_boxes contains bounding boxes and class labels:
            # Add batch index, cls (class), and bboxes to the batch_dict
            img_idx = torch.arange(global_imgs.size(0), device=yolo_boxes.device).unsqueeze(1).float()
            labels = torch.cat([img_idx, yolo_boxes], dim=1)  # [B, 5] -> [batch_idx, cls, x, y, w, h]
            
            batch_dict = {
                "img": global_imgs,                    # Tensor [B, 3, H, W]
                "batch_idx": img_idx,                # Tensor [B, 1]
                "cls": labels[:, 1].long(),            #
                "bboxes": labels[:, 2:],               # As coordenadas [x, y, w, h]
            }
            
            raw_preds = self.model.model(global_imgs)
            global_loss, loss_items = self.model.model.loss(batch_dict, raw_preds)
            global_loss = global_loss.sum()
            
            sim_loss = compute_similarity(
                raw_preds, global_imgs,
                padded_cropped_imgs, cropped_coords,
                padded_zoomed_imgs, zoomed_coords,
                self.model
            )

            if back_prop:
                total_loss = global_loss + sim_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
            det_loss += global_loss.item()
            sim_loss += sim_loss.item()
            box_loss += loss_items[0].item()
            cls_loss += loss_items[1].item()
            dfl_loss += loss_items[2].item()
            
        # Normalize losses
        det_loss /= len(dataset)
        sim_loss /= len(dataset)
        box_loss /= len(dataset)
        cls_loss /= len(dataset)
        dfl_loss /= len(dataset)
                    
        return det_loss, sim_loss, box_loss, cls_loss, dfl_loss

    def train(self, dataset, val_split=0.2):
        
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
        backbone_unfrozen = False
        val_loss_best = float('inf')
        
        base_checkpoint_path = self.train_cfg.logging.checkpoint_path
    
        # Cria diretoria com timestamp
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.weights_path = os.path.join(base_checkpoint_path, now)
        os.makedirs(self.weights_path, exist_ok=True)
        
        # split the dataset
        n = len(dataset)
        n_train = int(n * 1 - val_split)
        n_val = n - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.train_cfg.training.batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=self.train_cfg.training.batch_size, shuffle=False, collate_fn=custom_collate)
        
        # create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.training.learning_rate)
        # create scheduler
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.train_cfg.training.reduce_lr_patience, verbose=True)
        
        for epoch in range(self.train_cfg.training.epochs):
            print(f"Epoch {epoch+1}/{self.train_cfg.training.epochs}")
                     
            if epochs_no_improvement > self.train_cfg.training.unfreeze_patience:
                # Descongela o próximo bloco a partir do fim
                idx = len(self.blocks) - 1 - unfreeze_step
                if idx >= 0:
                    for param in self.blocks[idx].parameters():
                        param.requires_grad = True
                    print(f"Descongelado bloco {idx}")
                    unfreeze_step += 1
                else:
                    print("Todos os blocos já foram descongelados.")
                epochs_no_improvement = 0

                # load the best model weights
                self.model.load_state_dict(torch.load(os.path.join(self.weights_path, "best_model.pt")))
                self.model.to(self.device)
                
            # Train step
            train_det_loss, train_sim_loss, train_box_loss, train_cls_loss, train_dfl_loss = self.batch_train_loop(train_loader)
            print(f"Train Loss: {train_det_loss:.4f}, Sim Loss: {train_sim_loss.item():.4f}, Box Loss: {train_box_loss:.4f}, Cls Loss: {train_cls_loss:.4f}, DFL Loss: {train_dfl_loss:.4f}")    
                
            # Validation step
            self.model.model.eval()
            with torch.no_grad():
                val_det_loss, val_sim_loss, val_box_loss, val_cls_loss, val_dfl_loss = self.batch_train_loop(val_loader, back_prop=False)    
            scheduler.step(val_det_loss)
            print(f"Validation Loss: {val_det_loss:.4f}, Sim Loss: {val_sim_loss.item():.4f}, Box Loss: {val_box_loss:.4f}, Cls Loss: {val_cls_loss:.4f}, DFL Loss: {val_dfl_loss:.4f}")

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
    
    # TODO: implementar visualização de predições
    
    # TODO: implementar metricas de avaliação 
        
    # TODO: implementar o método de inferência