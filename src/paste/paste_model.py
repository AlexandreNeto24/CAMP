
import os
import sys
import random
import ipdb
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
    def __init__(self, yaml_path, model_name):
        train_cfg = TrainConfig.from_yaml(yaml_path, model_name)
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
        
        return self.model
    
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
        
        epochs_no_improvement = 0
        backbone_unfrozen = False
        val_loss_best = float('inf')
        
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
            
            if epochs_no_improvement > self.train_cfg.training.unfreeze_patience and not backbone_unfrozen:
                # Unfreeze the backbone
                for param in self.model.model.model.parameters():
                    param.requires_grad = True
                print("Backbone unfrozen.")
                backbone_unfrozen = True
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.training.learning_rate / 10)
                scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.train_cfg.training.reduce_lr_factor, patience=self.train_cfg.training.reduce_lr_patience, verbose=True)
                epochs_no_improvement = 0

                # load the best model weights
                self.model.load_state_dict(torch.load(os.path.join(self.train_cfg.logging.checkpoint_path, "best_model.pt")))
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
                torch.save(self.model.state_dict(), os.path.join(self.train_cfg.logging.checkpoint_path, f"best_model.pt"))
            else:
                epochs_no_improvement += 1
                print(f"No improvement in validation loss. Epochs without improvement: {epochs_no_improvement}")
                
            if epochs_no_improvement >= self.train_cfg.training.earlystop_patience:
                print(f"Early stopping triggered after {epochs_no_improvement} epochs without improvement.")
                break    
            
        return 
    
    # TODO: implementar o método de inferência
    
    # TODO: implementar visualização de predições
    
    # TODO: implementar metricas de avaliação 
    
    # TODO: infos summary modelo
    
    # TODO: salvar metricas do treino