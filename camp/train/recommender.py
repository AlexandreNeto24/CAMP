import os
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from camp.train.losses import compute_loss, compute_contrastive_loss


class AttentionRecommenderTrainer:
    """
    Treino SIMPLES e ROBUSTO para AttentionRecommender.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.patience = 30
        self.epochs_without_improvement = 0
    
    def fit(self, train_loader, val_loader, epochs=50, lr=5e-4, 
            accumulation_steps=4, run_dir=None):
        """Treina o modelo"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Reinicia a cada 10 épocas
            T_mult=2,    # Duplica o período a cada restart
            eta_min=1e-6 # LR mínimo
        )
        
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Treino
            train_loss = self._train_epoch(train_loader, optimizer, epoch, epochs, accumulation_steps)
            
            # Validação
            val_loss = self._validate_epoch(val_loader)
            
            scheduler.step(val_loss)
            
            # Guarda melhor modelo e verifica early stopping
            if run_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0  # ✅ Reseta contador
                self.model.save_model(os.path.join(run_dir, "best_model.pt"))
            else:
                self.epochs_without_improvement += 1  # ✅ Incrementa
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n⚠️ Early stopping: sem melhoria há {self.patience} épocas")
                    break  # ✅ Para o treino

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            lr_current = optimizer.param_groups[0]['lr']
            print(f"[{epoch+1}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} lr={lr_current:.2e}")
        
        # Carrega melhor modelo
        if run_dir:
            best_path = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(best_path):
                self.model.load_model(best_path)
                print(f"✓ Melhor modelo carregado")
            self._plot_losses(run_dir)
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, epoch, total_epochs, accumulation_steps):
        """Treino de uma época"""
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{total_epochs}", leave=False)):
            images = batch['images'].to(self.device)
            patches = batch['patches'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)
            sample_indices = batch['sample_indices'].to(self.device)
            segmentor_masks = batch.get('segmentor_target_mask', None)
            inpainting_masks = batch.get('inpainting_masks', None)  # ✅ PEGA A MÁSCARA
            
            if segmentor_masks is not None:
                segmentor_masks = segmentor_masks.to(self.device)
            
            if inpainting_masks is not None:
                inpainting_masks = inpainting_masks.to(self.device)

            if patches.numel() == 0:
                continue
            
            batch_images = images[sample_indices]
            patches_batch = patches
            
            if segmentor_masks is not None:
                segmentor_masks_aligned = segmentor_masks[sample_indices]
            else:
                segmentor_masks_aligned = None
            
            patches_ctx_batch = batch.get('patches_with_context', None)
            if patches_ctx_batch is not None:
                patches_ctx_batch = patches_ctx_batch.to(self.device)
            
            pred = self.model(
                batch_images, 
                patch=patches_batch,
                patch_with_context=patches_ctx_batch,
                return_intermediate=True
            )
            pred_heatmaps, patch_features, image_features = pred
            
            target = torch.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps
            target = target.unsqueeze(1)
            
            if target.shape[-2:] != pred_heatmaps.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=pred_heatmaps.shape[-2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Compute loss
            loss = compute_loss(
                pred_heatmaps, 
                target,
                segmentor_mask=segmentor_masks_aligned,
                inpainting_mask=inpainting_masks[sample_indices] if inpainting_masks is not None else None,
                patch_features=patch_features,
                image_features=image_features,
                contrastive_weight=0.5, # ✅ OPTION 1: Increased from 0.1 to 0.5 for harder negative mining
                context_weight=1.0 # Mantido em 1.0 conforme discussão prévia
            )
            
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        return epoch_loss / max(1, len(train_loader))
    
    def _validate_epoch(self, val_loader):
        """Validação de uma época"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch['images'].to(self.device)
                patches = batch['patches'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                sample_indices = batch['sample_indices'].to(self.device)
                segmentor_masks = batch.get('segmentor_target_mask', None)
                inpainting_masks = batch.get('inpainting_masks', None)  # ✅ PEGA A MÁSCARA
                
                if segmentor_masks is not None:
                    segmentor_masks = segmentor_masks.to(self.device)
                
                if inpainting_masks is not None:
                    inpainting_masks = inpainting_masks.to(self.device)
            
                if patches.numel() == 0:
                    continue
                
                batch_images = images[sample_indices]
                patches_batch = patches
                
                if segmentor_masks is not None:
                    segmentor_masks_aligned = segmentor_masks[sample_indices]
                else:
                    segmentor_masks_aligned = None
                
                patches_ctx_batch = batch.get('patches_with_context', None)
                if patches_ctx_batch is not None:
                    patches_ctx_batch = patches_ctx_batch.to(self.device)
                
                patches_mask_batch = batch.get('patches_context_masks', None)
                if patches_mask_batch is not None:
                    patches_mask_batch = patches_mask_batch.to(self.device)
                    
                pred = self.model(
                    batch_images, 
                    patch=patches_batch,
                    patch_with_context=patches_ctx_batch,
                    context_mask=patches_mask_batch, # ✅ Passa a máscara
                    return_intermediate=True
                )
                pred_heatmaps, patch_features, image_features = pred
                
                target = torch.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps
                target = target.unsqueeze(1)
                
                if target.shape[-2:] != pred_heatmaps.shape[-2:]:
                    target = F.interpolate(
                        target,
                        size=pred_heatmaps.shape[-2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Compute loss
                loss = compute_loss(
                    pred_heatmaps, 
                    target,
                    segmentor_mask=segmentor_masks_aligned,
                    inpainting_mask=inpainting_masks[sample_indices] if inpainting_masks is not None else None,
                    patch_features=patch_features,
                    image_features=image_features,
                    contrastive_weight=0.5, # ✅ ALIGNED with training
                    context_weight=1.0,     # ✅ ALIGNED with training
                )
                val_loss += loss.item()
        
        self.model.train()
        
        return val_loss / max(1, len(val_loader))
    
    def _plot_losses(self, run_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(self.val_losses, label='Val Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Curva de Loss (Treino vs Validação)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(run_dir, 'losses.png'), dpi=150, bbox_inches='tight')
        plt.close()