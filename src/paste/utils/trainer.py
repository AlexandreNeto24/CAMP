import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.paste.losses.heatmap_loss import compute_loss


class AttentionRecommenderTrainer:
    """
    Encapsula a lógica completa de treino para AttentionRecommender.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def fit(self, train_loader, val_loader, epochs=50, lr=5e-4, 
            accumulation_steps=4, run_dir=None):
        """
        Treina o modelo completo.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # Curriculum learning
            use_context_primary = self._get_loss_mode(epoch, epochs, switch_frac=0.10)
            context_wt = self._compute_context_weight(epoch, epochs)
            
            # Treino
            train_loss = self._train_epoch(
                train_loader, optimizer, epoch, epochs,
                use_context_primary, context_wt, accumulation_steps
            )
            
            # Validação
            val_loss = self._validate_epoch(
                val_loader, epoch, epochs,
                use_context_primary, context_wt
            )
            
            scheduler.step(val_loss)
            
            if run_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_model(os.path.join(run_dir, "best_model.pt"))
                print(f"✓ Best model saved")
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f} val={val_loss:.4f}")
        
        if run_dir:
            self._plot_losses(run_dir)
            best_path = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(best_path):
                self.model.load_model(best_path)
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, epoch, total_epochs,
                     use_context_primary, context_wt, accumulation_steps):
        """
        ✅ NOVO: Treino simples e direto.
        Loss sempre igual, sem curriculum learning confuso.
        """
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{total_epochs}")):
            images = batch['images'].to(self.device)              # [B, 3, H, W]
            patches = batch['patches'].to(self.device)            # [N, 3, h, w]
            heatmaps = batch['heatmaps'].to(self.device)          # [N, H, W]
            sample_indices = batch['sample_indices'].to(self.device)  # [N]
            segmentor_masks = batch.get('segmentor_target_mask', None)  # [B, 1, H, W]
            
            if segmentor_masks is not None:
                segmentor_masks = segmentor_masks.to(self.device)
            
            # Verificação de validade
            if patches.numel() == 0:
                continue
            
            # ===== ALINHAR IMAGENS E MÁSCARAS COM PATCHES =====
            # sample_indices diz qual imagem cada patch pertence
            batch_images = images[sample_indices]              # [N, 3, H, W]
            patches_batch = patches                             # [N, 3, h, w]
            
            if segmentor_masks is not None:
                segmentor_masks_aligned = segmentor_masks[sample_indices]  # [N, 1, H, W]
            else:
                segmentor_masks_aligned = None
            
            # ===== FORWARD PASS =====
            # Usa patch como está (com contexto original)
            pred = self.model(
                batch_images, 
                patch=patches_batch,
                return_intermediate=True
            )
            pred_heatmaps, patch_feats, img_feats = pred
            
            # ===== ALINHAR TARGETS =====
            target = torch.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps
            target = target.unsqueeze(1)  # [N, 1, H, W]
            
            if target.shape[-2:] != pred_heatmaps.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=pred_heatmaps.shape[-2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # ===== COMPUTE LOSS =====
            # Loss SIMPLES: compatibilidade de contexto + penalização do objeto
            loss = compute_loss(
                pred_heatmaps, 
                target,
                segmentor_mask=segmentor_masks_aligned,
                context_weight=0.5  # Peso fixo para penalidade do objeto
            )
            
            # ===== BACKWARD =====
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
        
        return epoch_loss / max(1, len(train_loader))
    
    def _validate_epoch(self, val_loader, epoch, total_epochs,
                   use_context_primary, context_wt):
        """
        ✅ NOVO: Validação simples.
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['images'].to(self.device)
                patches = batch['patches'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                sample_indices = batch['sample_indices'].to(self.device)
                segmentor_masks = batch.get('segmentor_target_mask', None)
                
                if segmentor_masks is not None:
                    segmentor_masks = segmentor_masks.to(self.device)
                
                if patches.numel() == 0:
                    continue
                
                # Alinhar
                batch_images = images[sample_indices]
                patches_batch = patches
                
                if segmentor_masks is not None:
                    segmentor_masks_aligned = segmentor_masks[sample_indices]
                else:
                    segmentor_masks_aligned = None
                
                # Forward
                pred = self.model(
                    batch_images, 
                    patch=patches_batch,
                    return_intermediate=True
                )
                pred_heatmaps, patch_feats, img_feats = pred
                
                # Targets
                target = torch.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps
                target = target.unsqueeze(1)
                
                if target.shape[-2:] != pred_heatmaps.shape[-2:]:
                    target = F.interpolate(
                        target,
                        size=pred_heatmaps.shape[-2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Loss
                loss = compute_loss(
                    pred_heatmaps, 
                    target,
                    segmentor_mask=segmentor_masks_aligned,
                    context_weight=0.5
                )
                val_loss += loss.item()
    
        return val_loss / max(1, len(val_loader))
    
    @staticmethod
    def _get_loss_mode(epoch, total_epochs, switch_frac=0.10):
        """Ignorado — sempre usa modo contexto"""
        return True

    @staticmethod
    def _compute_context_weight(epoch, total_epochs, switch_frac=0.10):
        """Sempre retorna peso fixo"""
        return 0.5
    
    def _plot_losses(self, run_dir):
        """Plota curva de loss"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Val')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
        plt.close()