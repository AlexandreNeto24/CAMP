import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.paste.losses.heatmap_loss import compute_loss


class AttentionRecommenderTrainer:
    """
    Treino SIMPLES e ROBUSTO para AttentionRecommender.
    
    Sem curriculum learning, sem truques.
    Apenas: Loss clara + Inpainting + Penalização do objeto.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
    
    def fit(self, train_loader, val_loader, epochs=50, lr=5e-4, 
            accumulation_steps=4, run_dir=None):
        """Treina o modelo"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
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
            
            # Guarda melhor modelo
            if run_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_model(os.path.join(run_dir, "best_model.pt"))
            
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
        # ✅ CRÍTICO: Coloca modelo em TRAIN mode
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{total_epochs}", leave=False)):
            images = batch['images'].to(self.device)                    # [B, 3, H, W]
            patches = batch['patches'].to(self.device)                  # [N, 3, h, w]
            heatmaps = batch['heatmaps'].to(self.device)                # [N, H, W]
            sample_indices = batch['sample_indices'].to(self.device)    # [N]
            segmentor_masks = batch.get('segmentor_target_mask', None)  # [B, 1, H, W]
            
            if segmentor_masks is not None:
                segmentor_masks = segmentor_masks.to(self.device)
            
            # Valida batch
            if patches.numel() == 0:
                continue
            
            # Alinha imagens e máscaras com patches
            batch_images = images[sample_indices]              # [N, 3, H, W]
            patches_batch = patches                             # [N, 3, h, w]
            
            if segmentor_masks is not None:
                segmentor_masks_aligned = segmentor_masks[sample_indices]  # [N, 1, H, W]
            else:
                segmentor_masks_aligned = None
            
            # Forward pass
            patches_ctx_batch = batch.get('patches_with_context', None)
            if patches_ctx_batch is not None:
                patches_ctx_batch = patches_ctx_batch.to(self.device)
            
            pred = self.model(
                batch_images, 
                patch=patches_batch,
                patch_with_context=patches_ctx_batch,  # ✅ NOVO
                return_intermediate=True
            )
            pred_heatmaps, patch_features, image_features = pred
            
            # Alinha targets
            target = torch.stack(heatmaps) if isinstance(heatmaps, list) else heatmaps
            target = target.unsqueeze(1)  # [N, 1, H, W]
            
            if target.shape[-2:] != pred_heatmaps.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=pred_heatmaps.shape[-2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Compute loss PRIMÁRIA
            loss_primary = compute_loss(
                pred_heatmaps, 
                target,
                segmentor_mask=segmentor_masks_aligned,
                context_weight=0.5
            )
            
            # ✅ NOVO: Batch Contrastive Loss (Hard Negatives)
            loss_contrastive = torch.tensor(0.0, device=self.device)
            
            if patch_features is not None and image_features is not None:
                from src.paste.losses.heatmap_loss import compute_contrastive_loss
                
                # 1. Loss INTRA-image (O patch pertence a ESTA imagem?)
                # (O teu código atual já faz isto)
                positive_mask = segmentor_masks_aligned if segmentor_masks_aligned is not None else target
                kernel = torch.ones(1, 1, 31, 31, device=self.device) / (31*31)
                negative_mask = F.conv2d(positive_mask, kernel, padding=15)
                negative_mask = (negative_mask > 0) & ~(positive_mask > 0.5)
                
                loss_intra = compute_contrastive_loss(
                    patch_features, image_features, positive_mask, negative_mask.float()
                )
                
                # 2. Loss INTER-image (O patch NÃO pertence à OUTRA imagem do batch!)
                # Se tivermos árvores na imagem B e um avião na imagem A, 
                # isto vai penalizar fortemente se o avião ativar nas árvores!
                if batch_images.shape[0] > 1:
                    # Roda as features das imagens (Shift)
                    # Imagem 0 recebe Patch 1 (que não lhe pertence)
                    image_features_shifted = torch.roll(image_features, shifts=1, dims=0)
                    
                    # Máscara Positiva = TUDO ZERO (Nada deve ativar)
                    # Máscara Negativa = TUDO UM (A imagem inteira é errada)
                    dummy_pos_mask = torch.zeros_like(positive_mask)
                    dummy_neg_mask = torch.ones_like(positive_mask) # Penaliza tudo!
                    
                    loss_inter = compute_contrastive_loss(
                        patch_features, image_features_shifted, dummy_pos_mask, dummy_neg_mask
                    )
                    
                    # Dá mais peso à loss Inter-image para corrigir o viés das árvores
                    loss_contrastive = 0.4 * loss_intra + 0.6 * loss_inter
                else:
                    loss_contrastive = loss_intra
            
            # Combina losses
            loss = loss_primary + 0.1 * loss_contrastive  # 10% peso para contrastiva
            
            # Backward
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
    
    def _validate_epoch(self, val_loader):
        """Validação de uma época"""
        # ✅ CRÍTICO: Coloca modelo em EVAL mode
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():  # ✅ CRÍTICO: Desativa gradientes
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch['images'].to(self.device)
                patches = batch['patches'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                sample_indices = batch['sample_indices'].to(self.device)
                segmentor_masks = batch.get('segmentor_target_mask', None)
                
                if segmentor_masks is not None:
                    segmentor_masks = segmentor_masks.to(self.device)
                
                if patches.numel() == 0:
                    continue
                
                # Alinha
                batch_images = images[sample_indices]
                patches_batch = patches
                
                if segmentor_masks is not None:
                    segmentor_masks_aligned = segmentor_masks[sample_indices]
                else:
                    segmentor_masks_aligned = None
                
                # Forward
                patches_ctx_batch = batch.get('patches_with_context', None)
                if patches_ctx_batch is not None:
                    patches_ctx_batch = patches_ctx_batch.to(self.device)
                pred = self.model(
                    batch_images, 
                    patch=patches_batch,
                    patch_with_context=patches_ctx_batch,
                    return_intermediate=True
                )
                pred_heatmaps, _, _ = pred
                
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
        
        # ✅ VOLTA A TRAIN MODE após validação
        self.model.train()
        
        return val_loss / max(1, len(val_loader))
    
    def _plot_losses(self, run_dir):
        """Plota curvas de loss"""
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