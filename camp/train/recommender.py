"""Recommender training.

The loss is not divided by ``accumulation_steps``. ``CosineAnnealingWarmRestarts``
is stepped with the validation loss. After validation the module is put back
in train mode, which leaves the encoder in eval via ``AttentionRecommender.train``.
"""

import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from camp.train.losses import compute_loss
from camp.train.progress import batch_bar

logger = logging.getLogger(__name__)


class AttentionRecommenderTrainer:
    """Train ``AttentionRecommender`` with early stopping on validation loss."""
    
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
        self.verbose = True
    
    def fit(self, train_loader, val_loader, epochs=50, lr=5e-4, 
            accumulation_steps=4, run_dir=None, phase=None):
        """Train for ``epochs``. ``phase`` labels the log line when set."""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs.
            T_mult=2,  # Double the period after each restart.
            eta_min=1e-6,  # Minimum learning rate.
        )
        
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_loss = self._train_epoch(
                train_loader, optimizer, epoch, epochs, accumulation_steps
            )
            val_loss = self._validate_epoch(val_loader)
            lr_current = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            
            improved = bool(run_dir) and val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.model.save_model(os.path.join(run_dir, "best_model.pt"))
            else:
                self.epochs_without_improvement += 1

            if self.verbose:
                phase_label = "" if phase is None else "phase %s  " % phase
                logger.info(
                    "%s/%s  %strain %.4f  val %.4f  lr %.1e%s",
                    epoch + 1,
                    epochs,
                    phase_label,
                    train_loss,
                    val_loss,
                    lr_current,
                    "  best" if improved else "",
                )

            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    logger.info(
                        "stop  no val loss gain for %s epochs", self.patience
                    )
                break

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
        
        if run_dir:
            best_path = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(best_path):
                self.model.load_model(best_path)
            self._plot_losses(run_dir)
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, epoch, total_epochs, accumulation_steps):
        """Train one epoch and return the mean loss."""
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(
            batch_bar(train_loader, "%s/%s train" % (epoch + 1, total_epochs))
        ):
            images = batch['images'].to(self.device)
            patches = batch['patches'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)
            sample_indices = batch['sample_indices'].to(self.device)
            segmentor_masks = batch.get('segmentor_target_mask', None)
            inpainting_masks = batch.get('inpainting_masks', None)

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
                context_mask=patches_mask_batch,
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
                contrastive_weight=0.5,
                context_weight=1.0,
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
        """Validate one epoch and return the mean loss."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in batch_bar(val_loader, "val"):
                images = batch['images'].to(self.device)
                patches = batch['patches'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                sample_indices = batch['sample_indices'].to(self.device)
                segmentor_masks = batch.get('segmentor_target_mask', None)
                inpainting_masks = batch.get('inpainting_masks', None)
                
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
                    context_mask=patches_mask_batch,
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
                    contrastive_weight=0.5,
                    context_weight=1.0,
                )
                val_loss += loss.item()
        
        self.model.train()
        
        return val_loss / max(1, len(val_loader))
    
    def _plot_losses(self, run_dir):
        """Write ``losses.png`` with the train and validation curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(self.val_losses, label='Val Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss (train vs validation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(run_dir, 'losses.png'), dpi=150, bbox_inches='tight')
        plt.close()