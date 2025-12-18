import torch
import torch.nn.functional as F


def compute_loss(pred_heatmap, target_heatmap, segmentor_mask=None, 
                 patch_features=None, image_features=None, 
                 contrastive_weight=0.0, context_weight=0.5, 
                 use_context_primary=False):
    """
    ✅ SIMPLES E FUNCIONA: Loss baseada em cor (o que funcionava antes).
    
    Apenas adiciona penalização do objeto (isto é bom).
    """
    
    pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1-1e-7)
    target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)
    
    # ===== LOSS PRIMÁRIA: MSE =====
    # Isto funcionava bem antes, mantém
    loss_primary = F.mse_loss(pred_heatmap, target_heatmap)
    
    # ===== LOSS SECUNDÁRIA: Penaliza ativação no objeto =====
    # Isto é novo e BOM (previne memorização)
    loss_object_penalty = 0.0
    
    if segmentor_mask is not None:
        segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)
        
        if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
            segmentor_mask = F.interpolate(
                segmentor_mask, 
                size=pred_heatmap.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Penaliza ativação NO objeto
        loss_object_penalty = F.mse_loss(
            pred_heatmap * segmentor_mask,
            torch.zeros_like(pred_heatmap)
        )
    
    # Total: MSE principal + penalização objeto
    total_loss = loss_primary + context_weight * loss_object_penalty
    
    return total_loss