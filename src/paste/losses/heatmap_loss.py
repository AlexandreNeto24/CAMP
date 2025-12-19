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


def compute_contrastive_loss(patch_features, image_features, positive_mask, negative_mask, temperature=0.07):
    """
    Loss contrastiva: força o modelo a aprender relações.
    
    ✅ VERSÃO ROBUSTA: Sem problemas de broadcasting
    
    Args:
        patch_features: [B, E, h, w] features do patch
        image_features: [B, E, H, W] features da imagem
        positive_mask: [B, 1, H, W] ou [B, H, W] máscara onde patch deveria ativar
        negative_mask: [B, 1, H, W] ou [B, H, W] máscara onde patch NÃO deveria ativar
        temperature: escala para softmax
        
    Returns:
        Loss escalar
    """
    
    B, E, h, w = patch_features.shape
    B, E_img, H, W = image_features.shape
    
    # Garante que ambas têm o mesmo E (usa o menor para ser seguro)
    E_min = min(E, E_img)
    patch_features = patch_features[:, :E_min, :, :]
    image_features = image_features[:, :E_min, :, :]
    
    # Trata máscaras: remove dimensão singleton se existir
    if positive_mask.dim() == 4 and positive_mask.shape[1] == 1:
        positive_mask = positive_mask.squeeze(1)  # [B, H, W]
    if negative_mask.dim() == 4 and negative_mask.shape[1] == 1:
        negative_mask = negative_mask.squeeze(1)  # [B, H, W]
    
    # Médias espaciais das features
    patch_avg = patch_features.mean(dim=(2, 3))  # [B, E_min]
    
    # Redimensiona image_features para o tamanho da máscara (H, W)
    if image_features.shape[-2:] != positive_mask.shape[-2:]:
        image_features = F.interpolate(
            image_features, 
            size=positive_mask.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, E_min, H, W]
    
    # Redimensiona máscaras
    pos_mask = F.interpolate(
        positive_mask.unsqueeze(1).float(),  # [B, 1, H, W]
        size=image_features.shape[-2:], 
        mode='nearest'
    ).squeeze(1) > 0.5  # [B, H, W]
    
    neg_mask = F.interpolate(
        negative_mask.unsqueeze(1).float(),  # [B, 1, H, W]
        size=image_features.shape[-2:], 
        mode='nearest'
    ).squeeze(1) > 0.5  # [B, H, W]
    
    # ✅ CORREÇÃO: Extrai features sem problemas de broadcasting
    
    # Positivas: média das features onde pos_mask > 0
    pos_mask_expanded = pos_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, H, W] - ERRADO, deixa como [B, 1, H, W]
    pos_mask_expanded = pos_mask.unsqueeze(1)  # [B, 1, H, W]
    
    # Multiplica features pela máscara e soma
    pos_features_sum = (image_features * pos_mask_expanded).sum(dim=(2, 3))  # [B, E_min]
    pos_count = pos_mask_expanded.sum(dim=(2, 3)).clamp(min=1)  # [B, 1] ✅ MANTÉM dimensão
    pos_features_avg = pos_features_sum / pos_count  # [B, E_min] / [B, 1] = [B, E_min]
    
    # Negativas: média das features onde neg_mask > 0
    neg_mask_expanded = neg_mask.unsqueeze(1)  # [B, 1, H, W]
    
    neg_features_sum = (image_features * neg_mask_expanded).sum(dim=(2, 3))  # [B, E_min]
    neg_count = neg_mask_expanded.sum(dim=(2, 3)).clamp(min=1)  # [B, 1] ✅ MANTÉM dimensão
    neg_features_avg = neg_features_sum / neg_count  # [B, E_min] / [B, 1] = [B, E_min]
    
    # Normaliza
    patch_norm = F.normalize(patch_avg, dim=1, p=2)  # [B, E_min]
    pos_norm = F.normalize(pos_features_avg, dim=1, p=2)  # [B, E_min]
    neg_norm = F.normalize(neg_features_avg, dim=1, p=2)  # [B, E_min]
    
    # Similaridades
    pos_sim = (patch_norm * pos_norm).sum(dim=1, keepdim=True)  # [B, 1]
    neg_sim = (patch_norm * neg_norm).sum(dim=1, keepdim=True)  # [B, 1]
    
    # Escala por temperatura
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    # NCE Loss: log(exp(pos) / (exp(pos) + exp(neg)))
    # Usa log_sum_exp para estabilidade numérica
    max_sim = torch.max(pos_sim, neg_sim)
    pos_contrib = torch.exp(pos_sim - max_sim)
    neg_contrib = torch.exp(neg_sim - max_sim)
    
    log_sum_exp = torch.log(pos_contrib + neg_contrib) + max_sim
    loss_nce = -pos_sim + log_sum_exp
    
    return loss_nce.mean()