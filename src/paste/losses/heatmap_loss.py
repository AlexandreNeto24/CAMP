import torch
import torch.nn.functional as F


def compute_loss(pred_heatmap, target_heatmap, segmentor_mask=None, 
                 patch_features=None, image_features=None, 
                 contrastive_weight=0.0, context_weight=0.5, 
                 use_context_primary=False):
    """
    ✅ NOVO: Loss SIMPLES e DIRECTA para compatibilidade de contexto.
    
    NÃO HÁ DOIS MODOS. Apenas um: treinar para ativar regiões compatíveis
    e desativar o próprio objeto.
    
    Args:
        pred_heatmap: [B, 1, H, W] predição do modelo
        target_heatmap: [B, 1, H, W] mapa de compatibilidade alvo (cor/textura similar)
        segmentor_mask: [B, 1, H, W] máscara do objeto (para penalizar ativação)
        patch_features: [B, C, h, w] features do patch (opcional)
        image_features: [B, C, H, W] features da imagem (opcional)
        contrastive_weight: peso da loss contrastiva (ignorado neste modo)
        context_weight: peso da penalização do objeto
        use_context_primary: ignorado (sempre usa modo contexto)
        
    Returns:
        Scalar loss
    """
    
    # ===== LOSS PRIMÁRIA: MSE entre predição e target de compatibilidade =====
    pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1-1e-7)
    target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)
    
    # Erro entre predição e target
    loss_primary = F.mse_loss(pred_heatmap, target_heatmap)
    
    # ===== LOSS SECUNDÁRIA: Penalizar ativação NO OBJETO =====
    # O modelo NÃO DEVE ativar a região do próprio objeto
    loss_object_penalty = 0.0
    if segmentor_mask is not None:
        segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)
        
        # Align shapes se necessário
        if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
            segmentor_mask = F.interpolate(
                segmentor_mask, 
                size=pred_heatmap.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # ✅ CRUCIAL: Penaliza FORTE qualquer ativação dentro do objeto
        # Queremos que pred_heatmap seja ~0 onde segmentor_mask=1
        loss_object_penalty = F.mse_loss(
            pred_heatmap * segmentor_mask,
            torch.zeros_like(pred_heatmap)
        )
    
    # ===== COMBINAÇÃO FINAL =====
    # Primária domina, secundária penaliza erros
    total_loss = loss_primary + context_weight * loss_object_penalty
    
    return total_loss