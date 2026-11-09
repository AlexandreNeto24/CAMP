import torch
import torch.nn.functional as F


def compute_loss(pred_heatmap, target_heatmap, segmentor_mask=None, 
                 inpainting_mask=None,
                 patch_features=None, image_features=None, 
                 contrastive_weight=0.0, 
                 context_weight=0.5,  # ✅ REDUZ DE 2.0 PARA 1.0
                 use_context_primary=False):
    """
    Loss ULTRA-SIMPLES: MSE + Penalização Objeto.
    """
    
    pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1-1e-7)
    target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)
    
    # 1. Loss MSE (Base)
    loss_primary = F.mse_loss(pred_heatmap, target_heatmap)
    
    # 2. Penaliza ativação no objeto
    loss_object_suppression = torch.tensor(0.0, device=pred_heatmap.device)
    
    if segmentor_mask is not None:
        segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)
        
        if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
            segmentor_mask = F.interpolate(
                segmentor_mask, 
                size=pred_heatmap.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Queremos pred_heatmap = 0 onde segmentor_mask > 0
        obj_activation = pred_heatmap * segmentor_mask
        loss_object_suppression = (obj_activation ** 2).mean()
    
    # Total: MSE + Penalização Objeto (SEM context reward)
    total_loss = loss_primary + context_weight * loss_object_suppression
    
    # 3. Loss Contrastiva (Hard Negatives)
    if contrastive_weight > 0 and patch_features is not None and image_features is not None:
        # Definir máscaras Positivas e Negativas
        
        # Positivos: Onde o target é alto (a região correta)
        positive_mask = (target_heatmap > 0.5).float()
        
        # Se não houver positivos (ex: slice vazio), pula
        if positive_mask.sum() > 0:
            # Negativos base: Tudo o que NÃO é positivo
            valid_negatives = 1.0 - positive_mask
            
            # HARD MINING: Onde o modelo erra com alta confiança?
            # Pegamos regiões onde pred é alto mas target é zero (Falsos Positivos)
            hard_negatives = (pred_heatmap.detach() > 0.3).float() * valid_negatives
            
            # Se tivermos poucos hard negatives, completamos com aleatórios validos
            if hard_negatives.sum() < 100: # Heurística simples
                 negative_mask = valid_negatives # Fallback para todos os negativos possíveis
            else:
                 negative_mask = hard_negatives
            
            # Calcula a loss contrastiva
            loss_contrastive = compute_contrastive_loss(
                patch_features, 
                image_features, 
                positive_mask, 
                negative_mask
            )
            
            total_loss += contrastive_weight * loss_contrastive
            
    return total_loss


def compute_contrastive_loss(patch_features, image_features, positive_mask, negative_mask, temperature=0.07):
    """
    Loss contrastiva SEMÂNTICA.
    Compara o CONTEXTO do patch com regiões da imagem.
    
    Args:
        patch_features (Tensor): Features do CONTEXTO do patch. [B, E, h, w]
        image_features (Tensor): Features da imagem inteira. [B, E, H, W]
        positive_mask (Tensor): Máscara de regiões onde o contexto é similar.
        negative_mask (Tensor): Máscara de regiões onde o contexto é dissimilar.
    """
    B, E, h, w = patch_features.shape
    B, E_img, H_feat, W_feat = image_features.shape
    
    # Garante que os canais são compatíveis
    E_min = min(E, E_img)
    patch_features = patch_features[:, :E_min, :, :]
    image_features = image_features[:, :E_min, :, :]
    
    # 1. Vetor descritor do CONTEXTO do patch
    # Média apenas dos pixels que não são zero (que são o contexto)
    patch_pixel_count = (patch_features.abs().sum(dim=1, keepdim=True) > 0).sum(dim=(2,3)).clamp(min=1)
    patch_context_avg = patch_features.sum(dim=(2, 3)) / patch_pixel_count # [B, E]

    feature_size = (H_feat, W_feat)
    
    if positive_mask.dim() == 3:
        positive_mask = positive_mask.unsqueeze(1)
    if negative_mask.dim() == 3:
        negative_mask = negative_mask.unsqueeze(1)
        
    pos_mask_small = F.interpolate(
        positive_mask.float(),
        size=feature_size,
        mode='nearest'
    )
    pos_mask = (pos_mask_small > 0.5).float()
    
    neg_mask_small = F.interpolate(
        negative_mask.float(),
        size=feature_size,
        mode='nearest'
    )
    neg_mask = (neg_mask_small > 0.5).float()
    
    # 2. Features médias das regiões POSITIVAS e NEGATIVAS na imagem
    pos_features_sum = (image_features * pos_mask).sum(dim=(2, 3))
    pos_count = pos_mask.sum(dim=(2, 3)).clamp(min=1)
    pos_features_avg = pos_features_sum / pos_count
    
    neg_features_sum = (image_features * neg_mask).sum(dim=(2, 3))
    neg_count = neg_mask.sum(dim=(2, 3)).clamp(min=1)
    neg_features_avg = neg_features_sum / neg_count
    
    # 3. Normaliza e calcula similaridade
    patch_norm = F.normalize(patch_context_avg, dim=1, p=2)
    pos_norm = F.normalize(pos_features_avg, dim=1, p=2)
    neg_norm = F.normalize(neg_features_avg, dim=1, p=2)
    
    pos_sim = (patch_norm * pos_norm).sum(dim=1, keepdim=True)
    neg_sim = (patch_norm * neg_norm).sum(dim=1, keepdim=True)
    
    # 4. Loss NCE (Noise Contrastive Estimation)
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    max_sim = torch.max(pos_sim, neg_sim)
    pos_contrib = torch.exp(pos_sim - max_sim)
    neg_contrib = torch.exp(neg_sim - max_sim)
    
    log_sum_exp = torch.log(pos_contrib + neg_contrib) + max_sim
    loss_nce = -pos_sim + log_sum_exp
    
    return loss_nce.mean()