import torch
import torch.nn.functional as F


def compute_contrastive_loss(patch_features, image_features, pred_heatmap, 
                            n_neg=64, tau=0.07):
    """
    InfoNCE-like loss para features contrastivas.
    
    Args:
        patch_features: [B, C, hp, wp]
        image_features: [B, C, Hf, Wf]
        pred_heatmap: [B, 1, H_img, W_img]
        n_neg: número de negativos
        tau: temperatura
        
    Returns:
        scalar loss
    """
    # Patch embedding: global avg pool
    patch_emb = F.adaptive_avg_pool2d(patch_features, 1).view(patch_features.size(0), -1)
    patch_emb = F.normalize(patch_emb, dim=1)

    # Positive embedding: weighted by heatmap
    _, _, Hf, Wf = image_features.shape
    heat_resized = F.interpolate(pred_heatmap, size=(Hf, Wf), mode='bilinear', align_corners=False)
    heat_flat = heat_resized.view(image_features.size(0), 1, -1)
    
    img_flat = image_features.view(image_features.size(0), image_features.size(1), -1)
    pos_emb = torch.bmm(img_flat, heat_flat.permute(0, 2, 1)).squeeze(-1)
    pos_emb = F.normalize(pos_emb, dim=1)

    # Negative embeddings: amostras aleatórias
    B, C, L = img_flat.shape
    neg_idxs = torch.randint(0, L, (B, n_neg), device=img_flat.device)
    neg_embs = []
    for b in range(B):
        neg_sel = img_flat[b:b+1, :, neg_idxs[b]]
        neg_mean = neg_sel.mean(dim=2).squeeze(0)
        neg_embs.append(neg_mean)
    neg_embs = torch.stack(neg_embs, dim=0)
    neg_embs = F.normalize(neg_embs, dim=1)

    # Similaridades
    pos_sim = (patch_emb * pos_emb).sum(dim=1) / tau
    neg_sim = (patch_emb * neg_embs).sum(dim=1) / tau

    # InfoNCE loss
    denom = torch.exp(pos_sim) + torch.exp(neg_sim)
    loss = -torch.log(torch.exp(pos_sim) / (denom + 1e-8)).mean()
    return loss