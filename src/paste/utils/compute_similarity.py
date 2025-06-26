import torch
import torchvision.ops as ops
from torchvision.ops import roi_align
import torch.nn.functional as F

def decode_boxes(raw_preds, img_idx, img_size=(640,640)):
    preds = raw_preds[0][img_idx].permute(1, 0).contiguous()  # [N, 5]
    x1 = preds[:, 0] - preds[:, 2] / 2
    y1 = preds[:, 1] - preds[:, 3] / 2
    x2 = preds[:, 0] + preds[:, 2] / 2
    y2 = preds[:, 1] + preds[:, 3] / 2
    conf = preds[:, 4]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [N, 4]
    return boxes, conf

def extract_roi_features(feature_map, boxes, batch_indices=None, output_size=(7, 7)):
    if batch_indices is None:
        batch_indices = torch.zeros((boxes.size(0),), dtype=torch.int, device=boxes.device)
    feature_map = feature_map.to(boxes.device)
    batch_indices = batch_indices.to(boxes.device)
    rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)  # [N, 5]
    spatial_scale = feature_map.shape[2] / 640  # assume input 640x640
    try:
        roi_feats = roi_align(feature_map, rois, output_size=output_size, spatial_scale=spatial_scale)
    except Exception as e:
        print(f"[DEBUG] roi_align error: {e}")
        return torch.zeros((boxes.size(0), feature_map.shape[1]), device=feature_map.device)
    roi_feats = F.adaptive_avg_pool2d(roi_feats, (1, 1)).view(roi_feats.size(0), -1)
    return F.normalize(roi_feats, dim=1)

def compute_similarity(
    raw_preds, global_imgs,
    padded_cropped_imgs, cropped_coords,
    padded_zoomed_imgs, zoomed_coords,
    model,
    img_indices,
    margin=1.0
):
    device = global_imgs.device
    backbone = model.model.model[:6]
    features = backbone(global_imgs)
    feature_map = features[-2] if len(features) >= 2 else features[-1]
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)

    all_boxes = torch.cat([cropped_coords[:, 1:], zoomed_coords[:, 1:]], dim=0).to(device)
    batch_indices = torch.zeros(all_boxes.size(0), dtype=torch.int, device=device)
    labels = torch.cat([torch.ones(cropped_coords.size(0)), torch.zeros(zoomed_coords.size(0))]).to(device)

    all_boxes = all_boxes.clone()
    all_boxes[:, 0::2] = all_boxes[:, 0::2].clamp(0, feature_map.shape[3] - 1)
    all_boxes[:, 1::2] = all_boxes[:, 1::2].clamp(0, feature_map.shape[2] - 1)

    embeddings = extract_roi_features(feature_map, all_boxes, batch_indices=batch_indices, output_size=(7,7))  # [N, D]

    # Gera pares positivos e negativos
    pos_pairs = []
    neg_pairs = []
    for i in range(embeddings.shape[0]):
        for j in range(i+1, embeddings.shape[0]):
            if labels[i] == labels[j]:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))

    loss = 0.0
    count = 0
    # Positivos (label=1)
    if pos_pairs:
        emb1 = torch.stack([embeddings[i] for i, j in pos_pairs])
        emb2 = torch.stack([embeddings[j] for i, j in pos_pairs])
        label = torch.ones(len(pos_pairs), device=device)
        loss += contrastive_loss(emb1, emb2, label, margin)
        count += 1
    # Negativos (label=0)
    if neg_pairs:
        emb1 = torch.stack([embeddings[i] for i, j in neg_pairs])
        emb2 = torch.stack([embeddings[j] for i, j in neg_pairs])
        label = torch.zeros(len(neg_pairs), device=device)
        loss += contrastive_loss(emb1, emb2, label, margin)
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=device)

def extract_multi_scale_features(backbone, imgs, N=3):
    """
    Extrai e concatena as últimas N features do backbone para um batch de imagens.
    backbone: módulo backbone (ex: model.model.model[:6])
    imgs: tensor [B, 3, H, W]
    N: número de features a usar (ex: 2 ou 3)
    Retorna: tensor [B, D], onde D = soma dos canais das N features
    """
    features = backbone(imgs)
    N = min(N, len(features))
    pooled = [F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1) for f in features[-N:]]
    embedding = torch.cat(pooled, dim=1)
    embedding = F.normalize(embedding, dim=1)
    return embedding, [f.shape[1] for f in features[-N:]]  # devolve também os canais

def contrastive_loss(emb1, emb2, label, margin=1.0):
    # emb1, emb2: [N, D]
    # label: [N] (1=similar, 0=dissimilar)
    D = F.pairwise_distance(emb1, emb2, p=2)
    loss = label * D**2 + (1 - label) * F.relu(margin - D)**2
    return loss.mean()