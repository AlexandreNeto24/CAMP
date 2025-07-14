import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def compute_similarity_matrix(emb1, boxes1, emb2, boxes2, weights=None):
    emb_dist = torch.cdist(emb1, emb2, p=2)
    iou = box_iou(boxes1, boxes2)
    iou_score = 1 - iou
    w = weights or {'emb': 1.0, 'iou': 1.0}
    sim_matrix = w['emb'] * emb_dist + w['iou'] * iou_score
    return sim_matrix

def similarity_loss_from_matrix(embeddings, boxes, labels, margin=1.0):
    sim_matrix = compute_similarity_matrix(embeddings, boxes, embeddings, boxes)  # [N, N]
    sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    N = labels.shape[0]
    device = embeddings.device
    labels = labels.view(-1)

    # Só pares positivos (labels==1) e negativos (labels==0)
    pos_mask = (labels.unsqueeze(1) == 1) & (labels.unsqueeze(0) == 1) & (~torch.eye(N, dtype=torch.bool, device=device))
    neg_mask = (labels.unsqueeze(1) == 1) & (labels.unsqueeze(0) == 0)

    pos_scores = sim_matrix[pos_mask]
    neg_scores = sim_matrix[neg_mask]

    # Contrastive loss: positivos querem score baixo, negativos querem score alto (> margin)
    loss_pos = (pos_scores ** 2).mean() if pos_scores.numel() > 0 else torch.tensor(0.0, device=device)
    loss_neg = (F.relu(margin - neg_scores) ** 2).mean() if neg_scores.numel() > 0 else torch.tensor(0.0, device=device)

    final_loss = (loss_pos + loss_neg) / (N if N > 0 else 1)

    if torch.isnan(final_loss): final_loss = torch.tensor(0.0, device=device)

    return final_loss