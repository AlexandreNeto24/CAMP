import numpy as np
import torch


def calculate_iou(box1, box2):
    """Calcula IoU (Intersection over Union) entre duas boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def calculate_context_activation(pred_heatmap, segmentor_mask):
    """Calcula ativação média no contexto"""
    # ✅ Garante que ambos os tensores estão no mesmo device
    device = pred_heatmap.device if isinstance(pred_heatmap, torch.Tensor) else 'cpu'
    
    if isinstance(pred_heatmap, torch.Tensor):
        pred_heatmap = pred_heatmap.to(device)
    else:
        pred_heatmap = torch.tensor(pred_heatmap, device=device)
    
    if isinstance(segmentor_mask, torch.Tensor):
        segmentor_mask = segmentor_mask.to(device)
    else:
        segmentor_mask = torch.tensor(segmentor_mask, device=device)
    
    seg_binary = (segmentor_mask > 0.5).float()
    context_activation = (pred_heatmap * seg_binary).sum() / (seg_binary.sum() + 1e-6)
    return float(context_activation)


def calculate_object_activation(pred_heatmap, object_mask):
    """Calcula ativação média no objeto"""
    # ✅ Garante que ambos os tensores estão no mesmo device
    device = pred_heatmap.device if isinstance(pred_heatmap, torch.Tensor) else 'cpu'
    
    if isinstance(pred_heatmap, torch.Tensor):
        pred_heatmap = pred_heatmap.to(device)
    else:
        pred_heatmap = torch.tensor(pred_heatmap, device=device)
    
    if isinstance(object_mask, torch.Tensor):
        object_mask = object_mask.to(device)
    else:
        object_mask = torch.tensor(object_mask, device=device)
    
    obj_binary = (object_mask > 0.5).float()
    object_activation = (pred_heatmap * obj_binary).sum() / (obj_binary.sum() + 1e-6)
    return float(object_activation)