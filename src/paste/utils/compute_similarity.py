import torch
import torchvision.ops as ops
from torchvision.ops import roi_align, box_iou
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

def extract_roi_features(feature_map, boxes, batch_idx, output_size=(20, 20)):
    spatial_scale = feature_map.shape[2] / 640
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    batch_index = torch.zeros((boxes.size(0), 1), dtype=torch.float32, device=boxes.device)
    boxes = torch.cat([batch_index, boxes], dim=1)
    boxes = boxes.to(feature_map.device)
    roi_map = roi_align(feature_map, boxes, output_size=output_size, spatial_scale=spatial_scale)
    return roi_map

def compute_similarity(
    raw_preds, global_imgs,
    padded_cropped_imgs, cropped_coords,
    padded_zoomed_imgs, zoomed_coords,
    model,
    img_indices,
    top_n=5,
    iou_thresh=0.5
):
    device = global_imgs.device
    global_feats = model.model.model[:6](global_imgs)
    cropped_feats = model.model.model[:6](padded_cropped_imgs)
    zoomed_feats = model.model.model[:6](padded_zoomed_imgs)

    N_inst = padded_cropped_imgs.shape[0]
    loss_total = 0.0
    count = 0

    for i in range(N_inst):
        img_idx = int(img_indices[i].item())
        # Decodifica as caixas e scores para a imagem original
        boxes, scores = decode_boxes(raw_preds, img_idx, img_size=(640,640))
        if scores.numel() > top_n:
            _, top_idx = torch.topk(scores, top_n)
            boxes = boxes[top_idx]
            scores = scores[top_idx]
        if boxes.shape[0] == 0:
            continue

        # Ground truth box da instância i (no formato xyxy)
        gt_box = cropped_coords[i, 1:].unsqueeze(0).to(device)  # [1, 4]

        # Matching por IoU
        ious = ops.box_iou(gt_box, boxes.to(device))  # [1, N_pred]
        max_iou, max_idx = ious.max(dim=1)  # [1]
        if max_iou.item() < iou_thresh:
            # Nenhuma predição cobre suficientemente a instância
            continue

        # Extrai a feature da ROI matched
        matched_box = boxes[max_idx.item()].unsqueeze(0)  # [1, 4]
        roi_feat = extract_roi_features(global_feats[img_idx].unsqueeze(0), matched_box, img_idx, output_size=(80, 80))
        roi_feat = F.adaptive_avg_pool2d(roi_feat, (1, 1)).view(1, -1)

        # Features do crop e zoomed da instância i
        ref_cropped = extract_roi_features(cropped_feats[i].unsqueeze(0), cropped_coords[i, 1:].unsqueeze(0), 0, output_size=(80, 80))
        ref_zoomed  = extract_roi_features(zoomed_feats[i].unsqueeze(0), zoomed_coords[i, 1:].unsqueeze(0), 0, output_size=(80, 80))
        ref_cropped = F.adaptive_avg_pool2d(ref_cropped, (1, 1)).view(1, -1)
        ref_zoomed  = F.adaptive_avg_pool2d(ref_zoomed, (1, 1)).view(1, -1)

        # Similaridade positiva e negativa
        sim_pos = F.cosine_similarity(roi_feat, ref_cropped, dim=1)
        sim_bg  = F.cosine_similarity(roi_feat, ref_zoomed, dim=1)
        loss = (1 - sim_pos) + sim_bg

        loss_total += loss
        count += 1

    return loss_total / count if count > 0 else torch.tensor(0.0, device=device)