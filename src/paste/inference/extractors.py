import numpy as np


def extract_recommendations_from_heatmap(heatmap, image_size, patch_size, top_k=5):
    """
    Extrai coordenadas de recomendação do mapa de calor.
    
    Args:
        heatmap: [H, W] tensor
        image_size: (H_img, W_img)
        patch_size: (h_patch, w_patch)
        top_k: número máximo de recomendações
        
    Returns:
        Lista de dicionários com recomendações
    """
    h, w = heatmap.shape
    patch_h, patch_w = patch_size

    y_scale = image_size[0] / h
    x_scale = image_size[1] / w

    scores = []
    for y in range(0, h - patch_h + 1, max(1, patch_h // 4)):
        for x in range(0, w - patch_w + 1, max(1, patch_w // 4)):
            window = heatmap[y:y+patch_h, x:x+patch_w]
            score = np.mean(window.cpu().numpy())
            scores.append((score, x, y))

    scores = sorted(scores, key=lambda tup: tup[0], reverse=True)
    recommendations = []
    used_boxes = []

    for i, (score, x, y) in enumerate(scores):
        center_y = int((y + patch_h // 2) * y_scale)
        center_x = int((x + patch_w // 2) * x_scale)

        half_h = int(patch_h * y_scale // 2)
        half_w = int(patch_w * x_scale // 2)
        box = [
            max(0, center_x - half_w),
            max(0, center_y - half_h),
            min(image_size[1], center_x + half_w),
            min(image_size[0], center_y + half_h)
        ]

        # NMS: evita boxes sobrepostas
        overlap = False
        for prev_box in used_boxes:
            iou = _calculate_iou(box, prev_box)
            if iou > 0.3:
                overlap = True
                break
        if overlap:
            continue

        recommendations.append({
            "box": box,
            "score": float(score),
            "rank": len(recommendations) + 1
        })
        used_boxes.append(box)

        if len(recommendations) >= top_k:
            break

    return recommendations


def _calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes"""
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