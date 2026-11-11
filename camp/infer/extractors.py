"""Turn a placement heatmap into boxes.

The heatmap is already a probability. Windows start at
``range(0, size - patch + 1, max(1, patch // 4))``, the score is the window
mean, and greedy NMS drops overlaps above IoU 0.3. Box order is
``(x1, y1, x2, y2)`` with x along width and y along height.
"""

import numpy as np
import torch


def extract_recommendations_from_heatmap(
    heatmap, image_size, patch_size, top_k=5, output_size=None
):
    """Return up to ``top_k`` non-overlapping boxes on ``heatmap``.

    ``heatmap`` is ``[H, W]``. ``image_size`` is ``(height, width)`` of the
    image the box coordinates are mapped into. ``patch_size`` is
    ``(patch_h, patch_w)`` in heatmap pixels. ``output_size`` is
    ``(height, width)`` of the box in image pixels; when it is omitted the
    box uses the patch projected onto the image.
    """
    if torch.is_tensor(heatmap):
        heatmap_np = heatmap.detach().cpu().numpy()
    else:
        heatmap_np = np.asarray(heatmap)

    height, width = heatmap_np.shape
    patch_h, patch_w = patch_size

    y_scale = image_size[0] / height
    x_scale = image_size[1] / width
    stride_y = max(1, patch_h // 4)
    stride_x = max(1, patch_w // 4)

    scores = []
    for y in range(0, height - patch_h + 1, stride_y):
        for x in range(0, width - patch_w + 1, stride_x):
            window = heatmap_np[y:y + patch_h, x:x + patch_w]
            scores.append((np.mean(window), x, y))

    scores = sorted(scores, key=lambda tup: tup[0], reverse=True)
    recommendations = []
    used_boxes = []

    for score, x, y in scores:
        center_y = int((y + patch_h // 2) * y_scale)
        center_x = int((x + patch_w // 2) * x_scale)

        if output_size is not None:
            out_h, out_w = output_size
            half_h = int(out_h / 2)
            half_w = int(out_w / 2)
        else:
            half_h = int(patch_h * y_scale / 2)
            half_w = int(patch_w * x_scale / 2)

        box = [
            max(0, center_x - half_w),
            max(0, center_y - half_h),
            min(image_size[1], center_x + half_w),
            min(image_size[0], center_y + half_h),
        ]

        overlap = False
        for prev_box in used_boxes:
            if _calculate_iou(box, prev_box) > 0.3:
                overlap = True
                break
        if overlap:
            continue

        recommendations.append({
            "box": box,
            "score": float(score),
            "rank": len(recommendations) + 1,
        })
        used_boxes.append(box)

        if len(recommendations) >= top_k:
            break

    return recommendations


def _calculate_iou(box1, box2):
    """Intersection over union for ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area > 0:
        return inter_area / union_area
    return 0
