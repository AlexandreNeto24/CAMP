import torch
import torchvision.ops as ops
from torchvision.ops import roi_align
import torch.nn.functional as F

def decode_boxes(raw_preds, img_idx, img_size=(640,640)):
    # raw_preds[0][img_idx] tem shape [5, N]; transpõe para [N, 5]
    preds = raw_preds[0][img_idx].permute(1, 0).contiguous()  # [N, 5]
    
    x1 = preds[:, 0] - preds[:, 2] / 2
    y1 = preds[:, 1] - preds[:, 3] / 2
    x2 = preds[:, 0] + preds[:, 2] / 2
    y2 = preds[:, 1] + preds[:, 3] / 2
    
    conf = preds[:, 4]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [N, 4]

    return boxes, conf

def extract_roi_features(feature_map, boxes, batch_idx, output_size=(20, 20)):
    spatial_scale = feature_map.shape[2] / 640  # Assumindo que o feature_map é [B, C, H, W]

    # Garante que boxes tenha shape [N, 4]
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)  # de [4] para [1, 4]

    # batch_index vai ser um full so de zeros do tamanho de boxes
    batch_index = torch.zeros((boxes.size(0), 1), dtype=torch.float32, device=boxes.device)
    boxes = torch.cat([batch_index, boxes], dim=1)  # [N, 5]
    
    # check if they are in the same device
    boxes = boxes.to(feature_map.device)
    roi_map = roi_align(feature_map, boxes, output_size=output_size, spatial_scale=spatial_scale)
    return roi_map

def compute_similarity(
    raw_preds, global_imgs,
    cropped_imgs, cropped_coords,
    zoomed_imgs, zoomed_coords,
    model, top_n=50, iou_thresh=0.7
    ):
    """
    Para cada imagem no batch, realiza os seguintes passos:
      1. Decodifica as raw_preds para obter as caixas (bounding boxes) e os scores.
      2. Filtra as top N caixas com maior confiança.
      3. Aplica NMS para remover caixas sobrepostas (usando apenas as coordenadas e scores).
      4. Usa as caixas filtradas para extrair os embeddings da região de interesse (ROI) 
         do mapa de features global (obtido do backbone).
      5. Calcula a similaridade (cosine similarity) entre o embedding da ROI e os embeddings
         do objeto (OI) e do background (OB) da mesma imagem.
    Retorna a loss média de similaridade para o batch.
    """
    batch_size = global_imgs.shape[0]
    loss_total = 0.0
    count = 0
    batch_loss = 0.0
    # mode.model.model to device
    model.model.model.to(global_imgs.device)

    # 1. Extrai o mapa de features global para o batch (usando as mesmas camadas do backbone)
    global_feats = model.model.model[:6](global_imgs)  # [B, D, H, W]

    # Extrair os embeddings para OI e OB
    cropped_feats = model.model.model[:6](cropped_imgs)
    zoomed_feats = model.model.model[:6](zoomed_imgs)

    # Processa imagem a imagem
    for img_idx in range(batch_size):
        # print(f'Processing image {img_idx}')
        # 2. Decodifica as caixas e os scores para a imagem atual
        boxes, scores = decode_boxes(raw_preds, img_idx, img_size=(640,640))

        # 3. Filtra as top N caixas com maior confiança
        if scores.numel() > top_n:
            _, top_idx = torch.topk(scores, top_n)
            boxes = boxes[top_idx]
            scores = scores[top_idx]
        # print(f'After top_n filtering: {boxes.shape[0]} boxes')

        # 4. Aplica NMS para remover caixas sobrepostas.
        # Como boxes tem formato [N, 5] com a primeira coluna sendo o índice da imagem (que é 0 aqui),
        # removemos essa coluna para o NMS, que espera um tensor [N, 4].
        keep_idx = ops.nms(boxes[:, 0:], scores, iou_thresh)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        # print(f'After NMS: {boxes.shape[0]} boxes')

        if boxes.shape[0] == 0:
            continue

        # 5. Extração das ROI features usando o mapa de features global.
        # global_feats[img_idx] tem shape [D, H, W]; adicionamos a dimensão de batch: [1, D, H, W].
        roi_feats = extract_roi_features(global_feats[img_idx].unsqueeze(0), boxes, img_idx, output_size=(80, 80))
        
        # remove the first collumn from cropped and zoomed coords that represents the id class
        current_cropped_coords = cropped_coords[img_idx, 1:]
        current_zoomed_coords = zoomed_coords[img_idx, 1:]

        # get the feature maps from the padding patches
        ref_cropped = extract_roi_features(cropped_feats[img_idx].unsqueeze(0), current_cropped_coords.unsqueeze(0), img_idx, output_size=(80, 80))
        ref_zoomed  = extract_roi_features(zoomed_feats[img_idx].unsqueeze(0), current_zoomed_coords.unsqueeze(0), img_idx, output_size=(80, 80))
        
        # reduce the number of features
        roi_feats = F.adaptive_avg_pool2d(roi_feats, (1, 1)).view(roi_feats.size(0), -1)  # [N_final, D]
        ref_cropped = F.adaptive_avg_pool2d(ref_cropped, (1, 1)).view(ref_cropped.size(0), -1)  # [1, D]
        ref_zoomed  = F.adaptive_avg_pool2d(ref_zoomed, (1, 1)).view(ref_zoomed.size(0), -1)  # [1, D]

        # 7. Para cada caixa filtrada, calcular a similaridade
        for box_feats in roi_feats:
            box_feats = box_feats.unsqueeze(0)
            sim_obj = F.cosine_similarity(box_feats, ref_cropped, dim=1)  # scalar
            sim_bg  = F.cosine_similarity(box_feats, ref_zoomed, dim=1)
            sim_avg = (sim_obj + sim_bg) / 2.0
            batch_loss += (1 - sim_avg)
            count += 1
        loss_total += (batch_loss / count)  # mean for the batch
        # mean for

    # sim loss mean for the batch
    loss_total = loss_total / batch_size

    # return loss_total / count if count > 0 else torch.tensor(0.0, device=device)
    return loss_total if count > 0 else torch.tensor(0.0, device=global_imgs.device)