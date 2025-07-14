import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align, box_iou

from src.paste.utils.similarity_module import similarity_loss_from_matrix
from src.paste.utils.context_head import ContextEmbeddingHead
from src.paste.utils.feature_extractors import (
    TextureExtractor, 
    ColorHistogramExtractor, 
    ShapeFeatureExtractor,
    MultiModalFusion
)

def decode_boxes(raw_preds, img_idx, img_size=(640,640)):
    preds = raw_preds[0][img_idx]
    # Adapta para diferentes formatos
    if preds.ndim == 2:
        if preds.shape[0] == 5:
            preds = preds.permute(1, 0).contiguous()
    elif preds.ndim == 3:
        preds = preds.view(-1, 5)
    else:
        raise ValueError(f"Formato inesperado de preds: {preds.shape}")

    x1 = preds[:, 0] - preds[:, 2] / 2
    y1 = preds[:, 1] - preds[:, 3] / 2
    x2 = preds[:, 0] + preds[:, 2] / 2
    y2 = preds[:, 1] + preds[:, 3] / 2
    conf = preds[:, 4]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes, conf  # Adicionado retorno explícito

def select_top_confident_preds(pred_boxes, pred_scores, top_k=40):
    # Seleciona top_k por confiança, sem filtro de IoU
    if pred_scores.numel() == 0:
        return torch.empty((0, 4), device=pred_boxes.device), torch.empty((0,), device=pred_boxes.device)
    topk = min(top_k, pred_scores.size(0))
    topk_idx = torch.topk(pred_scores, topk).indices
    top_boxes = pred_boxes[topk_idx]
    top_scores = pred_scores[topk_idx]
    return top_boxes, top_scores  # Adicione o retorno

def select_top_confident_and_iou_preds(pred_boxes, pred_scores, gt_boxes, top_k=40, iou_thresh=0.01):
    # Seleciona top_k por confiança
    if pred_scores.numel() == 0:
        return torch.empty((0, 4), device=pred_boxes.device), torch.empty((0,), device=pred_boxes.device)
    topk = min(top_k, pred_scores.size(0))
    topk_idx = torch.topk(pred_scores, topk).indices
    top_boxes = pred_boxes[topk_idx]
    top_scores = pred_scores[topk_idx]
    # Calcula IoU com todas as GT
    if gt_boxes.numel() == 0:
        return top_boxes, top_scores  # devolve sempre algo
    ious = box_iou(top_boxes, gt_boxes)  # [top_k, n_gt]
    max_ious, _ = ious.max(dim=1)  # máximo IoU para cada predição
    # Filtra por IoU threshold
    keep = max_ious > iou_thresh
    if keep.sum() > 0:
        filtered_boxes = top_boxes[keep]
        filtered_scores = top_scores[keep]
    else:
        # Se nenhuma passa o threshold, devolve as top_k por confiança
        filtered_boxes = top_boxes
        filtered_scores = top_scores
    # Debug opcional:
    print("max_ious:", max_ious.cpu().numpy())
    return filtered_boxes, filtered_scores

def extract_roi_features(feature_map, boxes, batch_indices=None, output_size=(7, 7)):
    if boxes.numel() == 0:
        return torch.empty((0, feature_map.shape[1]), device=feature_map.device)
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

def extract_multi_scale_roi_features(backbone_features, boxes, batch_indices=None, output_size=(7, 7), N=2):
    """
    Extrai e concatena as últimas N features do backbone para cada ROI.
    backbone_features: lista de feature maps do backbone
    boxes: [num_rois, 4]
    Retorna: [num_rois, D], D = soma dos canais das N features
    """
    if boxes.numel() == 0:
        return torch.empty((0, sum([f.shape[1] for f in backbone_features[-N:]])), device=boxes.device)
    if batch_indices is None:
        batch_indices = torch.zeros((boxes.size(0),), dtype=torch.int, device=boxes.device)
    batch_indices = batch_indices.to(boxes.device)
    roi_feats = []
    for f in backbone_features[-N:]:
        rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)
        spatial_scale = f.shape[2] / 640
        try:
            feats = roi_align(f, rois, output_size=output_size, spatial_scale=spatial_scale)
        except Exception as e:
            print(f"[DEBUG] roi_align error: {e}")
            feats = torch.zeros((boxes.size(0), f.shape[1], output_size[0], output_size[1]), device=f.device)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        roi_feats.append(pooled)
    embedding = torch.cat(roi_feats, dim=1)
    embedding = F.normalize(embedding, dim=1)
    return embedding

def compute_similarity(
    raw_preds, global_imgs,
    padded_cropped_imgs, cropped_coords,
    padded_zoomed_imgs, zoomed_coords,
    model,
    img_indices,
    yolo_boxes=None,
    margin=1.0,
    top_k=100,
    iou_pos_thresh=0.1,
    iou_bg_thresh=0.02,
    context_head=None,
    alpha_color=0.1,
    alpha_size=0.1
):
    device = global_imgs.device
    backbone = model.model.model[:6]
    
    # Liberamos memória CUDA antes de operações pesadas
    torch.cuda.empty_cache()
    
    features = backbone(global_imgs)
    feature_maps = features if isinstance(features, (list, tuple)) else [features]
    feature_map = feature_maps[-2] if len(feature_maps) >= 2 else feature_maps[-1]
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)

    boxes, conf = decode_boxes(raw_preds, 0, img_size=(640, 640))

    # GT boxes
    if yolo_boxes is not None and yolo_boxes.numel() > 0:
        cx = yolo_boxes[:, 1] * 640
        cy = yolo_boxes[:, 2] * 640
        w = yolo_boxes[:, 3] * 640
        h = yolo_boxes[:, 4] * 640
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    else:
        gt_boxes = torch.empty((0, 4), device=device)

    # Usar sampling balanceado com limites
    selected_boxes, selected_scores, num_pos = select_positive_negative_balanced(
        boxes, conf, gt_boxes, pos_needed=4, neg_needed=16)
    
    if selected_boxes.numel() == 0:
        return torch.tensor(0.0, device=device)

    batch_indices = torch.zeros(selected_boxes.size(0), dtype=torch.int, device=device)

    # Clamp boxes para evitar índices fora dos limites
    selected_boxes = selected_boxes.clone()
    x1 = selected_boxes[:, 0].clamp(0, feature_map.shape[3] - 1)
    y1 = selected_boxes[:, 1].clamp(0, feature_map.shape[2] - 1)
    x2 = selected_boxes[:, 2].clamp(0, feature_map.shape[3] - 1)
    y2 = selected_boxes[:, 3].clamp(0, feature_map.shape[2] - 1)
    selected_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Limita o número de boxes total para evitar OOM
    max_boxes = 150
    if selected_boxes.size(0) > max_boxes:
        keep_idx = torch.randperm(selected_boxes.size(0), device=device)[:max_boxes]
        selected_boxes = selected_boxes[keep_idx]
        batch_indices = batch_indices[keep_idx]
        selected_scores = selected_scores[keep_idx]

    # Embeddings multi-escala (concatena últimas 2 features)
    embeddings = extract_multi_scale_roi_features(feature_maps, selected_boxes, batch_indices=batch_indices, output_size=(7,7), N=2)

    # Passa pelos context embeddings
    if context_head is not None:
        embeddings = context_head(embeddings)

    # NOVO: Extrai patches para processamento de textura e cor
    patches = []
    for i, box in enumerate(selected_boxes):
        x1, y1, x2, y2 = box.int().clamp(min=0)
        
        # Garantir dimensões mínimas
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
            
        img_idx = batch_indices[i]
        if img_idx < global_imgs.size(0):
            # Extrai patch da imagem global
            try:
                patch = global_imgs[img_idx:img_idx+1, :, y1:y2, x1:x2]
                # Redimensiona para tamanho fixo
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                patches.append(patch)
            except:
                # Fallback para casos problemáticos
                patch = torch.zeros(1, 3, 64, 64, device=device)
                patches.append(patch)

    # Características de textura, cor e forma
    if patches:
        patches_tensor = torch.cat(patches, dim=0)
        patches_tensor = patches_tensor.to(device)  # Garante que está no dispositivo correto
    
        # Verifica e move os extratores para o dispositivo correto
        if texture_extractor is not None:
            texture_extractor = texture_extractor.to(device)
            texture_features = texture_extractor(patches_tensor)
        else:
            texture_features = torch.zeros((patches_tensor.size(0), 8), device=device)
            
        if color_extractor is not None:
            color_extractor = color_extractor.to(device)
            color_features = color_extractor(patches_tensor)
        else:
            color_features = torch.zeros((patches_tensor.size(0), 32), device=device)
            
        if shape_extractor is not None:
            shape_extractor = shape_extractor.to(device)
            shape_features = shape_extractor(selected_boxes)
        else:
            shape_features = torch.zeros((selected_boxes.size(0), 16), device=device)
    
        # Fusion module
        if fusion_module is not None:
            fusion_module = fusion_module.to(device)
            combined_embeddings = fusion_module(embeddings, texture_features, color_features, shape_features)
        else:
            combined_embeddings = embeddings
    else:
        combined_embeddings = embeddings

    # Calcula IoU para cada predição
    if gt_boxes.numel() > 0:
        ious = box_iou(selected_boxes, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        labels = torch.full((selected_boxes.size(0),), -1, device=device)
        labels[max_ious > iou_pos_thresh] = 1
        labels[max_ious < iou_bg_thresh] = 0
        valid_mask = labels >= 0
        combined_embeddings = combined_embeddings[valid_mask]
        selected_boxes = selected_boxes[valid_mask]
        embeddings = embeddings[valid_mask]  # ADICIONE ESTA LINHA
        labels = labels[valid_mask]
    else:
        labels = torch.zeros(selected_boxes.size(0), device=device)

    # Só calcula a loss se houver pelo menos 1 positivo e 1 negativo
    pos_count = (labels == 1).sum().item()
    neg_count = (labels == 0).sum().item()
    print(f"Num positivos: {pos_count} Num negativos: {neg_count}")
    
    if pos_count < 1 or neg_count < 1:
        print("Poucos positivos/negativos para similarity loss.")
        
        # Tenta forçar inclusão de GT boxes
        if yolo_boxes is not None and yolo_boxes.numel() > 0:
            print("Forçando inclusão de GT boxes como positivos...")
            try:
                # Limitar número para evitar OOM
                max_gt = min(5, gt_boxes.size(0))
                if gt_boxes.size(0) > max_gt:
                    selected_gt_idx = torch.randperm(gt_boxes.size(0), device=device)[:max_gt]
                    selected_gt_boxes = gt_boxes[selected_gt_idx]
                else:
                    selected_gt_boxes = gt_boxes
                
                # Extrair embeddings das GT boxes
                gt_batch_indices = torch.zeros(selected_gt_boxes.size(0), dtype=torch.int, device=device)
                gt_embeddings = extract_multi_scale_roi_features(feature_maps, selected_gt_boxes, 
                                                               batch_indices=gt_batch_indices, 
                                                               output_size=(7,7), N=2)
                if context_head is not None:
                    gt_embeddings = context_head(gt_embeddings)
                
                # Usar negativos existentes + GT boxes como positivos
                if neg_count > 0:
                    # Versão segura: guarda índices negativos antes de modificar
                    neg_mask = (labels == 0)
                    if torch.any(neg_mask):
                        neg_indices = neg_mask.nonzero().squeeze(1)
                        neg_embeddings = embeddings[neg_indices]
                        neg_boxes = selected_boxes[neg_indices]
                        
                        # Concatena com segurança
                        combined_embeddings = torch.cat([gt_embeddings, neg_embeddings])
                        combined_labels = torch.cat([
                            torch.ones(gt_embeddings.size(0), device=device),
                            torch.zeros(neg_embeddings.size(0), device=device)
                        ])
                        combined_boxes = torch.cat([selected_gt_boxes, neg_boxes])
                        
                        print(f"Criados {gt_embeddings.size(0)} embeddings positivos e {neg_embeddings.size(0)} negativos")
                        
                        # Recalcula loss com novos tensores
                        return triplet_loss_from_embeddings(combined_embeddings, combined_boxes, combined_labels, margin)
            except Exception as e:
                print(f"Erro ao processar GT boxes: {e}")
        
        # Fallback se tudo falhar
        return torch.tensor(0.0, device=device)

    # Imprime distâncias para debug
    if pos_count >= 2 and neg_count >= 1:
        pos_embed = embeddings[labels == 1]
        neg_embed = embeddings[labels == 0]
        with torch.no_grad():
            if pos_embed.size(0) >= 2:
                # Distância média entre positivos
                pos_dist = torch.cdist(pos_embed[:1], pos_embed[1:]).mean().item()
                print(f"Distância média entre positivos: {pos_dist:.4f}")
            
            if pos_embed.size(0) >= 1 and neg_embed.size(0) >= 1:
                # Distância média entre positivos e negativos
                neg_dist = torch.cdist(pos_embed[:1], neg_embed).mean().item()
                print(f"Distância média entre pos-neg: {neg_dist:.4f}")
    
    # Usa triplet loss com limites de memória
    sim_loss = triplet_loss_from_embeddings(embeddings, selected_boxes, labels, margin=1.0, max_triplets=1000)
    
    # Liberamos memória novamente
    torch.cuda.empty_cache()
    
    # Se não temos suficientes positivos por batch, força a inclusão de GT boxes
    if pos_count < 2:
        print("Poucos positivos detectados, adicionando GT boxes...")
        try:
            # LIMITAR MUITO MAIS GT BOXES para evitar OOM
            max_gt = min(8, gt_boxes.size(0))
            if gt_boxes.size(0) > max_gt:
                selected_gt_idx = torch.randperm(gt_boxes.size(0), device=device)[:max_gt]
                gt_boxes = gt_boxes[selected_gt_idx]
        
            # Libera memória antes de operação pesada
            torch.cuda.empty_cache()
        
            # Converte GT boxes para embeddings
            gt_batch_indices = torch.zeros(gt_boxes.size(0), dtype=torch.int, device=device)
            gt_embeddings = extract_multi_scale_roi_features(feature_maps, gt_boxes, batch_indices=gt_batch_indices, output_size=(7,7), N=2)
            if context_head is not None:
                gt_embeddings = context_head(gt_embeddings)
            
            # Adiciona aos embeddings existentes com labels=1
            embeddings = torch.cat([embeddings, gt_embeddings])
            labels = torch.cat([labels, torch.ones(gt_boxes.size(0), device=device)])
            
            pos_count = (labels == 1).sum().item()
            print(f"Agora temos {pos_count} positivos")
            
            # Recalcula a loss com os novos embeddings
            sim_loss = triplet_loss_from_embeddings(embeddings, torch.cat([selected_boxes, gt_boxes]), labels, margin=margin, max_triplets=1000)
        except Exception as e:
            print(f"Erro ao adicionar GT boxes: {e}")
            # Em caso de erro, mantém a loss original ou retorna zero
            sim_loss = torch.tensor(0.0, device=device)

    # Garantir que sempre retornamos um tensor
    if sim_loss is None:
        sim_loss = torch.tensor(0.0, device=device)

    return sim_loss

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

def color_histogram_loss(patch_imgs, region_imgs, bins=32):
    # patch_imgs, region_imgs: [N, 3, H, W], valores 0-1
    losses = []
    for patch, region in zip(patch_imgs, region_imgs):
        patch_hist = torch.histc(patch, bins=bins, min=0, max=1)
        region_hist = torch.histc(region, bins=bins, min=0, max=1)
        patch_hist = patch_hist / (patch_hist.sum() + 1e-8)
        region_hist = region_hist / (region_hist.sum() + 1e-8)
        losses.append(F.mse_loss(patch_hist, region_hist))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=patch_imgs.device)

def select_positive_negative_balanced(boxes, conf, gt_boxes, pos_needed=4, neg_needed=12):
    """
    Garante positivos incluindo as próprias caixas GT ou caixas próximas
    """
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.empty((0, 4), device=boxes.device), torch.empty((0,), device=boxes.device), 0
    
    device = boxes.device
    ious = box_iou(boxes, gt_boxes)
    max_ious, _ = ious.max(dim=1)
    
    # MUITO MAIS PERMISSIVO: 0.03 em vez de 0.1 ou 0.2
    pos_mask = max_ious > 0.03
    pos_boxes = boxes[pos_mask]
    pos_scores = conf[pos_mask]
    
    # SEMPRE inclui algumas caixas como positivas, pegando as top-IoU
    top_n = min(pos_needed, max_ious.size(0))
    top_iou_values, top_iou_indices = torch.topk(max_ious, top_n)
    pos_boxes = torch.cat([pos_boxes, boxes[top_iou_indices]], dim=0)
    pos_scores = torch.cat([pos_scores, conf[top_iou_indices]], dim=0)
    
    # Adiciona as próprias GT boxes (limitadas para evitar OOM)
    max_gt = min(8, gt_boxes.size(0))
    if gt_boxes.size(0) > max_gt:
        idx = torch.randperm(gt_boxes.size(0), device=device)[:max_gt]
        gt_boxes_subset = gt_boxes[idx]
        gt_scores_subset = torch.ones(max_gt, device=device)
    else:
        gt_boxes_subset = gt_boxes
        gt_scores_subset = torch.ones(gt_boxes.size(0), device=device)
    
    pos_boxes = torch.cat([pos_boxes, gt_boxes_subset], dim=0)
    pos_scores = torch.cat([pos_scores, gt_scores_subset], dim=0)
    
    # Remove duplicatas e limita número máximo
    if pos_boxes.size(0) > 1:
        # Remove duplicatas com IoU muito alto
        keep = []
        pos_ious = box_iou(pos_boxes, pos_boxes)
        for i in range(pos_boxes.size(0)):
            if i == 0 or (pos_ious[i, :i] < 0.9).all():
                keep.append(i)
        pos_boxes = pos_boxes[keep]
        pos_scores = pos_scores[keep]
    
    # Limita positivos para evitar OOM
    if pos_boxes.size(0) > 15:
        keep_idx = torch.randperm(pos_boxes.size(0), device=device)[:15]
        pos_boxes = pos_boxes[keep_idx]
        pos_scores = pos_scores[keep_idx]
    
    # Negativos também mais permissivos
    neg_mask = max_ious < 0.03
    neg_boxes = boxes[neg_mask]
    neg_scores = conf[neg_mask]
    
    # Limita número de negativos
    if neg_boxes.size(0) > neg_needed:
        top_neg_idx = torch.topk(neg_scores, neg_needed).indices
        neg_boxes = neg_boxes[top_neg_idx]
        neg_scores = neg_scores[top_neg_idx]
    
    # Combina positivos e negativos
    selected_boxes = torch.cat([pos_boxes, neg_boxes]) if neg_boxes.size(0) > 0 else pos_boxes
    selected_scores = torch.cat([pos_scores, neg_scores]) if neg_scores.size(0) > 0 else pos_scores
    
    return selected_boxes, selected_scores, pos_boxes.size(0)  # Adicionado retorno completo

def triplet_loss_from_embeddings(embeddings, boxes, labels, margin=1.0, max_triplets=1000):
    """
    Calcula triplet loss para embeddings com labels 1 (positivos) e 0 (negativos)
    """
    device = embeddings.device
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Se não temos positivos ou negativos, retorna zero
    if pos_mask.sum() < 1 or neg_mask.sum() < 1:
        return torch.tensor(0.0, device=device)
    
    pos_embeddings = embeddings[pos_mask]
    neg_embeddings = embeddings[neg_mask]
    
    # Se tivermos apenas 1 positivo, usamos uma estratégia diferente
    if pos_embeddings.size(0) == 1:
        # Usamos o único positivo como âncora
        anchor = pos_embeddings
        
        # Calculamos a distância para todos os negativos
        neg_dist = torch.cdist(anchor, neg_embeddings)
        
        # Queremos que todos os negativos estejam a pelo menos margin de distância
        hard_neg_loss = F.relu(margin - neg_dist).mean()
        return hard_neg_loss
    
    # Limitar número de amostras para evitar OOM
    if pos_embeddings.size(0) > 15:
        idx = torch.randperm(pos_embeddings.size(0), device=device)[:15]
        pos_embeddings = pos_embeddings[idx]
    
    if neg_embeddings.size(0) > 20:
        idx = torch.randperm(neg_embeddings.size(0), device=device)[:20]
        neg_embeddings = neg_embeddings[idx]
    
    # Calcular distâncias entre positivos
    pos_dist = torch.cdist(pos_embeddings, pos_embeddings)
    # Remover diagonais (distância de um embedding para si mesmo)
    mask = ~torch.eye(pos_embeddings.shape[0], dtype=torch.bool, device=device)
    pos_dist = pos_dist[mask].reshape(-1)
    
    # Calcular distâncias entre positivos e negativos
    pos_neg_dist = torch.cdist(pos_embeddings, neg_embeddings).reshape(-1)
    
    # Triplet loss simplificada
    pos_loss = pos_dist.mean() if pos_dist.numel() > 0 else torch.tensor(0.0, device=device)
    neg_loss = F.relu(margin - pos_neg_dist).mean()
    
    return pos_loss + neg_loss

def compute_similarity_enhanced(
    raw_preds, global_imgs,
    padded_cropped_imgs, cropped_coords,
    padded_zoomed_imgs, zoomed_coords,
    model,
    img_indices,
    yolo_boxes=None,
    margin=1.0,
    top_k=100,
    iou_pos_thresh=0.03,
    iou_bg_thresh=0.01,
    context_head=None,
    texture_extractor=None,
    color_extractor=None,
    shape_extractor=None,
    fusion_module=None
):
    device = global_imgs.device
    backbone = model.model.model[:6]
    
    # Liberamos memória CUDA antes de operações pesadas
    torch.cuda.empty_cache()
    
    features = backbone(global_imgs)
    feature_maps = features if isinstance(features, (list, tuple)) else [features]
    feature_map = feature_maps[-2] if len(feature_maps) >= 2 else feature_maps[-1]
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)

    boxes, conf = decode_boxes(raw_preds, 0, img_size=(640, 640))

    # GT boxes
    if yolo_boxes is not None and yolo_boxes.numel() > 0:
        cx = yolo_boxes[:, 1] * 640
        cy = yolo_boxes[:, 2] * 640
        w = yolo_boxes[:, 3] * 640
        h = yolo_boxes[:, 4] * 640
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    else:
        gt_boxes = torch.empty((0, 4), device=device)

    # Usar sampling balanceado
    selected_boxes, selected_scores, num_pos = select_positive_negative_balanced(
        boxes, conf, gt_boxes, pos_needed=4, neg_needed=16)
    
    if selected_boxes.numel() == 0:
        return torch.tensor(0.0, device=device)

    batch_indices = torch.zeros(selected_boxes.size(0), dtype=torch.int, device=device)

    # Clamp boxes para evitar índices fora dos limites
    selected_boxes = selected_boxes.clone()
    x1 = selected_boxes[:, 0].clamp(0, feature_map.shape[3] - 1)
    y1 = selected_boxes[:, 1].clamp(0, feature_map.shape[2] - 1)
    x2 = selected_boxes[:, 2].clamp(0, feature_map.shape[3] - 1)
    y2 = selected_boxes[:, 3].clamp(0, feature_map.shape[2] - 1)
    selected_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Limita o número de boxes total para evitar OOM
    max_boxes = 150
    if selected_boxes.size(0) > max_boxes:
        keep_idx = torch.randperm(selected_boxes.size(0), device=device)[:max_boxes]
        selected_boxes = selected_boxes[keep_idx]
        batch_indices = batch_indices[keep_idx]
        selected_scores = selected_scores[keep_idx]

    # Embeddings multi-escala (concatena últimas 2 features)
    embeddings = extract_multi_scale_roi_features(feature_maps, selected_boxes, batch_indices=batch_indices, output_size=(7,7), N=2)

    # Passa pelos context embeddings
    if context_head is not None:
        embeddings = context_head(embeddings)

    # NOVO: Extrai patches para processamento de textura e cor
    patches = []
    for i, box in enumerate(selected_boxes):
        x1, y1, x2, y2 = box.int().clamp(min=0)
        
        # Garantir dimensões mínimas
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
            
        img_idx = batch_indices[i]
        if img_idx < global_imgs.size(0):
            # Extrai patch da imagem global
            try:
                patch = global_imgs[img_idx:img_idx+1, :, y1:y2, x1:x2]
                # Redimensiona para tamanho fixo
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                patches.append(patch)
            except:
                # Fallback para casos problemáticos
                patch = torch.zeros(1, 3, 64, 64, device=device)
                patches.append(patch)

    # Características de textura, cor e forma
    if patches:
        patches_tensor = torch.cat(patches, dim=0)
        patches_tensor = patches_tensor.to(device)  # Garante que está no dispositivo correto
    
        # Verifica e move os extratores para o dispositivo correto
        if texture_extractor is not None:
            texture_extractor = texture_extractor.to(device)
            texture_features = texture_extractor(patches_tensor)
        else:
            texture_features = torch.zeros((patches_tensor.size(0), 8), device=device)
            
        if color_extractor is not None:
            color_extractor = color_extractor.to(device)
            color_features = color_extractor(patches_tensor)
        else:
            color_features = torch.zeros((patches_tensor.size(0), 32), device=device)
            
        if shape_extractor is not None:
            shape_extractor = shape_extractor.to(device)
            shape_features = shape_extractor(selected_boxes)
        else:
            shape_features = torch.zeros((selected_boxes.size(0), 16), device=device)
    
        # Fusion module
        if fusion_module is not None:
            fusion_module = fusion_module.to(device)
            combined_embeddings = fusion_module(embeddings, texture_features, color_features, shape_features)
        else:
            combined_embeddings = embeddings
    else:
        combined_embeddings = embeddings

    # Calcula IoU para cada predição
    if gt_boxes.numel() > 0:
        ious = box_iou(selected_boxes, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        labels = torch.full((selected_boxes.size(0),), -1, device=device)
        labels[max_ious > iou_pos_thresh] = 1
        labels[max_ious < iou_bg_thresh] = 0
        valid_mask = labels >= 0
        combined_embeddings = combined_embeddings[valid_mask]
        selected_boxes = selected_boxes[valid_mask]
        embeddings = embeddings[valid_mask]  # ADICIONE ESTA LINHA
        labels = labels[valid_mask]
    else:
        labels = torch.zeros(selected_boxes.size(0), device=device)

    # Só calcula a loss se houver pelo menos 1 positivo e 1 negativo
    pos_count = (labels == 1).sum().item()
    neg_count = (labels == 0).sum().item()
    print(f"Num positivos: {pos_count} Num negativos: {neg_count}")
    
    if pos_count < 1 or neg_count < 1:
        print("Poucos positivos/negativos para similarity loss.")
        
        # Tenta forçar inclusão de GT boxes
        if yolo_boxes is not None and yolo_boxes.numel() > 0:
            print("Forçando inclusão de GT boxes como positivos...")
            try:
                # Limitar número para evitar OOM
                max_gt = min(5, gt_boxes.size(0))
                if gt_boxes.size(0) > max_gt:
                    selected_gt_idx = torch.randperm(gt_boxes.size(0), device=device)[:max_gt]
                    selected_gt_boxes = gt_boxes[selected_gt_idx]
                else:
                    selected_gt_boxes = gt_boxes
                
                # Extrair embeddings das GT boxes
                gt_batch_indices = torch.zeros(selected_gt_boxes.size(0), dtype=torch.int, device=device)
                gt_embeddings = extract_multi_scale_roi_features(feature_maps, selected_gt_boxes, 
                                                               batch_indices=gt_batch_indices, 
                                                               output_size=(7,7), N=2)
                if context_head is not None:
                    gt_embeddings = context_head(gt_embeddings)
                
                # Usar negativos existentes + GT boxes como positivos
                if neg_count > 0:
                    # Versão segura: guarda índices negativos antes de modificar
                    neg_mask = (labels == 0)
                    if torch.any(neg_mask):
                        neg_indices = neg_mask.nonzero().squeeze(1)
                        neg_embeddings = embeddings[neg_indices]
                        neg_boxes = selected_boxes[neg_indices]
                        
                        # Concatena com segurança
                        combined_embeddings = torch.cat([gt_embeddings, neg_embeddings])
                        combined_labels = torch.cat([
                            torch.ones(gt_embeddings.size(0), device=device),
                            torch.zeros(neg_embeddings.size(0), device=device)
                        ])
                        combined_boxes = torch.cat([selected_gt_boxes, neg_boxes])
                        
                        print(f"Criados {gt_embeddings.size(0)} embeddings positivos e {neg_embeddings.size(0)} negativos")
                        
                        # Recalcula loss com novos tensores
                        return triplet_loss_from_embeddings(combined_embeddings, combined_boxes, combined_labels, margin)
            except Exception as e:
                print(f"Erro ao processar GT boxes: {e}")
        
        # Fallback se tudo falhar
        return torch.tensor(0.0, device=device)

    # Imprime distâncias para debug
    if pos_count >= 2 and neg_count >= 1:
        pos_embed = embeddings[labels == 1]
        neg_embed = embeddings[labels == 0]
        with torch.no_grad():
            if pos_embed.size(0) >= 2:
                # Distância média entre positivos
                pos_dist = torch.cdist(pos_embed[:1], pos_embed[1:]).mean().item()
                print(f"Distância média entre positivos: {pos_dist:.4f}")
            
            if pos_embed.size(0) >= 1 and neg_embed.size(0) >= 1:
                # Distância média entre positivos e negativos
                neg_dist = torch.cdist(pos_embed[:1], neg_embed).mean().item()
                print(f"Distância média entre pos-neg: {neg_dist:.4f}")
    
    # Usa triplet loss com limites de memória
    sim_loss = triplet_loss_from_embeddings(embeddings, selected_boxes, labels, margin=1.0, max_triplets=1000)
    
    # Liberamos memória novamente
    torch.cuda.empty_cache()
    
    # Se não temos suficientes positivos por batch, força a inclusão de GT boxes
    if pos_count < 2:
        print("Poucos positivos detectados, adicionando GT boxes...")
        try:
            # LIMITAR MUITO MAIS GT BOXES para evitar OOM
            max_gt = min(8, gt_boxes.size(0))
            if gt_boxes.size(0) > max_gt:
                selected_gt_idx = torch.randperm(gt_boxes.size(0), device=device)[:max_gt]
                gt_boxes = gt_boxes[selected_gt_idx]
        
            # Libera memória antes de operação pesada
            torch.cuda.empty_cache()
        
            # Converte GT boxes para embeddings
            gt_batch_indices = torch.zeros(gt_boxes.size(0), dtype=torch.int, device=device)
            gt_embeddings = extract_multi_scale_roi_features(feature_maps, gt_boxes, batch_indices=gt_batch_indices, output_size=(7,7), N=2)
            if context_head is not None:
                gt_embeddings = context_head(gt_embeddings)
            
            # Adiciona aos embeddings existentes com labels=1
            embeddings = torch.cat([embeddings, gt_embeddings])
            labels = torch.cat([labels, torch.ones(gt_boxes.size(0), device=device)])
            
            pos_count = (labels == 1).sum().item()
            print(f"Agora temos {pos_count} positivos")
            
            # Recalcula a loss com os novos embeddings
            sim_loss = triplet_loss_from_embeddings(embeddings, torch.cat([selected_boxes, gt_boxes]), labels, margin=margin, max_triplets=1000)
        except Exception as e:
            print(f"Erro ao adicionar GT boxes: {e}")
            # Em caso de erro, mantém a loss original ou retorna zero
            sim_loss = torch.tensor(0.0, device=device)

    # Garantir que sempre retornamos um tensor
    if sim_loss is None:
        sim_loss = torch.tensor(0.0, device=device)

    return sim_loss

def triplet_loss_adaptive(embeddings, boxes, labels, margin=1.0):
    """
    Versão adaptativa do triplet loss que ajusta dinamicamente os pesos com base 
    na distribuição das distâncias entre embeddings positivos e negativos.
    Implementa hard-negative mining e calibração de pesos.
    """
    device = embeddings.device
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Se não temos positivos ou negativos, retorna zero
    if pos_mask.sum() < 1 or neg_mask.sum() < 1:
        return torch.tensor(0.0, device=device)
    
    pos_embeddings = embeddings[pos_mask]
    neg_embeddings = embeddings[neg_mask]
    
    # Se tivermos apenas 1 positivo, usamos uma estratégia diferente
    if pos_embeddings.size(0) == 1:
        # Usamos o único positivo como âncora
        anchor = pos_embeddings
        
        # Calculamos a distância para todos os negativos
        neg_dist = torch.cdist(anchor, neg_embeddings)
        
        # Hard negative mining: foca nos negativos mais difíceis (próximos ao positivo)
        k = min(10, neg_dist.size(1))
        hard_neg_dist, _ = neg_dist.topk(k, dim=1, largest=False)
        hard_neg_loss = F.relu(margin - hard_neg_dist).mean()
        return hard_neg_loss
    
    # Limitar número de amostras para evitar OOM
    if pos_embeddings.size(0) > 15:
        idx = torch.randperm(pos_embeddings.size(0), device=device)[:15]
        pos_embeddings = pos_embeddings[idx]
    
    if neg_embeddings.size(0) > 20:
        idx = torch.randperm(neg_embeddings.size(0), device=device)[:20]
        neg_embeddings = neg_embeddings[idx]
    
    # Calcular distâncias entre positivos
    pos_dist = torch.cdist(pos_embeddings, pos_embeddings)
    # Remover diagonais (distância de um embedding para si mesmo)
    mask = ~torch.eye(pos_embeddings.shape[0], dtype=torch.bool, device=device)
    pos_dist = pos_dist[mask].reshape(-1)
    
    # Calcular distâncias entre positivos e negativos
    pos_neg_dist = torch.cdist(pos_embeddings, neg_embeddings)
    
    # Hard negative mining: para cada positivo, encontra os k negativos mais difíceis
    k = min(5, pos_neg_dist.size(1))
    hard_neg_dist = torch.topk(pos_neg_dist, k=k, dim=1, largest=False)[0]
    
    # Pesos adaptativos baseados na variância das distâncias
    pos_var = pos_dist.var() if pos_dist.numel() > 0 else torch.tensor(1.0, device=device)
    neg_var = hard_neg_dist.var() if hard_neg_dist.numel() > 0 else torch.tensor(1.0, device=device)
    
    # Pesos adaptativos: quanto menor a variância, maior o peso
    pos_weight = 1.0 / (pos_var.sqrt() + 1e-6)
    neg_weight = 1.0 / (neg_var.sqrt() + 1e-6)
    
    # Normaliza pesos
    total_weight = pos_weight + neg_weight
    pos_weight = pos_weight / total_weight
    neg_weight = neg_weight / total_weight
    
    # Calibração de pesos baseada no desempenho atual
    pos_dist_mean = pos_dist.mean() if pos_dist.numel() > 0 else torch.tensor(0.0, device=device)
    neg_dist_mean = hard_neg_dist.mean() if hard_neg_dist.numel() > 0 else torch.tensor(0.0, device=device)
    
    # Se os positivos estão mais separados que os negativos, focamos mais nos positivos
    if pos_dist_mean > neg_dist_mean:
        pos_weight *= 1.5
        neg_weight *= 0.5
    
    # Triplet loss com pesos adaptativos
    pos_loss = pos_dist.mean() if pos_dist.numel() > 0 else torch.tensor(0.0, device=device)
    neg_loss = F.relu(margin - hard_neg_dist).mean() if hard_neg_dist.numel() > 0 else torch.tensor(0.0, device=device)
    
    # Aplicação dos pesos
    weighted_loss = pos_weight * pos_loss + neg_weight * neg_loss
    
    return weighted_loss

def analyze_similarity_components(image_tensor, boxes, model, texture_extractor, 
                               color_extractor, shape_extractor, fusion_module):
    """
    Analisa e compara os diferentes componentes de similaridade (embedding, textura, cor, forma)
    para cada par de detecções na imagem.
    """
    device = image_tensor.device
    backbone = model.model.model[:6]
    
    # Extrai features do backbone
    with torch.no_grad():
        features = backbone(image_tensor)
        feature_maps = features if isinstance(features, (list, tuple)) else [features]
        
        batch_indices = torch.zeros(boxes.size(0), dtype=torch.int, device=device)
        
        # Extrai embeddings para cada box
        embeddings = extract_multi_scale_roi_features(
            feature_maps, boxes, batch_indices=batch_indices, output_size=(7,7), N=2)
        
        # Extrai patches para textura/cor
        patches = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().clamp(min=0)
            if x2 <= x1: x2 = x1 + 1
            if y2 <= y1: y2 = y1 + 1
            
            try:
                patch = image_tensor[:, :, y1:y2, x1:x2]
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                patches.append(patch)
            except:
                patch = torch.zeros(1, 3, 64, 64, device=device)
                patches.append(patch)
        
        if not patches:
            return {}
            
        patches_tensor = torch.cat(patches, dim=0)
        
        # Extrai características específicas
        results = {}
        
        # 1. Embeddings do modelo base
        emb_dist = torch.cdist(embeddings, embeddings)
        emb_sim = 1.0 - emb_dist / 2.0  # Normaliza para 0-1
        results['embedding'] = emb_sim.cpu().numpy()
        
        # 2. Características de textura, cor e forma se disponíveis
        if texture_extractor is not None:
            texture_features = texture_extractor(patches_tensor)
            texture_sim = 1.0 - torch.cdist(texture_features, texture_features) / 2.0
            results['texture'] = texture_sim.cpu().numpy()
        
        if color_extractor is not None:
            color_features = color_extractor(patches_tensor)
            color_sim = 1.0 - torch.cdist(color_features, color_features) / 2.0
            results['color'] = color_sim.cpu().numpy()
        
        if shape_extractor is not None:
            shape_features = shape_extractor(boxes)
            shape_sim = 1.0 - torch.cdist(shape_features, shape_features) / 2.0
            results['shape'] = shape_sim.cpu().numpy()
        
        # Similaridade espacial baseada em IoU
        iou = box_iou(boxes, boxes)
        results['spatial'] = iou.cpu().numpy()
        
        # Fusão multimodal se disponível
        if fusion_module is not None and all(k in results for k in ['texture', 'color', 'shape']):
            fused_features = fusion_module(
                embeddings, 
                texture_features, 
                color_features, 
                shape_features
            )
            fused_sim = 1.0 - torch.cdist(fused_features, fused_features) / 2.0
            results['fused'] = fused_sim.cpu().numpy()
            
        return results