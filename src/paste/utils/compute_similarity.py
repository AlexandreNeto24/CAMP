import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align, box_iou

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

def extract_multi_scale_roi_features(backbone_features, boxes, batch_indices=None, output_size=(7, 7), N=2):
    """
    Extrai e concatena as últimas N features do backbone para cada ROI.
    """
    if boxes.numel() == 0:
        return torch.empty((0, sum([f.shape[1] for f in backbone_features[-N:]])), device=boxes.device)
    
    if batch_indices is None:
        batch_indices = torch.zeros((boxes.size(0),), dtype=torch.int, device=boxes.device)
    
    # Converte todos para o mesmo tipo (float32)
    batch_indices = batch_indices.to(boxes.device)
    boxes = boxes.to(dtype=torch.float32)  # Garante que boxes seja float32
    
    roi_feats = []
    for f in backbone_features[-N:]:
        f = f.to(dtype=torch.float32)  # Garante que features seja float32
        
        # Cria ROIs e garante que esteja em float32
        rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)
        
        spatial_scale = f.shape[2] / 640
        try:
            feats = roi_align(f, rois, output_size=output_size, spatial_scale=spatial_scale)
        except Exception as e:
            print(f"[DEBUG] roi_align error: {e}")
            feats = torch.zeros((boxes.size(0), f.shape[1], *output_size), device=boxes.device)
            
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        roi_feats.append(pooled)
    
    embedding = torch.cat(roi_feats, dim=1)
    embedding = F.normalize(embedding.clone(), dim=1)
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
    all_boxes, all_labels, all_img_indices,
    model, margin=1.0,
    context_head=None,
    texture_extractor=None,  # Mantemos os parâmetros mas não usamos
    color_extractor=None,
    shape_extractor=None,
    fusion_module=None
):
    """
    Versão simplificada da similarity loss que calcula o contraste entre lesão e background
    usando apenas características do backbone.
    """
    device = global_imgs.device
    backbone = model.model.model[:6]
    
    # Extrai features do backbone
    features = backbone(global_imgs)
    feature_maps = features if isinstance(features, (list, tuple)) else [features]
    
    # Verifica se temos boxes suficientes
    if all_boxes.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    # Separar boxes de lesão e background
    lesion_mask = all_labels == 1
    bg_mask = all_labels == 0
    
    lesion_boxes = all_boxes[lesion_mask]
    lesion_indices = all_img_indices[lesion_mask]
    
    bg_boxes = all_boxes[bg_mask]
    bg_indices = all_img_indices[bg_mask]
    
    # Verifica se temos lesões e backgrounds correspondentes
    if lesion_boxes.numel() == 0:
        print("Nenhuma box de lesão encontrada")
        return torch.tensor(0.0, device=device)
    
    if bg_boxes.numel() == 0:
        print("Nenhuma box de background encontrada")
        return torch.tensor(0.0, device=device)
    
    # Mapeia backgrounds para suas lesões correspondentes
    # Assumindo que background_boxes são gerados expandindo lesion_boxes
    lesion_to_bg = {}
    for l_idx, l_box in enumerate(lesion_boxes):
        l_img = lesion_indices[l_idx].item()
        for b_idx, b_box in enumerate(bg_boxes):
            b_img = bg_indices[b_idx].item()
            if l_img == b_img:
                # Verifica se o background contém a lesão (expandido)
                l_center_x = l_box[1].item()
                l_center_y = l_box[2].item()
                b_center_x = b_box[1].item()
                b_center_y = b_box[2].item()
                
                # Se os centros estão próximos, considera que o background corresponde à lesão
                if abs(l_center_x - b_center_x) < 0.1 and abs(l_center_y - b_center_y) < 0.1:
                    if l_idx not in lesion_to_bg:
                        lesion_to_bg[l_idx] = b_idx
                    break
    
    # Converte coordenadas YOLO para formato [x1, y1, x2, y2]
    xyxy_lesion_boxes = torch.zeros_like(lesion_boxes[:, 1:])
    xyxy_lesion_boxes[:, 0] = (lesion_boxes[:, 1] - lesion_boxes[:, 3]/2) * 640  # x1
    xyxy_lesion_boxes[:, 1] = (lesion_boxes[:, 2] - lesion_boxes[:, 4]/2) * 640  # y1
    xyxy_lesion_boxes[:, 2] = (lesion_boxes[:, 1] + lesion_boxes[:, 3]/2) * 640  # x2
    xyxy_lesion_boxes[:, 3] = (lesion_boxes[:, 2] + lesion_boxes[:, 4]/2) * 640  # y2
    
    xyxy_bg_boxes = torch.zeros_like(bg_boxes[:, 1:])
    xyxy_bg_boxes[:, 0] = (bg_boxes[:, 1] - bg_boxes[:, 3]/2) * 640  # x1
    xyxy_bg_boxes[:, 1] = (bg_boxes[:, 2] - bg_boxes[:, 4]/2) * 640  # y1
    xyxy_bg_boxes[:, 2] = (bg_boxes[:, 1] + bg_boxes[:, 3]/2) * 640  # x2
    xyxy_bg_boxes[:, 3] = (bg_boxes[:, 2] + bg_boxes[:, 4]/2) * 640  # y2
    
    # Clamp para evitar valores fora dos limites
    xyxy_lesion_boxes = xyxy_lesion_boxes.clamp(min=0)
    xyxy_lesion_boxes[:, 2] = xyxy_lesion_boxes[:, 2].clamp(max=640)
    xyxy_lesion_boxes[:, 3] = xyxy_lesion_boxes[:, 3].clamp(max=640)
    
    xyxy_bg_boxes = xyxy_bg_boxes.clamp(min=0)
    xyxy_bg_boxes[:, 2] = xyxy_bg_boxes[:, 2].clamp(max=640)
    xyxy_bg_boxes[:, 3] = xyxy_bg_boxes[:, 3].clamp(max=640)
    
    # Extrai lesion_features e bg_features usando ROI Align
    lesion_features = extract_multi_scale_roi_features(
        feature_maps, xyxy_lesion_boxes, batch_indices=lesion_indices, output_size=(7, 7), N=2
    )
    
    bg_features = extract_multi_scale_roi_features(
        feature_maps, xyxy_bg_boxes, batch_indices=bg_indices, output_size=(7, 7), N=2
    )
    
    # Passa pelos context embeddings (mantemos isso)
    if context_head is not None:
        lesion_features = context_head(lesion_features)
        bg_features = context_head(bg_features)
    
    # Simplificar - não usamos mais textura/cor/forma
    # Agora lesion_embeddings e bg_embeddings são diretamente das features do backbone
    lesion_embeddings = lesion_features
    bg_embeddings = bg_features
    
    # AQUI ESTÁ A PARTE CRUCIAL: CALCULAR O CONTRASTE ENTRE LESÃO E BACKGROUND
    contrast_embeddings = []
    contrast_weights = []  # Para ponderação por importância

    # Aumentar peso da loss de contraste
    contrast_loss_weight = 2.0  # Era implicitamente 1.0 antes

    for l_idx in range(len(lesion_embeddings)):
        if l_idx in lesion_to_bg:
            b_idx = lesion_to_bg[l_idx]
            
            # Extrair embeddings correspondentes
            lesion_emb = lesion_embeddings[l_idx:l_idx+1]
            bg_emb = bg_embeddings[b_idx:b_idx+1]
            
            # Calcular contraste (diferença normalizada) - OPERAÇÃO CHAVE
            contrast = lesion_emb - bg_emb
            contrast = F.normalize(contrast.clone(), dim=1)
            
            # Cálculo do peso de importância baseado em quão distinto é o contraste
            with torch.no_grad():
                contrast_norm = torch.norm(contrast, p=2)
                weight = torch.sigmoid(contrast_norm - 0.5) + 0.5  # Normaliza para 0.5-1.5
            
            contrast_embeddings.append(contrast)
            contrast_weights.append(weight.item())

    if not contrast_embeddings:
        print("Nenhum par lesão-background para calcular contraste")
        return torch.tensor(0.0, device=device)
    
    contrast_embeddings = torch.cat(contrast_embeddings, dim=0)
    
    # Extrair propostas do modelo
    boxes, conf = decode_boxes(raw_preds, 0)
    if boxes.numel() == 0:
        print("Nenhuma proposta do modelo para calcular similaridade")
        return torch.tensor(0.0, device=device)
    
    # Seleciona top propostas para análise
    selected_boxes, selected_scores = select_top_confident_preds(boxes, conf, top_k=40)
    
    # Extrai features das propostas
    if selected_boxes.numel() == 0:
        print("Nenhuma proposta selecionada")
        return torch.tensor(0.0, device=device)
    
    batch_indices = torch.zeros(selected_boxes.size(0), dtype=torch.int, device=device)
    
    # Expandir as boxes de proposta para obter seu contexto (background)
    proposal_bg_boxes = expand_boxes(selected_boxes, ratio=1.5)
    
    # Extrai features das propostas e seus backgrounds
    proposal_features = extract_multi_scale_roi_features(
        feature_maps, selected_boxes, batch_indices=batch_indices, output_size=(7, 7), N=2
    )
    
    proposal_bg_features = extract_multi_scale_roi_features(
        feature_maps, proposal_bg_boxes, batch_indices=batch_indices, output_size=(7, 7), N=2
    )
    
    # Context head para propostas
    if context_head is not None:
        proposal_features = context_head(proposal_features)
        proposal_bg_features = context_head(proposal_bg_features)
    
    # NOVA SEÇÃO: Extração de características multimodais para propostas
    proposal_texture, proposal_color, proposal_shape = None, None, None
    proposal_bg_texture, proposal_bg_color, proposal_bg_shape = None, None, None
    
    # Processamento de características multimodais para propostas
    if texture_extractor is not None or color_extractor is not None:
        proposal_patches = extract_image_patches(global_imgs, selected_boxes, batch_indices)
        if proposal_patches and len(proposal_patches) > 0:
            proposal_patches_tensor = torch.cat(proposal_patches, dim=0)
            if texture_extractor is not None:
                proposal_texture = texture_extractor(proposal_patches_tensor)
            if color_extractor is not None:
                proposal_color = color_extractor(proposal_patches_tensor)
    
    # Características de forma para propostas
    if shape_extractor is not None:
        proposal_shape = shape_extractor(selected_boxes)
    
    # Processamento de características multimodais para backgrounds de propostas
    if texture_extractor is not None or color_extractor is not None:
        proposal_bg_patches = extract_image_patches(global_imgs, proposal_bg_boxes, batch_indices)
        if proposal_bg_patches:
            proposal_bg_patches_tensor = torch.cat(proposal_bg_patches, dim=0)
            
            # Características de textura para backgrounds de propostas
            if texture_extractor is not None:
                proposal_bg_texture = texture_extractor(proposal_bg_patches_tensor)
            
            # Características de cor para backgrounds de propostas
            if color_extractor is not None:
                proposal_bg_color = color_extractor(proposal_bg_patches_tensor)
    
    # Características de forma para backgrounds de propostas
    if shape_extractor is not None:
        proposal_bg_shape = shape_extractor(proposal_bg_boxes)
    
    # Verifica valores nulos e substitui por tensores zerados
    if proposal_texture is None:
        proposal_texture = torch.zeros((proposal_features.size(0), 8), device=device)
    if proposal_color is None:
        proposal_color = torch.zeros((proposal_features.size(0), 32), device=device)
    if proposal_shape is None:
        proposal_shape = torch.zeros((proposal_features.size(0), 16), device=device)
    
    if proposal_bg_texture is None:
        proposal_bg_texture = torch.zeros((proposal_bg_features.size(0), 8), device=device)
    if proposal_bg_color is None:
        proposal_bg_color = torch.zeros((proposal_bg_features.size(0), 32), device=device)
    if proposal_bg_shape is None:
        proposal_bg_shape = torch.zeros((proposal_bg_features.size(0), 16), device=device)
    
    # Fusão multimodal para propostas e seus backgrounds
    if fusion_module is not None:
        proposal_embeddings = fusion_module(
            proposal_features, proposal_texture, proposal_color, proposal_shape)
        proposal_bg_embeddings = fusion_module(
            proposal_bg_features, proposal_bg_texture, proposal_bg_color, proposal_bg_shape)
    else:
        proposal_embeddings = proposal_features
        proposal_bg_embeddings = proposal_bg_features
    
    # Calcular contraste para propostas
    proposal_contrasts = []
    for i in range(len(proposal_embeddings)):
        # Extrair embeddings
        prop_emb = proposal_embeddings[i:i+1]
        prop_bg_emb = proposal_bg_embeddings[i:i+1]
        
        # Calcular contraste (mesma operação usada para GTs)
        contrast = prop_emb - prop_bg_emb
        contrast = F.normalize(contrast.clone(), dim=1)  # clone() antes de normalizar
        
        proposal_contrasts.append(contrast)
    
    if not proposal_contrasts:
        print("Nenhum contraste de proposta calculado")
        return torch.tensor(0.0, device=device)
    
    proposal_contrasts = torch.cat(proposal_contrasts, dim=0)
    
    # Calcula IoU entre propostas e lesões para determinar positivos/negativos
    ious = box_iou(selected_boxes, xyxy_lesion_boxes)
    max_ious, _ = ious.max(dim=1)
    
    # Define limiar para considerar uma proposta como positiva
    iou_pos_thresh = 0.1
    
    # Identifica propostas positivas (alta IoU com GT)
    pos_mask = max_ious > iou_pos_thresh
    pos_contrasts = proposal_contrasts[pos_mask]
    neg_contrasts = proposal_contrasts[~pos_mask]
    
    # Se não tivermos positivos ou negativos suficientes
    if pos_contrasts.size(0) < 1:
        print("Nenhuma proposta positiva encontrada")
        # Em vez de:
        # return torch.tensor(0.0, device=device)
    
        # Faça:
        # Use algum tensor da rede para manter o grafo de computação
        dummy = proposal_features.mean() * 0
        return dummy
    # Para cada contraste positivo, calcular a similaridade com os contrastes GT
    pos_sim_loss = 0
    total_weights = sum(contrast_weights) if contrast_weights else 1.0

    for idx, pos_contrast in enumerate(pos_contrasts):
        # Calcula similaridade com cada GT contrast
        similarities = torch.mm(pos_contrast.unsqueeze(0), contrast_embeddings.t())
        
        # Aplica pesos de importância para cada contraste GT
        weighted_similarities = similarities * torch.tensor(contrast_weights, device=similarities.device)
        
        # Escolhe a maior similaridade (mais próxima de alguma GT)
        max_sim, _ = weighted_similarities.max(dim=1)
        
        # Loss para maximizar similaridade (1 - sim porque queremos maximizar)
        pos_sim_loss += (1 - max_sim)
    
    if pos_contrasts.size(0) > 0:
        pos_sim_loss /= pos_contrasts.size(0)
    
    # Para contrastes negativos, queremos que sejam diferentes dos GT
    neg_sim_loss = 0
    if neg_contrasts.size(0) > 0:
        for neg_contrast in neg_contrasts:
            # Similaridade com cada GT
            similarities = torch.mm(neg_contrast.unsqueeze(0), contrast_embeddings.t())
            # Pegamos a maior similaridade
            max_sim, _ = similarities.max(dim=1)
            # Queremos que seja menor que margin (1)
            neg_sim_loss += F.relu(max_sim - (1 - margin))
        
        neg_sim_loss /= neg_contrasts.size(0)
    
    # Loss combinada
    sim_loss = pos_sim_loss + neg_sim_loss
    
    # Aumenta o peso da loss de contraste
    sim_loss = sim_loss * contrast_loss_weight

    return sim_loss

def extract_image_patches(images, boxes, img_indices):
    """Extrai patches de imagem para as boxes especificadas"""
    patches = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().clamp(min=0)
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1
        
        img_idx = img_indices[i].item()
        if img_idx < images.size(0):
            try:
                patch = images[img_idx:img_idx+1, :, y1:y2, x1:x2]
                patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                patches.append(patch)
            except:
                device = images.device
                patch = torch.zeros(1, 3, 64, 64, device=device)
                patches.append(patch)
    
    return patches

def expand_boxes(boxes, ratio=1.5):
    """
    Expande boxes por um fator, mantendo dentro dos limites da imagem.
    Versão corrigida sem operações in-place.
    """
    # Calcular centros e dimensões
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    # Expandir mantendo o centro
    new_w = w * ratio
    new_h = h * ratio
    
    # Novas coordenadas - criamos novos tensores
    new_x1 = torch.clamp(cx - new_w / 2, min=0)  # x1
    new_y1 = torch.clamp(cy - new_h / 2, min=0)  # y1
    new_x2 = torch.clamp(cx + new_w / 2, max=640)  # x2
    new_y2 = torch.clamp(cy + new_h / 2, max=640)  # y2
    
    # Concatena em um novo tensor (sem modificar in-place)
    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

def compute_pairwise_similarity(features):
    """
    Calcula matriz de similaridade de cosseno entre vetores de características.
    
    Args:
        features: Tensor [N, D] de características
        
    Returns:
        Tensor [N, N] de similaridades
    """
    # Normalização para cálculo de similaridade de cosseno
    features = F.normalize(features, p=2, dim=1)
    
    # Similaridade de cosseno: produto escalar entre vetores normalizados
    similarity = torch.mm(features, features.t())
    
    return similarity

# Adicionar função auxiliar para ROI Align mais eficiente

def extract_roi_features(feature_maps, boxes, batch_indices=None, output_size=(7, 7)):
    """
    Extrai features usando ROI Align de forma mais eficiente.
    
    Args:
        feature_maps: Tensor de features [B, C, H, W]
        boxes: Coordenadas das boxes [N, 4] no formato [x1, y1, x2, y2]
        batch_indices: Índices de batch para cada box
        output_size: Tamanho de saída para ROI Align
        
    Returns:
        Tensor de features extraídas [N, C]
    """
    from torchvision.ops import roi_align
    
    if boxes.numel() == 0:
        return torch.empty((0, feature_maps.shape[1]), device=boxes.device)
    
    if batch_indices is None:
        batch_indices = torch.zeros(boxes.size(0), dtype=torch.int, device=boxes.device)
    
    # Formato para roi_align: [batch_idx, x1, y1, x2, y2]
    rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)
    
    # Calcula escala espacial
    spatial_scale = feature_maps.shape[2] / 640
    
    # Extrai features via ROI Align
    roi_features = roi_align(
        feature_maps,
        rois,
        output_size=output_size,
        spatial_scale=spatial_scale
    )
    
    # Pooling global para obter vetor de features
    pooled_features = F.adaptive_avg_pool2d(roi_features, (1, 1)).squeeze(-1).squeeze(-1)
    
    return pooled_features