import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def visualize_recommendations(model, image, patch, patch_with_context=None, recommendations=None, 
                            save_path=None, show_scores=True, show_heatmap=True, patch_mask=None, patch_box=None,
                            patch_display=None, patch_mask_display=None):
    """
    Visualiza recomendações e mapa de calor.
    
    Args:
        model: AttentionRecommender
        image: Tensor [3, H, W]
        patch: Tensor [3, h, w] (Usado na inferência, geralmente 128x128)
        patch_display: Tensor [3, h_iso, w_iso] (Opcional - Usado na visualização, resolução original)
        patch_mask_display: Tensor [1, h_iso, w_iso] (Opcional - Máscara de alta resolução)
    """
    device = model.device
    image = image.to(device)
    patch = patch.to(device)
    
    # Se patch_display não for fornecido, usa o patch de inferência
    if patch_display is None:
        patch_display = patch
    
    # Se patch_mask_display não for fornecido, usa o patch_mask padrão
    if patch_mask_display is None and patch_mask is not None:
        patch_mask_display = patch_mask

    if patch_mask is not None:
        patch_mask = patch_mask.to(device)
    
    if patch_with_context is not None:
        patch_with_context = patch_with_context.to(device)
    
    if recommendations is None:
        from src.paste.inference.recommender import RecommendationEngine
        engine = RecommendationEngine(model)
        # ✅ Passa o contexto e AGORA TAMBÉM a patch_box para manter a escala original
        recommendations = engine.recommend_placements(
            image.unsqueeze(0), 
            patch.unsqueeze(0), 
            patch_with_context=patch_with_context.unsqueeze(0) if patch_with_context is not None else None,
            patch_box=patch_box
        )
    
    with torch.no_grad():
        # ✅ Passa o contexto para o heatmap
        heatmap, _ = model(
            image.unsqueeze(0), 
            patch=patch.unsqueeze(0), 
            patch_with_context=patch_with_context.unsqueeze(0) if patch_with_context is not None else None,
            return_attention=True
        )
    
    if image.dim() == 4:
        image = image[0]
    if patch.dim() == 4:
        patch = patch[0]
    
    image_np = image.cpu().permute(1, 2, 0).numpy()
    patch_np = patch.cpu().permute(1, 2, 0).numpy()
    heatmap_np = heatmap[0, 0].cpu().numpy()
    
    # Modificado para 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    ax1, ax2, ax3 = axes.ravel()
    
    ax1.imshow(image_np)
    
    # Inset com patch
    patch_ax = fig.add_axes([0.01, 0.02, 0.08, 0.08]) # Ajustado para novo layout
    patch_ax.imshow(patch_np)
    patch_ax.set_title("Patch")
    patch_ax.axis('off')
    
    colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
    top_recommendation = None
    for i, rec in enumerate(recommendations[:5]):
        if i == 0:
            top_recommendation = rec
        box = rec["box"]
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        
        rect = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        if show_scores:
            score_text = f"#{i+1} Score: {rec['score']:.2f}"
            ax1.text(
                x1, y1-5, score_text, color='white', fontsize=10, 
                weight='bold', bbox=dict(facecolor=color, alpha=0.7)
            )
    
    ax1.set_title("Recomendações de Colagem")
    ax1.axis('off')
    
    # --- Subplot 2: Mapa de Calor ---
    if show_heatmap:
        hm = ax2.imshow(heatmap_np, cmap='hot')
        fig.colorbar(hm, ax=ax2, label='Score de Compatibilidade', shrink=0.8)
        ax2.set_title("Mapa de Calor")
    ax2.axis('off')

    # --- Subplot 3: Simulação da Colagem ---
    if top_recommendation:
        # Recupera a recomendação
        rec_box = top_recommendation['box']
        angle = top_recommendation.get('angle', 0)
        flip = top_recommendation.get('flip', False)
        scale = top_recommendation.get('scale', 1.0)
        
        # 1. Determina o Patch de Origem (Preferência por alta resolução)
        # Se patch_display vier do notebook, ele é o recorte original (crop) do objeto
        using_high_res = patch_display is not None and isinstance(patch_display, torch.Tensor)
        
        if using_high_res:
            sim_patch = patch_display.to(device).clone()
            # Se tiver máscara de alta resolução, usa-a
            sim_mask_tensor = patch_mask_display.to(device).clone().float() if patch_mask_display is not None else None
        else:
            sim_patch = patch.clone()
            sim_mask_tensor = patch_mask.clone().float() if patch_mask is not None else None

        # Garante dimensoes corretas da máscara para TTO
        if sim_mask_tensor is not None and sim_mask_tensor.dim() == 2:
            sim_mask_tensor = sim_mask_tensor.unsqueeze(0)

        # 2. Aplica Transformações Geométricas (TTO) ao Patch e à Máscara
        # Scale
        if scale != 1.0:
            sim_patch = F.interpolate(sim_patch.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
            if sim_mask_tensor is not None:
                sim_mask_tensor = F.interpolate(sim_mask_tensor.unsqueeze(0), scale_factor=scale, mode='nearest').squeeze(0)
        # Flip
        if flip:
            sim_patch = torch.flip(sim_patch, dims=[-1])
            if sim_mask_tensor is not None:
                sim_mask_tensor = torch.flip(sim_mask_tensor, dims=[-1])
        # Rotate
        if angle != 0:
            k = angle // 90
            sim_patch = torch.rot90(sim_patch, k=k, dims=[-2, -1])
            if sim_mask_tensor is not None:
                sim_mask_tensor = torch.rot90(sim_mask_tensor, k=k, dims=[-2, -1])

        # 3. Determina a Box de Destino (Onde colar?)
        # A box recomendada pela rede (rec_box) pode ser quadrada/padronizada (ex: 128x128).
        # Queremos colar o objeto (sim_patch) no seu tamanho real (preservando AR).
        
        # Centro da recomendação
        cx = (rec_box[0] + rec_box[2]) / 2
        cy = (rec_box[1] + rec_box[3]) / 2
        
        # Tamanho do patch transformado (Height, Width)
        ph, pw = sim_patch.shape[-2:]
        
        # Cria nova box centrada em (cx, cy) com tamanho (pw, ph)
        target_box = [cx - pw/2, cy - ph/2, cx + pw/2, cy + ph/2]
        
        # DEBUG: Print shapes
        # print(f"Viz Debug: HighRes={using_high_res}, SimPatch={sim_patch.shape}, TargetBox={target_box}")

        # 4. Prepara Máscara Numpy
        sim_mask_np = None
        if sim_mask_tensor is not None:
            sim_mask_np = sim_mask_tensor.squeeze(0).cpu().numpy()
            # Normaliza para 0-255 uint8
            if sim_mask_np.max() <= 1.0:
                sim_mask_np = (sim_mask_np * 255).astype(np.uint8)
            else:
                sim_mask_np = sim_mask_np.astype(np.uint8)

        # 5. Executa a Colagem
        simulated_result = simulate_paste_result(
            image=image,
            patch=sim_patch,
            patch_mask=sim_mask_np, # CRÍTICO: Máscara define transparência
            box=target_box,         # CRÍTICO: Box deriva do tamanho do patch, não da rede
            blend_mode='hard'
        )
        
        tto_info = []
        if scale != 1.0: tto_info.append(f"Scale: {scale}x")
        if angle != 0: tto_info.append(f"Rot: {angle}º")
        if flip: tto_info.append("Flip")
        tto_str = " | ".join(tto_info) if tto_info else "Original"
        
        # Log debug para o user ver
        if using_high_res:
            tto_str += " [High-Res]"
        else:
            tto_str += " [Low-Res]"

        ax3.imshow(simulated_result)
        ax3.set_title(f"Simulação (Score: {top_recommendation['score']:.2f})\n[{tto_str}]")
    else:

        ax3.text(0.5, 0.5, "Sem recomendações", ha='center', va='center')
    ax3.axis('off')
    
    plt.tight_layout(rect=[0.05, 0, 1, 1]) # Ajusta layout
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def simulate_paste_result(image, patch, box, patch_mask=None, blend_mode='hard', return_tensor=False):
    """
    Simula o resultado da colagem de um patch numa região com gestão robusta de bordas (Clipping).
    Modo 'hard' usa a máscara para colar apenas o objeto.
    """
    # Garante numpy e range 0-1
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    
    image_np = np.clip(image_np, 0, 1)
    patch_np = np.clip(patch_np, 0, 1)
    
    image_uint8 = (image_np * 255).astype(np.uint8)
    patch_uint8 = (patch_np * 255).astype(np.uint8)
    
    H, W = image_uint8.shape[:2]
    
    # Coordenadas da caixa (podem ser negativas ou > W/H)
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Dimensões teóricas do patch redimensionado
    target_w = x2 - x1
    target_h = y2 - y1
    
    if target_h <= 0 or target_w <= 0:
        return image_np if not return_tensor else image

    # 1. Redimensiona o patch (e máscara) para o tamanho TOTAL da caixa
    patch_resized = cv2.resize(patch_uint8, (target_w, target_h))
    
    if patch_mask is None:
        # Heurística se não houver máscara
        gray_patch = cv2.cvtColor(patch_resized, cv2.COLOR_RGB2GRAY)
        _, patch_mask_inferred = cv2.threshold(gray_patch, 1, 255, cv2.THRESH_BINARY)
        mask_resized = patch_mask_inferred
    else:
        mask_resized = cv2.resize(patch_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # 2. Calcula Clipping (Intersecção com a imagem)
    # Limites na imagem
    img_x1 = max(0, x1)
    img_y1 = max(0, y1)
    img_x2 = min(W, x2)
    img_y2 = min(H, y2)
    
    # Se estiver totalmente fora, aborta
    if img_x2 <= img_x1 or img_y2 <= img_y1:
        return image_np if not return_tensor else image
    
    # Offsets relativos no patch (para recortar a parte correspondente)
    patch_x1 = img_x1 - x1
    patch_y1 = img_y1 - y1
    patch_x2 = patch_x1 + (img_x2 - img_x1)
    patch_y2 = patch_y1 + (img_y2 - img_y1)

    # 3. Aplica a colagem na região válida
    result = image_uint8.copy()
    
    # Recortes
    roi = result[img_y1:img_y2, img_x1:img_x2]
    patch_crop = patch_resized[patch_y1:patch_y2, patch_x1:patch_x2]
    mask_crop = mask_resized[patch_y1:patch_y2, patch_x1:patch_x2]
    
    if mask_crop.size == 0 or patch_crop.size == 0 or roi.size == 0:
         return image_np if not return_tensor else image

    # Prepara máscara binária
    _, mask_binary = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)
    
    # Limpa o patch ("bitwise do objeto")
    patch_clean = cv2.bitwise_and(patch_crop, patch_crop, mask=mask_binary)
    
    # Máscara booleana para indexação
    mask_bool = mask_binary > 0
    
    # Broadcast para 3 canais se necessário
    if len(roi.shape) == 3 and len(mask_bool.shape) == 2:
        mask_bool = np.stack([mask_bool] * 3, axis=-1)

    # Verifica shapes (agora devem bater sempre) e copia
    if roi.shape == patch_clean.shape and roi.shape == mask_bool.shape:
        np.copyto(roi, patch_clean, where=mask_bool)
        result[img_y1:img_y2, img_x1:img_x2] = roi # Atualiza na imagem final
    
    result = result.astype(np.float32) / 255.0
    
    if return_tensor:
        return torch.tensor(result).permute(2, 0, 1)
    return result