import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def visualize_recommendations(model, image, patch, patch_with_context=None, recommendations=None, 
                            save_path=None, show_scores=True, show_heatmap=True, patch_mask=None, patch_box=None):
    """Visualiza recomendações e mapa de calor"""
    device = model.device
    image = image.to(device)
    patch = patch.to(device)
    
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
        # Se NENHUMA máscara foi fornecida externamente, tenta inferir
        if patch_mask is None:
            # Assume que o fundo é aproximadamente zero (preto)
            # Usa um threshold um pouco maior para evitar ruído de compressão/normalização
            patch_sum = patch.abs().sum(dim=0)
            patch_mask_np = (patch_sum > 0.05).cpu().numpy().astype(np.uint8) * 255
        else:
            # Se a máscara foi passada, converte para numpy corretamente
            if isinstance(patch_mask, torch.Tensor):
                if patch_mask.dim() == 3:
                    patch_mask = patch_mask[0] # [1, H, W] -> [H, W]
                patch_mask_np = patch_mask.cpu().numpy().astype(np.uint8)
                # Garante range 0-255
                if patch_mask_np.max() <= 1.0:
                    patch_mask_np = (patch_mask_np * 255).astype(np.uint8)
            else:
                patch_mask_np = patch_mask
        
        simulated_result = simulate_paste_result(
            image=image,
            patch=patch,
            patch_mask=patch_mask_np, # Passa a máscara processada
            box=top_recommendation['box'],
            blend_mode='hard'
        )
        ax3.imshow(simulated_result)
        ax3.set_title(f"Simulação (Score: {top_recommendation['score']:.2f})")
    else:
        ax3.text(0.5, 0.5, "Sem recomendações", ha='center', va='center')
    ax3.axis('off')
    
    plt.tight_layout(rect=[0.05, 0, 1, 1]) # Ajusta layout
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def simulate_paste_result(image, patch, box, patch_mask=None, blend_mode='hard', return_tensor=False):
    """
    Simula o resultado da colagem de um patch numa região.
    Modo 'hard' usa a máscara para colar apenas o objeto.
    """
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    
    image_np = np.clip(image_np, 0, 1)
    patch_np = np.clip(patch_np, 0, 1)
    
    image_uint8 = (image_np * 255).astype(np.uint8)
    patch_uint8 = (patch_np * 255).astype(np.uint8)
    
    x1, y1, x2, y2 = [int(coord) for coord in box]
    target_h, target_w = y2 - y1, x2 - x1
    
    if target_h <= 0 or target_w <= 0:
        return image_np if not return_tensor else image

    patch_resized = cv2.resize(patch_uint8, (target_w, target_h))
    result = image_uint8.copy()
    
    # Se a máscara do patch não for fornecida, tenta usar uma heurística
    # MAS: para resultado perfeito, o ideal é o próprio calling code passar o patch já limpo ou com máscara boa.
    if patch_mask is None:
        # Tenta inferir máscara de não-zeros do patch redimensionado
        gray_patch = cv2.cvtColor(patch_resized, cv2.COLOR_RGB2GRAY)
        # Threshold de 1 significa que qualquer cor > 1 conta como objeto
        _, patch_mask_inferred = cv2.threshold(gray_patch, 1, 255, cv2.THRESH_BINARY)
        mask_resized = patch_mask_inferred
    else:
        mask_resized = cv2.resize(patch_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Região de Interesse (ROI) na imagem de destino
    roi = result[y1:y2, x1:x2]

    # Garante que a máscara é binária 0-255
    _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    
    # 1. Limpa o patch usando a máscara ("bitwise do objeto")
    patch_clean = cv2.bitwise_and(patch_resized, patch_resized, mask=mask_binary)
    
    # MODO MAIS SIMPLES (pythonic override):
    # Onde a máscara é ativa, substitui pelo pixel do patch_clean
    mask_bool = mask_binary > 0
    
    # Garante dimensionalidade para broadcast
    if len(roi.shape) == 3 and len(mask_bool.shape) == 2:
        mask_bool = np.stack([mask_bool] * 3, axis=-1)

    # Verifica shapes antes de copiar
    if roi.shape == patch_clean.shape and roi.shape == mask_bool.shape:
        # AQUI ESTA O FIX:
        # Só copiamos pixels onde o patch TEM conteúdo (mask > 0)
        # O resto do ROI mantém-se original (fundo da imagem destino)
        np.copyto(roi, patch_clean, where=mask_bool)
        result[y1:y2, x1:x2] = roi
    
    result = result.astype(np.float32) / 255.0
    
    if return_tensor:
        return torch.tensor(result).permute(2, 0, 1)
    return result