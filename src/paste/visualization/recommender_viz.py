import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def visualize_recommendations(model, image, patch, recommendations=None, 
                            save_path=None, show_scores=True, show_heatmap=True):
    """Visualiza recomendações e mapa de calor"""
    device = model.device
    image = image.to(device)
    patch = patch.to(device)
    
    if recommendations is None:
        from src.paste.inference.recommender import RecommendationEngine
        engine = RecommendationEngine(model)
        recommendations = engine.recommend_placements(
            image.unsqueeze(0), patch.unsqueeze(0)
        )
    
    with torch.no_grad():
        heatmap, _ = model(image.unsqueeze(0), patch=patch.unsqueeze(0), return_attention=True)
    
    if image.dim() == 4:
        image = image[0]
    if patch.dim() == 4:
        patch = patch[0]
    
    image_np = image.cpu().permute(1, 2, 0).numpy()
    patch_np = patch.cpu().permute(1, 2, 0).numpy()
    heatmap_np = heatmap[0, 0].cpu().numpy()
    
    if show_heatmap:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    ax1.imshow(image_np)
    
    # Inset com patch
    patch_ax = fig.add_axes([0.02, 0.02, 0.1, 0.1])
    patch_ax.imshow(patch_np)
    patch_ax.set_title("Patch")
    patch_ax.axis('off')
    
    colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
    for i, rec in enumerate(recommendations[:5]):
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
    
    if show_heatmap:
        hm = ax2.imshow(heatmap_np, cmap='hot')
        plt.colorbar(hm, ax=ax2, label='Score de Compatibilidade')
        ax2.set_title("Mapa de Calor")
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def simulate_paste_result(image, patch, box, blend_mode='alpha', return_tensor=False):
    """Simula o resultado da colagem de um patch numa região"""
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    
    image_np = np.clip(image_np, 0, 1)
    patch_np = np.clip(patch_np, 0, 1)
    
    image_uint8 = (image_np * 255).astype(np.uint8)
    patch_uint8 = (patch_np * 255).astype(np.uint8)
    
    x1, y1, x2, y2 = [int(coord) for coord in box]
    target_h, target_w = y2 - y1, x2 - x1
    
    patch_resized = cv2.resize(patch_uint8, (target_w, target_h))
    result = image_uint8.copy()
    
    if blend_mode == 'alpha':
        alpha = 0.7
        result[y1:y2, x1:x2] = cv2.addWeighted(
            result[y1:y2, x1:x2], 1-alpha, 
            patch_resized, alpha, 0
        )
    elif blend_mode == 'poisson':
        try:
            mask = np.ones(patch_resized.shape, patch_resized.dtype) * 255
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            result = cv2.seamlessClone(
                patch_resized, result, mask, center, 
                cv2.NORMAL_CLONE
            )
        except Exception:
            alpha = 0.7
            result[y1:y2, x1:x2] = cv2.addWeighted(
                result[y1:y2, x1:x2], 1-alpha, 
                patch_resized, alpha, 0
            )
    else:
        result[y1:y2, x1:x2] = patch_resized
    
    result = result.astype(np.float32) / 255.0
    
    if return_tensor:
        return torch.tensor(result).permute(2, 0, 1)
    return result