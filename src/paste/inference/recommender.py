import torch
import torch.nn.functional as F
from .extractors import extract_recommendations_from_heatmap


class RecommendationEngine:
    """Encapsula lógica de recomendação para um modelo treinado"""
    
    def __init__(self, model):
        self.model = model
    
    def recommend_placements(self, image, patch, patch_box=None, top_k=5):
        """
        Recomenda os melhores locais para colagem do patch na imagem.
        
        Args:
            image: [B, 3, H, W]
            patch: [B, 3, h, w]
            patch_box: opcional [x1, y1, x2, y2]
            top_k: número máximo de recomendações
            
        Returns:
            Lista de dicionários com recomendações
        """
        device = self.model.device
        image = image.to(device)
        patch = patch.to(device)
        
        with torch.no_grad():
            heatmap, attention_map = self.model(image, patch=patch, return_attention=True)
        
        # Determina tamanho do patch
        if patch_box is not None:
            patch_h = int(patch_box[3] - patch_box[1])
            patch_w = int(patch_box[2] - patch_box[0])
            patch_size = (patch_h, patch_w)
        else:
            patch_size = patch.shape[-2:]
        
        recommendations = extract_recommendations_from_heatmap(
            heatmap[0, 0], 
            image.shape[-2:], 
            patch_size, 
            top_k=top_k
        )
        
        return recommendations