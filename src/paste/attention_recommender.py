import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import roi_align
import random
from tqdm.auto import tqdm 
# Configuração de caminhos para garantir importações corretas
project_root = Path(__file__).parent.parent.parent  # src/paste/ → src/ → project_root
sys.path.insert(0, str(project_root))

# Adiciona caminho para o ultralytics
ultralytics_path = os.path.abspath(os.path.join(project_root, "src", "ultralytics"))
sys.path.insert(0, ultralytics_path)

try:
    from ultralytics import YOLO
    # print("Ultralytics importado com sucesso no attention_recommender.py!")
except ImportError as e:
    # print(f"Erro ao importar ultralytics: {e}")
    # print(f"Caminhos de busca: {sys.path}")
    sys.exit(1)


class PositionalEncoding(nn.Module):
    """
    Codificação posicional para o mecanismo de atenção.
    
    Esta classe implementa a codificação posicional senoidal conforme proposto no artigo
    "Attention Is All You Need". Ela adiciona informações sobre a posição relativa
    das características no mapa de features, o que é crucial para o mecanismo de atenção
    preservar informações espaciais.
    
    Args:
        d_model (int): Dimensão do modelo (embedding)
        dropout (float): Taxa de dropout
        max_len (int): Tamanho máximo da sequência
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Cria matriz de codificação posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Codificação senoidal: usa sin para posições pares e cos para ímpares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registra como buffer (não é um parâmetro a ser otimizado)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Adiciona codificação posicional ao input
        
        Args:
            x: Tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Tensor com codificação posicional adicionada
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionRecommender(nn.Module):
    """
    Modelo de recomendação baseado em mecanismo de atenção para colagem de lesões em imagens.
    
    Este modelo utiliza o backbone de um detector YOLO pré-treinado para extrair características 
    e um mecanismo de atenção cruzada para relacionar patches de lesão com regiões candidatas em 
    imagens médicas. O resultado é um mapa de calor indicando os melhores locais para colagem.
    
    Características principais:
    - Extração in-situ de características para preservar detalhes do patch
    - Mecanismo de multi-head attention para capturar relações complexas
    - Decodificador para gerar mapa de calor de alta resolução
    
    Args:
        backbone_path (str): Caminho para o modelo YOLO pré-treinado
        embedding_dim (int): Dimensão do espaço de embedding
        num_heads (int): Número de cabeças no mecanismo de atenção
    """
    def __init__(self, backbone_model, embedding_dim=256, num_heads=8):
        super().__init__()
        
        self.backbone = backbone_model
        self.feature_extractor = self.backbone.get_encoder()
        
        # Congelar backbone para estabilidade de treinamento
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        feature_dim = self.backbone.feature_dim
        self.embedding_dim = embedding_dim
        
        # Camadas para projetar as features do patch (query) e da imagem (key)
        self.query_conv = nn.Conv2d(feature_dim, self.embedding_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(feature_dim, self.embedding_dim, kernel_size=1)
        
        # O decoder agora opera diretamente no mapa de correlação de 1 canal
        # e faz upsample para o tamanho da imagem original.
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        
        # Mecanismo de multi-head attention
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Codificação posicional para preservar informação espacial
        self.position_encoder = PositionalEncoding(embedding_dim, dropout=0.1)
        
    def extract_features(self, image):
        """
        Extrai features da imagem usando o backbone.
        
        Args:
            image: Tensor da imagem [B,3,H,W]
            
        Returns:
            Features extraídas [B,C,H/8,W/8]
        """
        device = next(self.feature_extractor.parameters()).device
        image = image.to(device)
        
        # Extrai features com backbone congelado
        with torch.no_grad():
            features = self.feature_extractor(image)
            
            # Garante formato consistente de saída
            if isinstance(features, (list, tuple)):
                return features[-1]  # Usa último nível de features
            return features

    # def extract_roi_features(self, features, boxes, batch_indices=None):
    #     """
    #     Extrai features de regiões específicas usando ROI Align.
        
    #     Esta função preserva a integridade das características extraídas na resolução
    #     original, evitando a perda de informação que ocorreria com redimensionamento.
        
    #     Args:
    #         features: Feature maps [B,C,H,W]
    #         boxes: Bounding boxes [N,4] formato [x1,y1,x2,y2]
    #         batch_indices: Índices de batch para cada box [N]
            
    #     Returns:
    #         Features das regiões [N,C,16,16]
    #     """
    #     if boxes.numel() == 0:
    #         return torch.empty((0, features.shape[1], 16, 16), device=features.device)
        
    #     device = features.device
    #     boxes = boxes.to(device)
        
    #     # Prepara índices de batch se não fornecidos
    #     if batch_indices is None:
    #         batch_indices = torch.zeros(boxes.size(0), device=device)
    #     else:
    #         batch_indices = batch_indices.to(device)
        
    #     # Combina índices de batch com boxes para formato esperado pelo ROI Align
    #     rois = torch.cat([batch_indices.unsqueeze(1), boxes], dim=1)
        
    #     # Calcula escala espacial (relação entre tamanho do feature map e imagem original)
    #     # Isso é crucial para o ROI Align saber como mapear as coordenadas
    #     spatial_scale = features.shape[2] / 640
        
    #     # Extrai features usando ROI Align
    #     # Tamanho de saída fixo (16,16) para todas as regiões, independente do tamanho original
    #     roi_features = roi_align(
    #         features, 
    #         rois, 
    #         output_size=(16, 16),
    #         spatial_scale=spatial_scale,
    #         sampling_ratio=2  # Número de pontos de amostragem por célula
    #     )
        
    #     return roi_features

    # def extract_patch_features(self, patch):
    #     """
    #     Extrai e codifica features de um patch.
        
    #     Args:
    #         patch: Tensor do patch [B,3,H,W]
            
    #     Returns:
    #         Features codificadas [B,embedding_dim]
    #     """
    #     # Garante formato correto
    #     if patch.dim() == 3:
    #         patch = patch.unsqueeze(0)
            
    #     device = next(self.parameters()).device
    #     patch = patch.to(device)
        
    #     # Extrai features usando backbone
    #     patch_features = self.extract_features(patch)
        
    #     # Codifica para espaço de embedding
    #     patch_embedding = self.patch_encoder(patch_features)
        
    #     # Normaliza para melhor estabilidade
    #     return F.normalize(patch_embedding, dim=1)

    def forward(self, image, lesion_coords=None, patch=None, weight_mask=None, return_attention=False):
        """
        Forward pass usando correlação cruzada espacial.
        """
        device = self.device
        image = image.to(device)
        patch = patch.to(device)

        # 1. Extrai features da imagem alvo e do patch
        image_features = self.extract_features(image)
        patch_features = self.extract_features(patch)

        # 2. Aplica a máscara de pesos às features do patch
        if weight_mask is not None:
            weight_mask = weight_mask.to(device)
            resized_weight_mask = F.interpolate(
                weight_mask.unsqueeze(1),
                size=patch_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            patch_features = patch_features * resized_weight_mask

        # 3. Projeta as features para o espaço de embedding
        query = self.query_conv(patch_features)
        key = self.key_conv(image_features)

        # Normaliza as features para a correlação ser um cosseno de similaridade
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        # 4. Calcula o mapa de compatibilidade usando correlação cruzada (convolução)
        # A saída é [B, B, H, W], onde B é o batch size.
        correlation_map_full = F.conv2d(key, weight=query, padding='same')

        # 5. Extrai a diagonal para obter a correlação de cada imagem com o seu próprio patch
        # Resultado é [B, H, W]
        correlation_map = torch.diagonal(correlation_map_full, offset=0, dim1=0, dim2=1).permute(2, 0, 1)
        
        # Adiciona a dimensão do canal: [B, 1, H, W]
        correlation_map = correlation_map.unsqueeze(1)

        # 6. Gera o heatmap de alta resolução a partir do mapa de correlação
        heatmap = self.heatmap_decoder(correlation_map)
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
        heatmap = torch.sigmoid(heatmap)

        if return_attention:
            return heatmap, correlation_map
            
        return heatmap

    def recommend_placements(self, image, patch, top_k=5):
        """
        Recomenda os melhores locais para colagem do patch na imagem.
        
        Este método processa o mapa de calor gerado pelo modelo e extrai
        as coordenadas dos pontos mais promissores para colagem.
        
        Args:
            image: Tensor da imagem destino [B,3,H,W]
            patch: Tensor do patch de lesão [B,3,H,W]
            top_k: Número máximo de recomendações
            
        Returns:
            Lista de dicionários com recomendações {box, score, rank}
        """
        # Converter tensores para o dispositivo do modelo
        device = next(self.parameters()).device
        image = image.to(device)
        patch = patch.to(device)
        
        # Gerar mapa de calor através do mecanismo de atenção
        with torch.no_grad():
            heatmap, attention_map = self.forward(image, patch=patch, return_attention=True)
            
        # Processar mapa de calor para extrair pontos de interesse
        recommendations = self._extract_recommendations_from_heatmap(
            heatmap[0, 0], 
            image.shape[-2:], 
            patch.shape[-2:], 
            top_k=top_k
        )
        
        return recommendations

    def _extract_recommendations_from_heatmap(self, heatmap, image_size, patch_size, top_k=5):
        """
        Extrai coordenadas de recomendação do mapa de calor.
        
        Args:
            heatmap: Mapa de calor [H,W]
            image_size: Tamanho da imagem original (H,W)
            patch_size: Tamanho do patch (h,w)
            top_k: Número máximo de recomendações
            
        Returns:
            Lista de dicionários com recomendações {box, score, rank}
        """
        # Aplica Non-Maximum Suppression para encontrar máximos locais
        # Isso evita que recomendações muito próximas sejam todas selecionadas
        nms_heatmap = self._non_max_suppression(heatmap.cpu().numpy(), window_size=20)
        heatmap_tensor = torch.from_numpy(nms_heatmap).to(heatmap.device)
        
        # Encontra top-k pontos
        h, w = heatmap_tensor.shape
        flat_heatmap = heatmap_tensor.flatten()
        values, indices = torch.topk(flat_heatmap, min(top_k*2, flat_heatmap.numel()))
        
        # Converte índices para coordenadas 2D
        coords_y = (indices // w).tolist()
        coords_x = (indices % w).tolist()
        
        # Calcula boxes com base nos pontos e no tamanho do patch
        recommendations = []
        for i, (y, x, score) in enumerate(zip(coords_y, coords_x, values.tolist())):
            if score < 0.2:  # Filtra scores muito baixos
                continue
                
            # Calcula coordenadas da box, centralizando no ponto
            patch_h, patch_w = patch_size
            
            # Escala do mapa de calor para imagem original
            y_scale = image_size[0] / h
            x_scale = image_size[1] / w
            
            # Coordenada central na imagem original
            center_y = int(y * y_scale)
            center_x = int(x * x_scale)
            
            # Gera box centralizada no ponto
            half_h = patch_h // 2
            half_w = patch_w // 2
            
            box = [
                max(0, center_x - half_w),               # x1
                max(0, center_y - half_h),               # y1
                min(image_size[1], center_x + half_w),   # x2
                min(image_size[0], center_y + half_h)    # y2
            ]
            
            recommendations.append({
                "box": box,
                "score": float(score),
                "rank": i + 1
            })
            
            if len(recommendations) >= top_k:
                break
                
        return recommendations

    def _non_max_suppression(self, heatmap, window_size=15):
        """
        Aplica non-maximum suppression no mapa de calor para encontrar máximos locais.
        
        Esta função permite identificar pontos que são definitivamente máximos locais,
        evitando múltiplas detecções em áreas vizinhas.
        
        Args:
            heatmap: Mapa de calor como array numpy
            window_size: Tamanho da janela para supressão
            
        Returns:
            Mapa de calor filtrado contendo apenas máximos locais
        """
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure, binary_erosion
        
        # Aplica filtro de máximo para detectar picos
        # O neighborhood define a conectividade (aqui, 8-conectado)
        neighborhood = generate_binary_structure(2, 2)
        local_max = maximum_filter(heatmap, footprint=neighborhood) == heatmap
        
        # Remove picos pequenos e background
        background = (heatmap < 0.1)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = local_max & ~eroded_background
        
        # Criar heatmap apenas com picos locais
        filtered_heatmap = np.zeros_like(heatmap)
        filtered_heatmap[detected_peaks] = heatmap[detected_peaks]
        
        return filtered_heatmap

    def visualize_recommendations(self, image, patch, recommendations=None, 
                                save_path=None, show_scores=True, show_heatmap=True):
        """
        Visualiza recomendações e mapa de calor.
        
        Esta função cria uma visualização completa das recomendações de colagem,
        incluindo a imagem original, o patch, e opcionalmente o mapa de calor.
        
        Args:
            image: Tensor da imagem [B,3,H,W] ou [3,H,W]
            patch: Tensor do patch [B,3,H,W] ou [3,H,W]
            recommendations: Lista de recomendações (opcional)
            save_path: Caminho para salvar visualização
            show_scores: Se deve mostrar scores
            show_heatmap: Se deve mostrar mapa de calor
            
        Returns:
            Figura matplotlib
        """
        # Gera recomendações se não fornecidas
        if recommendations is None:
            recommendations = self.recommend_placements(
                image.unsqueeze(0) if image.dim() == 3 else image,
                patch.unsqueeze(0) if patch.dim() == 3 else patch
            )
            
        # Gera também o mapa de calor para visualização
        with torch.no_grad():
            heatmap, attention_map = self.forward(
                image.unsqueeze(0) if image.dim() == 3 else image,
                patch=patch.unsqueeze(0) if patch.dim() == 3 else patch,
                return_attention=True
            )
            
        # Preparação para visualização
        if image.dim() == 4:
            image = image[0]
        if patch.dim() == 4:
            patch = patch[0]
            
        image_np = image.cpu().permute(1, 2, 0).numpy()
        patch_np = patch.cpu().permute(1, 2, 0).numpy()
        heatmap_np = heatmap[0, 0].cpu().numpy()
        
        # Criar figura
        if show_heatmap:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
            
        # Mostrar imagem original
        ax1.imshow(image_np)
        
        # Adicionar patch como inset
        patch_ax = fig.add_axes([0.02, 0.02, 0.1, 0.1])
        patch_ax.imshow(patch_np)
        patch_ax.set_title("Patch")
        patch_ax.axis('off')
        
        # Adicionar boxes para cada recomendação
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
        
        for i, rec in enumerate(recommendations[:5]):  # Limitado a 5 recomendações
            box = rec["box"]
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            # Desenha retângulo
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Adiciona texto com score
            if show_scores:
                score_text = f"#{i+1} Score: {rec['score']:.2f}"
                ax1.text(
                    x1, y1-5, score_text, color='white', fontsize=10, 
                    weight='bold', bbox=dict(facecolor=color, alpha=0.7)
                )
                
        ax1.set_title("Recomendações de Colagem")
        ax1.axis('off')
        
        # Mostrar mapa de calor se solicitado
        if show_heatmap and 'ax2' in locals():
            hm = ax2.imshow(heatmap_np, cmap='hot')
            plt.colorbar(hm, ax=ax2, label='Score de Compatibilidade')
            ax2.set_title("Mapa de Calor de Compatibilidade")
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        return fig

    def simulate_paste_result(self, image, patch, box, blend_mode='alpha', return_tensor=False):
        """
        Simula o resultado da colagem de um patch em uma região da imagem.
        
        Esta função permite visualizar como ficaria o resultado da colagem
        usando diferentes modos de blending.
        
        Args:
            image: Tensor da imagem [3,H,W]
            patch: Tensor do patch [3,h,w]
            box: Coordenadas [x1,y1,x2,y2]
            blend_mode: Método de blending ('alpha', 'poisson', 'direct')
            return_tensor: Se deve retornar tensor ou numpy array
            
        Returns:
            Imagem com patch colado
        """
        import cv2
        
        # Converter para numpy para processamento
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
        
        # Garantir que valores estão entre 0-1
        image_np = np.clip(image_np, 0, 1)
        patch_np = np.clip(patch_np, 0, 1)
        
        # Converter para uint8 (0-255)
        image_uint8 = (image_np * 255).astype(np.uint8)
        patch_uint8 = (patch_np * 255).astype(np.uint8)
        
        # Extrair coordenadas
        x1, y1, x2, y2 = [int(coord) for coord in box]
        target_h, target_w = y2 - y1, x2 - x1
        
        # Redimensionar patch para o tamanho da região-alvo
        patch_resized = cv2.resize(patch_uint8, (target_w, target_h))
        
        # Criar cópia da imagem para manipulação
        result = image_uint8.copy()
        
        if blend_mode == 'alpha':
            # Alpha blending simples
            alpha = 0.7  # Opacidade do patch
            result[y1:y2, x1:x2] = cv2.addWeighted(
                result[y1:y2, x1:x2], 1-alpha, 
                patch_resized, alpha, 0
            )
        
        elif blend_mode == 'poisson':
            try:
                # Tentar Poisson blending se OpenCV suportar
                # Converte para BGR para OpenCV
                source = patch_resized
                target = result
                mask = np.ones(source.shape, source.dtype) * 255
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Aplica Poisson blending
                result = cv2.seamlessClone(
                    source, target, mask, center, 
                    cv2.NORMAL_CLONE
                )
            except Exception as e:
                print(f"Erro no Poisson blending: {e}")
                # Fallback para alpha blending
                alpha = 0.7
                result[y1:y2, x1:x2] = cv2.addWeighted(
                    result[y1:y2, x1:x2], 1-alpha, 
                    patch_resized, alpha, 0
                )
        
        else:  # modo 'direct'
            # Simplesmente sobrepõe o patch
            result[y1:y2, x1:x2] = patch_resized
        
        # Normaliza de volta para 0-1
        result = result.astype(np.float32) / 255.0
        
        # Retornar no formato apropriado
        if return_tensor:
            return torch.tensor(result).permute(2, 0, 1)
        return result
    
    # Adicione estes métodos à classe AttentionRecommender, após os métodos existentes

    def create_gaussian_heatmap(self, image_size, center, sigma=20):
        """
        Cria um mapa de calor gaussiano centrado em 'center'.
        
        Args:
            image_size: Tamanho da imagem (H, W)
            center: Centro da gaussiana (x, y)
            sigma: Desvio padrão da gaussiana
            
        Returns:
            Tensor do mapa de calor [H, W]
        """
        h, w = image_size
        x = torch.arange(0, w, dtype=torch.float32, device=self.device)
        y = torch.arange(0, h, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        x0, y0 = center
        
        # Calcula gaussiana 2D
        gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        # Normaliza para max=1
        if gaussian.max() > 0:
            gaussian = gaussian / gaussian.max()
            
        return gaussian

    def compute_loss(self, pred_heatmap, target_heatmap, negative_mask=None, use_focal=True, neg_weight=1.0):
        """
        Combina MSE, perda focal e uma perda de penalidade de background.
        
        Args:
            pred_heatmap: Mapa de calor predito [B,1,H,W]
            target_heatmap: Mapa de calor alvo [B,1,H,W]
            negative_mask: Máscara de áreas a penalizar [B,1,H,W]
            use_focal: Se deve usar perda focal
            neg_weight: Peso para a perda de background
            
        Returns:
            Tensor com valor da perda
        """
        # MSE padrão
        mse_loss = F.mse_loss(pred_heatmap, target_heatmap)
        
        # Perda de penalidade de background
        background_loss = 0.0
        if negative_mask is not None:
            # Redimensiona a máscara negativa para o tamanho do heatmap
            if negative_mask.shape[-2:] != pred_heatmap.shape[-2:]:
                negative_mask = F.interpolate(
                    negative_mask, 
                    size=pred_heatmap.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calcula a perda nas áreas "proibidas"
            # Queremos que a predição seja zero nessas áreas
            background_loss = F.mse_loss(pred_heatmap * negative_mask, torch.zeros_like(pred_heatmap))

        focal_loss = 0.0
        if use_focal:
            # Perda focal para lidar com desbalanceamento positivo/negativo
            eps = 1e-7
            pred_heatmap_focal = pred_heatmap.clamp(min=eps, max=1-eps)
            
            pos_mask = (target_heatmap >= 0.5).float()
            neg_mask = (target_heatmap < 0.5).float()
            
            pos_loss = -pos_mask * (1 - pred_heatmap_focal) ** 2 * torch.log(pred_heatmap_focal)
            neg_loss = -neg_mask * pred_heatmap_focal ** 2 * torch.log(1 - pred_heatmap_focal)
            
            focal_loss = torch.mean(pos_loss + neg_loss)
        
        # Combina as perdas: 50% MSE, 20% Focal, 30% Background
        return 0.5 * mse_loss + 0.2 * focal_loss + neg_weight * background_loss

    def prepare_training_sample(self, image, patch, original_box):
        """
        Prepara amostras para treinamento a partir de uma imagem e um patch.
        
        Args:
            image: Tensor da imagem [B,3,H,W] ou [3,H,W]
            patch: Tensor do patch [B,3,H,W] ou [3,H,W]
            original_box: Coordenadas originais do patch [x1,y1,x2,y2]
            
        Returns:
            Tensor do mapa de calor alvo [B,1,H,W]
        """
        # Garantir formato correto (adicionar dimensão de batch se necessário)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        
        # Calcular centro do box
        x1, y1, x2, y2 = original_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Criar mapa de calor alvo (gaussiana centrada na posição original)
        target_heatmap = self.create_gaussian_heatmap(
            image.shape[-2:],
            (center_x, center_y),
            sigma=max(x2-x1, y2-y1) / 2  # Sigma proporcional ao tamanho do patch
        )
        
        return target_heatmap.unsqueeze(0).unsqueeze(0)  # [B,1,H,W]

    def batch_process_recommendations(self, images, patches, top_k=5):
        """
        Processa múltiplas imagens e patches eficientemente em batch.
        
        Args:
            images: Tensor de imagens destino [B,3,H,W]
            patches: Tensor de patches [B,3,h,w]
            top_k: Número máximo de recomendações por imagem
            
        Returns:
            Lista de recomendações para cada imagem no batch
        """
        device = next(self.parameters()).device
        batch_size = images.shape[0]
        
        # Mover para o dispositivo correto
        images = images.to(device)
        patches = patches.to(device)
        
        # Gerar mapas de calor em batch
        with torch.no_grad():
            heatmaps, _ = self.forward(images, patch=patches, return_attention=True)
        
        # Extrair recomendações para cada imagem
        all_recommendations = []
        for b in range(batch_size):
            recommendations = self._extract_recommendations_from_heatmap(
                heatmaps[b, 0],
                images.shape[-2:],
                patches[b].shape[-2:],
                top_k=top_k
            )
            all_recommendations.append(recommendations)
        
        return all_recommendations

    def evaluate_recommendations(self, image, patch, original_box, recommendations=None):
        """
        Avalia a qualidade das recomendações comparando com a posição original.
        
        Args:
            image: Tensor da imagem [3,H,W]
            patch: Tensor do patch [3,h,w]
            original_box: Box original do patch [x1,y1,x2,y2]
            recommendations: Lista de recomendações (opcional)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        # Gera recomendações se não fornecidas
        if recommendations is None:
            recommendations = self.recommend_placements(image.unsqueeze(0), patch.unsqueeze(0))
        
        metrics = {}
        
        # Calcula centro da box original
        orig_cx = (original_box[0] + original_box[2]) / 2
        orig_cy = (original_box[1] + original_box[3]) / 2
        
        # Calcula distâncias para cada recomendação
        distances = []
        ious = []
        
        for rec in recommendations:
            box = rec['box']
            
            # Calcula centro da recomendação
            rec_cx = (box[0] + box[2]) / 2
            rec_cy = (box[1] + box[3]) / 2
            
            # Distância euclidiana normalizada
            img_diagonal = np.sqrt(image.shape[1]**2 + image.shape[2]**2)
            distance = np.sqrt((rec_cx - orig_cx)**2 + (rec_cy - orig_cy)**2) / img_diagonal
            distances.append(distance)
            
            # Calcula IoU
            iou = self._calculate_iou(box, original_box)
            ious.append(iou)
        
        # Calcula métricas
        if distances:
            metrics['min_distance'] = float(min(distances))
            metrics['mean_distance'] = float(np.mean(distances))
        
        if ious:
            metrics['max_iou'] = float(max(ious))
            metrics['mean_iou'] = float(np.mean(ious))
        
        # Calcula precision@k
        for k in [1, 3, 5]:
            if len(ious) >= k:
                metrics[f'p@{k}'] = float(sum(1 for iou in ious[:k] if iou > 0.5) / k)
        
        return metrics
    
    # Adicionar após o método evaluate_recommendations

    def evaluate_exogenous_patches(self, dataset, num_samples=15, save_dir=None):
        """
        Avalia o desempenho do recomendador usando patches exógenos (de outras imagens).
        
        Args:
            dataset: Dataset de validação
            num_samples: Número de exemplos exógenos para avaliar
            save_dir: Diretório para salvar resultados
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        metrics = {'recommendation_scores': []}
        
        num_images = len(dataset)
        if num_images < 2:
            print("Dataset muito pequeno para avaliação exógena.")
            return metrics
        
        for i in tqdm(range(num_samples), desc="Avaliando patches exógenos"):
            # Seleciona aleatoriamente uma imagem de origem e uma de destino
            source_idx, target_idx = random.sample(range(num_images), 2)
            
            source_sample = dataset[source_idx]
            target_sample = dataset[target_idx]
            
            # Pega um patch da imagem de origem
            if not source_sample['patches']:
                continue
            
            patch = source_sample['patches'][0].to(device)
            patch_box = source_sample['patch_boxes'][0].tolist()
            source_img = source_sample['image'].to(device)
            
            # Pega a imagem de destino
            target_img = target_sample['image'].to(device)
            
            # Gera recomendações
            recommendations = self.recommend_placements(
                target_img.unsqueeze(0), patch.unsqueeze(0), top_k=5
            )
            
            if not recommendations:
                continue
            
            metrics['recommendation_scores'].extend([rec['score'] for rec in recommendations])
            
            # Gera o heatmap para visualização
            with torch.no_grad():
                heatmap, _ = self.forward(
                    target_img.unsqueeze(0), patch=patch.unsqueeze(0), return_attention=True
                )
            
            # Salva a visualização
            if save_dir:
                fig = self._create_exogenous_visualization(
                    source_img, patch, patch_box, target_img, recommendations,
                    source_idx, target_idx, heatmap[0, 0]
                )
                fig_path = os.path.join(save_dir, f"exogenous_s{source_idx}_t{target_idx}.png")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # Calcula estatísticas finais
        results = {}
        if metrics['recommendation_scores']:
            scores = metrics['recommendation_scores']
            results['recommendation_scores'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        
        if save_dir:
            import json
            with open(os.path.join(save_dir, 'exogenous_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)
                
        return results

    def _create_exogenous_visualization(self, source_img, patch, patch_box, target_img, recommendations, source_idx, target_idx, heatmap=None):
        """Cria uma visualização detalhada da recomendação exógena."""
        source_np = source_img.cpu().permute(1, 2, 0).numpy()
        patch_np = patch.cpu().permute(1, 2, 0).numpy()
        target_np = target_img.cpu().permute(1, 2, 0).numpy()
        
        fig = plt.figure(figsize=(24, 6))
        
        # Imagem de Origem
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(source_np)
        ax1.set_title(f"Imagem de Origem (idx: {source_idx})")
        x1, y1, x2, y2 = patch_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax1.add_patch(rect)
        ax1.axis('off')
        
        # Patch Extraído
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(patch_np)
        ax2.set_title("Patch Extraído")
        ax2.axis('off')
        
        # Imagem de Destino
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(target_np)
        ax3.set_title(f"Imagem de Destino (idx: {target_idx})")
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
        for i, rec in enumerate(recommendations[:5]):
            box = rec['box']
            score = rec.get('score', 0)
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor=colors[i], facecolor='none')
            ax3.add_patch(rect)
            ax3.text(box[0], box[1]-5, f"#{i+1}: {score:.2f}", color='white', fontsize=9, weight='bold', bbox=dict(facecolor=colors[i], alpha=0.7, pad=1))
        ax3.axis('off')
        
        # Mapa de Compatibilidade
        ax4 = fig.add_subplot(1, 4, 4)
        if heatmap is not None:
            im = ax4.imshow(heatmap.cpu().numpy(), cmap='hot')
            fig.colorbar(im, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
        ax4.set_title("Mapa de Compatibilidade")
        ax4.axis('off')
        
        plt.tight_layout()
        return fig

    def _calculate_iou(self, box1, box2):
        """
        Calcula IoU (Intersection over Union) entre duas boxes.
        
        Args:
            box1: Primeira box [x1,y1,x2,y2]
            box2: Segunda box [x1,y1,x2,y2]
            
        Returns:
            Valor de IoU entre 0 e 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calcula área da interseção
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calcula áreas das boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calcula união
        union_area = box1_area + box2_area - inter_area
        
        # Calcula IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou

    def save_model(self, path):
        """
        Salva o modelo em um arquivo.
        
        Args:
            path: Caminho para salvar o modelo
        """
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva apenas os parâmetros treináveis (não o backbone)
        state_dict = {
            key: value for key, value in self.state_dict().items()
            if not key.startswith('backbone.') and not key.startswith('feature_extractor.')
        }
        
        # Adiciona metadados
        metadata = {
            'embedding_dim': self.embedding_dim,  # Corrigido: índice 2 é a camada Linear
            'num_heads': self.num_heads,
            'version': '1.0',
        }
        
        # Salva estado do modelo e metadados
        torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
        print(f"Modelo salvo em {path}")

    def load_model(self, path):
        """
        Carrega o modelo de um arquivo.
        
        Args:
            path: Caminho para o arquivo do modelo
            
        Returns:
            self (para permitir chamadas encadeadas)
        """
        # Carrega estado do modelo
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        
        # Carrega apenas os parâmetros treináveis
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        
        print(f"Modelo carregado de {path}")
        return self

    @property
    def device(self):
        """
        Retorna o dispositivo onde o modelo está.
        """
        return next(self.parameters()).device
    
    def train(self, mode=True):
        """
        Sobrescreve nn.Module.train() para definir o modo dos módulos treináveis,
        mantendo o backbone em modo de avaliação.
        """
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.eval()
        return self
