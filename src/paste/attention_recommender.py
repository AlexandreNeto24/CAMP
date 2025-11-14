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
import json
# Configuração de caminhos para garantir importações corretas
project_root = Path(__file__).parent.parent.parent  # src/paste/ → src/ → project_root
sys.path.insert(0, str(project_root))

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
    def __init__(self, backbone_model, embedding_dim=2048, num_heads=16, dropout_rate=0.3):
        super().__init__()
        self.backbone = backbone_model
        self.feature_extractor = self.backbone.get_encoder()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        feature_dim = self.backbone.feature_dim
        self.embedding_dim = embedding_dim

        # segurança: embedding divisible by num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            features = self.feature_extractor(dummy)
            if isinstance(features, (list, tuple)):
                concat_dim = features[-2].shape[1] + features[-1].shape[1]
            else:
                concat_dim = features.shape[1]

        # Camadas para projetar as features do patch (query) e da imagem (key)
        # Use GroupNorm em vez de BatchNorm para estabilidade com small batches
        gn_groups = min(32, max(1, embedding_dim // 16))
        self.query_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )

        # branch para source/context token (crop expandido)
        self.src_query_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )
        # embedding da box normalizada (cx,cy,w,h) -> E
        self.box_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embedding_dim)
        )

        # Fusion: reduz 2*E -> mid_channels antes do decoder (mantém informação)
        mid_channels = 64
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embedding_dim * 2, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder do heatmap agora espera mid_channels em vez de 1
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
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
        
        # Adiciona o transformer encoder aqui:
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads),
            num_layers=2
        )
        
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
                # Redimensiona features[-2] para o tamanho de features[-1]
                target_size = features[-1].shape[-2:]
                upsampled = F.interpolate(features[-2], size=target_size, mode='bilinear', align_corners=False)
                return torch.cat([upsampled, features[-1]], dim=1)
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

    def forward(self, image, lesion_coords=None, patch=None, source_context=None, source_box=None, weight_mask=None, return_attention=False, return_intermediate=False):
        """
        Forward pass:
          - image: target image tensor [B,3,H,W]
          - patch: patch tensor [B,3,ph,pw]
          - source_context: optional crop around original box [B,3,ch,cw] (conditioning)
          - source_box: optional box [B,4] em pixels na escala da imagem original
        """
        device = self.device
        image = image.to(device)
        patch = patch.to(device)
        if source_context is not None:
            source_context = source_context.to(device)
        if source_box is not None:
            source_box = source_box.to(device)

        # 1. Extrai features da imagem alvo e do patch
        image_features = self.extract_features(image)   # [B, C, Hf, Wf]
        patch_features = self.extract_features(patch)   # [B, C, hp, wp]

        # 2. Projeta para embedding
        query = self.query_conv(patch_features)         # [B,E,hq,wq]
        key = self.key_conv(image_features)             # [B,E,H,W]

        # 3. Cria token de contexto a partir de source_context ou source_box (ROIAlign)
        src_token = None
        if source_context is not None:
            src_feat = self.extract_features(source_context)   # [B,C,hs,ws]
            src_proj = self.src_query_conv(src_feat)           # [B,E,hs,ws]
            src_token = F.adaptive_avg_pool2d(src_proj, 1).view(src_proj.size(0), -1)  # [B,E]
        elif source_box is not None:
            # espera source_box em pixels (x1,y1,x2,y2) por amostra
            Bf, _, Hf, Wf = image_features.shape
            # converte boxes para rois (batch_idx, x1,y1,x2,y2)
            rois = []
            for b in range(source_box.size(0)):
                x1,y1,x2,y2 = source_box[b].tolist()
                rois.append(torch.tensor([b, x1, y1, x2, y2], device=device, dtype=torch.float32))
            rois = torch.stack(rois, dim=0)
            spatial_scale = float(Hf) / float(image.shape[-2])
            roi_feats = roi_align(image_features, rois, output_size=(8,8), spatial_scale=spatial_scale, aligned=True)
            src_proj = self.src_query_conv(roi_feats)   # [B,E,8,8]
            src_token = F.adaptive_avg_pool2d(src_proj, 1).view(src_proj.size(0), -1)  # [B,E]

            # add box embedding (normalize by image size)
            img_h, img_w = image.shape[-2], image.shape[-1]
            cx = ((source_box[:,0] + source_box[:,2]) / 2.0) / float(img_w)
            cy = ((source_box[:,1] + source_box[:,3]) / 2.0) / float(img_h)
            bw = ((source_box[:,2] - source_box[:,0]) / float(img_w)).clamp(min=1e-6)
            bh = ((source_box[:,3] - source_box[:,1]) / float(img_h)).clamp(min=1e-6)
            box_norm = torch.stack([cx, cy, bw, bh], dim=1)  # [B,4]
            box_emb = self.box_mlp(box_norm)                  # [B,E]
            src_token = src_token + box_emb

        # 4. Flatten + codificação posicional
        B, E, h, w = query.shape
        query_flat = query.view(B, E, -1).permute(0, 2, 1)  # [B, qlen, E]
        B, E, H, W = key.shape
        key_flat = key.view(B, E, -1).permute(0, 2, 1)      # [B, klen, E]

        # se houver token de contexto, anexa ao final da sequência de query
        if src_token is not None:
            src_token_exp = src_token.unsqueeze(1)  # [B,1,E]
            query_flat = torch.cat([query_flat, src_token_exp], dim=1)  # [B, qlen+1, E]

        query_flat = self.position_encoder(query_flat)
        key_flat = self.position_encoder(key_flat)

        # 5. Transformer encoder nas features da imagem
        key_trans = self.transformer_encoder(key_flat)  # [B, klen, E]
        key_trans_spatial = key_trans.permute(0, 2, 1).view(B, E, H, W)

        # 6. Cross-attention entre patch (queries) e imagem (keys/values)
        attn_out, attn_weights = self.attention(query=query_flat, key=key_flat, value=key_flat)
        # se existiu token extra no final da query, ignorar quando reconstruir espacial
        qlen = h * w
        attn_spatial = attn_out[:, :qlen, :].permute(0, 2, 1).view(B, E, h, w)

        # 7. Upsample + fusion + decoder (como antes)
        attn_out_upsampled = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=False)
        concat = torch.cat([key_trans_spatial, attn_out_upsampled], dim=1)  # [B, 2E, H, W]
        fused = self.fusion_conv(concat)
        heatmap = self.heatmap_decoder(fused)
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)

        # comportamento compatível:
        # - padrão: retorna heatmap
        # - return_attention=True: retorna (heatmap, attn_weights)
        # - return_intermediate=True: retorna (heatmap, patch_features, image_features)  (útil para perdas extra)
        if return_attention and return_intermediate:
            return heatmap, attn_weights, patch_features, image_features
        if return_attention:
            return heatmap, attn_weights
        if return_intermediate:
            return heatmap, patch_features, image_features
        return heatmap

    def recommend_placements(self, image, patch, patch_box=None, top_k=5):
        """
        Recomenda os melhores locais para colagem do patch na imagem,
        considerando o tamanho original do patch (bounding box).
        
        Args:
            image: Tensor da imagem destino [B,3,H,W]
            patch: Tensor do patch de lesão [B,3,H,W]
            patch_box: Bounding box original do patch [x1, y1, x2, y2] (opcional)
            top_k: Número máximo de recomendações
            
        Returns:
            Lista de dicionários com recomendações {box, score, rank}
        """
        device = next(self.parameters()).device
        image = image.to(device)
        patch = patch.to(device)
        
        # Gerar mapa de calor através do mecanismo de atenção
        with torch.no_grad():
            heatmap, attention_map = self.forward(image, patch=patch, return_attention=True)
        
        # Determina o tamanho do patch para recomendação
        if patch_box is not None:
            patch_h = int(patch_box[3] - patch_box[1])
            patch_w = int(patch_box[2] - patch_box[0])
            patch_size = (patch_h, patch_w)
        else:
            patch_size = patch.shape[-2:]
        
        # Extrai recomendações usando o tamanho original do patch
        recommendations = self._extract_recommendations_from_heatmap(
            heatmap[0, 0], 
            image.shape[-2:], 
            patch_size, 
            top_k=top_k
        )
        
        return recommendations

    def _extract_recommendations_from_heatmap(self, heatmap, image_size, patch_size, top_k=5):
        """
        Extrai coordenadas de recomendação do mapa de calor considerando regiões de alta compatibilidade.
        """
        import numpy as np

        h, w = heatmap.shape
        patch_h, patch_w = patch_size

        # Escala do mapa de calor para imagem original
        y_scale = image_size[0] / h
        x_scale = image_size[1] / w

        # Percorre o heatmap com janela do tamanho do patch
        scores = []
        for y in range(0, h - patch_h + 1, max(1, patch_h // 4)):
            for x in range(0, w - patch_w + 1, max(1, patch_w // 4)):
                window = heatmap[y:y+patch_h, x:x+patch_w]
                score = np.mean(window.cpu().numpy())
                scores.append((score, x, y))

        # Seleciona top_k regiões com maior média
        scores = sorted(scores, key=lambda tup: tup[0], reverse=True)
        recommendations = []
        used_boxes = []

        for i, (score, x, y) in enumerate(scores):
            # Coordenada central na imagem original
            center_y = int((y + patch_h // 2) * y_scale)
            center_x = int((x + patch_w // 2) * x_scale)

            # Gera box centralizada na região
            half_h = int(patch_h * y_scale // 2)
            half_w = int(patch_w * x_scale // 2)
            box = [
                max(0, center_x - half_w),
                max(0, center_y - half_h),
                min(image_size[1], center_x + half_w),
                min(image_size[0], center_y + half_h)
            ]

            # NMS: evita boxes muito sobrepostas
            overlap = False
            for prev_box in used_boxes:
                iou = self._calculate_iou(box, prev_box)
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


    def compute_loss(self, pred_heatmap, target_heatmap, segmentor_mask=None, 
                 patch_features=None, image_features=None, 
                 contrastive_weight=0.0, context_weight=0.7, 
                 use_context_primary=False):  # NOVO FLAG
        """
        Loss com dois modos:
          - use_context_primary=False (início): target_heatmap dominante (gaussiana)
          - use_context_primary=True (final): segmentor_mask dominante (contexto)
        """
        pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1-1e-7)
        target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)
        
        if use_context_primary and segmentor_mask is not None:
            # MODO: contexto como target primário
            segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)
            if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
                segmentor_mask = F.interpolate(segmentor_mask, 
                                              size=pred_heatmap.shape[-2:], 
                                              mode='bilinear', align_corners=False)
            
            # Loss primária: CONTEXTO com soft target 0.9
            loss_primary = F.mse_loss(pred_heatmap, segmentor_mask)
            
            # Loss secundária: penaliza FORA do contexto COM PESO MAIOR
            outside_context_mask = 1.0 - segmentor_mask
            loss_regularization = F.mse_loss(pred_heatmap * outside_context_mask, 
                                             torch.zeros_like(pred_heatmap))
            
            # AJUSTADO: aumenta peso da regularização de 0.2 para 0.8
            total_loss = loss_primary + 0.8 * loss_regularization
            
        else:
            # MODO CLÁSSICO: gaussiana como target (primeiras épocas)
            loss_primary = F.mse_loss(pred_heatmap, target_heatmap)
            
            # Loss contextual: regularização leve
            loss_context = 0.0
            if segmentor_mask is not None:
                segmentor_mask = segmentor_mask.clamp(min=0.0, max=1.0)
                if segmentor_mask.shape[-2:] != pred_heatmap.shape[-2:]:
                    segmentor_mask = F.interpolate(segmentor_mask, 
                                                  size=pred_heatmap.shape[-2:], 
                                                  mode='bilinear', align_corners=False)
                outside_context_mask = 1.0 - segmentor_mask
                loss_context = F.mse_loss(pred_heatmap * outside_context_mask, 
                                         torch.zeros_like(pred_heatmap))
            
            # REDUZIDO: em modo gaussiana, contexto é regularização fraca (0.1)
            total_loss = loss_primary + 0.1 * loss_context
        
        if contrastive_weight > 0.0 and patch_features is not None and image_features is not None:
            c_loss = self.compute_contrastive_loss(patch_features, image_features, pred_heatmap)
            total_loss = total_loss + contrastive_weight * c_loss
        
        return total_loss

    def compute_contrastive_loss(self, patch_features, image_features, pred_heatmap, n_neg=64, tau=0.07):
        """
        InfoNCE-like loss:
          - patch_features: [B, C, hp, wp] (before query_conv)
          - image_features: [B, C, Hf, Wf] (before key_conv)
          - pred_heatmap: [B, 1, H_img, W_img] (output heatmap in image coords)
        Returns scalar loss that pushes patch_emb near positive region emb and away from negatives.
        """
        # Reduce to embeddings
        # patch emb: global avg pool -> [B, C]
        patch_emb = F.adaptive_avg_pool2d(patch_features, 1).view(patch_features.size(0), -1)  # [B, C]
        patch_emb = F.normalize(patch_emb, dim=1)

        # Resize pred_heatmap to feature map size and compute weighted region embedding (positive)
        _, _, Hf, Wf = image_features.shape
        heat_resized = F.interpolate(pred_heatmap, size=(Hf, Wf), mode='bilinear', align_corners=False)  # [B,1,Hf,Wf]
        heat_flat = heat_resized.view(image_features.size(0), 1, -1)  # [B,1,Hf*Wf]

        # image features -> [B, C, Hf*Wf]
        img_flat = image_features.view(image_features.size(0), image_features.size(1), -1)  # [B,C,L]
        # positive embedding: weighted average by heatmap
        pos_emb = torch.bmm(img_flat, heat_flat.permute(0,2,1)).squeeze(-1)  # [B, C]
        pos_emb = F.normalize(pos_emb, dim=1)

        # negatives: sample random spatial positions from img_flat
        B, C, L = img_flat.shape
        neg_idxs = torch.randint(0, L, (B, n_neg), device=img_flat.device)
        neg_embs = []
        for b in range(B):
            neg_sel = img_flat[b:b+1, :, neg_idxs[b]]  # [1, C, n_neg]
            neg_mean = neg_sel.mean(dim=2).squeeze(0)  # [C]
            neg_embs.append(neg_mean)
        neg_embs = torch.stack(neg_embs, dim=0)  # [B, C]
        neg_embs = F.normalize(neg_embs, dim=1)

        # compute similarities
        pos_sim = (patch_emb * pos_emb).sum(dim=1) / tau  # [B]
        neg_sim = (patch_emb * neg_embs).sum(dim=1) / tau  # [B]

        # InfoNCE: -log (exp(pos) / (exp(pos) + sum exp(neg)))
        logits_pos = pos_sim
        logits_neg = neg_sim
        denom = torch.exp(logits_pos) + torch.exp(logits_neg)
        loss = -torch.log(torch.exp(logits_pos) / (denom + 1e-8)).mean()
        return loss

    # def compute_loss(self, pred_heatmap, target_heatmap, negative_mask=None, use_focal=True, neg_weight=0.0, use_ranking=True):
    #     """
    #     Combina MSE, perda focal, penalidade de background e entropia para robustez.
    #     Garante que não há NaN/Inf nos cálculos.
    #     """
    #     # Clamp para evitar NaN/Inf
    #     pred_heatmap = pred_heatmap.clamp(min=1e-7, max=1-1e-7)
    #     target_heatmap = target_heatmap.clamp(min=0.0, max=1.0)

    #     # Debug: verifica valores
    #     if torch.isnan(pred_heatmap).any() or torch.isinf(pred_heatmap).any():
    #         print("NaN/Inf detectado em pred_heatmap!")
    #         pred_heatmap = torch.nan_to_num(pred_heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    #     if torch.isnan(target_heatmap).any() or torch.isinf(target_heatmap).any():
    #         print("NaN/Inf detectado em target_heatmap!")
    #         target_heatmap = torch.nan_to_num(target_heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    #     # MSE padrão
    #     mse_loss = F.mse_loss(pred_heatmap, target_heatmap)

    #     # Perda de penalidade de background
    #     background_loss = 0.0
    #     if negative_mask is not None:
    #         if negative_mask.shape[-2:] != pred_heatmap.shape[-2:]:
    #             negative_mask = F.interpolate(
    #                 negative_mask, 
    #                 size=pred_heatmap.shape[-2:], 
    #                 mode='bilinear', 
    #                 align_corners=False
    #             )
    #         background_loss = F.mse_loss(pred_heatmap * negative_mask, torch.zeros_like(pred_heatmap))

    #     # Focal loss
    #     focal_loss = 0.0
    #     if use_focal:
    #         eps = 1e-7
    #         pred_heatmap_focal = pred_heatmap.clamp(min=eps, max=1-eps)
    #         pos_mask = (target_heatmap >= 0.5).float()
    #         neg_mask = (target_heatmap < 0.5).float()
    #         pos_loss = -pos_mask * (1 - pred_heatmap_focal) ** 2 * torch.log(pred_heatmap_focal)
    #         neg_loss = -neg_mask * pred_heatmap_focal ** 2 * torch.log(1 - pred_heatmap_focal)
    #         focal_loss = torch.mean(pos_loss + neg_loss)

    #     # Ranking loss
    #     ranking_loss = 0.0
    #     if use_ranking:
    #         pos_score = torch.max(pred_heatmap)
    #         neg_score = torch.mean(pred_heatmap[target_heatmap < 0.1])
    #         margin = 0.1
    #         ranking_loss = F.relu(margin - (pos_score - neg_score))

    #     # Entropy loss para penalizar mapas uniformes
    #     entropy_loss = -torch.mean(pred_heatmap * torch.log(pred_heatmap + 1e-8))

    #     # Combina as perdas
    #     total_loss = (
    #         0.5 * mse_loss +
    #         0.2 * focal_loss +
    #         neg_weight * background_loss +
    #         0.3 * ranking_loss +
    #         0.1 * entropy_loss
    #     )

    #     # Debug: verifica se a loss é NaN
    #     if torch.isnan(total_loss):
    #         print("NaN detectado na loss! Retornando zero.")
    #         return torch.tensor(0.0, device=pred_heatmap.device, requires_grad=True)

    #     return total_loss

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
            original_box: Box original do patch [x1, y1, x2, y2]
            recommendations: Lista de recomendações (opcional)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        # Gera recomendações se não fornecidas
        if recommendations is None:
            recommendations = self.recommend_placements(image.unsqueeze(0), patch.unsqueeze(0), patch_box=original_box)
        
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
                target_img.unsqueeze(0), patch.unsqueeze(0), patch_box=patch_box, top_k=5
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
    
    def evaluate_patch_on_normal(self, val_dataset, normal_images_dir, save_dir, num_samples=10):
        """
        Avalia a colagem de patches de instância (da validação) em imagens normais.
        Salva subplots mostrando: imagem de origem, bounding box do patch, patch extraído,
        imagem normal (destino) com recomendações, e mapa de compatibilidade.
        """
        import glob
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)
        normal_img_paths = sorted(glob.glob(os.path.join(normal_images_dir, "*.jpg")) +
                                glob.glob(os.path.join(normal_images_dir, "*.png")))
        if not normal_img_paths:
            print(f"Nenhuma imagem encontrada em {normal_images_dir}")
            return

        device = self.device
        metrics = {'recommendation_scores': []}

        for i in range(num_samples):
            # Seleciona aleatoriamente uma imagem de validação e uma normal
            val_idx = random.randint(0, len(val_dataset)-1)
            val_sample = val_dataset[val_idx]
            normal_img_path = random.choice(normal_img_paths)

            # Carrega imagem de validação (origem)
            val_image = val_sample['image'].to(device)
            if not val_sample['patches']:
                continue
            patch = val_sample['patches'][0].to(device)
            patch_box = val_sample['patch_boxes'][0].tolist()

            # Carrega imagem normal (destino)
            normal_image = Image.open(normal_img_path).convert('RGB')
            normal_image = normal_image.resize((512, 512))
            normal_np = np.array(normal_image)
            normal_tensor = torch.from_numpy(normal_np).permute(2, 0, 1).float() / 255.0
            normal_tensor = normal_tensor.to(device)

            # Recomenda colagem
            recommendations = self.recommend_placements(
                normal_tensor.unsqueeze(0), patch.unsqueeze(0), patch_box=patch_box, top_k=5
            )
            metrics['recommendation_scores'].extend([rec['score'] for rec in recommendations])

            # Gera heatmap
            with torch.no_grad():
                heatmap, _ = self.forward(
                    normal_tensor.unsqueeze(0), patch=patch.unsqueeze(0), return_attention=True
                )

            # Visualização
            fig = plt.figure(figsize=(22, 6))
            # 1. Imagem de origem + bounding box
            ax1 = fig.add_subplot(1, 5, 1)
            val_np = val_image.cpu().permute(1, 2, 0).numpy()
            ax1.imshow(val_np)
            x1, y1, x2, y2 = patch_box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
            ax1.set_title("Imagem de Validação (Origem)")
            ax1.axis('off')

            # 2. Patch extraído
            ax2 = fig.add_subplot(1, 5, 2)
            patch_np = patch.cpu().permute(1, 2, 0).numpy()
            ax2.imshow(patch_np)
            ax2.set_title("Patch Extraído")
            ax2.axis('off')

            # 3. Imagem normal (destino)
            ax3 = fig.add_subplot(1, 5, 3)
            normal_np_disp = normal_tensor.cpu().permute(1, 2, 0).numpy()
            ax3.imshow(normal_np_disp)
            ax3.set_title("Imagem Normal (Destino)")
            ax3.axis('off')

            # 4. Recomendações na imagem normal
            ax4 = fig.add_subplot(1, 5, 4)
            ax4.imshow(normal_np_disp)
            colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
            for j, rec in enumerate(recommendations[:5]):
                box = rec['box']
                score = rec.get('score', 0)
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor=colors[j], facecolor='none')
                ax4.add_patch(rect)
                ax4.text(box[0], box[1]-5, f"#{j+1}: {score:.2f}", color='white', fontsize=9, weight='bold', bbox=dict(facecolor=colors[j], alpha=0.7, pad=1))
            ax4.set_title("Recomendações na Imagem Normal")
            ax4.axis('off')

            # 5. Mapa de compatibilidade
            ax5 = fig.add_subplot(1, 5, 5)
            hm = ax5.imshow(heatmap[0, 0].cpu().numpy(), cmap='hot')
            plt.colorbar(hm, ax=ax5, orientation='vertical', fraction=0.046, pad=0.04)
            ax5.set_title("Mapa de Compatibilidade")
            ax5.axis('off')

            plt.tight_layout()
            fig_path = os.path.join(save_dir, f"normal_patch_eval_{i}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

        # Estatísticas finais
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
        with open(os.path.join(save_dir, 'normal_patch_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print("Avaliação patch em imagens normais concluída.")
        print(json.dumps(results, indent=2))
        
    def train_recommender(self, train_dataset, val_dataset, run_dir,
                          epochs=30, batch_size=4, lr=5e-4, device='cuda',
                          accumulation_steps=4, num_workers=4, save_name="best_model.pt"):
        """
        Treina este AttentionRecommender usando os datasets fornecidos.
        """
        import time
        from torch.utils.data import DataLoader
        from src.paste.attention_dataset import attention_collate

        os.makedirs(run_dir, exist_ok=True)
        
        # NOVO: Cria pasta para debug/visualização de inputs
        debug_inputs_dir = os.path.join(run_dir, "debug_training_inputs")
        os.makedirs(debug_inputs_dir, exist_ok=True)
        
        if isinstance(device, str) and device.isdigit():
            device = f'cuda:{device}'
        if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        device = torch.device(device)
        self.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=attention_collate, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=attention_collate, num_workers=max(1, num_workers//2), pin_memory=True)

        # NOVO: Visualiza primeiras amostras de treino e validação
        self._visualize_training_inputs(train_loader, val_loader, debug_inputs_dir, device, num_samples=6)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # ALTERADO: começa contexto MUITO mais cedo (10% das épocas)
            switch_frac = 0.10  # começa contexto na época 5 (em vez de 15)
            use_context_primary = AttentionRecommender.get_loss_mode(epoch, epochs, switch_epoch_frac=switch_frac)
            
            # NOVO: context_weight com ramp agressivo
            if use_context_primary:
                # Em modo CONTEXTO: começa em 0.5, vai até 1.0
                progress_in_context = (epoch - int(epochs * switch_frac)) / (epochs - int(epochs * switch_frac))
                context_wt = 0.5 + 0.5 * progress_in_context  # 0.5 -> 1.0 (mais rápido)
                contrastive_wt = 0.02
            else:
                # Em modo GAUSSIANA: contexto é MUITO fraco
                context_wt = 0.05  # REDUZIDO de 0.15
                contrastive_wt = 0.05
            
            mode_str = "CONTEXTO" if use_context_primary else "GAUSSIANA"
            print(f"\nÉpoca {epoch+1}/{epochs} (mode={mode_str}, ctx_wt={context_wt:.2f})")
            
            t0 = time.time()
            self.train()
            epoch_loss = 0.0
            
            for i, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch+1}/{epochs}")):
                images = batch['images'].to(device)
                patches = batch['patches'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                sample_indices = batch['sample_indices'].to(device)
                weight_masks = batch.get('weight_masks', None)
                segmentor_masks = batch.get('segmentor_target_mask', None)
                
                if weight_masks is not None:
                    weight_masks = weight_masks.to(device)
                if segmentor_masks is not None:
                    segmentor_masks = segmentor_masks.to(device)

                if patches.numel() == 0:
                    continue

                batch_images = images[sample_indices]
                pred = self(batch_images, patch=patches, weight_mask=weight_masks, return_intermediate=True)
                pred_heatmaps, patch_feats, img_feats = pred
                
                target = torch.nn.functional.interpolate(heatmaps.unsqueeze(1),
                                                     size=pred_heatmaps.shape[-2:], mode='bilinear', align_corners=False)
                
                if segmentor_masks is not None:
                    segmentor_masks_aligned = segmentor_masks[sample_indices]
                else:
                    segmentor_masks_aligned = None
                
                # MUDANÇA: passa context_weight e contrastive_weight dinâmicos
                loss = self.compute_loss(pred_heatmaps, target, 
                                        segmentor_mask=segmentor_masks_aligned,
                                        patch_features=patch_feats, 
                                        image_features=img_feats, 
                                        contrastive_weight=contrastive_wt,  # DINÂMICO
                                        context_weight=context_wt,          # DINÂMICO
                                        use_context_primary=use_context_primary)

                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps

            avg_train_loss = epoch_loss / max(1, len(train_loader))
            train_losses.append(avg_train_loss)

            # Validation (mesmos parâmetros de loss que treino)
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    images = batch['images'].to(device)
                    patches = batch['patches'].to(device)
                    heatmaps = batch['heatmaps'].to(device)
                    sample_indices = batch['sample_indices'].to(device)
                    weight_masks = batch.get('weight_masks', None)
                    segmentor_masks = batch.get('segmentor_target_mask', None)
                    
                    if weight_masks is not None:
                        weight_masks = weight_masks.to(device)
                    if segmentor_masks is not None:
                        segmentor_masks = segmentor_masks.to(device)
                    
                    if patches.numel() == 0:
                        continue
                    
                    batch_images = images[sample_indices]
                    pred = self(batch_images, patch=patches, weight_mask=weight_masks, return_intermediate=True)
                    pred_heatmaps, patch_feats, img_feats = pred
                    
                    target = torch.nn.functional.interpolate(heatmaps.unsqueeze(1),
                                                             size=pred_heatmaps.shape[-2:], mode='bilinear', align_corners=False)
                    
                    if segmentor_masks is not None:
                        segmentor_masks_aligned = segmentor_masks[sample_indices]
                    else:
                        segmentor_masks_aligned = None
                    
                    loss = self.compute_loss(pred_heatmaps, target, 
                                            segmentor_mask=segmentor_masks_aligned,
                                            patch_features=patch_feats, 
                                            image_features=img_feats, 
                                            contrastive_weight=contrastive_wt,   # DINÂMICO
                                            context_weight=context_wt,           # DINÂMICO
                                            use_context_primary=use_context_primary)
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} time={time.time()-t0:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(os.path.join(run_dir, save_name))
                print("Saved best model ->", os.path.join(run_dir, save_name))

        # try to plot losses
        try:
            plt.figure(figsize=(8,4)); plt.plot(train_losses,label='train'); plt.plot(val_losses,label='val'); plt.legend()
            plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()
        except Exception:
            pass

        # load best
        best_path = os.path.join(run_dir, save_name)
        if os.path.exists(best_path):
            try:
                self.load_model(best_path)
            except Exception:
                try:
                    self.load_state_dict(torch.load(best_path, map_location=device), strict=False)
                except Exception:
                    print("Warning: couldn't fully load best model")
        return self

    def _visualize_training_inputs(self, train_loader, val_loader, debug_dir, device, num_samples=6):
        """
        Visualiza amostras de treino e validação: imagens, patches, máscaras, targets.
        Guarda subplots organizados em debug_dir.
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            debug_dir: Diretório para guardar visualizações
            device: Dispositivo (cpu/cuda)
            num_samples: Número de batches a visualizar de cada split
        """
        os.makedirs(debug_dir, exist_ok=True)
        
        # --- TREINO ---
        train_viz_dir = os.path.join(debug_dir, "train_samples")
        os.makedirs(train_viz_dir, exist_ok=True)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_samples:
                break
            
            images = batch['images']  # [B, 3, H, W]
            patches = batch['patches']  # [N, 3, h, w]
            heatmaps = batch['heatmaps']  # [N, H, W]
            segmentor_masks = batch.get('segmentor_target_mask', None)  # [B, 1, H, W]
            sample_indices = batch['sample_indices']  # [N]
            
            # Número de imagens e patches neste batch
            B = images.shape[0]
            N = patches.shape[0] if patches.numel() > 0 else 0
            
            # Cria figura com múltiplos subplots
            # Mostra: imagem, segmentor_mask, patches, heatmaps
            n_cols = 4
            n_rows = min(B, 3)  # máx 3 imagens por batch
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for row_idx in range(n_rows):
                img_idx = row_idx
                img = images[img_idx].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
                img = np.clip(img, 0, 1)
                
                # Col 0: Imagem original
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f"Imagem {img_idx}")
                axes[row_idx, 0].axis('off')
                
                # Col 1: Máscara de segmentação (contexto)
                if segmentor_masks is not None:
                    seg_mask = segmentor_masks[img_idx, 0].cpu().numpy()
                    axes[row_idx, 1].imshow(seg_mask, cmap='gray')
                    axes[row_idx, 1].set_title(f"Seg Mask {img_idx}")
                else:
                    axes[row_idx, 1].text(0.5, 0.5, "N/A", ha='center', va='center')
                    axes[row_idx, 1].set_title("Seg Mask (N/A)")
                axes[row_idx, 1].axis('off')
                
                # Col 2: Patches da imagem (mostra primeiro patch)
                patches_from_img = []
                for p_idx in range(N):
                    if sample_indices[p_idx] == img_idx:
                        patches_from_img.append(p_idx)
                
                if patches_from_img:
                    patch_idx = patches_from_img[0]
                    patch = patches[patch_idx].cpu().numpy()
                    patch = np.transpose(patch, (1, 2, 0))
                    patch = np.clip(patch, 0, 1)
                    axes[row_idx, 2].imshow(patch)
                    axes[row_idx, 2].set_title(f"Patch (1º de {len(patches_from_img)})")
                else:
                    axes[row_idx, 2].text(0.5, 0.5, "Sem patches", ha='center', va='center')
                    axes[row_idx, 2].set_title("Patch (N/A)")
                axes[row_idx, 2].axis('off')
                
                # Col 3: Target heatmap (gaussiana) do 1º patch
                if patches_from_img:
                    patch_idx = patches_from_img[0]
                    heatmap = heatmaps[patch_idx].cpu().numpy()
                    axes[row_idx, 3].imshow(heatmap, cmap='hot')
                    axes[row_idx, 3].set_title(f"Target Heatmap (Gaussiana)")
                else:
                    axes[row_idx, 3].text(0.5, 0.5, "N/A", ha='center', va='center')
                    axes[row_idx, 3].set_title("Target Heatmap (N/A)")
                axes[row_idx, 3].axis('off')
            
            plt.suptitle(f"Treino - Batch {batch_idx} | B={B}, N={N} patches", fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(train_viz_dir, f"batch_{batch_idx:02d}.png"), dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Visualizações de treino salvas em: {train_viz_dir}")
        
        # --- VALIDAÇÃO ---
        val_viz_dir = os.path.join(debug_dir, "val_samples")
        os.makedirs(val_viz_dir, exist_ok=True)
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
            
            images = batch['images']  # [B, 3, H, W]
            patches = batch['patches']  # [N, 3, h, w]
            heatmaps = batch['heatmaps']  # [N, H, W]
            segmentor_masks = batch.get('segmentor_target_mask', None)  # [B, 1, H, W]
            sample_indices = batch['sample_indices']  # [N]
            
            B = images.shape[0]
            N = patches.shape[0] if patches.numel() > 0 else 0
            
            n_cols = 4
            n_rows = min(B, 3)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for row_idx in range(n_rows):
                img_idx = row_idx
                img = images[img_idx].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0, 1)
                
                # Col 0: Imagem original
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f"Imagem {img_idx}")
                axes[row_idx, 0].axis('off')
                
                # Col 1: Máscara de segmentação
                if segmentor_masks is not None:
                    seg_mask = segmentor_masks[img_idx, 0].cpu().numpy()
                    axes[row_idx, 1].imshow(seg_mask, cmap='gray')
                    axes[row_idx, 1].set_title(f"Contexto Mask")
                else:
                    axes[row_idx, 1].text(0.5, 0.5, "N/A", ha='center', va='center')
                    axes[row_idx, 1].set_title("Contexto (N/A)")
                axes[row_idx, 1].axis('off')
                
                # Col 2: Patches
                patches_from_img = []
                for p_idx in range(N):
                    if sample_indices[p_idx] == img_idx:
                        patches_from_img.append(p_idx)
                
                if patches_from_img:
                    patch_idx = patches_from_img[0]
                    patch = patches[patch_idx].cpu().numpy()
                    patch = np.transpose(patch, (1, 2, 0))
                    patch = np.clip(patch, 0, 1)
                    axes[row_idx, 2].imshow(patch)
                    axes[row_idx, 2].set_title(f"Patch ({len(patches_from_img)} total)")
                else:
                    axes[row_idx, 2].text(0.5, 0.5, "Sem patches", ha='center', va='center')
                    axes[row_idx, 2].set_title("Patch (N/A)")
                axes[row_idx, 2].axis('off')
                
                # Col 3: Target heatmap
                if patches_from_img:
                    patch_idx = patches_from_img[0]
                    heatmap = heatmaps[patch_idx].cpu().numpy()
                    axes[row_idx, 3].imshow(heatmap, cmap='hot')
                    axes[row_idx, 3].set_title(f"Target (Gaussiana)")
                else:
                    axes[row_idx, 3].text(0.5, 0.5, "N/A", ha='center', va='center')
                    axes[row_idx, 3].set_title("Target (N/A)")
                axes[row_idx, 3].axis('off')
            
            plt.suptitle(f"Validação - Batch {batch_idx} | B={B}, N={N} patches", fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(val_viz_dir, f"batch_{batch_idx:02d}.png"), dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Visualizações de validação salvas em: {val_viz_dir}")
        print(f"\n✓ Todas as visualizações de input guardam em: {debug_dir}")

    def evaluate_recommender(self, val_dataset, num_samples=10, save_dir=None):
        """
        Avalia com diagnóstico visual de contexto vs objeto.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
        metrics = {
            'recommendation_scores': [],
            'context_activation': [],
            'object_activation': [],
            'context_to_object_ratio': []
        }
        
        for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
            sample = val_dataset[idx]
            image = sample['image'].to(device)
            patches = sample['patches']
            patch_boxes = sample['patch_boxes']
            segmentor_mask = sample.get('segmentor_target_mask', None)
            
            if len(patches) == 0:
                continue
            
            patch = patches[0].to(device)
            orig_box = patch_boxes[0].tolist()
            
            recs = self.recommend_placements(image.unsqueeze(0), patch.unsqueeze(0), patch_box=orig_box)
            
            # --- Diagnóstico ---
            if segmentor_mask is not None and recs:
                with torch.no_grad():
                    pred_heatmap = self(image.unsqueeze(0), patch=patch.unsqueeze(0))
                
                if segmentor_mask.dim() == 4:
                    seg_mask_binary = (segmentor_mask[0, 0] > 0.5).float()
                elif segmentor_mask.dim() == 3:
                    seg_mask_binary = (segmentor_mask[0] > 0.5).float()
                else:
                    seg_mask_binary = (segmentor_mask > 0.5).float()
                
                seg_mask_binary = seg_mask_binary.to(device)
                
                # Objeto = região original do patch
                obj_mask_binary = torch.zeros_like(seg_mask_binary)
                x1, y1, x2, y2 = map(int, orig_box)
                obj_mask_binary[y1:y2, x1:x2] = 1.0
                
                pred_heat_2d = pred_heatmap[0, 0]
                
                context_act = (pred_heat_2d * seg_mask_binary).sum() / (seg_mask_binary.sum() + 1e-6)
                object_act = (pred_heat_2d * obj_mask_binary).sum() / (obj_mask_binary.sum() + 1e-6)
                ratio = context_act / (object_act + 1e-6)
                
                metrics['context_activation'].append(float(context_act))
                metrics['object_activation'].append(float(object_act))
                metrics['context_to_object_ratio'].append(float(ratio))
                
                # Visualiza se save_dir fornecido
                if save_dir:
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # 1. Imagem original
                    axes[0].imshow(image.cpu().permute(1, 2, 0).numpy())
                    axes[0].set_title("Imagem")
                    axes[0].axis('off')
                    
                    # 2. Máscara de contexto
                    axes[1].imshow(seg_mask_binary.cpu().numpy(), cmap='gray')
                    axes[1].set_title("Contexto (esperado)")
                    axes[1].axis('off')
                    
                    # 3. Máscara de objeto
                    axes[2].imshow(obj_mask_binary.cpu().numpy(), cmap='gray')
                    axes[2].set_title("Objeto (original)")
                    axes[2].axis('off')
                    
                    # 4. Heatmap predito com overlay
                    hm_np = pred_heat_2d.cpu().numpy()
                    axes[3].imshow(hm_np, cmap='hot')
                    axes[3].set_title(f"Pred (Ctx:{context_act:.3f} vs Obj:{object_act:.3f})")
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"diag_{i:02d}_ratio{ratio:.2f}.png"), dpi=100)
                    plt.close(fig)
        
        if recs:
            metrics['recommendation_scores'].extend([r.get('score', 0.0) for r in recs])
    
        # Calcula estatísticas finais
        results = {}
        if metrics['recommendation_scores']:
            results['recommendation_scores'] = {
                'mean': float(np.mean(metrics['recommendation_scores'])),
                'std': float(np.std(metrics['recommendation_scores']))
            }
        
        if metrics['context_activation']:
            results['context_activation'] = {
                'mean': float(np.mean(metrics['context_activation'])),
                'std': float(np.std(metrics['context_activation']))
            }
            results['object_activation'] = {
                'mean': float(np.mean(metrics['object_activation'])),
                'std': float(np.std(metrics['object_activation']))
            }
            results['context_to_object_ratio'] = {
                'mean': float(np.mean(metrics['context_to_object_ratio'])),
                'std': float(np.std(metrics['context_to_object_ratio'])),
                'min': float(np.min(metrics['context_to_object_ratio'])),
                'max': float(np.max(metrics['context_to_object_ratio']))
            }
            
            print(f"\n*** DIAGNÓSTICO FINAL ***")
            print(f"Ativação CONTEXTO (média): {results['context_activation']['mean']:.4f} ± {results['context_activation']['std']:.4f}")
            print(f"Ativação OBJETO (média): {results['object_activation']['mean']:.4f} ± {results['object_activation']['std']:.4f}")
            print(f"Razão CONTEXTO/OBJETO (média): {results['context_to_object_ratio']['mean']:.2f}")
            print(f"  Intervalo: [{results['context_to_object_ratio']['min']:.2f}, {results['context_to_object_ratio']['max']:.2f}]")
            if results['context_to_object_ratio']['mean'] > 1.0:
                print("✓ SUCESSO: Modelo aprendeu a preferir contexto!")
            else:
                print("✗ PROBLEMA: Modelo ainda prefere objeto. Aumenta context_weight ou usa_context_primary mais cedo.")
        
        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
        
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

    @staticmethod
    def get_loss_mode(epoch, total_epochs, switch_epoch_frac=0.3):
        """
        Retorna True quando deve treinar com contexto como primário.
        Transição: primeiras 60% das épocas = gaussiana, últimas 40% = contexto
        """

        switch_epoch = int(total_epochs * switch_epoch_frac)
        return epoch >= switch_epoch

    def _create_exogenous_visualization(self, source_img, patch, patch_box, target_img, 
                                       recommendations, source_idx, target_idx, heatmap):
        """
        Cria visualização comparativa para avaliação exógena.
        
        Args:
            source_img: Imagem de origem (tensor)
            patch: Patch extraído (tensor)
            patch_box: Bounding box do patch [x1,y1,x2,y2]
            target_img: Imagem de destino (tensor)
            recommendations: Lista de recomendações

            source_idx: Índice da imagem de origem
            target_idx: Índice da imagem de destino
            heatmap: Mapa de compatibilidade (tensor)
            
        Returns:
            Figura matplotlib
        """
        import matplotlib.patches as mpatches
        
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Imagem de origem com patch marcado
        ax1 = fig.add_subplot(1, 5, 1)
        source_np = source_img.cpu().permute(1, 2, 0).numpy()
        ax1.imshow(source_np)
        if patch_box:
            x1, y1, x2, y2 = patch_box
            rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
        ax1.set_title(f"Origem (idx={source_idx})")
        ax1.axis('off')
        
        # 2. Patch extraído
        ax2 = fig.add_subplot(1, 5, 2)
        patch_np = patch.cpu().permute(1, 2, 0).numpy()
        ax2.imshow(patch_np)
        ax2.set_title("Patch")
        ax2.axis('off')
        
        # 3. Imagem de destino
        ax3 = fig.add_subplot(1, 5, 3)
        target_np = target_img.cpu().permute(1, 2, 0).numpy()
        ax3.imshow(target_np)
        ax3.set_title(f"Destino (idx={target_idx})")
        ax3.axis('off')
        
        # 4. Recomendações na imagem de destino
        ax4 = fig.add_subplot(1, 5, 4)
        ax4.imshow(target_np)
        colors = ['lime', 'cyan', 'yellow', 'magenta', 'white']
        for i, rec in enumerate(recommendations[:5]):
            box = rec['box']
            score = rec.get('score', 0)
            color = colors[i % len(colors)]
            rect = mpatches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax4.add_patch(rect)
            ax4.text(box[0], box[1]-5, f"#{i+1}: {score:.2f}", 
                    color='white', fontsize=9, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))
        ax4.set_title("Recomendações")
        ax4.axis('off')
        
        # 5. Mapa de compatibilidade
        ax5 = fig.add_subplot(1, 5, 5)
        hm_np = heatmap.cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap
        im = ax5.imshow(hm_np, cmap='hot')
        plt.colorbar(im, ax=ax5, label='Compatibilidade')
        ax5.set_title("Mapa de Compatibilidade")
        ax5.axis('off')
        
        plt.tight_layout()
        return fig
