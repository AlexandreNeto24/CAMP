import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .backbone import PositionalEncoding
from src.paste.utils.trainer import AttentionRecommenderTrainer


class AttentionRecommender(nn.Module):
    """
    Modelo de recomendação baseado em atenção para colagem de lesões.
    
    RESPONSABILIDADES:
    - Definir arquitetura do modelo
    - Extrair features com backbone
    - Implementar forward pass
    - Gerenciar estado do modelo
    
    NÃO inclui:
    - Treino (vai para training/trainer.py)
    - Loss (vai para losses/)
    - Inferência/recomendações (vai para inference/)
    - Visualização (vai para visualization/)
    - Avaliação (vai para evaluation/)
    """
    
    def __init__(self, backbone_model, embedding_dim=2048, num_heads=16, dropout_rate=0.3):
        super().__init__()
        self.backbone = backbone_model
        self.feature_extractor = self.backbone.get_encoder()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        feature_dim = self.backbone.feature_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Segurança: embedding divisível por num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Detecta dimensão de entrada
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            features = self.feature_extractor(dummy)
            if isinstance(features, (list, tuple)):
                concat_dim = features[-2].shape[1] + features[-1].shape[1]
                # ✅ CORREÇÃO: Detecta canais para skip connection dinamicamente (índice 1)
                skip_channels = features[1].shape[1]
            else:
                concat_dim = features.shape[1]
                skip_channels = 32 # Fallback seguro

        # Projeção Query/Key
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

        # Source/Context token
        self.src_query_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embedding_dim)
        )

        # Fusion: reduz 2*E -> mid_channels
        mid_channels = 64
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embedding_dim * 2, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # # ✅ CORREÇÃO: Usa skip_channels detectado (40) em vez de hardcoded (32)
        # self.skip_proj = nn.Sequential(
        #     nn.Conv2d(skip_channels, 32, kernel_size=1), 
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )

        # # ✅ ATUALIZAR DECODER PARA RECEBER SKIP (64 + 32 canais)
        # self.heatmap_decoder = nn.Sequential(
        #     # Entrada agora é mid_channels (64) + skip_channels (32) = 96
        #     nn.ConvTranspose2d(mid_channels + 32, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )
        
        # ✅ NOVO: Decoder simples
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, 64, kernel_size=4, stride=2, padding=1),  # [64, H/8, W/8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # [1, H, W]
            nn.Sigmoid()
        )
        
        # Mecanismo de atenção
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional encoding + Transformer
        self.position_encoder = PositionalEncoding(embedding_dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads),
            num_layers=2
        )
        
    def extract_features(self, image, return_skip=False):
        """
        ✅ NOVO: Extração simples sem skip.
        """
        device = next(self.feature_extractor.parameters()).device
        image = image.to(device)
        
        with torch.no_grad():
            features = self.feature_extractor(image)
            
            if isinstance(features, (list, tuple)):
                target_size = features[-1].shape[-2:]
                upsampled = F.interpolate(features[-2], size=target_size, mode='bilinear', align_corners=False)
                final_feat = torch.cat([upsampled, features[-1]], dim=1)
                return final_feat
            
            return features

    def forward(self, image, lesion_coords=None, patch=None, patch_with_context=None,
                source_context=None, source_box=None, weight_mask=None, 
                return_attention=False, return_intermediate=False):
        """
        Forward pass do modelo.
        
        Args:
            image: [B, 3, H, W]
            patch: [B, 3, ph, pw] (APENAS O OBJETO, fundo mascarado)
            patch_with_context: [B, 3, ch, cw] (✅ NOVO: Objeto + contexto original)
            source_context: opcional [B, 3, ch, cw]
            source_box: opcional [B, 4]
            return_attention: bool
            return_intermediate: bool
            
        Returns:
            heatmap ou (heatmap, attn_weights) ou (heatmap, patch_feats, img_feats)
        """
        device = self.device
        image = image.to(device)
        patch = patch.to(device)
        
        # 1. Extrai features
        image_features = self.extract_features(image)   # [B, E, H, W]
        patch_features = self.extract_features(patch)   # [B, E, h, w]

        # 2. Projeta
        query = self.query_conv(patch_features)         # [B, E, h, w]
        key = self.key_conv(image_features)             # [B, E, H, W]

        # 3. Flatten + positional encoding
        B, E, h, w = query.shape
        query_flat = query.view(B, E, -1).permute(0, 2, 1)  # [B, h*w, E]
        B, E, H, W = key.shape
        key_flat = key.view(B, E, -1).permute(0, 2, 1)      # [B, H*W, E]

        query_flat = self.position_encoder(query_flat)
        key_flat = self.position_encoder(key_flat)

        # 4. Transformer encoder na imagem
        key_trans = self.transformer_encoder(key_flat)  # [B, H*W, E]
        key_trans_spatial = key_trans.permute(0, 2, 1).view(B, E, H, W)

        # 5. Cross-attention
        attn_out, attn_weights = self.attention(query=query_flat, key=key_flat, value=key_flat)
        qlen = h * w
        attn_spatial = attn_out[:, :qlen, :].permute(0, 2, 1).view(B, E, h, w)

        # 6. Upsample + fusion + decoder
        attn_out_upsampled = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=False)
        concat = torch.cat([key_trans_spatial, attn_out_upsampled], dim=1)  # [B, 2E, H, W]
        fused = self.fusion_conv(concat)                                     # [B, 64, H, W]
        
        heatmap = self.heatmap_decoder(fused)
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)

        # Retorna conforme necessário
        if return_attention and return_intermediate:
            return heatmap, attn_weights, patch_features, image_features
        if return_attention:
            return heatmap, attn_weights
        if return_intermediate:
            return heatmap, patch_features, image_features
        return heatmap

    def train(self, mode=True):
        """Override: mantém backbone em eval"""
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.eval()
        return self

    @property
    def device(self):
        """Retorna dispositivo do modelo"""
        return next(self.parameters()).device

    def save_model(self, path):
        """Salva modelo (apenas parâmetros treináveis)"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            key: value for key, value in self.state_dict().items()
            if not key.startswith('backbone.') and not key.startswith('feature_extractor.')
        }
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'version': '1.0',
        }
        
        torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
        print(f"Modelo salvo em {path}")

    def load_model(self, path):
        """Carrega modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.load_state_dict(state_dict, strict=False)
        print(f"Modelo carregado de {path}")
        return self
    
    def fit_trainer(self, train_loader, val_loader, epochs=50, lr=5e-4, 
                          accumulation_steps=4, run_dir=None, device='cuda'):
        """
        Treina o modelo usando AttentionRecommenderTrainer.
        
        Este método é um wrapper conveniente que encapsula o treino completo,
        permitindo usar: model.train_with_trainer(train_loader, val_loader, ...)
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            epochs: Número de épocas
            lr: Learning rate
            accumulation_steps: Passos de gradient accumulation
            run_dir: Diretório para guardar resultados
            device: Dispositivo ('cuda', 'cpu', etc)
            
        Returns:
            self (modelo treinado) para permitir method chaining
            
        Example:
            >>> model = AttentionRecommender(...)
            >>> trained_model = model.train_with_trainer(
            ...     train_loader=train_loader,
            ...     val_loader=val_loader,
            ...     epochs=50,
            ...     run_dir='./results',
            ...     device='cuda'
            ... )
        """
        # Cria instância do trainer
        trainer = AttentionRecommenderTrainer(self, device=device)
        
        # Executa treino
        trained_model = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            accumulation_steps=accumulation_steps,
            run_dir=run_dir
        )
        
        # Retorna self para permitir chaining
        return self
    
    @property
    def trainer(self):
        """
        Propriedade que retorna uma instância do Trainer.
        Uso: model.trainer.fit(...)
        
        Returns:
            AttentionRecommenderTrainer vinculado a este modelo
        """
        return AttentionRecommenderTrainer(self, device=self.device)
    
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
        device = self.device
        image = image.to(device)
        patch = patch.to(device)
        
        with torch.no_grad():
            heatmap, _ = self.forward(image, patch=patch, return_attention=True)
        
        # Determina tamanho do patch
        if patch_box is not None:
            patch_h = int(patch_box[3] - patch_box[1])
            patch_w = int(patch_box[2] - patch_box[0])
            patch_size = (patch_h, patch_w)
        else:
            patch_size = patch.shape[-2:]
        
        # Extrai recomendações do heatmap
        recommendations = self._extract_recommendations_from_heatmap(
            heatmap[0, 0],  # [H, W]
            image.shape[-2:],
            patch_size,
            top_k=top_k
        )
        
        return recommendations
    
    def _extract_recommendations_from_heatmap(self, heatmap, image_size, patch_size, top_k=5):
        """
        Extrai coordenadas de recomendação do mapa de calor.
        """
        import numpy as np
        
        h, w = heatmap.shape
        patch_h, patch_w = patch_size
        
        # Escala do mapa de calor para imagem original
        y_scale = image_size[0] / h
        x_scale = image_size[1] / w
        
        # Percorre o heatmap com janela do tamanho do patch
        scores = []
        for y in range(0, max(1, h - patch_h + 1), max(1, patch_h // 4)):
            for x in range(0, max(1, w - patch_w + 1), max(1, patch_w // 4)):
                y_end = min(y + patch_h, h)
                x_end = min(x + patch_w, w)
                window = heatmap[y:y_end, x:x_end]
                
                if window.numel() == 0:
                    continue
                
                score = float(window.mean())
                scores.append((score, x, y))
        
        # Seleciona top_k regiões
        scores = sorted(scores, key=lambda tup: tup[0], reverse=True)
        recommendations = []
        used_boxes = []
        
        for i, (score, x, y) in enumerate(scores):
            # Coordenada central na imagem original
            center_y = int((y + patch_h // 2) * y_scale)
            center_x = int((x + patch_w // 2) * x_scale)
            
            # Gera box centrada na região
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
    
    def _calculate_iou(self, box1, box2):
        """Calcula IoU entre duas boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou