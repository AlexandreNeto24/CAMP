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
            # Geralmente features[2] ou [3] tem boa resolução e semântica
            # EfficientNet features: [stem, layer1, layer2, layer3, layer4]
            # Vamos usar layer2 (stride 8) ou layer1 (stride 4)
            self.skip_idx = -3  # Tenta pegar uma feature intermédia
            skip_channels = features[self.skip_idx].shape[1]
            # print(f"Skip connection channels: {skip_channels}")
            
            # Define concat_dim baseado nas features concatenadas
            target_size = features[-1].shape[-2:]
            upsampled = F.interpolate(features[-2], size=target_size, mode='bilinear', align_corners=False)
            
            # ✅ OPTION 2: Add +2 for coordinate channels (X, Y)
            concat_dim = upsampled.shape[1] + features[-1].shape[1] + 2
            # print(f"Concatenated feature dimension (with coords): {concat_dim}")

        # Projeção Query/Key
        gn_groups = min(32, max(1, embedding_dim // 16))
        self.query_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1), # ✅ CORRIGIDO: usa concat_dim (520), não skip_channels (48)
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1),
            nn.GroupNorm(gn_groups, self.embedding_dim),
            nn.Dropout2d(dropout_rate)
        )

        # ✅ NOVO: Camadas de projeção para o Cross-Attention
        self.query_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(concat_dim, self.embedding_dim, kernel_size=1)

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

        # ✅ 2. PROJEÇÃO DO SKIP
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
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
        
        # ✅ NOVO: Projeção do similarity_map (1 canal → 64 canais)
        self.similarity_proj = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 1 → 64 canais
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Refina
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ✅ 3. DECODER MELHORADO (Recebe 64 do fusion + 64 do skip = 128)
        self.heatmap_decoder = nn.Sequential(
            # Entrada: mid_channels (64)
            nn.ConvTranspose2d(mid_channels, 64, kernel_size=4, stride=2, padding=1), # x2 upsample
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # AQUI VAMOS CONCATENAR O SKIP (64+64=128)
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # x2 upsample
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # x2 upsample
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Final resolution
            nn.Sigmoid()
        )
        
        # Mecanismo de atenção
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.context_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

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
        
        # ✅ NOVO: Camada para processar o token de relação
        self.relationship_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        # TODO: REMOVE COORDINATES 
    def _add_coordinate_channels(self, features):
        """
        Adiciona canais de coordenadas normalizadas (x, y) ao tensor de features.
        Isso permite ao mecanismo de atenção aprender viés espacial (ex: céu vs terra).
        """
        B, C, H, W = features.shape
        # Gera grids normalizados [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=features.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        x_coords = torch.linspace(-1, 1, W, device=features.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        
        # Concatena nos canais: [B, C+2, H, W]
        return torch.cat([features, x_coords, y_coords], dim=1)

    def extract_features(self, image, return_skip=False):
        """
        ✅ NOVO: Extração simples sem skip.
        Agora suporta Injeção de Coordenadas (CoordConv) na Entrada.
        """
        device = next(self.feature_extractor.parameters()).device
        image = image.to(device)
        
        # ✅ Lógica CoordConv (Input Injection)
        # Se a imagem tem 3 canais, tentamos adicionar X,Y (para total 5)
        # Isso ative o "Semântica Espacial" desde a primeira camada
        if image.shape[1] == 3:
            # Cria Bx5xHxW
            image_input = self._add_coordinate_channels(image)
        else:
            image_input = image

        try:
            with torch.no_grad():
                features = self.feature_extractor(image_input)
        except RuntimeError as e:
            # Fallback: Se o backbone explodir com 5 canais (ex: foi instanciado com 3),
            # tentamos usar apenas a imagem original de 3 canais.
            if "channels" in str(e) or "shape" in str(e):
                # print("⚠️ Backbone não aceita 5 canais. Usando RGB padrão (sem CoordConv).")
                with torch.no_grad():
                    features = self.feature_extractor(image)
            else:
                raise e
            
        # Pega a feature final (topo)
        final_feat = features[-1]
        
        # Se for lista, processa como antes para garantir tamanho
        if isinstance(features, (list, tuple)):
            target_size = features[-1].shape[-2:]
            upsampled = F.interpolate(features[-2], size=target_size, mode='bilinear', align_corners=False)
            final_feat = torch.cat([upsampled, features[-1]], dim=1)
        
        if return_skip:
            # Retorna também a feature intermédia para skip connection
            skip_feat = features[self.skip_idx]
            return final_feat, skip_feat
        
        return final_feat

    def forward(self, image, lesion_coords=None, patch=None, patch_with_context=None,
                source_context=None, source_box=None, weight_mask=None, 
                context_mask=None, # ✅ NOVO: Máscara explícita (1=Objeto, 0=Fundo)
                return_attention=False, return_intermediate=False):
        """
        Forward SIMPLIFICADO baseado em feature matching.
        """
        device = self.device
        image = image.to(device)
        patch = patch.to(device)
        
        if patch_with_context is None:
            patch_with_context = patch
        else:
            patch_with_context = patch_with_context.to(device)
            
        if context_mask is not None:
            context_mask = context_mask.to(device)
    
        # 1. Extrai features espaciais da imagem
        image_features_raw, image_skip = self.extract_features(image, return_skip=True)  # [B, E, H, W]
        
        # ✅ OPTION 2: Inject Coordinate Channels (X, Y) into Image Features
        image_features = self._add_coordinate_channels(image_features_raw) # [B, E+2, H, W]
        
        # 2. Gera máscara de exclusão (Objeto)
        # Se passarmos context_mask, usamos ela. Caso contrário, tentamos heurística (muito falível)
        exclusion_mask = None
        if context_mask is not None:
             exclusion_mask = context_mask # [B, 1, H, W]
        elif patch is not None:
             # HEURÍSTICA DE PÂNICO: Tenta adivinhar fundo preto (falha se patch for recorte da foto normal)
             # Idealmente, devemos sempre passar context_mask
             exclusion_mask = (patch.abs().sum(dim=1, keepdim=True) > 0.1).float()

        # 3. Extrai features do patch (objeto e contexto)
        patch_obj_features_raw, patch_ctx_features_raw = self.get_patch_features(
            patch=patch,
            patch_with_context=patch_with_context,
            context_mask=exclusion_mask
        )
        
        # ✅ OPTION 2: Inject Coordinate Channels into Patch Features too
        # Even though patches are locally Translation Invariant, for the Dot Product to work 
        # the channel dimensions must match. 
        # We fill them with zeros or relative coords? Relative coords (-1 to 1) make sense for structure.
        patch_obj_features = self._add_coordinate_channels(patch_obj_features_raw)
        patch_ctx_features = self._add_coordinate_channels(patch_ctx_features_raw)

        # 4. Prepara os tensores para o Cross-Attention
        # Projeta as features para a dimensão de embedding correta (ex: 520 -> 384)
        query_proj = self.query_proj(patch_ctx_features)
        key_proj = self.key_proj(image_features)
        value_proj = self.value_proj(image_features)

        B, E, H_img, W_img = key_proj.shape
        _, _, H_patch, W_patch = query_proj.shape

        # Query: [B, E, H_patch, W_patch] -> [B, H_patch*W_patch, E]
        # Apenas tokens onde a máscara de contexto é ativa
        ctx_mask_small = F.interpolate(exclusion_mask.float(), size=(H_patch, W_patch), mode='nearest')
        context_mask_inv = (1.0 - ctx_mask_small).squeeze(1) > 0.5 # [B, H_patch, W_patch]
        
        # Achatamos e pegamos apenas os tokens válidos
        query_flat = query_proj.permute(0, 2, 3, 1) # [B, H_patch, W_patch, E]
        
        # Para simplificar, usamos a média dos tokens de contexto como Query
        # Isto evita problemas com número variável de tokens por batch item
        valid_query_tokens = [q[m] for q, m in zip(query_flat, context_mask_inv)]
        query_avg_list = [tokens.mean(dim=0, keepdim=True) if len(tokens) > 0 else torch.zeros(1, E, device=self.device) for tokens in valid_query_tokens]
        query_avg = torch.stack(query_avg_list) # [B, 1, E]

        # Key/Value: [B, E, H_img, W_img] -> [B, H_img*W_img, E]
        key = key_proj.view(B, E, -1).permute(0, 2, 1)
        value = value_proj.view(B, E, -1).permute(0, 2, 1)

        # 5. Aplica Cross-Attention
        # Query: [B, 1, E] (Contexto do Patch)
        # Key/Value: [B, H*W, E] (Imagem de Destino)
        attn_output, attn_weights = self.context_attention(query=query_avg, key=key, value=value)
        
        # attn_output é [B, 1, E], representa a "mistura" da imagem de destino
        # que melhor corresponde ao contexto do patch. Não é o que queremos para um mapa de calor.
        # O que queremos é o `attn_weights`: [B, 1, H_img*W_img]
        # Ele diz-nos, para o nosso contexto médio, quais pixels da imagem de destino foram mais importantes.
        
        similarity_map = attn_weights.view(B, 1, H_img, W_img)
        
        # 6. Projeta de 1 canal para 64 canais
        x = self.similarity_proj(similarity_map)  # [B, 1, H, W] → [B, 64, H, W]
    
        # 7. Refina com decoder (usa skip connection)
        x = self.heatmap_decoder[0](x)
        x = self.heatmap_decoder[1](x)
        x = self.heatmap_decoder[2](x)
        
        # Skip connection
        skip = self.skip_proj(image_skip)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        # Resto do decoder
        for i in range(3, len(self.heatmap_decoder)):
            x = self.heatmap_decoder[i](x)
    
        heatmap = x
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
    
        # Retorna conforme necessário
        if return_attention and return_intermediate:
            # Retorna as features de CONTEXTO para a loss
            return heatmap, None, patch_ctx_features, image_features
        if return_attention:
            return heatmap, None
        if return_intermediate:
            # Retorna as features de CONTEXTO para a loss
            return heatmap, patch_ctx_features, image_features
        return heatmap

    def get_patch_features(self, patch, patch_with_context, context_mask):
        """
        Extrai features separadas para o objeto e para o seu contexto.
        
        Args:
            patch (Tensor): O patch contendo apenas o objeto.
            patch_with_context (Tensor): O patch com o objeto e seu fundo original.
            context_mask (Tensor): Máscara que é 1 no objeto e 0 no fundo.

        Returns:
            tuple: (patch_object_features, patch_context_features)
                   - patch_object_features: Features do objeto. [B, E, h, w]
                   - patch_context_features: Features do contexto/fundo. [B, E, h, w]
        """
        # Features do patch completo (objeto + contexto)
        features_with_context = self.extract_features(patch_with_context) # [B, E, h_f, w_f]
        
        # Redimensiona a máscara para o tamanho das features
        mask_small = F.interpolate(
            context_mask.float(), 
            size=features_with_context.shape[-2:], 
            mode='nearest'
        )
        
        # Máscara para o objeto (onde context_mask é 1)
        object_mask = (mask_small > 0.5).float()
        
        # Máscara para o contexto (onde context_mask é 0)
        context_mask_inv = 1.0 - object_mask
        
        # Extrai features do objeto
        patch_object_features = features_with_context * object_mask
        
        # Extrai features do contexto
        patch_context_features = features_with_context * context_mask_inv
        
        return patch_object_features, patch_context_features

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
    
    def _compute_relationship_token(self, patch_obj_features, patch_ctx_features):
        """
        Computa um token que captura a relação entre objeto e contexto.
        VERSÃO FINAL E ROBUSTA - Garante [B, E] SEMPRE.
        """
        B = patch_obj_features.shape[0]
        E = self.embedding_dim  # ✅ Usa self.embedding_dim
        
        # 0. ✅ NOVO: Reduz ambas para o MESMO número de canais PRIMEIRO
        if patch_obj_features.shape[1] != E:
            obj_feat = F.conv2d(patch_obj_features, torch.ones(E, patch_obj_features.shape[1], 1, 1, device=patch_obj_features.device) / patch_obj_features.shape[1])
        else:
            obj_feat = patch_obj_features
        
        if patch_ctx_features.shape[1] != E:
            ctx_feat = F.conv2d(patch_ctx_features, torch.ones(E, patch_ctx_features.shape[1], 1, 1, device=patch_ctx_features.device) / patch_ctx_features.shape[1])
        else:
            ctx_feat = patch_ctx_features
        
        # 1. Redimensiona para o mesmo tamanho espacial
        target_size = (
            max(obj_feat.shape[2], ctx_feat.shape[2]),
            max(obj_feat.shape[3], ctx_feat.shape[3])
        )
        
        obj_feat = F.interpolate(obj_feat, size=target_size, mode='bilinear', align_corners=False)
        ctx_feat = F.interpolate(ctx_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # ✅ VERIFICAÇÃO: Agora ambas têm [B, E, h, w]
        assert obj_feat.shape[1] == E, f"obj_feat channels: {obj_feat.shape[1]} != {E}"
        assert ctx_feat.shape[1] == E, f"ctx_feat channels: {ctx_feat.shape[1]} != {E}"
        
        # 2. Contraste
        contrast = torch.abs(obj_feat - ctx_feat).mean(dim=(2, 3))  # [B, E]
        contrast = torch.clamp(contrast, 0, 1)
        
        # 3. Magnitude
        obj_mag = torch.norm(obj_feat, p=2, dim=(2, 3))  # [B, E]
        ctx_mag = torch.norm(ctx_feat, p=2, dim=(2, 3))  # [B, E]
        magnitude = obj_mag / (ctx_mag + 1e-6)
        magnitude = torch.clamp(magnitude, 0.1, 10)
        
        # 4. Similaridade (AGORA SEGURO porque ambas têm E canais)
        obj_norm = F.normalize(obj_feat.view(B, E, -1), dim=2, p=2)  # [B, E, N]
        ctx_norm = F.normalize(ctx_feat.view(B, E, -1), dim=2, p=2)  # [B, E, N]
        similarity = (obj_norm * ctx_norm).mean(dim=2)  # [B, E]
        similarity = torch.clamp(similarity, -1, 1)
        
        # 5. Combina
        w1, w2, w3 = 1.0, 0.5, 0.5
        combined = (
            w1 * contrast +
            w2 * (magnitude / 10.0) +
            w3 * (similarity + 1) / 2.0
        ) / (w1 + w2 + w3)
        
        # 6. Normaliza
        combined = torch.clamp(combined, 0, 1)
        relationship_token = torch.tanh(combined)  # [-1, 1]
        
        # ✅ VERIFICAÇÃO FINAL
        assert relationship_token.shape == (B, E), \
            f"❌ Shape ERRADO! Expected [{B}, {E}], got {relationship_token.shape}"
        
        return relationship_token

    def encode_context_descriptor(self, image, exclusion_mask=None):
        """
        Encode imagem para um vetor global focado no CONTEXTO (ignorando o objeto).
        Args:
            image: [B, 3, H, W] (Patch com contexto)
            exclusion_mask: [B, 1, H, W] (1 onde é o objeto, 0 onde é fundo)
        Returns:
            [B, E] Vetor descritor do contexto
        """
        features = self.extract_features(image)  # [B, E, h, w]
        
        if exclusion_mask is not None:
            # Redimensiona máscara para tamanho das features
            mask_small = F.interpolate(
                exclusion_mask.float(), 
                size=features.shape[-2:], 
                mode='nearest'
            )
            # Inverte: 1 onde é FUNDO (contexto)
            context_mask = 1.0 - mask_small
            
            # Evita divisão por zero se não houver contexto (patch ocupa tudo)
            valid_pixels = context_mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            valid_pixels = torch.clamp(valid_pixels, min=1.0)
            
            # Masked Global Average Pooling
            # Zera features do objeto
            active_features = features * context_mask
            global_vec = active_features.sum(dim=(2, 3)) / valid_pixels.squeeze(-1).squeeze(-1) # [B, E]
            
            # Fallback seguro: se contexto for < 10% da área, mistura com média global
            total_pixels = features.shape[-2] * features.shape[-1]
            ratio = valid_pixels.squeeze() / total_pixels
            
            # Se ratio < 0.1, usa média global tradicional
            if (ratio < 0.1).any():
                global_avg = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                # Mistura suave baseada no ratio
                # w = torch.clamp(ratio / 0.1, 0, 1).unsqueeze(-1) # 0 se sem background, 1 se >10%
                # global_vec = w * global_vec + (1 - w) * global_avg
                
                # Simplesmente substitui onde o ratio é muito baixo
                replace_mask = (ratio < 0.1).float().unsqueeze(-1)
                global_vec = (1 - replace_mask) * global_vec + replace_mask * global_avg
                
        else:
            # Global Average Pooling padrão
            global_vec = F.adaptive_avg_pool2d(features, (1, 1))  # [B, E, 1, 1]
            global_vec = global_vec.squeeze(-1).squeeze(-1)       # [B, E]
        
        # Normaliza para cosine similarity (ou L2 matching)
        global_vec = F.normalize(global_vec, p=2, dim=1)
        
        return global_vec