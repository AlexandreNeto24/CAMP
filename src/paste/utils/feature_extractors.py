import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextureExtractor(nn.Module):
    """
    Extrai características de textura usando filtros Gabor em múltiplas orientações.
    Adaptado para trabalhar com patches de ROI Align de tamanhos variáveis.
    """
    def __init__(self, num_filters=8):
        super().__init__()
        self.num_filters = num_filters
        
        # Inicializa filtros Gabor com diferentes orientações
        self.conv_layers = nn.ModuleList()
        
        # Diferentes orientações para capturar padrões direcionais
        orientations = torch.linspace(0, np.pi, num_filters)
        
        for theta in orientations:
            # Cria kernel Gabor com orientação específica
            kernel = self._get_gabor_kernel(
                ksize=9, sigma=2.0, 
                theta=theta, lambd=4.0,
                gamma=0.5, psi=0
            )
            
            # Cria camada convolucional com este kernel
            conv = nn.Conv2d(3, 1, kernel_size=9, padding=4, bias=False)
            with torch.no_grad():
                conv.weight.data = kernel
            self.conv_layers.append(conv)
        
    def _get_gabor_kernel(self, ksize, sigma, theta, lambd, gamma, psi):
        # Implementação do kernel Gabor para detecção de texturas
        y, x = torch.meshgrid(
            torch.linspace(-ksize//2, ksize//2, ksize),
            torch.linspace(-ksize//2, ksize//2, ksize),
            indexing='ij'
        )
        
        # Rotação
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
        
        # Função Gabor
        gb = torch.exp(
            -0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2
        ) * torch.cos(2 * np.pi * x_theta / lambd + psi)
        
        # Normalização
        gb = gb / gb.abs().max()
        
        # Formato para convolução (out_ch, in_ch, h, w)
        kernel = gb.expand(1, 3, ksize, ksize)
        
        return kernel
        
    def forward(self, x):
        # Verificação adicional: lidar com patches de tamanhos variados
        if x.numel() == 0:
            return torch.zeros((0, self.num_filters), device=x.device)
            
        batch_size = x.shape[0]
        responses = []
        
        # Aplica cada filtro Gabor
        for conv in self.conv_layers:
            response = conv(x)
            
            # Lidando com tamanhos pequenos de patch
            if response.numel() == 0:
                responses.append(torch.zeros((batch_size, 1), device=x.device))
                continue
                
            pooled = F.adaptive_avg_pool2d(torch.abs(response), (1, 1)).view(batch_size, -1)
            responses.append(pooled)
            
        # Concatena todos os resultados
        texture_feats = torch.cat(responses, dim=1)
        return F.normalize(texture_feats, dim=1)

class ColorHistogramExtractor(nn.Module):
    """
    Extrai características de cor usando histogramas em múltiplos espaços de cor.
    Essencial para metaplasia que frequentemente apresenta colorações características.
    """
    def __init__(self, bins=8):
        super().__init__()
        self.bins = bins
    
    def forward(self, x):
        batch_size = x.shape[0]
        features = []
        
        # Histogramas RGB
        for c in range(3):
            channel_values = x[:, c].view(batch_size, -1)
            histograms = []
            
            # Calcula histograma para cada imagem no batch
            for i in range(batch_size):
                hist = torch.histc(channel_values[i], bins=self.bins, min=0, max=1)
                hist = hist / (hist.sum() + 1e-8)  # Normaliza
                histograms.append(hist)
                
            # Stack do batch
            channel_hists = torch.stack(histograms, dim=0)
            features.append(channel_hists)
                
        # Calcula características de matiz/saturação
        # Aproximação de HSV: diferença entre canais
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        max_rgb, _ = torch.max(x, dim=1)
        min_rgb, _ = torch.min(x, dim=1)
        chroma = (max_rgb - min_rgb).view(batch_size, -1)
        
        # Histograma de saturação (chroma)
        sat_histograms = []
        for i in range(batch_size):
            hist = torch.histc(chroma[i], bins=self.bins, min=0, max=1)
            hist = hist / (hist.sum() + 1e-8)
            sat_histograms.append(hist)
        
        features.append(torch.stack(sat_histograms, dim=0))
        
        # Concatena todos os histogramas [batch_size, bins*4]
        color_feats = torch.cat(features, dim=1)
        return color_feats

class ShapeFeatureExtractor(nn.Module):
    """
    Extrai características de forma e tamanho das bounding boxes,
    importante para analisar a morfologia das lesões metaplásicas.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 16)  # Projeta para espaço de maior dimensão
    
    def forward(self, boxes):
        # boxes: [batch_size, 4] no formato [x1, y1, x2, y2]
        if boxes.size(0) == 0:
            return torch.empty((0, 16), device=boxes.device)
            
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Características de forma
        aspect_ratios = widths / (heights + 1e-6)
        compactness = (widths + heights) / (torch.sqrt(areas) + 1e-6)
        
        # Combina características [batch_size, 3]
        shape_features = torch.stack([aspect_ratios, compactness, areas], dim=1)
        
        # Normaliza área para 0-1
        shape_features[:, 2] = shape_features[:, 2] / (640*640)
        
        # Projeta para dimensão maior
        return self.fc(shape_features)

class MultiModalFusion(nn.Module):
    """
    Módulo de fusão que combina embeddings, textura, cor e forma de maneira adaptativa.
    """
    def __init__(self, embedding_dim=128, texture_dim=8, color_dim=32, shape_dim=16):
        super().__init__()
        
        # Projeção das diferentes modalidades
        self.embedding_proj = nn.Linear(embedding_dim, 64)
        self.texture_proj = nn.Sequential(
            nn.Linear(texture_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.color_proj = nn.Sequential(
            nn.Linear(color_dim, 32), 
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.shape_proj = nn.Sequential(
            nn.Linear(shape_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        # Gerador de pesos de atenção entre modalidades
        self.attention = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 16, 4),  # 4 pesos para 4 modalidades
            nn.Softmax(dim=1)
        )
        
        # Fusão final
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 16, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        
    def forward(self, embeddings, texture_feats, color_feats, shape_feats):
        # Projeção de cada modalidade
        emb = F.normalize(self.embedding_proj(embeddings), dim=1)
        tex = F.normalize(self.texture_proj(texture_feats), dim=1)
        col = F.normalize(self.color_proj(color_feats), dim=1)
        shp = F.normalize(self.shape_proj(shape_feats), dim=1)
        
        # Concatenação para atenção e fusão
        combined = torch.cat([emb, tex, col, shp], dim=1)
        
        # Atenção adaptativa entre modalidades
        weights = self.attention(combined)  # [batch_size, 4]
        
        # Concatenação ponderada
        weighted_emb = emb * weights[:, 0].unsqueeze(1)
        weighted_tex = tex * weights[:, 1].unsqueeze(1)
        weighted_col = col * weights[:, 2].unsqueeze(1)
        weighted_shp = shp * weights[:, 3].unsqueeze(1)
        
        # Fusão final
        weighted_combined = torch.cat([
            weighted_emb, weighted_tex, weighted_col, weighted_shp
        ], dim=1)
        
        fused = self.fusion(weighted_combined)
        return F.normalize(fused, dim=1)