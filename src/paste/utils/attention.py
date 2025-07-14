import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    """
    Implementação do módulo Coordinate Attention que melhora a detecção de bordas sutis
    em lesões de metaplasia através de atenção espacial coordenada.
    
    Referência: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pooling horizontal
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pooling vertical
        
        mid_channels = max(8, in_channels // reduction)
        
        # Projeção e transformação de características
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        # Mapas de atenção separados para dimensões h e w
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        identity = x
        
        # Pooling para capturar dependências horizontais e verticais
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, 1, w] -> [n, c, w, 1]
        
        # Concatena e processa as informações posicionais
        x_cat = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)
        
        # Divide de volta em representações h e w
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, w, 1] -> [n, c, 1, w]
        
        # Gera mapas de atenção específicos para cada direção
        att_h = torch.sigmoid(self.conv_h(x_h))  # Atenção vertical
        att_w = torch.sigmoid(self.conv_w(x_w))  # Atenção horizontal
        
        # Aplica atenção por multiplicação
        out = identity * att_h * att_w
        return out