import torch
import torch.nn as nn
import torch.nn.functional as F
from src.paste.utils.attention import CoordinateAttention

class ContextEmbeddingHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        
        # Adicionando Coordinate Attention (NOVO)
        self.coord_attention = CoordinateAttention(in_dim)
        
        # Arquitetura mais complexa com residual connections
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Skip connection
        self.skip = nn.Linear(in_dim, 256) if in_dim != 256 else nn.Identity()

    def forward(self, x):
        # Se input for 4D (convolução), aplica coordinate attention
        if len(x.shape) == 4:
            x = self.coord_attention(x)
            # Flatten para fully connected
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            
        identity = self.skip(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x) + identity))
        x = self.fc3(x)
        x = self.norm(x)
        x = F.normalize(x, dim=1)
        return x