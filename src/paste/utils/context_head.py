import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEmbeddingHead(nn.Module):
    """
    Versão simplificada do Context Embedding Head para extrair características de contexto.
    Evita uso de módulos de atenção complexos que podem ser instáveis com poucos dados.
    """
    def __init__(self, in_dim, out_dim=128, dropout=0.1):
        super().__init__()
        
        # Arquitetura simplificada com skip connections
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc_out = nn.Linear(256, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Garantir que x tem dimensão de batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Primeira camada
        h1 = self.activation(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1)
        
        # Segunda camada com residual
        h2 = self.fc2(h1)
        h2 = self.bn2(h2 + h1)  # Skip connection
        h2 = self.activation(h2)
        h2 = self.dropout(h2)
        
        # Camada final
        out = self.fc_out(h2)
        
        # Normalizar embeddings
        return F.normalize(out, p=2, dim=1)