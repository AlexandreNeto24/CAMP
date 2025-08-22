import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    """
    Encoder especializado para compreender relações objeto-contexto,
    generalizável para diferentes domínios.
    """
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # Encoder de objeto
        self.object_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )
        
        # Encoder de contexto - usa kernels maiores para capturar informação contextual
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),  # Campo receptivo maior
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )
        
        # Módulo de relação
        self.relation_module = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def encode_object(self, x):
        return F.normalize(self.object_encoder(x), dim=1)
        
    def encode_context(self, x):
        return F.normalize(self.context_encoder(x), dim=1)
        
    def compute_compatibility(self, obj_embedding, ctx_embedding):
        # Concatena os embeddings e calcula compatibilidade
        combined = torch.cat([obj_embedding, ctx_embedding], dim=1)
        return self.relation_module(combined)
        
    def forward(self, object_patch, context_patch):
        obj_embedding = self.encode_object(object_patch)
        ctx_embedding = self.encode_context(context_patch)
        compatibility = self.compute_compatibility(obj_embedding, ctx_embedding)
        return obj_embedding, ctx_embedding, compatibility