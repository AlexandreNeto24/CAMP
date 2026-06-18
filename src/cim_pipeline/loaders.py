# Imports
import torch
import numpy as np
import os
from pathlib import Path
from typing import Optional

from src.paste import PretrainedUNet, AttentionRecommender

def find_latest_weights(exp_dir: Path, pattern: str) -> Optional[Path]:
    """Encontra os pesos mais recentes que correspondem ao padrão."""
    candidates = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_segmentor(class_name, exp_dir, device, encoder_name, verbose):
    
    # 1. Nova estrutura: weights/<class>/segmentor/best.pth
    weights_path = None
    weights_dir = exp_dir.parent / 'weights' / class_name / 'segmentor'
    for name in ['best.pth', 'best_segmentor.pth']:
        if (weights_dir / name).exists():
            weights_path = weights_dir / name
            break
    
    # 2. Fallback: experiments/ (estrutura antiga)
    if weights_path is None:
        weights_path = find_latest_weights(
            exp_dir, f"{class_name}_segmentation_*/best_segmentor.pth"
        )
    
    # Construir em CPU — AttentionRecommender.__init__ faz forward com dummy CPU
    model = PretrainedUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    
    if weights_path:
        try:
            state = torch.load(weights_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state, strict=False)
        except Exception:
            model.load_adapted_state_dict(str(weights_path), strict=False)
        if verbose: print(f"  ✓ Pesos: {weights_path.relative_to(exp_dir.parent)}")
    else:
        model = PretrainedUNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        if verbose: print(f"  ⚠ Sem pesos para '{class_name}' — usando ImageNet features")
    
    model.eval()
    return model  # Fica em CPU; será movido para device após construir o recommender

def load_recommender(class_name, exp_dir, device, segmentor, verbose):
    
    # 1. Nova estrutura: weights/<class>/recommender/best.pth
    weights_dir = exp_dir.parent / 'weights' / class_name / 'recommender'
    weights_path = None
    for name in ['best.pth', 'best_model.pt']:
        if (weights_dir / name).exists():
            weights_path = weights_dir / name
            break
    
    # 2. Fallback: experiments/ (estrutura antiga)
    if weights_path is None:
        weights_path = find_latest_weights(
            exp_dir, f"*{class_name}*attention_recommender_*/best_model.pt"
        )
    if weights_path is None:
        weights_path = find_latest_weights(
            exp_dir, f"{class_name}_attention_recommender_*/best_model.pt"
        )
    if weights_path is None:
        weights_path = find_latest_weights(
            exp_dir, "attention_recommender_*/best_model.pt"
        )
    
    # Construir em CPU (segmentor já está em CPU)
    model = AttentionRecommender(
        backbone_model=segmentor,
        embedding_dim=2048,
        num_heads=16,
        dropout_rate=0.3
    )
    
    if weights_path:
        model.load_model(str(weights_path))
        if verbose: print(f"  ✓ Pesos: {weights_path.relative_to(exp_dir.parent)}")
    else:
        if verbose: print(f"  ⚠ Sem pesos de recommender — inferência sem treino")
    
    # Agora sim, mover tudo (segmentor + recommender) para device
    model.to(device)
    model.eval()
    return model