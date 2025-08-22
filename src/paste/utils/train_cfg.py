import yaml
from dataclasses import dataclass, field
from typing import Any, Dict
import os

@dataclass
class TrainParams:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.0005  # Reduzir de 0.001 para 0.0005
    img_size: int = 640
    earlystop_patience: int = 10
    unfreeze_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.1
    
    # Parâmetros específicos para treinamento de compatibilidade
    contrast_loss_weight: float = 2.0  # Peso da loss de contraste
    contrast_margin: float = 0.3  # Margem para triplet loss de contraste
    use_intraimage_negatives: bool = True  # Usar negativos da mesma imagem
    negative_selection: str = "hard"  # Estratégia de seleção (easy/hard/semi-hard)

@dataclass
class LoggingParams:
    log_dir: str = 'logs/'
    save_checkpoint: bool = True
    checkpoint_path: str = 'checkpoints/'

@dataclass
class TrainConfig:
    """Configuration for training a model."""
    # Model parameters
    model_name: str = field(default="yolov8s")
    model_weights: str = field(default="yolov8s.pt")
    hyp_path: str = field(default="hyp.yaml")
    device: str = field(default="cuda:0")

    training: TrainParams = field(default_factory=TrainParams)
    logging: LoggingParams = field(default_factory=LoggingParams)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, model_key: str) -> "TrainConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        if model_key not in config_dict:
            raise ValueError(f"Model key '{model_key}' not found in the YAML file.")

        
        m = config_dict[model_key]
        tp = TrainParams(**m.get("training", {}))
        lp = LoggingParams(**m.get("logging", {}))
        
        name = m.get("model_name", cls.model_name)
        weights = m.get("model_weights", cls.model_weights)
        
        lp.log_dir = lp.log_dir.format(model_name=name)
        lp.checkpoint_path = lp.checkpoint_path.format(model_name=name)
        
        return cls(
            model_name    = name,
            model_weights = weights.format(model_name=name),
            hyp_path      = m.get("hyp_path", cls.hyp_path),
            device        = m.get("device", cls.device),
            training      = tp,
            logging       = lp
        )