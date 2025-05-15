import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TrainParams:
    epochs_warmup: int = 50
    epochs_finetune: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    img_size: int = 640
    
@dataclass
class LoggingParams:
    log_dir: str = 'logs/'
    save_checkpoint: bool = True
    checkpoint_path: str = 'checkpoints/'

@dataclass
class TrainConfig:
    """Configuration for training a model."""
    # Model parameters
    model_name: str = field(default="yolov8s.pt")
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
        
        model_config = config_dict[model_key]
        training_config = model_config.get("training", {})
        logging_config = model_config.get("logging", {})
        
        training_params = TrainParams(**training_config)
        logging_params = LoggingParams(**logging_config)
        
        return cls(
            model_name=model_config.get("model_name", "yolov8s.pt"),
            hyp_path=model_config.get("hyp_path", "hyp.yaml"),
            device=model_config.get("device", "cuda:0"),
            training=training_params,
            logging=logging_params
        ) 