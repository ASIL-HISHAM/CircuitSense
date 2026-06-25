"""
CircuitSense — Centralized Configuration Management
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_channels: int = 1
    sequence_length: int = 1024
    num_classes: int = 8
    cnn_channels: list = None
    cnn_kernel_size: int = 7
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    attention_heads: int = 8
    classifier_dropout: float = 0.4

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128, 256]


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "data/checkpoints"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Synthetic generator
    samples_per_class: int = 1250
    sample_rate: int = 10000
    duration: float = 0.1024
    noise_std: float = 0.02
    
    # Augmentation
    augment: bool = True
    aug_noise_prob: float = 0.4
    aug_scale_prob: float = 0.3
    aug_timewarp_prob: float = 0.2
    aug_dropout_prob: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1.0e-6
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    loss: str = "focal"  # focal | cross_entropy
    focal_gamma: float = 2.0


@dataclass
class InferenceConfig:
    """Inference configuration."""
    device: str = "auto"  # auto | cuda | cpu | mps
    threshold: float = 0.6
    batch_size: int = 32


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    max_batch_size: int = 1000
    request_timeout: int = 30


class Config:
    """Main configuration class that loads from YAML and environment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config from YAML file and environment variables.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        
        # Load from YAML if provided
        if config_path:
            self.load_yaml(config_path)
        else:
            self._load_defaults()
        
        # Override with environment variables
        self._load_env()
        
        logger.info("Configuration loaded successfully")

    def load_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        
        # Update dataclass fields from YAML
        if "model" in cfg:
            for key, value in cfg["model"].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if "data" in cfg:
            for key, value in cfg["data"].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if "training" in cfg:
            for key, value in cfg["training"].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if "inference" in cfg:
            for key, value in cfg["inference"].items():
                if hasattr(self.inference, key):
                    setattr(self.inference, key, value)
        
        if "api" in cfg:
            for key, value in cfg["api"].items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
        
        logger.info(f"Config loaded from: {config_path}")

    def _load_defaults(self) -> None:
        """Load default configuration from configs/default.yaml if exists."""
        default_config = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if default_config.exists():
            logger.info("Loading default configuration")
            self.load_yaml(str(default_config))

    def _load_env(self) -> None:
        """Override configuration with environment variables."""
        # API settings
        if env_host := os.getenv("API_HOST"):
            self.api.host = env_host
        if env_port := os.getenv("API_PORT"):
            self.api.port = int(env_port)
        if env_log := os.getenv("LOG_LEVEL"):
            self.api.log_level = env_log
        
        # Inference settings
        if env_device := os.getenv("DEVICE"):
            self.inference.device = env_device
        if env_threshold := os.getenv("CONFIDENCE_THRESHOLD"):
            self.inference.threshold = float(env_threshold)
        
        # Data settings
        if env_workers := os.getenv("NUM_WORKERS"):
            self.data.num_workers = int(env_workers)
        
        logger.debug("Environment variables applied to configuration")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "api": self.api.__dict__,
        }

    def __repr__(self) -> str:
        return f"Config(model={self.model}, data={self.data}, training={self.training}, inference={self.inference}, api={self.api})"


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
