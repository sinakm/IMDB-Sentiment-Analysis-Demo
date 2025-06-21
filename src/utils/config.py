"""
Configuration classes for the sentiment analysis project.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    # Common parameters
    embed_dim: int = 200
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    max_length: int = 512
    
    # LSTM specific
    bidirectional: bool = True
    use_attention: bool = True
    vocab_size: int = 20000
    
    # Verbalizer specific
    transformer_model: str = "bert-base-uncased"
    freeze_transformer: bool = False
    
    # Pre-trained embeddings
    glove_dim: int = 300
    glove_version: str = "6B"
    freeze_embeddings: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training parameters
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Custom loss parameters
    confidence_penalty: float = 2.0
    length_weight: float = 0.1
    
    # Optimization
    gradient_clip_norm: float = 1.0
    early_stopping_patience: Optional[int] = None
    
    # Data
    val_ratio: float = 0.1
    num_workers: int = 0
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    cache_dir: str = "./cache"
    
    # Reproducibility
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    model: ModelConfig
    training: TrainingConfig
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize configuration.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(model_config, training_config)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """
        Load config from JSON file.
        
        Args:
            json_path: Path to JSON config file
            
        Returns:
            Config instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
    
    def to_json(self, json_path: str) -> None:
        """
        Save config to JSON file.
        
        Args:
            json_path: Path to save JSON config file
        """
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update_from_args(self, args) -> None:
        """
        Update config from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Update model config
        if hasattr(args, 'embed_dim'):
            self.model.embed_dim = args.embed_dim
        if hasattr(args, 'hidden_dim'):
            self.model.hidden_dim = args.hidden_dim
        if hasattr(args, 'num_layers'):
            self.model.num_layers = args.num_layers
        if hasattr(args, 'dropout'):
            self.model.dropout = args.dropout
        if hasattr(args, 'vocab_size'):
            self.model.vocab_size = args.vocab_size
        if hasattr(args, 'max_length'):
            self.model.max_length = args.max_length
        if hasattr(args, 'transformer_model'):
            self.model.transformer_model = args.transformer_model
        if hasattr(args, 'freeze_transformer'):
            self.model.freeze_transformer = args.freeze_transformer
        
        # Update training config
        if hasattr(args, 'batch_size'):
            self.training.batch_size = args.batch_size
        if hasattr(args, 'num_epochs'):
            self.training.num_epochs = args.num_epochs
        if hasattr(args, 'learning_rate'):
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'confidence_penalty'):
            self.training.confidence_penalty = args.confidence_penalty
        if hasattr(args, 'length_weight'):
            self.training.length_weight = args.length_weight
        if hasattr(args, 'seed'):
            self.training.random_seed = args.seed


def create_default_configs() -> Dict[str, Config]:
    """
    Create default configurations for different scenarios.
    
    Returns:
        Dictionary of default configurations
    """
    configs = {}
    
    # Demo configuration (fast training)
    demo_model = ModelConfig(
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        max_length=256,
        vocab_size=10000
    )
    demo_training = TrainingConfig(
        batch_size=32,
        num_epochs=2,
        learning_rate=2e-3
    )
    configs['demo'] = Config(demo_model, demo_training)
    
    # Full configuration (best performance)
    full_model = ModelConfig(
        embed_dim=300,
        hidden_dim=256,
        num_layers=2,
        max_length=512,
        vocab_size=50000
    )
    full_training = TrainingConfig(
        batch_size=16,
        num_epochs=5,
        learning_rate=1e-3,
        early_stopping_patience=3
    )
    configs['full'] = Config(full_model, full_training)
    
    # Fast configuration (balanced speed/performance)
    fast_model = ModelConfig(
        embed_dim=200,
        hidden_dim=200,
        num_layers=2,
        max_length=384,
        vocab_size=30000
    )
    fast_training = TrainingConfig(
        batch_size=24,
        num_epochs=3,
        learning_rate=1.5e-3
    )
    configs['fast'] = Config(fast_model, fast_training)
    
    return configs


def get_config_for_mode(mode: str) -> Config:
    """
    Get configuration for specific mode.
    
    Args:
        mode: Mode name ('demo', 'full', 'fast', 'compare')
        
    Returns:
        Configuration for the mode
    """
    configs = create_default_configs()
    
    if mode == 'demo':
        return configs['demo']
    elif mode == 'full':
        return configs['full']
    elif mode in ['compare', 'fast']:
        return configs['fast']
    else:
        # Default configuration
        return Config()


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Model validation
    if config.model.embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    
    if config.model.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    
    if config.model.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    
    if not 0 <= config.model.dropout <= 1:
        raise ValueError("dropout must be between 0 and 1")
    
    if config.model.max_length <= 0:
        raise ValueError("max_length must be positive")
    
    if config.model.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    
    # Training validation
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config.training.confidence_penalty < 0:
        raise ValueError("confidence_penalty must be non-negative")
    
    if config.training.length_weight < 0:
        raise ValueError("length_weight must be non-negative")
    
    if not 0 < config.training.val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")


def print_config(config: Config) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration to print
    """
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    print("\nModel Configuration:")
    print("-" * 30)
    for key, value in config.model.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    print("-" * 30)
    for key, value in config.training.__dict__.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)


# Example configuration files
EXAMPLE_CONFIGS = {
    "demo": {
        "model": {
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_layers": 1,
            "max_length": 256,
            "vocab_size": 10000
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 2,
            "learning_rate": 0.002
        }
    },
    "production": {
        "model": {
            "embed_dim": 300,
            "hidden_dim": 512,
            "num_layers": 3,
            "max_length": 512,
            "vocab_size": 50000,
            "transformer_model": "bert-large-uncased"
        },
        "training": {
            "batch_size": 8,
            "num_epochs": 10,
            "learning_rate": 0.0005,
            "early_stopping_patience": 5,
            "confidence_penalty": 3.0,
            "length_weight": 0.2
        }
    }
}


def save_example_configs(output_dir: str = "./configs") -> None:
    """
    Save example configuration files.
    
    Args:
        output_dir: Directory to save config files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, config_dict in EXAMPLE_CONFIGS.items():
        config_file = output_path / f"{name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved example config: {config_file}")
