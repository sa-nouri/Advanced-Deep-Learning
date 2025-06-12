from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Base configuration for models."""
    model_type: str
    model_name: str
    num_classes: int
    hidden_size: int = 768
    dropout: float = 0.1
    max_length: int = 512
    pretrained: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'max_length': self.max_length,
            'pretrained': self.pretrained
        }

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.001
    num_workers: int = 4
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'gradient_clip_val': self.gradient_clip_val,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_delta': self.early_stopping_delta,
            'num_workers': self.num_workers,
            'seed': self.seed
        }

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: Union[str, Path]
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    random_state: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_path': str(self.data_path),
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'shuffle': self.shuffle,
            'random_state': self.random_state
        }

class ConfigManager:
    """Manager for handling configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.model_config: Optional[ModelConfig] = None
        self.training_config: Optional[TrainingConfig] = None
        self.data_config: Optional[DataConfig] = None
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from file."""
        try:
            config_path = Path(config_path) if config_path else self.config_path
            if not config_path:
                raise ValueError("No config path provided")
                
            with open(config_path, 'r') as f:
                if config_path.suffix == '.yaml':
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError("Unsupported file format. Use .yaml or .json")
                    
            self.model_config = ModelConfig.from_dict(config_dict['model'])
            self.training_config = TrainingConfig.from_dict(config_dict['training'])
            self.data_config = DataConfig.from_dict(config_dict['data'])
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def save_config(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        try:
            save_path = Path(save_path) if save_path else self.config_path
            if not save_path:
                raise ValueError("No save path provided")
                
            config_dict = {
                'model': self.model_config.to_dict(),
                'training': self.training_config.to_dict(),
                'data': self.data_config.to_dict()
            }
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                if save_path.suffix == '.yaml':
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(config_dict, f, indent=4)
                else:
                    raise ValueError("Unsupported file format. Use .yaml or .json")
                    
            logger.info(f"Saved configuration to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise 
