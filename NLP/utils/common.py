import torch
import logging
import random
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union
import yaml

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")
    except Exception as e:
        logger.error(f"Error setting seed: {str(e)}")
        raise

def setup_logging(log_file: Union[str, Path], level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    try:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging set up to {log_file}")
    except Exception as e:
        logger.error(f"Error setting up logging: {str(e)}")
        raise

def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.suffix == '.json':
                json.dump(config, f, indent=4)
            elif save_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
                
        logger.info(f"Saved configuration to {save_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
                
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def get_device() -> torch.device:
    """Get the appropriate device for training."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device
    except Exception as e:
        logger.error(f"Error getting device: {str(e)}")
        raise

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        logger.error(f"Error counting parameters: {str(e)}")
        raise

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: Union[str, Path],
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Save model checkpoint."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if metrics:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, Path]
) -> int:
    """Load model checkpoint."""
    try:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise 
