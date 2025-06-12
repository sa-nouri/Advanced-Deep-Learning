import torch
import numpy as np
from typing import List, Dict, Union, Tuple
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_metrics(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    average: str = 'weighted'
) -> Dict[str, float]:
    """Calculate common classification metrics."""
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average)
        }
        
        logger.info("Calculated metrics successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def plot_confusion_matrix(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    labels: List[str],
    save_path: Union[str, Path] = None
) -> None:
    """Plot confusion matrix."""
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_metrics_history(
    history: Dict[str, List[float]],
    save_path: Union[str, Path] = None
) -> None:
    """Plot training metrics history."""
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved metrics history plot to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting metrics history: {str(e)}")
        raise

def calculate_class_weights(
    y: Union[List, np.ndarray, torch.Tensor],
    method: str = 'balanced'
) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    try:
        y = np.array(y)
        classes = np.unique(y)
        n_samples = len(y)
        
        if method == 'balanced':
            weights = n_samples / (len(classes) * np.bincount(y))
        elif method == 'inverse':
            weights = 1.0 / np.bincount(y)
        else:
            raise ValueError("Unsupported method. Use 'balanced' or 'inverse'")
            
        weights = torch.FloatTensor(weights)
        logger.info("Calculated class weights successfully")
        return weights
        
    except Exception as e:
        logger.error(f"Error calculating class weights: {str(e)}")
        raise 
