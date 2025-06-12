"""
Model export functionality for the TWR-VAE model.
Supports exporting to ONNX and TorchScript formats.
"""

import torch
import torch.onnx
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def export_to_onnx(
    model: torch.nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int],
    opset_version: int = 12,
    dynamic_axes: Optional[Dict] = None
) -> None:
    """
    Export the model to ONNX format.
    
    Args:
        model: The TWR-VAE model to export
        save_path: Path to save the ONNX model
        input_shape: Shape of the input tensor (batch_size, seq_len, hidden_size)
        opset_version: ONNX opset version to use
        dynamic_axes: Dictionary specifying dynamic axes
    """
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        logger.info(f"Successfully exported model to ONNX format at {save_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {str(e)}")
        raise

def export_to_torchscript(
    model: torch.nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int],
    optimize: bool = True
) -> None:
    """
    Export the model to TorchScript format.
    
    Args:
        model: The TWR-VAE model to export
        save_path: Path to save the TorchScript model
        input_shape: Shape of the input tensor (batch_size, seq_len, hidden_size)
        optimize: Whether to optimize the model
    """
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save the model
        traced_model.save(str(save_path))
        logger.info(f"Successfully exported model to TorchScript format at {save_path}")
    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {str(e)}")
        raise

def quantize_model(
    model: torch.nn.Module,
    save_path: Path,
    input_shape: Tuple[int, int, int],
    dtype: torch.dtype = torch.qint8
) -> None:
    """
    Quantize the model for better inference performance.
    
    Args:
        model: The TWR-VAE model to quantize
        save_path: Path to save the quantized model
        input_shape: Shape of the input tensor (batch_size, seq_len, hidden_size)
        dtype: Quantization data type
    """
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Prepare the model for quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=dtype
        )
        
        # Save the quantized model
        torch.save(quantized_model.state_dict(), save_path)
        logger.info(f"Successfully quantized and saved model at {save_path}")
    except Exception as e:
        logger.error(f"Failed to quantize model: {str(e)}")
        raise 
