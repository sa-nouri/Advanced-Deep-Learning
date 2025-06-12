# Image Classification Module

This module implements state-of-the-art deep learning models for image classification tasks. Currently, it features an EfficientNet-B0 implementation with plans to add more architectures.

## Features

- Pre-trained EfficientNet-B0 model
- Transfer learning support
- Data augmentation pipeline
- Comprehensive metrics tracking
- Training visualization tools
- Model checkpointing and early stopping
- Learning rate scheduling
- TensorBoard integration

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on the Cars196 dataset:

```bash
python train.py \
    --dataset cars196 \
    --batch-size 64 \
    --epochs 14 \
    --img-size 224 \
    --learning-rate 1e-2 \
    --dropout-rate 0.2 \
    --output-dir output
```

### Command Line Arguments

- `--dataset`: Dataset name (default: "cars196")
- `--batch-size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 14)
- `--img-size`: Image size (default: 224)
- `--learning-rate`: Initial learning rate (default: 1e-2)
- `--dropout-rate`: Dropout rate for regularization (default: 0.2)
- `--output-dir`: Directory to save outputs (default: "output")
- `--weights`: Path to pre-trained weights (optional)

### Using Pre-trained Weights

To use pre-trained weights:

```bash
python train.py \
    --weights path/to/weights.h5 \
    --output-dir output
```

### Monitoring Training

The training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir output/logs
```

## Model Architecture

The EfficientNet model consists of:

1. **Data Augmentation**:
   - Random rotation
   - Random translation
   - Random flip
   - Random contrast

2. **Base Model**:
   - EfficientNet-B0 backbone
   - Pre-trained on ImageNet
   - Transfer learning support

3. **Classification Head**:
   - Global average pooling
   - Batch normalization
   - Dropout regularization
   - Dense layer with softmax activation

## Evaluation Metrics

The model tracks the following metrics during training:

- Accuracy
- Precision
- Recall
- AUC
- PR curve
- False positives/negatives
- True positives/negatives

## Output Directory Structure

```
output/
├── checkpoints/     # Model checkpoints
├── logs/           # Training logs and TensorBoard files
└── final_model.h5  # Final trained model
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
