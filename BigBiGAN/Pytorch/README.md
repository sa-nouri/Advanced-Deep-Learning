# BigBiGAN-PyTorch

An unofficial PyTorch implementation of [BigBiGAN](https://arxiv.org/abs/1907.02544) for unsupervised representation learning.

## Overview

BigBiGAN extends the BigGAN architecture by adding an encoder network that learns to map images to latent representations. This implementation includes both BigBiGAN and BigGAN architectures, with support for various datasets and training configurations.

## Architecture

### BigBiGAN
![BigBiGAN Architecture](https://github.com/RKorzeniowski/BigBiGAN-PyTorch/blob/main/imgs/bigbigan_arch.png)

### BigGAN
![BigGAN Architecture](https://github.com/RKorzeniowski/BigBiGAN-PyTorch/blob/main/imgs/biggan_arch.jpg)

## Features

### Model Components
- **Generator**: Based on BigGAN architecture with conditional batch normalization
- **Discriminator**: Implements spectral normalization and self-attention
- **Encoder**: RevNet-based architecture for image-to-latent mapping
- **Loss Functions**: Supports both BiGAN and WGAN losses

### Training Features
- Support for multiple datasets
- Configurable model architecture
- Comprehensive logging and visualization
- Checkpoint management
- Flexible training pipeline

### Supported Datasets
- MNIST (32x32)
- Fashion MNIST (32x32)
- CIFAR10 (32x32)
- CIFAR100 (32x32)
- Imagewoof (64x64)
- Imagenette (64x64)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Advanced-Deep-Learning.git
cd Advanced-Deep-Learning/BigBiGAN/Pytorch

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_gan.py --dataset CIFAR10 --data_path /path/to/data --model_architecture bigbigan
```

### Command Line Arguments

- `--dataset`: Dataset to use (choices: ["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof"])
- `--data_path`: Path to dataset root folder
- `--model_architecture`: Type of architecture (choices: ["bigbigan", "biggan"])

## Results

### CIFAR10
![CIFAR10 Results](https://github.com/RKorzeniowski/BigBiGAN-PyTorch/blob/main/imgs/CIFAR10_sample.png)

### Imagewoof
![Imagewoof Results](https://github.com/RKorzeniowski/BigBiGAN-PyTorch/blob/main/imgs/imagewoof_sample.png)

## Code Structure

```
BigBiGAN/
├── src/
│   ├── models/          # Model implementations
│   ├── data/           # Data loading and processing
│   ├── training/       # Training utilities
│   ├── utils/          # Helper functions
│   └── visualization/  # Visualization tools
├── configs/            # Configuration files
├── scripts/            # Training and evaluation scripts
└── tests/             # Unit tests
```

## Requirements

- Python 3.7+
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- tensorboard >= 2.3.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [BigGAN](https://github.com/taki0112/BigGAN-Tensorflow)
- [RevNet](https://github.com/google/revisiting-self-supervised)
- [BigBiGAN TensorFlow](https://github.com/LEGO999/BigBiGAN-TensorFlow2.0)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
