# Text Classification Module

This module provides implementations of various text classification models using different approaches and architectures. It supports multiple model types including BERT, FastText, Word2Vec, and GloVe-based models.

## Features

- Multiple model architectures:
  - BERT-based classification with transfer learning
  - FastText model with n-gram features
  - Word2Vec model with CNN layers
  - GloVe model with LSTM layers
- Comprehensive data processing pipeline
- Training utilities with checkpointing and learning rate scheduling
- Support for both CSV and JSON data formats
- Proper error handling and logging
- Training history tracking and visualization

## Requirements

All required packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch >= 1.7.0
- Transformers >= 4.5.0
- scikit-learn >= 0.24.0
- Gensim >= 4.0.0 (for Word2Vec and FastText)
- NLTK >= 3.5

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

### Data Format

The module expects input data in either CSV or JSON format with the following structure:
- CSV: Must contain 'text' and 'label' columns
- JSON: Must contain a list of objects with 'text' and 'label' fields

### Training

To train a model, use the `main.py` script with appropriate arguments:

```bash
python main.py \
    --data_path path/to/your/data.csv \
    --model_type bert \
    --num_classes 2 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --save_dir checkpoints
```

### Command Line Arguments

- `--data_path`: Path to the data file (CSV or JSON)
- `--model_type`: Type of model to use (bert, fasttext, word2vec, glove)
- `--num_classes`: Number of classes for classification
- `--batch_size`: Batch size for training (default: 32)
- `--max_length`: Maximum sequence length (default: 512)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: 0)
- `--save_dir`: Directory to save model checkpoints (default: checkpoints)

## Model Architecture

### BERT Model
- Uses pre-trained BERT model as encoder
- Adds a classification head on top
- Supports transfer learning with fine-tuning

### FastText Model
- Uses character n-grams for feature extraction
- Includes embedding layer and fully connected network
- Good for handling out-of-vocabulary words

### Word2Vec Model
- Uses pre-trained Word2Vec embeddings
- Implements CNN layers for feature extraction
- Effective for capturing local patterns

### GloVe Model
- Uses pre-trained GloVe embeddings
- Implements LSTM layers for sequence modeling
- Good for capturing long-range dependencies

## Training Process

1. Data Loading and Preprocessing:
   - Loads data from CSV or JSON
   - Splits into training and validation sets
   - Tokenizes text based on model type
   - Creates data loaders

2. Model Training:
   - Initializes model and moves to appropriate device
   - Sets up optimizer and learning rate scheduler
   - Trains for specified number of epochs
   - Saves best model based on validation accuracy
   - Tracks training history

3. Evaluation:
   - Computes accuracy and loss on validation set
   - Logs metrics for monitoring
   - Saves training history for visualization

## Future Improvements

1. Model Enhancements:
   - Add support for more pre-trained models
   - Implement attention mechanisms
   - Add support for multi-label classification

2. Training Improvements:
   - Add support for mixed precision training
   - Implement gradient accumulation
   - Add support for distributed training

3. Evaluation Metrics:
   - Add support for more metrics (F1, precision, recall)
   - Add confusion matrix visualization
   - Add support for cross-validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
