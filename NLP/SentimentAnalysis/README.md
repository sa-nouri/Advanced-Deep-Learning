# Sentiment Analysis

This module implements a BERT-based sentiment analysis model for binary classification of text sentiment. The model uses transfer learning from pre-trained BERT models and includes comprehensive training and evaluation utilities.

## Features

- BERT-based text classification
- Transfer learning from pre-trained models
- Comprehensive logging and error handling
- Training history tracking
- Model checkpointing
- Learning rate scheduling with warmup
- Support for CSV and JSON data formats

## Requirements

See `requirements.txt` for the complete list of dependencies. Key requirements include:
- PyTorch >= 1.7.0
- Transformers >= 4.5.0
- scikit-learn >= 0.24.0
- Other dependencies for data processing and visualization

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

The model expects input data in either CSV or JSON format with the following columns:
- `text`: The input text to classify
- `label`: The sentiment label (0 for negative, 1 for positive)

### Training

To train the model, use the `main.py` script:

```bash
python main.py \
    --data_path /path/to/data.csv \
    --save_dir checkpoints \
    --model_name bert-base-uncased \
    --batch_size 32 \
    --max_length 128 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 0
```

### Model Architecture

The sentiment analysis model consists of:
1. **BERT Encoder**:
   - Pre-trained BERT model (default: bert-base-uncased)
   - Configurable model size and architecture

2. **Classification Head**:
   - Dropout layer
   - Linear classification layer
   - Binary classification output

### Training Process

The training process includes:
1. Data loading and preprocessing
2. Tokenization using BERT tokenizer
3. Model training with:
   - AdamW optimizer
   - Learning rate scheduling with warmup
   - Gradient clipping
   - Model checkpointing
   - Early stopping

### Evaluation

The model's performance can be monitored through:
- Training and validation loss
- Accuracy metrics
- Training history plots
- Model checkpoints

## Future Improvements

1. Model Enhancements:
   - Support for multi-class classification
   - Additional pre-trained models (RoBERTa, ALBERT, etc.)
   - Ensemble methods

2. Training Improvements:
   - Cross-validation
   - Hyperparameter optimization
   - Multi-GPU training

3. Evaluation:
   - Additional metrics (F1, precision, recall)
   - Confusion matrix visualization
   - Error analysis tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
