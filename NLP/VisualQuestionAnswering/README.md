# Visual Question Answering (VQA)

This module implements a Visual Question Answering model that can answer questions about images. The model combines a ResNet-based image encoder with a GRU-based question encoder to generate answers.

## Features

- ResNet-18 based image feature extraction
- GRU-based question processing
- Attention mechanism for focusing on relevant image regions
- Comprehensive logging and error handling
- Training history visualization
- Attention map visualization
- Model checkpointing and early stopping

## Requirements

See `requirements.txt` for the complete list of dependencies. Key requirements include:
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- NLTK >= 3.5
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

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

### Training

To train the model, use the `main.py` script:

```bash
python main.py \
    --image_dir /path/to/images \
    --questions_file /path/to/questions.json \
    --answers_file /path/to/answers.json \
    --save_dir checkpoints \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --weight_decay 1e-4
```

### Visualization

The module provides several visualization tools:

1. Training History:
```python
from visualize import plot_training_history
plot_training_history('checkpoints/training_history.json', 'training_plot.png')
```

2. Attention Maps:
```python
from visualize import visualize_attention
visualize_attention(
    image_path='path/to/image.jpg',
    question='What color is the car?',
    attention_map=model.get_attention_map(),
    save_path='attention.png'
)
```

3. Predictions:
```python
from visualize import visualize_predictions
visualize_predictions(
    model=model,
    image_path='path/to/image.jpg',
    question='What color is the car?',
    vocab=vocab,
    idx2word=idx2word,
    device=device,
    save_path='prediction.png'
)
```

## Model Architecture

The VQA model consists of three main components:

1. **Image Encoder**:
   - ResNet-18 backbone
   - Feature projection layer
   - Output: 512-dimensional image features

2. **Question Encoder**:
   - Word embedding layer (GloVe)
   - GRU layer
   - Output: 150-dimensional question features

3. **Answer Generator**:
   - Combined feature processing
   - Multi-layer classifier
   - Softmax output for answer prediction

## Performance

The model's performance can be monitored through:
- Training and validation loss
- Accuracy metrics
- Attention map visualizations
- Example predictions

## Future Improvements

1. Model Enhancements:
   - Implement more sophisticated attention mechanisms
   - Add support for multiple answers
   - Incorporate pre-trained language models

2. Training Improvements:
   - Add data augmentation
   - Implement cross-validation
   - Add support for multi-GPU training

3. Evaluation:
   - Add more evaluation metrics
   - Implement human evaluation
   - Add support for VQA challenge metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
