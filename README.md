## Advanced Deep Learning

Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example.
The repo contains some advanced topics' implementation about deep neural networks.

### Selected Projects

- Implementing Autoregressive Generative models such as PixelCNN, and PixelRNN
- Implementing Contrastive Representation Learning algorithms such as Contrastive Predictive Coding, and
SimCLR
- Implementing Unsupervised Representation Learning algorithms such as CutMix, and Image Rotation
Prediction
- Implementing Text (Language) Modelling models such as Neural Variational Document Model, Multi-Level
Latent Variable, and Timestep-Wise Regularization
- Implementing Generative Adversarial Networks such as BiGAN, BigBiGAN, InfoGAN-BigBiGAN
- Implementing Image Classification models such as EfficientNet-B0, VGGNet, ResNet, Inception, and Xception 
- Implementing Text Classification models by BERT, Fasttext, Word2vec, and Glove
- Implementing Visual Question Answering models such as ResNet+Glove

### Natural Language Processing (NLP) Module

The NLP module provides comprehensive implementations of various natural language processing tasks, including text classification, sentiment analysis, and visual question answering. Each submodule is designed with modularity, extensibility, and best practices in mind.

#### Structure
```
NLP/
├── TextClassification/     # Multi-model text classification
├── SentimentAnalysis/      # BERT-based sentiment analysis
├── VisualQuestionAnswering/# Image-based question answering
└── utils/                  # Common utilities and tools
```

#### Technical Implementation Details

1. **Text Classification**
   - **BERT Model**:
     - Uses pre-trained BERT model as encoder
     - Implements custom classification head with dropout
     - Supports transfer learning with fine-tuning
     - Uses AdamW optimizer with weight decay
     - Implements learning rate scheduling with warmup
     - Handles sequence lengths up to 512 tokens
     - Supports both single-label and multi-label classification

   - **FastText Model**:
     - Implements character n-gram features (3-6 grams)
     - Uses hierarchical softmax for efficient training
     - Supports subword embeddings
     - Implements bag of n-grams with TF-IDF weighting
     - Uses adaptive learning rate

   - **Word2Vec Model**:
     - Implements CBOW and Skip-gram architectures
     - Uses negative sampling for efficiency
     - Supports both pre-trained and custom embeddings
     - Implements CNN layers for feature extraction
     - Uses max pooling for feature aggregation

   - **GloVe Model**:
     - Supports pre-trained GloVe embeddings
     - Implements LSTM layers for sequence modeling
     - Uses bidirectional processing
     - Implements attention mechanism
     - Supports variable length sequences

2. **Sentiment Analysis**
   - **Model Architecture**:
     - BERT-based encoder with custom classification head
     - Implements attention mechanism for focus on important tokens
     - Uses dropout for regularization (p=0.1)
     - Supports binary and multi-class classification
     - Implements label smoothing for better generalization

   - **Training Process**:
     - Uses AdamW optimizer with weight decay
     - Implements learning rate scheduling with warmup
     - Uses gradient clipping (max_norm=1.0)
     - Implements early stopping with patience
     - Supports mixed precision training (FP16)

3. **Visual Question Answering**
   - **Image Processing**:
     - ResNet-18 backbone with pre-trained weights
     - Feature extraction from last convolutional layer
     - Spatial attention mechanism
     - Image feature projection to 512 dimensions
     - Supports multiple image sizes with adaptive pooling

   - **Question Processing**:
     - Word embedding layer (300 dimensions)
     - Bidirectional GRU for sequence modeling
     - Question encoding to 150 dimensions
     - Attention mechanism for question understanding
     - Supports variable length questions

   - **Answer Generation**:
     - Multi-modal fusion of image and question features
     - Multi-layer classifier with ReLU activation
     - Dropout for regularization (p=0.5)
     - Softmax output for answer prediction
     - Supports both open-ended and multiple-choice answers

4. **Common Utilities**
   - **Configuration Management**:
     - YAML and JSON support
     - Type-safe configuration classes
     - Validation of configuration parameters
     - Support for nested configurations
     - Environment variable integration

   - **Metrics and Evaluation**:
     - Accuracy, precision, recall, F1 score
     - Confusion matrix visualization
     - ROC and PR curves
     - Learning curve analysis
     - Model comparison tools

   - **Training Utilities**:
     - Checkpoint management
     - Early stopping implementation
     - Learning rate scheduling
     - Gradient accumulation
     - Mixed precision training support

   - **Data Processing**:
     - Efficient data loading with caching
     - Multi-process data loading
     - Custom dataset implementations
     - Data augmentation pipelines
     - Tokenization utilities

### Text Generation Module

The Text Generation module implements a Timestep-Wise Regularized Variational Autoencoder (TWR-VAE) for text generation tasks. This module focuses on generating coherent and contextually relevant text responses in dialogue systems.

#### Features

1. **Model Architecture**:
   - Bidirectional GRU for utterance encoding
   - Context encoder with GRU for dialogue history
   - Recognition network for posterior distribution
   - Prior network for prior distribution
   - Decoder with GRU for response generation

2. **Key Features**:
   - Timestep-wise regularization for better temporal dependencies
   - Support for multiple datasets (PTB, Yelp, Yahoo)
   - Comprehensive evaluation metrics
   - Model export to ONNX and TorchScript
   - Model quantization for efficient inference

3. **Training Process**:
   - Reconstruction loss for response generation
   - KL divergence loss with annealing
   - Beta parameter for balancing losses
   - Adam optimizer with gradient clipping
   - Learning rate scheduling

4. **Evaluation Metrics**:
   - BLEU scores (recall and precision)
   - Bag-of-words similarity metrics
   - Distinct metrics for diversity
   - Perplexity for language modeling

5. **Code Quality**:
   - Type hints and documentation
   - Modular design with reusable components
   - Comprehensive error handling
   - Logging and monitoring
   - Unit tests and validation

6. **Model Export**:
   - ONNX export for cross-platform deployment
   - TorchScript export for optimized inference
   - Model quantization for reduced size
   - Dynamic axes support for variable lengths
   - Batch processing optimization

#### Usage

Each module can be used independently and includes:
- Clear installation instructions
- Example usage with command-line arguments
- Configuration examples
- Training and evaluation scripts
- Visualization tools

For detailed usage instructions, please refer to the README files in each module's directory.

#### Requirements

Common requirements across all modules:
- Python 3.7+
- PyTorch >= 1.7.0
- Transformers >= 4.5.0
- scikit-learn >= 0.24.0
- Additional requirements are specified in each module's requirements.txt

#### Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Each module follows consistent coding standards and includes proper documentation.

#### License

This project is licensed under the MIT License - see the LICENSE file for details.
