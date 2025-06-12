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

#### Features

1. **Text Classification**
   - Multiple model architectures (BERT, FastText, Word2Vec, GloVe)
   - Comprehensive data processing pipeline
   - Training utilities with checkpointing
   - Support for both CSV and JSON data formats
   - Proper error handling and logging

2. **Sentiment Analysis**
   - BERT-based text classification
   - Transfer learning from pre-trained models
   - Training history tracking
   - Model checkpointing
   - Learning rate scheduling with warmup

3. **Visual Question Answering**
   - ResNet-18 based image feature extraction
   - GRU-based question processing
   - Attention mechanism for focusing on relevant image regions
   - Training history visualization
   - Attention map visualization

4. **Common Utilities**
   - Configuration management
   - Metrics calculation and visualization
   - Logging and error handling
   - Model checkpointing
   - Data processing utilities

#### Key Features Across All Modules

- **Modular Design**: Each module is self-contained and follows consistent design patterns
- **Error Handling**: Comprehensive error handling and logging throughout
- **Documentation**: Detailed README files and inline documentation
- **Configuration**: Flexible configuration management using YAML/JSON
- **Metrics**: Common evaluation metrics and visualization tools
- **Training**: Support for model checkpointing and early stopping
- **Data Processing**: Standardized data loading and preprocessing pipelines

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
