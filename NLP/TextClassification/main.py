import torch
import logging
import argparse
from pathlib import Path
import json
from model import (
    BertForTextClassification,
    FastTextModel,
    Word2VecModel,
    GloVeModel,
    create_embedding_layer
)
from data import (
    load_data,
    create_data_loaders,
    prepare_tokenizer,
    save_tokenizer
)
from train import TextClassificationTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification Training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the data file (CSV or JSON)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='bert',
                      choices=['bert', 'fasttext', 'word2vec', 'glove'],
                      help='Type of model to use')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='Name of the pre-trained model (for BERT)')
    parser.add_argument('--num_classes', type=int, required=True,
                      help='Number of classes for classification')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0,
                      help='Number of warmup steps for learning rate scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Set random seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load data
        train_texts, val_texts, train_labels, val_labels = load_data(
            args.data_path,
            model_type=args.model_type
        )
        
        # Prepare tokenizer
        tokenizer = prepare_tokenizer(
            model_type=args.model_type,
            model_name=args.model_name
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_texts, val_texts, train_labels, val_labels,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_type=args.model_type,
            num_workers=args.num_workers
        )
        
        # Initialize model
        if args.model_type == 'bert':
            model = BertForTextClassification(
                model_name=args.model_name,
                num_labels=args.num_classes
            )
        elif args.model_type == 'fasttext':
            model = FastTextModel(
                vocab_size=len(train_loader.dataset.word2idx),
                embedding_dim=300,
                num_classes=args.num_classes
            )
        elif args.model_type == 'word2vec':
            model = Word2VecModel(
                vocab_size=len(train_loader.dataset.word2idx),
                embedding_dim=300,
                num_classes=args.num_classes
            )
        elif args.model_type == 'glove':
            model = GloVeModel(
                vocab_size=len(train_loader.dataset.word2idx),
                embedding_dim=300,
                num_classes=args.num_classes
            )
            
        model = model.to(device)
        logger.info(f"Initialized {args.model_type} model")
        
        # Initialize trainer
        trainer = TextClassificationTrainer(
            model=model,
            device=device,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Save configuration
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        # Save tokenizer if using BERT
        if args.model_type == 'bert':
            save_tokenizer(tokenizer, save_dir)
            
        # Train model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=save_dir,
            warmup_steps=args.warmup_steps
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
