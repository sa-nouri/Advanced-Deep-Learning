import torch
import logging
import argparse
from pathlib import Path
import json
from model import SentimentAnalysisModel
from data import load_data, create_data_loaders, prepare_tokenizer, save_tokenizer
from train import SentimentAnalysisTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to data file (CSV or JSON)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='BERT model name')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0,
                      help='Number of warmup steps')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')
        
        # Load and prepare data
        train_df, val_df = load_data(args.data_path)
        
        # Prepare tokenizer
        tokenizer = prepare_tokenizer(args.model_name)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Initialize model
        model = SentimentAnalysisModel(
            bert_model_name=args.model_name,
            num_labels=2  # Binary classification
        ).to(device)
        
        # Initialize trainer
        trainer = SentimentAnalysisTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create save directory
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration
        config = vars(args)
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        # Save tokenizer
        save_tokenizer(tokenizer, save_dir / 'tokenizer')
        
        # Train model
        logger.info('Starting training...')
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=save_dir,
            warmup_steps=args.warmup_steps
        )
        logger.info('Training completed successfully')
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
