import torch
import logging
import argparse
from pathlib import Path
import json
import numpy as np
from model import VQAModel
from data import build_vocab, create_data_loaders
from train import VQATrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vqa_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VQA model')
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Directory containing images')
    parser.add_argument('--questions_file', type=str, required=True,
                      help='Path to questions JSON file')
    parser.add_argument('--answers_file', type=str, required=True,
                      help='Path to answers JSON file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')
        
        # Build vocabulary
        vocab = build_vocab(args.questions_file)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            args.image_dir,
            args.questions_file,
            args.answers_file,
            vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Initialize model
        embedding_matrix = np.random.randn(len(vocab), 300)  # Replace with actual GloVe embeddings
        model = VQAModel(
            embedding_matrix=embedding_matrix,
            embedding_dim=300,
            hidden_dim=150,
            num_classes=1000  # Adjust based on your answer vocabulary
        ).to(device)
        
        # Initialize trainer
        trainer = VQATrainer(
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
            
        # Train model
        logger.info('Starting training...')
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=save_dir
        )
        logger.info('Training completed successfully')
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
