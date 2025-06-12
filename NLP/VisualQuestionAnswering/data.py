import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import logging
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

logger = logging.getLogger(__name__)

class VQADataset(Dataset):
    def __init__(self, image_dir, questions_file, answers_file, vocab, max_question_length=20):
        self.image_dir = Path(image_dir)
        self.max_question_length = max_question_length
        
        try:
            # Load questions and answers
            with open(questions_file, 'r') as f:
                self.questions = json.load(f)
            with open(answers_file, 'r') as f:
                self.answers = json.load(f)
                
            # Image transformations
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Vocabulary
            self.vocab = vocab
            self.word2idx = {word: idx for idx, word in enumerate(vocab)}
            
            logger.info(f"Dataset initialized with {len(self.questions)} samples")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise
            
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        try:
            # Load and transform image
            image_path = self.image_dir / f"{self.questions[idx]['image_id']}.jpg"
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # Process question
            question = self.questions[idx]['question']
            tokens = word_tokenize(question.lower())
            question_length = min(len(tokens), self.max_question_length)
            
            # Convert tokens to indices
            question_indices = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                              for token in tokens[:self.max_question_length]]
            question_indices.extend([self.word2idx['<PAD>']] * 
                                  (self.max_question_length - len(question_indices)))
            
            # Get answer
            answer = self.answers[idx]['answer']
            
            return {
                'image': image,
                'question': torch.LongTensor(question_indices),
                'question_length': question_length,
                'answer': answer
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

def build_vocab(questions_file, min_word_freq=5):
    try:
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Load questions
        with open(questions_file, 'r') as f:
            questions = json.load(f)
            
        # Tokenize questions and count word frequencies
        word_freq = Counter()
        for q in questions:
            tokens = word_tokenize(q['question'].lower())
            word_freq.update(tokens)
            
        # Build vocabulary
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab.extend([word for word, freq in word_freq.items() 
                     if freq >= min_word_freq])
        
        logger.info(f"Vocabulary built with {len(vocab)} words")
        return vocab
        
    except Exception as e:
        logger.error(f"Error building vocabulary: {str(e)}")
        raise

def create_data_loaders(image_dir, questions_file, answers_file, vocab, 
                       batch_size=32, num_workers=4):
    try:
        # Create datasets
        dataset = VQADataset(image_dir, questions_file, answers_file, vocab)
        
        # Split into train and validation sets (80-20 split)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise 
