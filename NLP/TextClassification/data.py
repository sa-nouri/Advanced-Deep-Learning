import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
import re

logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, model_type='bert'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        # For non-BERT models, create vocabulary
        if model_type != 'bert':
            self.word2idx = self._create_vocabulary()
            
    def _create_vocabulary(self):
        try:
            # Tokenize all texts
            all_words = []
            for text in self.texts:
                words = re.findall(r'\w+', text.lower())
                all_words.extend(words)
            
            # Create vocabulary
            word_counts = Counter(all_words)
            word2idx = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
            word2idx['<PAD>'] = 0
            word2idx['<UNK>'] = len(word2idx)
            
            logger.info(f"Created vocabulary with {len(word2idx)} tokens")
            return word2idx
            
        except Exception as e:
            logger.error(f"Error creating vocabulary: {str(e)}")
            raise
            
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            if self.model_type == 'bert':
                # BERT tokenization
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'token_type_ids': encoding['token_type_ids'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            else:
                # Simple tokenization for other models
                words = re.findall(r'\w+', text.lower())
                word_ids = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
                
                # Pad or truncate
                if len(word_ids) > self.max_length:
                    word_ids = word_ids[:self.max_length]
                else:
                    word_ids = word_ids + [self.word2idx['<PAD>']] * (self.max_length - len(word_ids))
                
                return {
                    'input_ids': torch.tensor(word_ids, dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long),
                    'lengths': torch.tensor(len(words), dtype=torch.long)
                }
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            raise

def load_data(data_path, model_type='bert', test_size=0.2, random_state=42):
    try:
        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
            
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples")
        return train_texts, val_texts, train_labels, val_labels
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_data_loaders(train_texts, val_texts, train_labels, val_labels, tokenizer, 
                       batch_size=32, max_length=512, model_type='bert', num_workers=4):
    try:
        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, tokenizer, max_length, model_type
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, max_length, model_type
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

def prepare_tokenizer(model_type='bert', model_name='bert-base-uncased'):
    try:
        if model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized BERT tokenizer with model {model_name}")
        else:
            tokenizer = None
            logger.info("No tokenizer needed for non-BERT models")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error preparing tokenizer: {str(e)}")
        raise

def save_tokenizer(tokenizer, save_dir):
    try:
        if tokenizer is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Saved tokenizer to {save_dir}")
            
    except Exception as e:
        logger.error(f"Error saving tokenizer: {str(e)}")
        raise 
