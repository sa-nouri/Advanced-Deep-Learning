import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from transformers import BertTokenizer
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            raise

def load_data(data_path, test_size=0.2, random_state=42):
    try:
        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
            
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        logger.info(f"Loaded {len(df)} samples, split into {len(train_df)} train and {len(val_df)} validation samples")
        return train_df, val_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_data_loaders(train_df, val_df, tokenizer, batch_size=32, max_length=128):
    try:
        # Create datasets
        train_dataset = SentimentDataset(
            texts=train_df['text'].values,
            labels=train_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_dataset = SentimentDataset(
            texts=val_df['text'].values,
            labels=val_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

def prepare_tokenizer(model_name='bert-base-uncased'):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized tokenizer with {model_name}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error preparing tokenizer: {str(e)}")
        raise

def save_tokenizer(tokenizer, save_dir):
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved tokenizer to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving tokenizer: {str(e)}")
        raise 
