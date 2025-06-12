import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class SentimentAnalysisTrainer:
    def __init__(self, model, device, learning_rate=2e-5, weight_decay=0.01):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_epoch(self, train_loader, scheduler=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs[0]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                # Calculate metrics
                total_loss += loss.item()
                logits = outputs[1]
                _, predicted = torch.max(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 100 == 0:
                    logger.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}, '
                              f'Accuracy = {100.*correct/total:.2f}%')
                    
            return total_loss / len(train_loader), 100. * correct / total
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                    loss = outputs[0]
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    logits = outputs[1]
                    _, predicted = torch.max(logits, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            return total_loss / len(val_loader), 100. * correct / total
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def train(self, train_loader, val_loader, num_epochs, save_dir, warmup_steps=0):
        best_val_acc = 0
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create scheduler
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            for epoch in range(num_epochs):
                logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
                
                # Training
                train_loss, train_acc = self.train_epoch(train_loader, scheduler)
                logger.info(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
                
                # Validation
                val_loss, val_acc = self.evaluate(val_loader)
                logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, save_dir / 'best_model.pth')
                    
                # Save training history
                history = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                with open(save_dir / 'training_history.json', 'a') as f:
                    json.dump(history, f)
                    f.write('\n')
                    
        except Exception as e:
            logger.error(f"Error during training loop: {str(e)}")
            raise 
