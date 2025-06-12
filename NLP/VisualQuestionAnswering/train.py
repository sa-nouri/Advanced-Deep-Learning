import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VQATrainer:
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            for batch_idx, (images, questions, question_lengths, answers) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images, questions, question_lengths)
                loss = self.criterion(outputs, answers)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += answers.size(0)
                correct += predicted.eq(answers).sum().item()
                
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
                for images, questions, question_lengths, answers in tqdm(val_loader):
                    images = images.to(self.device)
                    questions = questions.to(self.device)
                    answers = answers.to(self.device)
                    
                    outputs = self.model(images, questions, question_lengths)
                    loss = self.criterion(outputs, answers)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += answers.size(0)
                    correct += predicted.eq(answers).sum().item()
                    
            return total_loss / len(val_loader), 100. * correct / total
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        best_val_acc = 0
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for epoch in range(num_epochs):
                logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
                
                # Training
                train_loss, train_acc = self.train_epoch(train_loader)
                logger.info(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
                
                # Validation
                val_loss, val_acc = self.evaluate(val_loader)
                logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
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
