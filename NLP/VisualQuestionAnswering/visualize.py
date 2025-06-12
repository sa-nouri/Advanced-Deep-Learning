import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging
from pathlib import Path
import seaborn as sns

logger = logging.getLogger(__name__)

def visualize_attention(image_path, question, attention_map, save_path=None):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = transform(image)
        
        # Convert attention map to numpy array
        attention_map = attention_map.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot attention map
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(attention_map, alpha=0.5, cmap='jet')
        plt.title(f'Attention Map\nQuestion: {question}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            logger.info(f'Saved attention visualization to {save_path}')
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing attention: {str(e)}")
        raise

def plot_training_history(history_file, save_path=None):
    try:
        # Load training history
        with open(history_file, 'r') as f:
            history = [json.loads(line) for line in f]
            
        # Extract metrics
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, train_loss, label='Training Loss')
        ax1.plot(epochs, val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, train_acc, label='Training Accuracy')
        ax2.plot(epochs, val_acc, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved training history plot to {save_path}')
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def visualize_predictions(model, image_path, question, vocab, idx2word, device, save_path=None):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Process question
        tokens = question.lower().split()
        question_indices = [vocab.index(token) if token in vocab else vocab.index('<UNK>')
                          for token in tokens]
        question_tensor = torch.LongTensor(question_indices).unsqueeze(0).to(device)
        question_length = torch.LongTensor([len(question_indices)]).to(device)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            output = model(image_tensor, question_tensor, question_length)
            predicted_idx = output.argmax(dim=1).item()
            confidence = output[0][predicted_idx].item()
            
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Plot prediction
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.6, f'Question: {question}', fontsize=12)
        plt.text(0.1, 0.5, f'Predicted Answer: {idx2word[predicted_idx]}', fontsize=12)
        plt.text(0.1, 0.4, f'Confidence: {confidence:.2%}', fontsize=12)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            logger.info(f'Saved prediction visualization to {save_path}')
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")
        raise 
