import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class VQAModel(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim=300, hidden_dim=150, num_classes=1000):
        super(VQAModel, self).__init__()
        try:
            # Image processing branch
            self.resnet = models.resnet18(pretrained=True)
            self.resnet.fc = nn.Sequential()  # Remove final classification layer
            self.image_projection = nn.Linear(512, hidden_dim)
            
            # Question processing branch
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix),
                freeze=True
            )
            self.gru = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            
            # Answer prediction
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
            
            logger.info("VQA model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VQA model: {str(e)}")
            raise

    def forward(self, image, question, question_lengths):
        try:
            # Process image
            image_features = self.resnet(image)
            image_features = self.image_projection(image_features)
            image_features = image_features.unsqueeze(0)  # Add sequence dimension
            
            # Process question
            embedded = self.embedding(question)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                question_lengths,
                batch_first=True,
                enforce_sorted=False
            )
            _, question_features = self.gru(packed, image_features)
            question_features = question_features.squeeze(0)
            
            # Combine features
            combined = torch.cat([question_features, image_features.squeeze(0)], dim=1)
            
            # Predict answer
            logits = self.classifier(combined)
            return F.softmax(logits, dim=1)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

def create_embedding_layer(weights_matrix, non_trainable=False):
    try:
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        embedding_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        if non_trainable:
            embedding_layer.weight.requires_grad = False
        return embedding_layer, num_embeddings, embedding_dim
    except Exception as e:
        logger.error(f"Error creating embedding layer: {str(e)}")
        raise 
