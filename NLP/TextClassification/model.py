import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BertForTextClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        try:
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.init_weights()
            logger.info("BERT text classification model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BERT model: {str(e)}")
            raise

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        try:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None):
        super().__init__()
        try:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                self.embedding.weight.requires_grad = False

            self.fc = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            logger.info("FastText model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FastText model: {str(e)}")
            raise

    def forward(self, x):
        try:
            embedded = self.embedding(x)
            pooled = torch.mean(embedded, dim=1)
            return self.fc(pooled)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None):
        super().__init__()
        try:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                self.embedding.weight.requires_grad = False

            self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            
            self.fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
            logger.info("Word2Vec model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Word2Vec model: {str(e)}")
            raise

    def forward(self, x):
        try:
            embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
            
            x = F.relu(self.conv1(embedded))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)
            return self.fc(x)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None):
        super().__init__()
        try:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                self.embedding.weight.requires_grad = False

            self.lstm = nn.LSTM(
                embedding_dim,
                256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.5
            )
            
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            logger.info("GloVe model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GloVe model: {str(e)}")
            raise

    def forward(self, x, lengths):
        try:
            embedded = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed)
            
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            return self.fc(hidden)
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
