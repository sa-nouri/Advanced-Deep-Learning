import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
import logging

logger = logging.getLogger(__name__)

class BertForSentimentAnalysis(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        try:
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.init_weights()
            logger.info("BERT sentiment analysis model initialized successfully")
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

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions)
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class SentimentAnalysisModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        try:
            self.bert = BertForSentimentAnalysis.from_pretrained(
                bert_model_name,
                num_labels=num_labels
            )
            logger.info(f"Sentiment analysis model initialized with {bert_model_name}")
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis model: {str(e)}")
            raise

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        try:
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        try:
            self.eval()
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs[0]
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise 
