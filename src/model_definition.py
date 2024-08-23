# model_definition.py
import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('asafaya/bert-base-arabic')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits)