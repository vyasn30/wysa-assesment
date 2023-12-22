import torch.nn as nn 
from transformers import RobertaModel 


class RobertaClass(nn.Module):
    """
    Based on Roberta which is a BeRT derivative. Added two linear layers 
    to learn sentiment and product.
    """

    def __init__(self, num_classes_sentiment=5, num_classes_product=10):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier_sentiment = nn.Linear(768, num_classes_sentiment) # Layer for sentiment classification
        self.classifier_product = nn.Linear(768, num_classes_product)  # Layer for product classification

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)

        # Output for sentiment classification
        output_sentiment = self.classifier_sentiment(pooler)

        # Output for product classification
        output_product = self.classifier_product(pooler)

        return output_sentiment, output_product

