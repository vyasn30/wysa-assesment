from src.ml.models.RobertaClass import RobertaClass
from transformers import RobertaTokenizer
import torch
import pandas as pd

# PRODUCT_MAPPING = {
#     'iPhone': 1,
#     'iPad or iPhone App': 2,
#     'iPad': 3,
#     'Google': 4,
#     'Android': 5,
#     'Apple': 6,
#     'Android App': 7,
#     'Other Google product or service': 8,
#     'Other Apple product or service': 9
# }

# EMOTION_MAPPING = {
#     'Positive emotion': 1,
#     'Negative emotion': 2, 
#     'No emotion toward brand or product': 3
# }

PRODUCT_MAPPING = {
    1: 'iPhone',
    2: 'iPad or iPhone App',
    3: 'iPad',
    4: 'Google',
    5: 'Android',
    6: 'Apple',
    7: 'Android App',
    8: 'Other Google product or service',
    9: 'Other Apple product or service'
}

EMOTION_MAPPING = {
    1: 'Positive emotion',
    2: 'Negative emotion',
    3: 'No emotion toward brand or product'
}


def load_model():
    output_model_file = 'src/ml/models/pytorch_roberta_sentiment.bin'
    loaded_model = torch.load(output_model_file)
    output_vocab_file = './'
    # Extract the state dictionary from the loaded model
    state_dict = loaded_model.state_dict()

    # Create an instance of your model and load the state dictionary
    model = RobertaClass(num_classes_sentiment=5, num_classes_product=10)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(output_vocab_file)

    return model, tokenizer

def predict(model, tokenizer, tweet):
    
    inputs = tokenizer.encode_plus(
                tweet,
                None,
                add_special_tokens=True,
                max_length=256,
                pad_to_max_length=True,
                return_token_type_ids=True
    )


    ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

    ids = ids.unsqueeze(0)
    mask = mask.unsqueeze(0)

    # Forward pass through the loaded model
    with torch.no_grad():
        outputs_sentiment, outputs_product = model(ids, mask, token_type_ids)

    # Get predicted classes
    _, predicted_sentiment = torch.max(outputs_sentiment, 1)
    _, predicted_product = torch.max(outputs_product, 1)

    predicted_product = PRODUCT_MAPPING.get(predicted_product.item())
    predicted_sentiment = EMOTION_MAPPING.get(predicted_sentiment.item())
    # Print the results
    # print(f"Predicted Sentiment: {predicted_sentiment}")
    # print(f"Predicted Product: {predicted_product}")

    return predicted_sentiment, predicted_product


if __name__ == "__main__":
    model, tokenizer = load_model()
    df = pd.read_csv("ML Assignment Dataset - Test.csv")
    df[['Emotion', 'Product']] = df['Tweet'].apply(lambda tweet: pd.Series(predict(model, tokenizer, tweet)))

    df.to_csv("answer.csv")