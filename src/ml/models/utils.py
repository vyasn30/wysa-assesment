from src.ml.models.RobertaClass import RobertaClass
from transformers import RobertaTokenizer
import torch
from src.constants import PRODUCT_MAPPING, EMOTION_MAPPING

def load_model():
    output_model_file = 'src/ml/models/pytorch_roberta_sentiment.bin'
    loaded_model = torch.load(output_model_file)
    output_vocab_file = 'src/ml/models/vocab/'
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

    with torch.no_grad():
        outputs_sentiment, outputs_product = model(ids, mask, token_type_ids)

    # Get predicted classes
    _, predicted_sentiment = torch.max(outputs_sentiment, 1)
    _, predicted_product = torch.max(outputs_product, 1)

    predicted_product = PRODUCT_MAPPING.get(predicted_product.item())
    predicted_sentiment = EMOTION_MAPPING.get(predicted_sentiment.item())

    return predicted_sentiment, predicted_product




