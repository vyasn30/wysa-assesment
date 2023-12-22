from torch.utils.data import Dataset
import torch

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.tweet_text
        self.max_len = max_len
        self.sentiment_targets = self.data.sentiment 
        self.product_targets = self.data.emotion_at
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())


        # Use RobertaTokenizer to tokenize the tweet_text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        # Covert it to torch.tensors. Wasted 2 hours. The model unpacks these
        # and it needs attributes for shapes and lens and simple numpy arrs 
        # aren't compatible
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'sentiment_targets': torch.tensor(self.sentiment_targets[index], dtype=torch.float),
            'emotion_targets': torch.tensor(self.product_targets[index], dtype=torch.long)
        }