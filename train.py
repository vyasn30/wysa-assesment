from transformers import RobertaTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from src.ml.data.SentimentData import SentimentData
from src.ml.models.RobertaClass import RobertaClass
import torch

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct



def train_model(epoch):
    tr_loss = 0
    n_correct_sentiment = 0
    n_correct_emotion = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for batch_idx, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets_sentiment = data['sentiment_targets'].to(device, dtype=torch.long)
        targets_emotion = data['emotion_targets'].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs_sentiment, outputs_emotion = model(ids, mask, token_type_ids)

        loss_sentiment = loss_function(outputs_sentiment, targets_sentiment)
        loss_emotion = loss_function(outputs_emotion, targets_emotion)

        # Combine the losses, for example, add them
        loss = loss_sentiment + loss_emotion
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()

        _, predicted_sentiment = torch.max(outputs_sentiment.data, 1)
        _, predicted_emotion = torch.max(outputs_emotion.data, 1)

        n_correct_sentiment += (predicted_sentiment == targets_sentiment).sum().item()
        n_correct_emotion += (predicted_emotion == targets_emotion).sum().item()

        nb_tr_steps += 1
        nb_tr_examples += targets_sentiment.size(0)

        if batch_idx % 500 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_sentiment_step = (n_correct_sentiment * 100) / nb_tr_examples
            accu_emotion_step = (n_correct_emotion * 100) / nb_tr_examples

            print(f"Epoch [{epoch + 1}/{EPOCHS}] - Batch [{batch_idx}/{len(training_loader)}]")
            print(f"  Loss: {loss_step:.4f}")
            print(f"  Sentiment Accuracy: {accu_sentiment_step:.2f}%")
            print(f"  Emotion Accuracy: {accu_emotion_step:.2f}%")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu_sentiment = (n_correct_sentiment * 100) / nb_tr_examples
    epoch_accu_emotion = (n_correct_emotion * 100) / nb_tr_examples

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}] - Summary:")
    print(f"  Total Loss: {epoch_loss:.4f}")
    print(f"  Total Sentiment Accuracy: {epoch_accu_sentiment:.2f}%")
    print(f"  Total Emotion Accuracy: {epoch_accu_emotion:.2f}%")
    print("--------------------------------------------------------")

    return

if __name__ == "__main__":
    df = pd.read_csv(".data/train.csv")
    print(df.head())


    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    train_size = 0.8
    train_data=df.sample(frac=train_size,random_state=200)
    test_data=df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = SentimentData(train_data, tokenizer, MAX_LEN)
    testing_set = SentimentData(test_data, tokenizer, MAX_LEN) 
    
    training_loader = DataLoader(training_set, **train_params)

    print(training_loader)

    model = RobertaClass()
    device = "cpu"
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    EPOCHS = 1
    for epoch in range(EPOCHS):
        train_model(epoch) 

    output_model_file = 'pytorch_roberta_sentiment.bin'
    output_vocab_file = './'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)