import uvicorn

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.app.repository.RedisRepo import RedisRepository
from typing import Optional
from src.ml.models.RobertaClass import RobertaClass
from transformers import RobertaTokenizer
from src.constants import PRODUCT_MAPPING, EMOTION_MAPPING
from src.ml.models.utils import load_model, predict

repository = RedisRepository()

class Tweet(BaseModel):
    tweet_id: str
    tweet_text: str
    sentiment: Optional[str] = None
    sentiment_at: Optional[str] = None

app = FastAPI()

model, tokenizer = load_model()

@app.post("/tweets/", response_model=dict)
def insert_tweet(tweet: Tweet):
    sentiment, product = predict(model, tokenizer, tweet.tweet_text)
    repository.insert_tweet(
        tweet.tweet_id, 
        tweet.tweet_text, 
        sentiment, 
        product
    )
    return {'message': 'Tweet inserted successfully'}

@app.get("/tweets/{tweet_id}", response_model=dict)
def get_tweet_by_id(tweet_id: str):
    tweet = repository.get_tweet_by_id(tweet_id)
    if tweet:
        return tweet
    else:
        raise HTTPException(status_code=404, detail='Tweet not found')

@app.get("/tweets/sentiment/{sentiment}", response_model=list)
def get_tweets_by_sentiment(sentiment: str):
    tweets = repository.get_tweets_by_sentiment(sentiment)
    return tweets

@app.put("/tweets/{tweet_id}", response_model=dict)
def update_sentiment_by_id(tweet_id: str, new_sentiment: str):
    success = repository.update_sentiment_by_id(tweet_id, new_sentiment)
    if success:
        return {'message': 'Sentiment updated successfully'}
    else:
        raise HTTPException(status_code=404, detail='Tweet not found')

@app.delete("/tweets/{tweet_id}", response_model=dict)
def delete_tweet_by_id(tweet_id: str):
    deleted = repository.delete_tweet_by_id(tweet_id)
    if deleted:
        return {'message': 'Tweet deleted successfully'}
    else:
        raise HTTPException(status_code=404, detail='Tweet not found')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)