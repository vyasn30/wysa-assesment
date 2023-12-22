import redis
import json

class RedisRepository:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def insert_tweet(self, tweet_id, tweet_text, sentiment, sentiment_at):
        tweet = {
            'tweet_id': tweet_id,
            'tweet_text': tweet_text,
            'sentiment': sentiment,
            'sentiment_at': sentiment_at
        }
        serialized_tweet = json.dumps(tweet)
        self.redis_client.hset('tweets', tweet_id, serialized_tweet)

    def get_tweet_by_id(self, tweet_id):
        serialized_tweet = self.redis_client.hget('tweets', tweet_id)
        if serialized_tweet:
            return json.loads(serialized_tweet)
        else:
            return None

    def get_tweets_by_sentiment(self, sentiment):
        tweets = []
        for tweet_id, serialized_tweet in self.redis_client.hscan_iter('tweets'):
            tweet = json.loads(serialized_tweet)
            if tweet['sentiment'] == sentiment:
                tweets.append(tweet)
        return tweets

    def update_sentiment_by_id(self, tweet_id, new_sentiment):
        serialized_tweet = self.redis_client.hget('tweets', tweet_id)
        if serialized_tweet:
            tweet = json.loads(serialized_tweet)
            tweet['sentiment'] = new_sentiment
            updated_serialized_tweet = json.dumps(tweet)
            self.redis_client.hset('tweets', tweet_id, updated_serialized_tweet)
            return True
        else:
            return False

    def delete_tweet_by_id(self, tweet_id):
        deleted_count = self.redis_client.hdel('tweets', tweet_id)
        return deleted_count > 0

