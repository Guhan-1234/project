import tweepy
from transformers import pipeline
import os

# ========================
# 1. Twitter API Setup
# ========================
# Replace with your Twitter API credentials
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Tweepy
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# ========================
# 2. Fetch Tweets
# ========================
def fetch_tweets(keyword, count=10):
    tweets = api.search_tweets(q=keyword, lang="en", count=count, tweet_mode='extended')
    return [tweet.full_text for tweet in tweets]

# ========================
# 3. Emotion Detection
# ========================
# Load the emotion classifier pipeline from Hugging Face
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def analyze_emotions(texts):
    results = []
    for text in texts:
        emotions = emotion_classifier(text)[0]
        emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)
        results.append({
            "text": text,
            "top_emotion": emotions_sorted[0]['label'],
            "score": emotions_sorted[0]['score'],
            "all_emotions": emotions_sorted
        })
    return results

# ========================
# 4. Run Analysis
# ========================
if __name__ == "__main__":
    keyword = "climate change"  # Replace with any topic
    tweets = fetch_tweets(keyword, count=5)
    emotion_results = analyze_emotions(tweets)

    for idx, result in enumerate(emotion_results):
        print(f"\nTweet #{idx+1}:")
        print(result["text"])
        print(f"Top Emotion: {result['top_emotion']} ({result['score']:.2f})")
        print("All Emotions:")
        for emotion in result["all_emotions"]:
            print(f"  {emotion['label']}: {emotion['score']:.2f}")
