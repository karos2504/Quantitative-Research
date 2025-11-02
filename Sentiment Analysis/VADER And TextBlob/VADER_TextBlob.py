from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER sentiment analyzer
analyser = SentimentIntensityAnalyzer()

# Sample sentences to analyze with VADER
sentences = [
    "This is a good course",
    "This is an awesome course",       # degree modifier
    "The instructor is so cool",
    "The instructor is so cool!!",    # exclamation changes score
    "The instructor is so COOL!!",    # capitalization changes score
    "Machine learning makes me :)",   # emoticons
    "His antics had me ROFL",          # slang
    "The movie SUX"                   # slang
]

print("VADER Sentiment Scores:")
for sentence in sentences:
    score = analyser.polarity_scores(sentence)
    print(f"Sentence: {sentence}")
    print(f"Scores: {score}\n")

# Sample words and sentence for TextBlob sentiment analysis
words = ["His", "remarkable", "work", "ethic", "impressed", "me"]
sentence = "His remarkable work ethic impressed me"

print("TextBlob Sentiment Scores:")
for word in words:
    sentiment = TextBlob(word).sentiment
    print(f"Word: {word}, Sentiment: {sentiment}")

sentence_sentiment = TextBlob(sentence).sentiment
print(f"\nSentence: '{sentence}'")
print(f"Sentiment: {sentence_sentiment}")
