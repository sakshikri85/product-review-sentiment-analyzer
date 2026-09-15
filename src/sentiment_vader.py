import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_analyzer = SentimentIntensityAnalyzer()


def vader_sentiment(text: str):
    """Returns (label, compound_score) using VADER's compound score."""
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, compound


def textblob_sentiment(text: str):
    """Returns (label, polarity) using TextBlob polarity score."""
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return label, polarity


if __name__ == "__main__":
    samples = [
        "I absolutely loved this product, works great!",
        "Terrible quality, broke in two days.",
        "It's okay, does the job.",
    ]
    for s in samples:
        print(s, "->", vader_sentiment(s), textblob_sentiment(s))
