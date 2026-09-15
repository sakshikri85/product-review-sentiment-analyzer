import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """Make sure required NLTK corpora are available; download if missing."""
    packages = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for pkg, path in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# keep a few negation words - they matter a LOT for sentiment
NEGATIONS = {"not", "no", "nor", "never", "none"}
STOPWORDS = STOPWORDS - NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase + strip URLs/HTML/punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str):
    """Tokenize, remove stopwords, lemmatize. Returns list of tokens."""
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess_review(text: str) -> str:
    """Full pipeline: clean -> tokenize -> lemmatize -> join back to string."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "This product is NOT good!! Delivery was late :( Visit http://example.com"
    print("Original :", sample)
    print("Cleaned  :", preprocess_review(sample))
