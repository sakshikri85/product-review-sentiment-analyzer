"""
setup_nltk.py
--------------
Run this ONCE after installing requirements.txt to download the NLTK
language data needed by this project (stopwords, tokenizer, lemmatizer,
VADER sentiment lexicon).

    python setup_nltk.py
"""

import nltk

PACKAGES = ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab", "vader_lexicon"]

if __name__ == "__main__":
    for pkg in PACKAGES:
        print(f"Downloading NLTK package: {pkg} ...")
        nltk.download(pkg)
    print("\n[OK] All NLTK data downloaded successfully.")
