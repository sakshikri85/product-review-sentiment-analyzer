import nltk

PACKAGES = ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab", "vader_lexicon"]

if __name__ == "__main__":
    for pkg in PACKAGES:
        print(f"Downloading NLTK package: {pkg} ...")
        nltk.download(pkg)
    print("\n[OK] All NLTK data downloaded successfully.")
