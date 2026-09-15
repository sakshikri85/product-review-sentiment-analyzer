import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_dataset
from src.preprocessing import preprocess_review
from src.sentiment_vader import vader_sentiment, textblob_sentiment
from src.ml_model import train_and_evaluate, rating_to_label
from src.visualize import (
    plot_sentiment_distribution,
    plot_rating_distribution,
    plot_wordclouds,
)
from src.report_generator import generate_report

DATA_PATH = "data/product_reviews.csv"
OUTPUT_DIR = "output"


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"[INFO] Dataset not found at {path}, generating a sample one...")
        return generate_dataset(n=600, out_path=path)
    print(f"[OK] Loaded dataset -> {path}")
    return pd.read_csv(path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_data()
    print(df.head(3).to_string())
    print(f"\nTotal reviews: {len(df)}")

    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing text")
    print("=" * 60)
    df["clean_review"] = df["review_text"].apply(preprocess_review)
    print(df[["review_text", "clean_review"]].head(3).to_string())

    print("\n" + "=" * 60)
    print("STEP 3: Rule-based sentiment (VADER + TextBlob)")
    print("=" * 60)
    vader_results = df["review_text"].apply(vader_sentiment)
    df["vader_sentiment"] = vader_results.apply(lambda x: x[0])
    df["vader_score"] = vader_results.apply(lambda x: x[1])

    tb_results = df["review_text"].apply(textblob_sentiment)
    df["textblob_sentiment"] = tb_results.apply(lambda x: x[0])
    df["textblob_score"] = tb_results.apply(lambda x: x[1])

    print(df["vader_sentiment"].value_counts())

    print("\n" + "=" * 60)
    print("STEP 4: Ground-truth label from star rating")
    print("=" * 60)
    df["true_sentiment"] = df["rating"].apply(rating_to_label)
    vader_acc = (df["vader_sentiment"] == df["true_sentiment"]).mean()
    print(f"VADER vs rating-based label agreement: {vader_acc:.2%}")

    print("\n" + "=" * 60)
    print("STEP 5: Training ML classifier (TF-IDF + Logistic Regression / NB)")
    print("=" * 60)
    best_name, best_acc, summary = train_and_evaluate(
        df, text_col="clean_review", label_col="true_sentiment",
        output_dir=OUTPUT_DIR,
    )
    print(f"\n>>> Best model: {best_name} (accuracy: {best_acc:.4f})")

    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations")
    print("=" * 60)
    plot_sentiment_distribution(df, "vader_sentiment", OUTPUT_DIR)
    plot_rating_distribution(df, "rating", OUTPUT_DIR)
    plot_wordclouds(df, "clean_review", "vader_sentiment", OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("STEP 7: Saving final labeled dataset")
    print("=" * 60)
    final_path = os.path.join(OUTPUT_DIR, "reviews_with_sentiment.csv")
    df.to_csv(final_path, index=False)
    print(f"[OK] Saved -> {final_path}")

    print("\n" + "=" * 60)
    print("STEP 8: Generating easy-to-read HTML report")
    print("=" * 60)
    generate_report(output_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total reviews analyzed : {len(df)}")
    print(f"VADER sentiment split  :\n{df['vader_sentiment'].value_counts().to_string()}")
    print(f"ML model accuracies    : {summary}")
    print(f"Best ML model          : {best_name} ({best_acc:.2%} accuracy)")
    print(f"\nAll outputs saved in ./{OUTPUT_DIR}/ :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("   -", f)
    print("\nDone! Pipeline finished with no errors.")


if __name__ == "__main__":
    main()
