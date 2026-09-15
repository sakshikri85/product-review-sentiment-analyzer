import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f1c40f"}


def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str,
                                 output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    counts = df[sentiment_col].value_counts()
    colors = [COLORS.get(k, "#3498db") for k in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Sentiment Distribution (Bar Chart)")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Reviews")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.5, str(v), ha="center")

    axes[1].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Sentiment Distribution (Pie Chart)")

    plt.tight_layout()
    path = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Saved -> {path}")


def plot_rating_distribution(df: pd.DataFrame, rating_col="rating",
                              output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    counts = df[rating_col].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#3498db")
    plt.title("Star Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    path = os.path.join(output_dir, "rating_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Saved -> {path}")


def plot_wordclouds(df: pd.DataFrame, text_col: str, sentiment_col: str,
                     output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    for sentiment in ["positive", "negative"]:
        subset = df[df[sentiment_col] == sentiment][text_col]
        text_blob = " ".join(subset.astype(str).tolist()).strip()

        if not text_blob:
            continue

        wc = WordCloud(width=900, height=500, background_color="white",
                        colormap="Greens" if sentiment == "positive" else "Reds",
                        max_words=100).generate(text_blob)

        plt.figure(figsize=(9, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {sentiment.capitalize()} Reviews")
        plt.tight_layout()
        path = os.path.join(output_dir, f"wordcloud_{sentiment}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[OK] Saved -> {path}")
