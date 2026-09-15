import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def rating_to_label(rating: int) -> str:
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def train_and_evaluate(df: pd.DataFrame, text_col="clean_review",
                        label_col="true_sentiment", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    X = df[text_col].fillna("")
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    best_model_name, best_model, best_acc = None, None, -1

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        results[name] = {"accuracy": acc, "report": report, "preds": preds}

        print(f"\n{'=' * 50}\n{name} -> Accuracy: {acc:.4f}\n{'=' * 50}")
        print(report)

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    # Confusion matrix for the best model
    best_preds = results[best_model_name]["preds"]
    labels_order = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels_order)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix -> {cm_path}")

    # persist model + vectorizer for reuse
    joblib.dump(best_model, os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    summary = {name: r["accuracy"] for name, r in results.items()}
    return best_model_name, best_acc, summary
