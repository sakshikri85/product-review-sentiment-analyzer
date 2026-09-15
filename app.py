import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_review
from src.sentiment_vader import vader_sentiment, textblob_sentiment
from src.ml_model import rating_to_label, train_and_evaluate
from src.data_generator import generate_dataset

OUTPUT_DIR = "output"
DATA_PATH = "data/product_reviews.csv"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
VEC_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")

st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
)

# ---------- Helper: load or train the ML model ----------
@st.cache_resource(show_spinner="Model taiyaar ho raha hai (pehli baar thoda time lagega)...")
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        return model, vectorizer

    # Fallback: train fresh if not already trained
    if not os.path.exists(DATA_PATH):
        generate_dataset(n=600, out_path=DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["clean_review"] = df["review_text"].apply(preprocess_review)
    df["true_sentiment"] = df["rating"].apply(rating_to_label)
    train_and_evaluate(df, text_col="clean_review", label_col="true_sentiment",
                        output_dir=OUTPUT_DIR)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


@st.cache_data(show_spinner="Dataset load ho raha hai...")
def load_full_dataset():
    result_path = os.path.join(OUTPUT_DIR, "reviews_with_sentiment.csv")
    if os.path.exists(result_path):
        return pd.read_csv(result_path)
    return None


SENTIMENT_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}


def render_result_card(label, method_name, extra=""):
    color = SENTIMENT_COLOR.get(label, "#3498db")
    emoji = SENTIMENT_EMOJI.get(label, "")
    st.markdown(
        f"""
        <div style="background:{color}22; border-left:6px solid {color};
                    border-radius:8px; padding:14px 18px; margin-bottom:10px;">
            <div style="font-size:14px; color:#555;">{method_name}</div>
            <div style="font-size:22px; font-weight:bold; color:{color};">
                {emoji} {label.upper()}
            </div>
            <div style="font-size:13px; color:#777;">{extra}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================= UI =================
st.title("📊 Product Review Sentiment Analyzer")
st.caption("Real-time NLP sentiment analysis — Amazon/Flipkart style product reviews")

tab1, tab2 = st.tabs(["🔴 Live Analyzer (type/paste any review)", "📈 Full Dataset Dashboard"])

# ---------------- TAB 1: LIVE ANALYZER ----------------
with tab1:
    st.subheader("Apna review type karo ya kahin se copy-paste karo")
    st.write("Apna khud ka sentence likho, ya kisi bhi real Amazon/Flipkart review "
             "ko copy karke yahan paste karo — turant result dikhega.")

    example_reviews = {
        "-- Select an example --": "",
        "😊 Positive example": "This product exceeded my expectations! Great quality and fast delivery.",
        "😠 Negative example": "Worst purchase ever. It broke after two days and customer service didn't help.",
        "😐 Neutral example": "It's an okay product, does the job but nothing extraordinary.",
    }
    choice = st.selectbox("Ya ek example try karo:", list(example_reviews.keys()))

    user_review = st.text_area(
        "Review yahan likho:",
        value=example_reviews[choice] if choice != "-- Select an example --" else "",
        height=120,
        placeholder="e.g. Amazing sound quality, battery life bhi bahut acchi hai, highly recommend!",
    )

    analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary")

    if analyze_clicked:
        if not user_review.strip():
            st.warning("Pehle koi review likho ya paste karo!")
        else:
            with st.spinner("Analyzing..."):
                model, vectorizer = load_or_train_model()

                v_label, v_score = vader_sentiment(user_review)
                tb_label, tb_score = textblob_sentiment(user_review)

                cleaned = preprocess_review(user_review)
                vec = vectorizer.transform([cleaned])
                ml_label = model.predict(vec)[0]

            st.markdown("### Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                render_result_card(v_label, "VADER (rule-based)",
                                    f"compound score: {v_score:.3f}")
            with col2:
                render_result_card(tb_label, "TextBlob (rule-based)",
                                    f"polarity score: {tb_score:.3f}")
            with col3:
                render_result_card(ml_label, "ML Model (Logistic Regression)",
                                    "trained on 600 reviews")

            labels = [v_label, tb_label, ml_label]
            final = max(set(labels), key=labels.count)
            st.markdown(
                f"""
                <div style="text-align:center; padding:16px; margin-top:10px;
                            background:{SENTIMENT_COLOR[final]}; border-radius:10px;">
                    <span style="font-size:26px; color:white; font-weight:bold;">
                        Overall Verdict: {SENTIMENT_EMOJI[final]} {final.upper()}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Cleaned/preprocessed text (jo model ko diya gaya)"):
                st.code(cleaned if cleaned else "(empty after cleaning)")

# ---------------- TAB 2: FULL DASHBOARD ----------------
with tab2:
    df = load_full_dataset()
    if df is None:
        st.warning(
            "Pehle terminal mein `python main.py` chalao taaki poora dataset "
            "process ho aur ye dashboard bhar sake."
        )
    else:
        total = len(df)
        counts = df["vader_sentiment"].value_counts()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", total)
        c2.metric("😊 Positive", int(counts.get("positive", 0)),
                   f"{counts.get('positive', 0)/total:.1%}")
        c3.metric("😠 Negative", int(counts.get("negative", 0)),
                   f"{counts.get('negative', 0)/total:.1%}")
        c4.metric("😐 Neutral", int(counts.get("neutral", 0)),
                   f"{counts.get('neutral', 0)/total:.1%}")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Sentiment Distribution")
            st.bar_chart(counts)
        with col_b:
            st.subheader("Rating Distribution")
            st.bar_chart(df["rating"].value_counts().sort_index())

        st.markdown("---")
        st.subheader("Charts (generated by main.py)")
        img_col1, img_col2 = st.columns(2)
        wc_pos = os.path.join(OUTPUT_DIR, "wordcloud_positive.png")
        wc_neg = os.path.join(OUTPUT_DIR, "wordcloud_negative.png")
        if os.path.exists(wc_pos):
            img_col1.image(wc_pos, caption="Positive Reviews Word Cloud")
        if os.path.exists(wc_neg):
            img_col2.image(wc_neg, caption="Negative Reviews Word Cloud")

        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.subheader("ML Model Confusion Matrix")
            st.image(cm_path, width=500)

        st.markdown("---")
        st.subheader("Browse reviews")
        sentiment_filter = st.selectbox(
            "Filter by sentiment:", ["all", "positive", "negative", "neutral"]
        )
        show_df = df if sentiment_filter == "all" else df[df["vader_sentiment"] == sentiment_filter]
        st.dataframe(
            show_df[["product_name", "rating", "review_text", "vader_sentiment"]].head(50),
            width="stretch",
        )
