# Sentiment Analysis on Product Reviews (Amazon/Flipkart)

Ek complete Python NLP project jo product reviews ka sentiment analyze karta
hai — Positive / Negative / Neutral. Ye VS Code mein direct run ho jaayega,
bina kisi error ke (already tested end-to-end).

## Project kya karta hai (Pipeline)

1. **Data load** — `data/product_reviews.csv` (600 realistic Amazon/Flipkart
   style reviews). Agar file nahi milti to khud-ba-khud generate ho jaati hai.
2. **Text Preprocessing** — lowercase, punctuation/URL removal, stopword
   removal, tokenization, lemmatization (NLTK).
3. **Rule-based Sentiment** — NLTK VADER + TextBlob se polarity score nikalta
   hai (positive/negative/neutral).
4. **Machine Learning Model** — TF-IDF vectorizer + Logistic Regression aur
   Naive Bayes classifier train karta hai (rating ko ground-truth label
   maan kar), accuracy/precision/recall report deta hai.
5. **Visualizations** — sentiment distribution (bar + pie chart), rating
   distribution, word clouds (positive & negative reviews) — sab
   `output/` folder mein PNG images ke roop mein save hote hain.
6. **Final Output** — `output/reviews_with_sentiment.csv` mein har review ke
   saath uska sentiment label + score save ho jaata hai.

## Kyun scraping ke bajaye dataset use kiya?

Amazon/Flipkart bots ko block karte hain (captcha, IP ban) aur unki
HTML site structure baar-baar change hoti rehti hai — isliye live scraper
"guaranteed zero error" nahi ho sakta. Isliye main pipeline ek realistic
generated dataset use karta hai jo har baar 100% reliably chalta hai.

Agar aap phir bhi scraping try karna chahte ho, ek **bonus/optional**
script diya gaya hai: `src/scraper_flipkart.py` (best-effort, isse
main pipeline ka koi lena-dena nahi, alag se run hota hai).

Agar aapke paas apna khud ka reviews CSV file hai (columns: `review_text`,
`rating`), to bas usse `data/product_reviews.csv` naam se save kar do —
pipeline usi se chal jaayega.

## VS Code mein Setup & Run (Step-by-step)

### 1. Project folder VS Code mein open karo
`File > Open Folder` → `sentiment_project` folder select karo.

### 2. (Recommended) Virtual environment banao
Terminal open karo (`` Ctrl+` ``) aur ye run karo:

```bash
python -m venv venv
```

Activate karo:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

VS Code niche-right corner mein Python interpreter select karne ka
prompt dega — usme `venv` wala interpreter select kar lena.

### 3. Requirements install karo

```bash
pip install -r requirements.txt
```

### 4. NLTK data download karo (ek baar)

```bash
python setup_nltk.py
```

### 5. Project run karo

```bash
python main.py
```

Bas! Terminal mein pura pipeline step-by-step chalega aur end mein
`output/` folder mein ye files ban jaayengi:

| File | Description |
|---|---|
| `reviews_with_sentiment.csv` | Final dataset with sentiment labels |
| `sentiment_distribution.png` | Bar + pie chart of sentiments |
| `rating_distribution.png` | Star rating distribution |
| `wordcloud_positive.png` | Word cloud of positive reviews |
| `wordcloud_negative.png` | Word cloud of negative reviews |
| `confusion_matrix.png` | ML model confusion matrix |
| `best_model.joblib` | Saved trained ML model |
| `tfidf_vectorizer.joblib` | Saved TF-IDF vectorizer |

## Project Structure

```
sentiment_project/
├── data/
│   └── product_reviews.csv        (auto-generated if missing)
├── src/
│   ├── data_generator.py          # sample dataset generator
│   ├── preprocessing.py           # text cleaning/tokenizing/lemmatizing
│   ├── sentiment_vader.py         # VADER + TextBlob rule-based sentiment
│   ├── ml_model.py                # TF-IDF + Logistic Regression / Naive Bayes
│   ├── visualize.py               # charts & word clouds
│   └── scraper_flipkart.py        # optional/bonus scraper
├── output/                        # generated charts, CSV, model files
├── main.py                        # runs the full pipeline
├── setup_nltk.py                  # one-time NLTK data download
├── requirements.txt
└── README.md
```

## 🔴 NEW: Real-time Interactive Web App

Ab ek proper interactive web app bhi hai jisme aap **live** koi bhi review
type ya paste karke turant result dekh sakte ho — VADER, TextBlob aur ML
model teeno ka result ek saath, plus ek pura dashboard tab.

Chalane ke liye (`python main.py` chalane ke baad):

```bash
streamlit run app.py
```

Ye automatically browser mein khul jaayega (`http://localhost:8501`).
Do tabs milenge:
- **Live Analyzer** — apna review type karo ya kisi real Amazon/Flipkart
  review ko copy-paste karo, "Analyze Sentiment" dabao, turant result.
- **Full Dataset Dashboard** — sare 600 reviews ka interactive summary,
  charts, word clouds, aur filter karke reviews browse karne ka option.

Band karne ke liye terminal mein `Ctrl + C` dabao.

## Apna khud ka CSV use karna ho to

Bas ye 2 columns hone chahiye:
- `review_text` — review ka text
- `rating` — 1 to 5 star rating (number)

File ko `data/product_reviews.csv` naam se save karo aur `python main.py`
run karo.

## Note on accuracy

Is project ka sample dataset template-based hai (realistic bhasha ke saath
generate kiya gaya hai), isliye ML model ki accuracy bahut high (~100%)
dikh sakti hai — real-world scraped data par accuracy generally 75-90%
ke beech hoti hai kyunki real reviews zyada noisy/mixed-sentiment hote
hain. Apna khud ka real dataset daalne par realistic accuracy milegi.

## Troubleshooting

- **`ModuleNotFoundError`** → `pip install -r requirements.txt` phir se run karo, aur check karo ki VS Code sahi Python interpreter (venv wala) use kar raha hai.
- **NLTK `LookupError` / resource not found** → `python setup_nltk.py` chalao.
- **Charts nahi khul rahe** → `output/` folder ke andar dekho, VS Code file explorer se PNG files par click karke directly preview ho jaayengi.
