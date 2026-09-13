"""
report_generator.py
---------------------
Creates ONE clean, easy-to-read HTML report (output/report.html) that
shows all charts + results together with simple explanations.
Just double-click output/report.html to open it in your browser -
no coding knowledge needed to read it.
"""

import os
import base64
import pandas as pd


def _img_to_base64(path):
    """Embed image directly into the HTML so the report is a single
    self-contained file (works even if you move/share just this file)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_report(output_dir="output"):
    csv_path = os.path.join(output_dir, "reviews_with_sentiment.csv")
    df = pd.read_csv(csv_path)

    total = len(df)
    counts = df["vader_sentiment"].value_counts()
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)

    pos_pct = round(positive / total * 100, 1)
    neg_pct = round(negative / total * 100, 1)
    neu_pct = round(neutral / total * 100, 1)

    # a few example reviews for each sentiment, so it feels concrete
    def sample_reviews(label, n=3):
        rows = df[df["vader_sentiment"] == label].head(n)
        return rows["review_text"].tolist()

    pos_examples = sample_reviews("positive")
    neg_examples = sample_reviews("negative")
    neu_examples = sample_reviews("neutral")

    images = {
        "sentiment": _img_to_base64(os.path.join(output_dir, "sentiment_distribution.png")),
        "rating": _img_to_base64(os.path.join(output_dir, "rating_distribution.png")),
        "wc_pos": _img_to_base64(os.path.join(output_dir, "wordcloud_positive.png")),
        "wc_neg": _img_to_base64(os.path.join(output_dir, "wordcloud_negative.png")),
        "confusion": _img_to_base64(os.path.join(output_dir, "confusion_matrix.png")),
    }

    def img_tag(key, alt):
        if images[key]:
            return f'<img src="data:image/png;base64,{images[key]}" alt="{alt}">'
        return f"<p><em>({alt} not found)</em></p>"

    def examples_html(items, css_class):
        if not items:
            return "<p><em>No examples found.</em></p>"
        lis = "".join(f'<li>"{txt}"</li>' for txt in items)
        return f'<ul class="{css_class}">{lis}</ul>'

    html = f"""<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<title>Product Review Sentiment Analysis - Report</title>
<style>
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f4f6f8;
    color: #222;
    margin: 0;
    padding: 0;
  }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 30px 20px 60px;
  }}
  header {{
    background: linear-gradient(135deg, #2c3e50, #4ca1af);
    color: white;
    padding: 40px 20px;
    text-align: center;
  }}
  header h1 {{ margin: 0 0 8px; font-size: 28px; }}
  header p {{ margin: 0; opacity: 0.9; }}

  .summary-cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 30px 0;
  }}
  .card {{
    flex: 1;
    min-width: 140px;
    background: white;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .card .num {{ font-size: 30px; font-weight: bold; }}
  .card .label {{ color: #666; margin-top: 4px; font-size: 14px; }}
  .card.total .num {{ color: #34495e; }}
  .card.positive .num {{ color: #2ecc71; }}
  .card.negative .num {{ color: #e74c3c; }}
  .card.neutral .num {{ color: #f39c12; }}

  section {{
    background: white;
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  section h2 {{
    margin-top: 0;
    color: #2c3e50;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 10px;
  }}
  img {{
    max-width: 100%;
    border-radius: 8px;
    display: block;
    margin: 12px auto;
  }}
  ul.examples {{ padding-left: 20px; line-height: 1.8; }}
  ul.pos li {{ color: #1e8449; }}
  ul.neg li {{ color: #c0392b; }}
  ul.neu li {{ color: #b9770e; }}

  .explain {{
    background: #eef6fb;
    border-left: 4px solid #4ca1af;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 14px 0;
    font-size: 15px;
  }}
  footer {{
    text-align: center;
    color: #888;
    font-size: 13px;
    padding: 20px;
  }}
</style>
</head>
<body>

<header>
  <h1>📊 Product Review Sentiment Analysis</h1>
  <p>Amazon / Flipkart Style Customer Reviews Report</p>
</header>

<div class="container">

  <div class="summary-cards">
    <div class="card total">
      <div class="num">{total}</div>
      <div class="label">Total Reviews</div>
    </div>
    <div class="card positive">
      <div class="num">{positive}</div>
      <div class="label">Positive ({pos_pct}%)</div>
    </div>
    <div class="card negative">
      <div class="num">{negative}</div>
      <div class="label">Negative ({neg_pct}%)</div>
    </div>
    <div class="card neutral">
      <div class="num">{neutral}</div>
      <div class="label">Neutral ({neu_pct}%)</div>
    </div>
  </div>

  <section>
    <h2>1️⃣ Sentiment Distribution</h2>
    <div class="explain">
      Ye chart batata hai ki total reviews mein se kitne customers khush the
      (positive), kitne naraz the (negative), aur kitne ne mixed/plain
      review diya (neutral).
    </div>
    {img_tag('sentiment', 'Sentiment Distribution Chart')}
  </section>

  <section>
    <h2>2️⃣ Star Rating Distribution</h2>
    <div class="explain">
      Ye chart batata hai ki customers ne kitne 1-star, 2-star, 3-star,
      4-star, 5-star reviews diye hain.
    </div>
    {img_tag('rating', 'Rating Distribution Chart')}
  </section>

  <section>
    <h2>3️⃣ Sabse Zyada Use Hue Words - Positive Reviews Mein</h2>
    <div class="explain">
      Jitna bada word dikh raha hai, utni baar wo word positive reviews
      mein use hua hai. Jaise "great", "perfect", "satisfied" etc.
    </div>
    {img_tag('wc_pos', 'Positive Word Cloud')}
    <p><strong>Kuch example positive reviews:</strong></p>
    {examples_html(pos_examples, 'examples pos')}
  </section>

  <section>
    <h2>4️⃣ Sabse Zyada Use Hue Words - Negative Reviews Mein</h2>
    <div class="explain">
      Isi tarah, ye words batate hain customers negative reviews mein
      sabse zyada kya complain karte hain - jaise "broke", "poor",
      "disappointed" etc.
    </div>
    {img_tag('wc_neg', 'Negative Word Cloud')}
    <p><strong>Kuch example negative reviews:</strong></p>
    {examples_html(neg_examples, 'examples neg')}
  </section>

  <section>
    <h2>5️⃣ Model Accuracy Check</h2>
    <div class="explain">
      Humne ek Machine Learning model bhi train kiya jo khud review padh
      kar sentiment predict karna seekhta hai. Ye chart batata hai model
      ne kitne reviews sahi predict kiye aur kitne galat - jitna zyada
      diagonal (top-left se bottom-right) mein number hoga, utna better
      model hai.
    </div>
    {img_tag('confusion', 'Confusion Matrix')}
  </section>

  <section>
    <h2>📁 Baaki Files</h2>
    <div class="explain">
      Full detailed data (har review ka sentiment label ke saath)
      <code>output/reviews_with_sentiment.csv</code> file mein hai -
      Excel ya Google Sheets mein khol kar dekh sakte ho.
    </div>
  </section>

</div>

<footer>
  Generated automatically by report_generator.py
</footer>

</body>
</html>
"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Report generated -> {report_path}")
    print("Ab 'output' folder mein jaake 'report.html' par double-click karo -")
    print("browser mein khul jaayega poora sundar report ke saath.")
    return report_path


if __name__ == "__main__":
    generate_report()
