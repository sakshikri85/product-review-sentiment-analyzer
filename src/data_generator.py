"""
data_generator.py
------------------
Generates a realistic e-commerce (Amazon/Flipkart style) product review
dataset and saves it as data/product_reviews.csv

Why a generated dataset instead of live scraping?
  1. Amazon/Flipkart aggressively block bots/scrapers (captchas, IP bans),
     so a scraper breaks often and cannot guarantee "zero errors".
  2. This keeps the whole project fully self-contained and runnable
     offline in VS Code without depending on a live website's HTML
     structure (which changes constantly and breaks scrapers).
A best-effort scraper (src/scraper_flipkart.py) is still included as a
bonus module if you want to try pulling real reviews.
"""

import random
import pandas as pd
import os

random.seed(42)

PRODUCTS = [
    "Wireless Bluetooth Earbuds", "Smartphone Back Cover", "Laptop Backpack",
    "Smartwatch", "Bluetooth Speaker", "USB-C Charging Cable", "Running Shoes",
    "Kitchen Mixer Grinder", "LED Desk Lamp", "Air Fryer", "Cotton T-Shirt",
    "Power Bank 10000mAh", "Wireless Mouse", "Gaming Keyboard", "Office Chair"
]

POSITIVE_TEMPLATES = [
    "Absolutely loved this {p}! Works perfectly and arrived on time.",
    "Excellent quality {p}, totally worth the price. Highly recommended!",
    "Best {p} I have bought online. Great build quality and fast delivery.",
    "Superb product! The {p} exceeded my expectations, five stars.",
    "Very happy with this {p}, will definitely buy again from this brand.",
    "Amazing value for money. The {p} looks premium and works great.",
    "This {p} is fantastic, exactly as described. Very satisfied customer.",
    "Great purchase! The {p} is durable and easy to use.",
    "Impressed by the quality of the {p}. Packaging was also very good.",
    "Perfect {p} for daily use, no issues at all after a month.",
]

NEGATIVE_TEMPLATES = [
    "Very disappointed with this {p}. Stopped working within a week.",
    "Waste of money, the {p} quality is really poor.",
    "The {p} arrived damaged and customer service was unhelpful.",
    "Not worth the price, this {p} broke on the second day.",
    "Terrible experience, the {p} does not match the product description.",
    "Poor build quality, the {p} feels cheap and flimsy.",
    "I regret buying this {p}, it stopped functioning after few uses.",
    "Delivery was late and the {p} was defective on arrival.",
    "Really bad {p}, would not recommend to anyone.",
    "The {p} is a total scam, completely different from the pictures.",
]

NEUTRAL_TEMPLATES = [
    "The {p} is okay, does the job but nothing special.",
    "Average {p}, quality is decent for the price range.",
    "It's a fine {p}, met basic expectations, could be better.",
    "The {p} works fine, though delivery took longer than expected.",
    "Decent {p} for the price, some minor issues but usable.",
    "The {p} is average, packaging could have been better.",
    "Okay-ish product, the {p} performs as expected, nothing more.",
    "Reasonable {p}, matches the description mostly.",
]

REVIEWERS = ["Amit", "Priya", "Rahul", "Sneha", "Vikram", "Anjali", "Rohit",
             "Kavya", "Arjun", "Neha", "Suresh", "Pooja", "Karan", "Divya",
             "Manoj", "Riya", "Sanjay", "Isha", "Deepak", "Nisha"]


def _make_row(review_id):
    product = random.choice(PRODUCTS)
    sentiment_bucket = random.choices(
        ["positive", "negative", "neutral"], weights=[0.55, 0.30, 0.15], k=1
    )[0]

    if sentiment_bucket == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
        rating = random.choice([4, 5, 5, 5])
    elif sentiment_bucket == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
        rating = random.choice([1, 1, 2, 2])
    else:
        template = random.choice(NEUTRAL_TEMPLATES)
        rating = 3

    review_text = template.format(p=product)

    # add a little natural noise/variation sometimes
    if random.random() < 0.25:
        review_text += " " + random.choice([
            "Shipping was quick.", "Packaging was neat.",
            "Would consider buying again.", "Not sure if I will repurchase.",
            "Customer support responded quickly.", "",
        ])

    return {
        "review_id": review_id,
        "product_name": product,
        "reviewer_name": random.choice(REVIEWERS),
        "rating": rating,
        "review_text": review_text.strip(),
        "platform": random.choice(["Amazon", "Flipkart"]),
    }


def generate_dataset(n=600, out_path="data/product_reviews.csv"):
    rows = [_make_row(i + 1) for i in range(n)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {len(df)} reviews -> {out_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
