"""
scraper_flipkart.py
---------------------
OPTIONAL / BONUS module - best-effort scraper for Flipkart product review
pages using requests + BeautifulSoup.

IMPORTANT (please read before using):
  - E-commerce sites change their HTML structure often and actively block
    scrapers (captchas, rate limits, login walls). This script is a
    starting point, NOT a guaranteed-working tool.
  - Scraping may violate a website's Terms of Service - use responsibly,
    for personal/educational purposes only, and respect robots.txt.
  - This module is NOT used by main.py's default pipeline (which relies
    on the bundled/generated dataset) so the core project always runs
    with zero errors even without internet access or if the site
    structure changes.

Usage:
    python src/scraper_flipkart.py "<flipkart_product_review_url>"
"""

import sys
import time
import csv

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def scrape_flipkart_reviews(url: str, max_pages: int = 3, delay: float = 1.5):
    """Attempts to scrape review text + rating from a Flipkart product page.
    Returns a list of dicts: [{"rating": int|None, "review_text": str}, ...]
    NOTE: Flipkart's CSS class names change frequently; you may need to
    inspect the page (right-click -> Inspect) and update the selectors
    below (`review_div_class`, `rating_class`) if this stops working.
    """
    all_reviews = []
    review_div_class = "ZmyHeo"     # review text container (may need updating)
    rating_class = "XQDdHH"         # rating badge (may need updating)

    for page in range(1, max_pages + 1):
        page_url = f"{url}&page={page}" if "?" in url else f"{url}?page={page}"
        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[WARN] Could not fetch page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        review_blocks = soup.find_all("div", class_=review_div_class)

        if not review_blocks:
            print(f"[INFO] No reviews found on page {page} "
                  f"(site structure may have changed, or page ended).")
            break

        rating_tags = soup.find_all("div", class_=rating_class)

        for i, block in enumerate(review_blocks):
            text = block.get_text(" ", strip=True)
            rating = None
            if i < len(rating_tags):
                try:
                    rating = int(rating_tags[i].get_text(strip=True))
                except ValueError:
                    rating = None
            all_reviews.append({"rating": rating, "review_text": text})

        time.sleep(delay)  # be polite, avoid hammering the server

    return all_reviews


def save_to_csv(reviews, out_path="data/scraped_reviews.csv"):
    if not reviews:
        print("[WARN] No reviews scraped, nothing to save.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rating", "review_text"])
        writer.writeheader()
        writer.writerows(reviews)
    print(f"[OK] Saved {len(reviews)} scraped reviews -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scraper_flipkart.py \"<flipkart_review_url>\"")
        sys.exit(0)

    target_url = sys.argv[1]
    print("[INFO] Attempting to scrape... this may fail if Flipkart's "
          "page structure has changed or if requests are blocked.")
    reviews = scrape_flipkart_reviews(target_url)
    save_to_csv(reviews)
