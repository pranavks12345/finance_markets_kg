"""
Pull last month of company news from Finnhub in weekly chunks
+ Deduplicate results
"""

import requests
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Configuration
API_KEY = os.getenv("FINNHUB_API_KEY")
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
DAYS_BACK = 90   # last month
CHUNK_SIZE = 7   # days per API call

SAVE_DIR = Path("data/raw/news")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def pull_news_for_ticker(ticker: str):
    """Pull last month of news for a ticker in weekly chunks"""
    print(f"\n📰 Getting news for {ticker}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)

    # Break into weekly chunks
    all_articles = []
    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_SIZE), end_date)

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "token": API_KEY,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"  📦 {len(data)} articles {chunk_start.date()} → {chunk_end.date()}")
            all_articles.extend(data)
        else:
            print(f"  ❌ Error {response.status_code} for {chunk_start.date()} → {chunk_end.date()}")

        # be nice to API
        time.sleep(1)
        chunk_start = chunk_end

    # Deduplicate
    unique_articles = {}
    for art in all_articles:
        key = art.get("id") or f"{art.get('headline','')}_{art.get('datetime','')}"
        unique_articles[key] = art

    deduped_articles = list(unique_articles.values())

    # Save
    filename = SAVE_DIR / f"{ticker}_news.json"
    with open(filename, "w") as f:
        json.dump(deduped_articles, f, indent=2)

    print(f"✅ Saved {len(deduped_articles)} unique articles to {filename}")
    return deduped_articles


def main():
    print(f"🚀 Pulling last {DAYS_BACK} days of news (weekly chunks)...")
    print(f"📊 Tickers: {', '.join(TICKERS)}")

    total_articles = 0
    for ticker in TICKERS:
        articles = pull_news_for_ticker(ticker)
        total_articles += len(articles)
        print()

    print(f"🎉 Done! Total unique articles collected: {total_articles}")


if __name__ == "__main__":
    main()

