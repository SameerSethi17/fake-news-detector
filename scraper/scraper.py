import feedparser
from newspaper import Article
from datetime import datetime
import time
import re
import sys

# DB
sys.path.append('../database')
from db import get_collection

collection = get_collection()


# 🔹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# 🔹 Check duplicate
def is_duplicate(url):
    return collection.find_one({"url": url}) is not None


# 🔹 RSS scraper
def scrape_rss(feed_url, source_name):
    print(f"\n🔍 Scraping {source_name}...")

    feed = feedparser.parse(feed_url)

    count = 0

    for entry in feed.entries[:10]:  # limit
        try:
            url = entry.link

            if is_duplicate(url):
                print("⚠️ Duplicate skipped")
                continue

            article = Article(url)
            article.download()
            article.parse()

            if len(article.text.strip()) < 200:
                continue

            data = {
                "title": article.title,
                "content": article.text,
                "cleaned_text": clean_text(article.text),
                "source": source_name,
                "date": datetime.now(),
                "url": url,
                "predicted": False
            }

            collection.insert_one(data)
            print("✅ Saved:", article.title)

            count += 1
            time.sleep(1)

        except Exception:
            print("⚠️ Skipped:", url)

    print(f"✅ {count} new articles saved from {source_name}")


# 🔹 Run ALL sources
if __name__ == "__main__":
    sources = [
        ("http://feeds.bbci.co.uk/news/rss.xml", "BBC"),
        ("http://rss.cnn.com/rss/edition.rss", "CNN"),
        ("https://www.theguardian.com/world/rss", "Guardian"),
        ("https://www.aljazeera.com/xml/rss/all.xml", "Al Jazeera"),
        ("https://www.thehindu.com/news/national/feeder/default.rss", "The Hindu"),
    ]

    for feed_url, name in sources:
        scrape_rss(feed_url, name)