import sys
import os
import time
import requests

# ✅ Ensure backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_loader import predict
from backend.db import save_prediction


# 🌐 News API
NEWS_API_URL = "https://newsapi.org/v2/top-headlines?country=us&pageSize=20&apiKey=f0be9bc0d176494fb840c3af2eb3644e"


# -------------------- LOGGER --------------------
def log(msg):
    print(f"[AUTO] {msg}")


# -------------------- FETCH NEWS --------------------
def fetch_news():
    try:
        response = requests.get(NEWS_API_URL, timeout=10)

        # ✅ API validation
        if response.status_code != 200:
            log(f"❌ API failed with status code: {response.status_code}")
            return []

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            log("⚠️ Empty API response — using fallback dataset")

            return [
                "ISRO successfully launches new communication satellite",
                "Scientists discover breakthrough in quantum computing chip",
                "Fake claim: humans can teleport using sound energy waves",
                "India announces new digital economy reform policy",
                "Researchers develop AI system for medical diagnosis improvement"
            ]

        news_list = []

        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""

            text = f"{title}. {description}".strip()

            if text and text != ".":
                news_list.append(text)

        log(f"Fetched {len(news_list)} articles")
        return news_list

    except Exception as e:
        log(f"❌ Fetch error: {e}")

        return [
            "ISRO successfully launches satellite mission",
            "Scientists discover breakthrough in AI technology",
            "Fake news example: humans can live without oxygen",
            "India launches new space exploration program"
        ]


# -------------------- PIPELINE --------------------
def run_pipeline():
    log("🚀 Pipeline started")

    news_list = fetch_news()

    if not news_list:
        log("⚠️ No data to process")
        return

    for text in news_list:
        try:
            log(f"📰 News: {text}")

            # ---------------- MODEL PREDICTION ----------------
            result = predict(text)

            # ✅ SAFE OUTPUT HANDLING (VERY IMPORTANT)
            label = result.get("label") or result.get("prediction") or "UNKNOWN"
            confidence = result.get("confidence") or result.get("score") or 0.0

            log(f"➡ Prediction: {label} ({confidence})")

            # ---------------- SAVE TO DB ----------------
            save_prediction(
                text=text,
                label=label,
                confidence=confidence,
                source="NewsAPI"
            )

        except Exception as e:
            log(f"❌ Processing error: {e}")


# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    log("Automation Service Started")

    while True:
        run_pipeline()

        log("⏳ Sleeping for 10 minutes...\n")
        time.sleep(600)