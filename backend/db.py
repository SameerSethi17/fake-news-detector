from pymongo import MongoClient
from datetime import datetime


import os
from pymongo import MongoClient

MONGO_URI = os.getenv("mongodb+srv://sameersethi:Csio%4012345@cluster0.04gjeho.mongodb.net/?retryWrites=true&w=majority")

client = MongoClient(MONGO_URI)
db = client["fake_news_db"]
collection = db["news"]


def save_prediction(text, label, confidence, source="manual"):
    try:
        record = {
            "text": text,
            "label": label,
            "confidence": float(confidence),
            "source": source,
            "timestamp": datetime.utcnow()
        }

        collection.insert_one(record)
        print("✅ Saved to DB")

    except Exception as e:
        print("❌ DB insert failed:", e)


def get_collection():
    return collection
    


