from fastapi import FastAPI
import os

from backend.model_loader import predict
from backend.db import save_prediction, get_collection
from backend.schemas import InputText

app = FastAPI()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "Fake News Detection API Running"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict_api(data: InputText):
    try:
        print("📥 Received request:", data.text)

        # 🔮 Prediction
        result = predict(data.text)

        print("🔮 Prediction done:", result)

        # 💾 Save to DB (safe guard)
        try:
            save_prediction(
                text=data.text,
                label=result["label"],
                confidence=result["confidence"],
                source="frontend"
            )
            print("💾 Saved to DB")
        except Exception as db_error:
            print("⚠️ DB ERROR (ignored):", db_error)

        return {
            "prediction": result["label"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        print("❌ ERROR in /predict:", e)
        return {"error": str(e)}


# =========================
# HISTORY ENDPOINT
# =========================
@app.get("/history")
def get_history():
    try:
        collection = get_collection()
        data = list(collection.find().sort("_id", -1).limit(50))

        for item in data:
            item["_id"] = str(item["_id"])

        return data

    except Exception as e:
        print("❌ HISTORY ERROR:", e)
        return {"error": str(e)}


# =========================
# RUN SERVER (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
