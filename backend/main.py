from fastapi import FastAPI
from backend.model_loader import predict
from backend.db import save_prediction
from backend.schemas import InputText
from backend.db import get_collection
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Fake News Detection API Running"}


@app.post("/predict")
def predict_api(data: InputText):
    try:
        print("📥 Received request")

        # 🔮 Prediction
        result = predict(data.text)

        print("🔮 Prediction done:", result)

        # 💾 Save to DB
        save_prediction(
            text=data.text,
            label=result["label"],
            confidence=result["confidence"],
            source="frontend"
        )

        print("💾 Saved to DB")

        return result

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}
    


@app.get("/history")
def get_history():
    collection = get_collection()

    data = list(collection.find().sort("_id", -1).limit(50))

    # Convert ObjectId to string
    for item in data:
        item["_id"] = str(item["_id"])

    return data


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
