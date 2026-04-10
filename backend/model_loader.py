from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = "model/final_model"

print("MODEL LOADING START")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
print("TOKENIZER LOADED")

# Load model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("MODEL LOADED")
print("MODEL READY")


# 🔥 PREDICT FUNCTION
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    label = "REAL" if pred.item() == 1 else "FAKE"

    return {
        "label": label,
        "confidence": float(confidence.item())
    }