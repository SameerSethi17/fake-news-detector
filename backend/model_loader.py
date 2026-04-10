from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

# 🔥 LOAD FROM HUGGING FACE
MODEL_NAME = "sameersethi/fake-news-detector"

print("MODEL LOADING START")

# Load tokenizer from HF
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
print("TOKENIZER LOADED")

# Load model from HF
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
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
