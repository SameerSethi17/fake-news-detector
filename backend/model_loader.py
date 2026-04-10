from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "sameersethi/fake-news-detector"

device = torch.device("cpu")

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        print("MODEL LOADING START")

        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        print("TOKENIZER LOADED")

        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

        print("MODEL LOADED")
        print("MODEL READY")


def predict(text):
    load_model()  # 🔥 lazy loading

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    label = "REAL" if pred.item() == 1 else "FAKE"

    return {
        "label": label,
        "confidence": float(confidence.item())
    }
