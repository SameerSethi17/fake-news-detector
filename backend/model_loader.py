import os
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_NAME = "sameersethi/fake-news-detector"
device = torch.device("cpu")

print("🚀 LOADING MODEL AT STARTUP (DO NOT TOUCH REQUEST TIME)")

# FORCE FAST LOAD SETTINGS
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True
)

model.to(device)
model.eval()

print("✅ MODEL READY - SERVER SAFE")


def predict(text):
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
