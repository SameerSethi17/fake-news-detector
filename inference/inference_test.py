

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = r"D:\fake-news-detector\model\model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

text = "India has launched a new satellite for communication improvements."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=1)

print("Probabilities:", probs)