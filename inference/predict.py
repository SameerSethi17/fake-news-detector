import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_loader import predict

print("MODEL LOADING START")

# 🔥 TEST PREDICTION
result = predict("Scientists discovered a new planet similar to Earth")

print("RESULT:", result)