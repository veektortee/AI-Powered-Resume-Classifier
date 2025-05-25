#!/bin/bash

echo "🚀 Starting Resume Classifier Training..."

# Step 1: Train the model
python train.py

echo "✅ Training completed."

# Step 2: Load latest best model and evaluate
echo "🚀 Evaluating the latest saved best model..."
python -c "
import tensorflow as tf
from train import load_latest_model, test_ds

model = load_latest_model()
loss, accuracy = model.evaluate(test_ds)
print(f'Test Accuracy: {accuracy:.4f}')
"

echo "✅ Evaluation completed!"