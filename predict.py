# predict.py
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
from preprocess_extraction import preprocess
import pickle

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model('resume_classifier_model.keras', custom_objects={'TFBertModel': TFBertModel})

# Load label encoder from pickle
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def predict_resume(file_path, sequence_length=512):
    # Preprocess and clean the file text
    cleaned_text = preprocess(file_path)
    
    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        max_length=sequence_length,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )
    
    # Predict
    preds = model.predict({
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask']
    })
    
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
    
    return pred_class

if __name__ == "__main__":
    # Example usage
    file_path = "2.png"  # Replace with actual file path
    prediction = predict_resume(file_path)
    print(f"Predicted category: {prediction}")