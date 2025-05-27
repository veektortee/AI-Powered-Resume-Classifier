import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
from preprocess_extraction import preprocess
import pickle
import os
import tempfile

# Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model('resume_classifier_model.keras', custom_objects={'TFBertModel': TFBertModel})
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_model_and_tokenizer()

# App Title
st.title("Resume Classification App")

# File Upload
uploaded_file = st.file_uploader("Upload a Resume (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file with correct extension
    file_suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    cleaned_text = preprocess(tmp_file_path)
    st.text_area("Extracted Text", cleaned_text, height=200)

    # Tokenize and predict
    sequence_length = 512
    inputs = tokenizer(
        cleaned_text,
        max_length=sequence_length,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )
    preds = model.predict({
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask']
    })
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]

    st.write(f"**Predicted Category:** {pred_class}")
