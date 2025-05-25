# app.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from preprocessing_utils import preprocess_resume, label_encoder

app = FastAPI()

# Load model
model = tf.keras.models.load_model("resume_classifier_model.keras")

@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    contents = await file.read()
    inputs = preprocess_resume(contents, file.filename.split(".")[-1])

    predictions = model(inputs).numpy()
    pred_class = np.argmax(predictions, axis=1)
    label = label_encoder.inverse_transform(pred_class)[0]

    return {"predicted_label": label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)