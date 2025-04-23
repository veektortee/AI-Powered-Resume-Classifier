import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib

# Load cleaned dataset
CSV_PATH = "resume_dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Please ensure the file exists.")

df = pd.read_csv(CSV_PATH)

# Check for required columns
if 'label' not in df.columns or 'text' not in df.columns:
    raise KeyError("The dataset must contain 'label' and 'text' columns.")

# Encode labels
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values,
    df['label_enc'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label_enc']
)

# Load BERT encoder from TF Hub
BERT_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
bert_layer = hub.KerasLayer(BERT_MODEL_URL, trainable=False)

# Build model using Functional API
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
embedding = tf.keras.layers.Lambda(lambda x: bert_layer(x))(input_text)
x = tf.keras.layers.Dense(256, activation='relu')(embedding)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

model = tf.keras.Model(inputs=input_text, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model + label encoder
os.makedirs("models", exist_ok=True)
#model.save("models/resume_bert_model")
joblib.dump(label_encoder, "models/label_encoder.pkl")