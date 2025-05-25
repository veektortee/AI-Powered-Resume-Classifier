import os
import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        # Extract text from PDF
        text = extract_text(file_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Extract text from image using OCR
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    else:
        raise ValueError("Unsupported file format. Only PDF, PNG, JPG, and JPEG are allowed.")
    return text

def preprocess(file_path):
    # Extract raw text
    raw_text = extract_text_from_file(file_path)
    # Clean text
    cleaned_text = clean_text(raw_text)
    return cleaned_text

if __name__ == "__main__":
    # Example usage
    file_path = "2.png" 
    print(preprocess(file_path))