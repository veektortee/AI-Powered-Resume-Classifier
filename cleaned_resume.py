import os
import pandas as pd
import re
from PyPDF2 import PdfReader
import zipfile

# Input directory where resumes are stored by job title
RESUME = "archive.zip"  
EXTRACT_PATH = "archive"
OUTPUT_CSV = "resume_dataset.csv"

# Automatically unzip if not already extracted
def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Done unzipping.")
    else:
        print("📁 Dataset already unzipped.")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = text.lower().strip()  # Lowercase and trim
    return text

def normalize_label(label: str) -> str:
    return label.replace('_', ' ').strip().title()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def load_resumes(resume_dir: str) -> pd.DataFrame:
    data = []
    for root, dirs, files in os.walk(resume_dir):
        for file in files:
            if file.endswith(".pdf"):
                label = os.path.basename(root)
                label = normalize_label(label)
                file_path = os.path.join(root, file)
                try:
                    text = extract_text_from_pdf(file_path)
                    cleaned = clean_text(text)
                    data.append({"text": cleaned, "label": label})
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if not data:
        print("No valid PDF files found in the directory.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df

def main():
    unzip_dataset(RESUME_DIR, EXTRACT_PATH)  # unzip if needed
    df = load_resumes(EXTRACT_PATH)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved cleaned data to {OUTPUT_CSV}")
        print(df['label'].value_counts())
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()