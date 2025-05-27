# Resume Classification with Fine-Tuned BERT

This project presents a robust resume classification system powered by a fine-tuned BERT model. Designed for scalability and real-world application, the system classifies resumes into one of 43 distinct categories. It features an interactive Streamlit app for seamless user interaction, enabling quick resume uploads and instant predictions.
---

## Key Features
- **Dataset**: Preprocessed data adapted from a [GitHub user’s repository](https://github.com/noran-mohamed/Resume-Classification-Dataset).  
- **Model**: Fine-tuned `BERT` for multiclass text classification.  
- **Preprocessing**:  
  - Cleaned text from provided data.  
- **Training**:  
  - Initial phase: train classification head only.  
  - Fine-tuning phase: unfreeze BERT layers for deeper learning.  
  - Achieved ~97% train accuracy and ~91% validation accuracy.  
- **Classification**:  
  - Supports **43 predefined resume categories**.  
  - Resumes outside these categories will not be accurately classified.  
- **Deployment**: Streamlit app for uploading and classifying resumes.  
- **Optimizations**:  
  - Reduced learning rate during training.  
  - Early stopping and dropout regularization.  

---

## Supported Categories (43)
- Accountant  
- Advocate  
- Arts  
- Automation Testing  
- Blockchain  
- Business Analyst  
- Civil Engineer  
- Data Analyst  
- Data Engineer  
- Data Science  
- Database Administrator  
- DevOps Engineer  
- DotNet Developer  
- Electrical Engineering  
- ETL Developer  
- Finance  
- Hadoop Developer  
- Health and Fitness  
- Human Resources  
- Java Developer  
- Mechanical Engineer  
- Network Security Engineer  
- Operations Manager  
- Python Developer  
- SAP Developer  
- Salesforce Developer  
- SQL Developer  
- Web Developer  
- Testing  
- Functional Consultant  
- System Administrator  
- Technical Support  
- Project Manager  
- Product Manager  
- Quality Assurance  
- Business Development  
- Content Writer  
- Graphic Designer  
- Digital Marketing  
- Frontend Developer  
- Backend Developer  
- Full Stack Developer  
- Cloud Engineer  

---

## Setup
### Prerequisites
- Python 3.8+
- Refer to `requirements.txt` for full list of dependencies.

### Installation
```bash
git clone <https://github.com/veektortee/AI-Powered-Resume-Classifier.git>
cd AI-Powered-Resume-Classifier
pip install -r requirements.txt
```
---

## File Structure
- app.py: Streamlit app for live resume classification.
-	preprocess_extraction.py: Extracts and cleans text from uploaded resumes.
-	predict.py: Standalone script for classification.
- label_encoder.pkl: Label encoder used for class mapping.
-	resume_classifier_model.keras: Fine-tuned BERT model checkpoint.
-	requirements.txt: All necessary dependencies.

---

## Challenges
- Had to downgrade Colab environment for compatibility with specific library versions (CUDA 11.8 and cuDNN 8.6).
- Managed large model size by saving externally.
- Applied layer-wise freezing/unfreezing to optimize fine-tuning.

---

## Credits
- Dataset courtesy of [Noran Mohamed](https://github.com/noran-mohamed/Resume-Classification-Dataset)
- Preprocessing and fine-tuning logic designed and optimized for resume classification.
- Hugging Face Transformers for pretrained BERT models.

---

## Example Usage
### Streamlit App:
```bash
streamlit run streamlit_app.py
```
Upload your resume (.pdf, .png, .jpg) and instantly get the predicted category.

---

## Results
- Achieved 97% training accuracy and 91% validation accuracy.
- Handled over 12,000 preprocessed resumes.
- Balanced class weights to manage imbalance.

---

## Notes
- The model supports classification for 43 classes only. Resumes outside these categories will not be accurately classified.
- OCR and text extraction are included to handle image and PDF resumes.
- Model and label encoder are saved externally and not pushed to the repository.
