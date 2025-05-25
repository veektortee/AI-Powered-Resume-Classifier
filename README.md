# Resume Classification with Fine-Tuned BERT

This project showcases a **resume classification system** fine-tuned on preprocessed data provided by (https://github.com/noran-mohamed/Resume-Classification-Dataset). It classifies resumes into predefined categories using a **Streamlit app** for user interaction.

---

## Key Features
- **Dataset**: Preprocessed data adapted from a GitHub user’s repository.  
- **Model**: Fine-tuned `BERT` for multiclass text classification.  
- **Preprocessing**:  
  - Cleaned text from provided data.  
  - Augmentation with GPT-2, synonym replacement, and random deletion.  
  - Applied sliding window tokenization for long text inputs.  
- **Training**:  
  - Initial phase: train classification head only.  
  - Fine-tuning phase: unfreeze BERT layers for deeper learning.  
  - Achieved ~96% train accuracy and ~90% validation accuracy.  
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
git clone <your-repo-link>
cd resume-classifier
pip install -r requirements.txt
