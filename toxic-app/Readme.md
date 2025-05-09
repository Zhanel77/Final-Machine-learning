# Toxic Comment Classifier (Multi-Label Classification)

This project is a **Machine Learning web application** that detects toxic characteristics in text comments.  
The model performs **multi-label classification** to identify the following categories:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Built using **Random Forest + TF-IDF + FastAPI + HTML/CSS frontend**.
---

## Project Structure

toxic-app/
├── app/
│ ├── main.py # FastAPI web server
│ ├── predict.py # Predict function and formatting
│ ├── model_loader.py # Loads model and vectorizer
│ ├── emotion_model.pkl # Trained ML model
│ └── vectorizer.pkl # TF-IDF vectorizer
├── templates/
│ └── form.html # HTML frontend
├── static/
│ └── style.css # UI styling
├── train_model.py # Training script
├── train_clean.csv # Training dataset
├── test_clean.csv # Test dataset (features)
├── test_labels.csv # Test dataset (labels)
└── requirements.txt # Dependencies
---

## How It Works

1. Text is preprocessed and vectorized using **TF-IDF**
2. A **Random Forest** model predicts one or more labels (multi-label)
3. User sees predictions + probabilities via a clean FastAPI frontend

---

## How to Run Locally

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```
**This saves:**
app/emotion_model.pkl
app/vectorizer.pkl

### 3. Run FastAPI
```bash
uvicorn app.main:app --reload
```
**Go to: http://127.0.0.1:8000** 

 ### Model Performance
Evaluated on filtered test set (excluding label=-1):
Algorithm: Random Forest (MultiOutputClassifier)
Vectorization: TF-IDF (top 10,000 features)
Metrics: Accuracy, Precision, Recall, F1-score per label
(See train_model.py console output for full report)

### Future Improvements
--Replace RandomForest with XGBoost or BERT
--Add language filter and token normalization
--API version for bulk comment predictions

### Author
Zhanel Kuandyk
IT-2303