# Toxic Comment Classifier (Multi-Label Classification)

A Machine Learning web application that detects toxic characteristics in user-generated text comments. The model performs **multi-label classification** to identify toxic attributes such as:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Built using **Random Forest**, **TF-IDF**, **FastAPI**, and a lightweight **HTML/CSS frontend**.

---

## Table of Contents

- [File/Folder Structure](#filefolder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## File/Folder Structure

```
toxic-app/
├── app/
│   ├── main.py              # FastAPI web server
│   ├── predict.py           # Predict function and formatting
│   ├── model_loader.py      # Loads model and vectorizer
│   ├── emotion_model.pkl    # Trained ML model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── templates/
│   └── form.html            # HTML frontend
├── static/
│   └── style.css            # UI styling
├── train_model.py           # Training script
├── train_clean.csv          # Training dataset
├── test_clean.csv           # Test dataset (features)
├── test_labels.csv          # Test dataset (labels)
└── requirements.txt         # Python dependencies
```

---

## Features

- Multi-label classification of toxic comments
- Clean, responsive frontend interface
- Real-time probability display of each label
- Easily extendable model pipeline
- FastAPI backend for RESTful service

---

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd toxic-app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Train the model

```bash
python train_model.py
```

This will save the trained model and vectorizer to:

- `app/emotion_model.pkl`
- `app/vectorizer.pkl`

### Step 2: Run the web server

```bash
uvicorn app.main:app --reload
```

Then open your browser and go to:

```
http://127.0.0.1:8000
```

---

###  Deployment
The application is live at:
https://final-machine-learning.onrender.com

## Model Performance

- **Algorithm**: Random Forest via `MultiOutputClassifier`
- **Vectorization**: TF-IDF (Top 10,000 features)
- **Evaluation**: Accuracy, Precision, Recall, F1-score per label
- **Dataset**: Custom cleaned dataset excluding invalid labels (`label = -1`)

Detailed metrics are printed in the console during training via `train_model.py`.

---

## Configuration

To change the model or vectorizer:

- Replace `emotion_model.pkl` and `vectorizer.pkl` with your custom models.
- Ensure that the format is compatible with `predict.py` and `model_loader.py`.

---

## Future Improvements

- Replace RandomForest with XGBoost or BERT
- Add multilingual support and language filtering
- Deploy a RESTful API endpoint for batch predictions

---

## Troubleshooting

- **Module not found**: Ensure you're in the correct directory and using a virtual environment.
- **Model not loading**: Run `train_model.py` before launching the server.
- **Server not starting**: Check FastAPI and Uvicorn installation with `pip show fastapi uvicorn`.

---

## Example

Positive:
- **I really enjoyed this article, thanks for sharing.**
- **Looking forward to your next post!**
Negative:
- **You're smart, but sometimes you act like a jerk.**
- **I hate you and your entire family.**

## Contributors

- **Zhanel Kuandyk** – IT-2303 
---
