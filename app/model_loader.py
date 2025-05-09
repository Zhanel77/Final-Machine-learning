import joblib

# Загрузка модели и TF-IDF
model = joblib.load("app/emotion_model.pkl")
vectorizer = joblib.load("app/vectorizer.pkl")
