import joblib

# === Загрузка модели и TF-IDF вектора ===
model = joblib.load("app/emotion_model.pkl")
vectorizer = joblib.load("app/vectorizer.pkl")

# === Названия меток ===
class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_labels(text):
    # Векторизация текста
    X_vec = vectorizer.transform([text])
    
    # Получение вероятностей по каждому классу
    proba = model.predict_proba(X_vec)
    probabilities = {label: round(float(p[0][1]), 3) for label, p in zip(class_names, proba)}
    
    # Порог классификации: более чувствительный (0.1)
    threshold = 0.1
    prediction = {label: int(prob > threshold) for label, prob in probabilities.items()}

    return prediction, probabilities
