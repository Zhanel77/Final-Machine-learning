import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# === Загрузка данных ===
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")
test_labels = pd.read_csv("test_labels.csv")

# === Сэмплируем 10 000 строк для ускорения обучения ===
train_df = train_df.sample(n=10000, random_state=42)

# === Удаляем строки с -1 в метках ===
test_labels = test_labels[(test_labels != -1).all(axis=1)]
test_df = test_df[test_df["id"].isin(test_labels["id"])]

# === Подготовка признаков и меток ===
X_train = train_df["comment_text"]
y_train = train_df.drop(columns=["id", "comment_text"])

X_test = test_df["comment_text"]
y_test = test_labels.drop(columns=["id"])

# === TF-IDF векторизация ===
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Обучение модели ===
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputClassifier(rf)
model.fit(X_train_vec, y_train)

# === Оценка ===
y_pred = model.predict(X_test_vec)
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# === Сохранение ===
joblib.dump(model, "app/emotion_model.pkl")
joblib.dump(vectorizer, "app/vectorizer.pkl")
print("Модель и вектор сохранены в папку app/")
