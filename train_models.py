import os
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_FILE = "data/final_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)
X = df["text"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
)

# =====================
# TF-IDF
# =====================
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

# =====================
# RANDOM FOREST
# =====================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
print("✅ RandomForest trained")

# =====================
# XGBOOST
# =====================
xgb = XGBClassifier(eval_metric="logloss")
xgb.fit(X_train_tfidf, y_train)
joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")
print("✅ XGBoost trained")

# =====================
# BiLSTM
# =====================
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=200)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=200)

joblib.dump(tokenizer, f"{MODEL_DIR}/tokenizer.pkl")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=200),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=3, batch_size=32)

model.save(f"{MODEL_DIR}/bilstm_model.keras")
print("✅ BiLSTM trained")

print("\n✅ All models saved in models/")
