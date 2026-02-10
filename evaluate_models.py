import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# =====================
# CONFIG
# =====================
DATA_FILE = "data/final_dataset.csv"
RESULT_DIR = "results"
MODEL_DIR = "models"
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================
# SAFE UTILITIES
# =====================
def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except:
        try:
            preds = model.predict(X)
            if preds.ndim == 1:
                return np.column_stack([1 - preds, preds])
            return preds
        except:
            return None


def safe_auc(y_true, y_probs):
    try:
        if len(np.unique(y_true)) < 2:
            return None, None, None
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        return fpr, tpr, auc(fpr, tpr)
    except:
        return None, None, None


def plot_confusion(y_true, y_pred, name):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{name.lower()}_confusion.png")
    plt.close()


def plot_roc(y_true, y_probs, name):
    fpr, tpr, auc_val = safe_auc(y_true, y_probs)
    if auc_val is None:
        print(f"⚠️ ROC skipped for {name} (single class)")
        return None

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.savefig(f"{RESULT_DIR}/{name.lower()}_roc.png")
    plt.close()
    return auc_val


def plot_pdf(y_probs, name):
    try:
        pos_probs = y_probs[:, 1]
        neg_probs = y_probs[:, 0]

        plt.figure()
        plt.hist(pos_probs, bins=30, alpha=0.6, density=True, label="Positive")
        plt.hist(neg_probs, bins=30, alpha=0.6, density=True, label="Negative")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title(f"{name} Probability Density Function")
        plt.legend()
        plt.savefig(f"{RESULT_DIR}/{name.lower()}_pdf.png")
        plt.close()
    except:
        print(f"⚠️ PDF skipped for {name}")


# =====================
# LOAD DATA
# =====================
print("📥 Loading test data...")
df = pd.read_csv(DATA_FILE)

X = df["text"].astype(str)
y = df["label"]

# =====================
# LOAD VECTORIZERS
# =====================
tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
tokenizer = joblib.load(f"{MODEL_DIR}/tokenizer.pkl")

X_tfidf = tfidf.transform(X)

seqs = tokenizer.texts_to_sequences(X)
X_seq = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=200)

# =====================
# LOAD MODELS
# =====================
rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
bilstm = tf.keras.models.load_model(f"{MODEL_DIR}/bilstm_model.keras")

# =====================
# EVALUATION CORE
# =====================
results = []

def evaluate_model(name, model, X_input, is_dl=False):
    print(f"\n📊 {name}")

    preds = model.predict(X_input)
    if is_dl:
        preds = (preds > 0.5).astype(int).reshape(-1)

    probs = safe_predict_proba(model, X_input)
    if probs is None and is_dl:
        probs = np.column_stack([1 - preds, preds])

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    plot_confusion(y, preds, name)

    auc_val = None
    if probs is not None:
        auc_val = plot_roc(y, probs, name)
        plot_pdf(probs, name)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc_val
    })


# =====================
# RUN EVALUATIONS
# =====================
evaluate_model("RandomForest", rf, X_tfidf)
evaluate_model("XGBoost", xgb, X_tfidf)
evaluate_model("BiLSTM", bilstm, X_seq, is_dl=True)

# =====================
# SAVE METRICS
# =====================
results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULT_DIR}/metrics_table.csv", index=False)

# =====================
# MODEL COMPARISON PLOT
# =====================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(results_df))

plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(x, results_df[metric], marker="o", label=metric)

plt.xticks(x, results_df["Model"])
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/model_comparison.png")
plt.close()

print("\n✅ Evaluation complete. All plots saved in /results/")
