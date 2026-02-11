FinTech Intelligent Document Processing (IDP) System

An end-to-end AI-powered document classification pipeline for financial documents using Machine Learning and Deep Learning, including dataset generation, model training, evaluation, and visualization with enterprise-grade robustness.

This project demonstrates a complete production ML workflow suitable for FinTech automation, OCR pipelines, compliance systems, and document intelligence platforms.

🚀 Key Features

✅ Synthetic financial dataset generation
✅ Multi-class document classification
✅ Models: Random Forest, XGBoost, BiLSTM
✅ Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
✅ Confusion matrices, ROC curves, probability density plots
✅ Crash-proof evaluation pipeline
✅ Ready-to-deploy model artifacts
✅ Professional documentation & notebooks

🏗️ Project Architecture

📂 Repository Structure
fintech-intelligent-document-processing/
│
├── pipeline/
│   ├── generate_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── utils.py
│   ├── config.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   └── final_dataset.csv
│
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── bilstm_model.keras
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── docs/
│   ├── architecture.png
│   └── evaluation_report.pdf
│
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── probability_distributions/
│   └── metrics_summary.csv
│
├── requirements.txt
├── README.md
└── .gitignore

🧠 Document Classes
Class	Description
invoice	Commercial invoices
receipt	Purchase receipts
bank_statement	Bank account statements
tax_document	Tax filings
id_document	Identity documents
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/fintech-intelligent-document-processing.git
cd fintech-intelligent-document-processing

2️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Pipeline Execution
Step 1 — Generate Dataset
python pipeline/generate_dataset.py


Creates:

data/final_dataset.csv

Step 2 — Train Models
python pipeline/train_models.py


Trains and saves:

models/
├── rf_model.pkl
├── xgb_model.pkl
├── bilstm_model.keras
├── tfidf_vectorizer.pkl
└── tokenizer.pkl

Step 3 — Evaluate Models
python pipeline/evaluate_models.py


Generates:

Accuracy, Precision, Recall, F1

Confusion matrices

ROC-AUC curves

Probability density plots

Metrics summary CSV

All saved under:

results/

📊 Sample Results
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Random Forest	1.00	1.00	1.00	1.00	1.00
XGBoost	      1.00	1.00	1.00	1.00	1.00
BiLSTM	      1.00	1.00	1.00	1.00	1.00

(Synthetic dataset → perfect separability)

📈 Visualizations Generated

✔ Confusion matrices
✔ ROC & AUC curves
✔ Precision-Recall curves
✔ Probability density functions (PDFs)
✔ Model comparison bar charts

🧪 Notebook

Interactive analysis available in:

notebooks/EDA.ipynb


Includes:

Class distributions

Token statistics

Text length analysis

Dataset sanity checks

📄 Documentation
File	Purpose
docs/architecture.png	System architecture diagram
docs/evaluation_report.pdf	Full professional evaluation report
🏢 Real-World Use Cases

✔ FinTech document ingestion systems
✔ OCR post-processing pipelines
✔ Compliance automation
✔ KYC verification systems
✔ Invoice & receipt classification engines

🛡️ Robustness Guarantees

This pipeline:

Handles missing classes safely

Works with binary and multiclass outputs

Handles single-class edge cases

Prevents ROC/AUC crashes

Supports variable model outputs

Supports CPU-only execution

🧰 Tech Stack

Python 3.9+

TensorFlow / Keras

Scikit-learn

XGBoost

Pandas / NumPy

Matplotlib / Seaborn

🧑‍💻 Author

Manjula Srinivasan
Data Science and Machine Learning

📜 License

MIT License — Free to use, modify, and distribute.
