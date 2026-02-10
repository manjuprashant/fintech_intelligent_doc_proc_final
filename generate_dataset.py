import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_FILE = "data/final_dataset.csv"
os.makedirs("data", exist_ok=True)

records = []
for file in os.listdir(RAW_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(RAW_DIR, file))
        records.append(df)

final_df = pd.concat(records, ignore_index=True)
final_df.to_csv(OUT_FILE, index=False)

print("✅ Dataset generated at data/final_dataset.csv")
