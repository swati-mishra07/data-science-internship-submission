
from pathlib import Path

import pandas as pd
from sklift.datasets import fetch_hillstrom

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "uplift"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Hillstrom dataset...")

dataset = fetch_hillstrom(
    target_col="conversion",
    data_home=str(DATA_DIR),
    download_if_missing=True,
)

# The Bunch object stores features, outcome and treatment separately.
df = dataset.data.copy()
df["conversion"] = dataset.target
df["segment"] = dataset.treatment

output_path = DATA_DIR / "hillstrom.csv"
df.to_csv(output_path, index=False)

print("\nDataset saved successfully!")
print("Shape:", df.shape)
print("Saved to:", output_path)

print("\nTreatment group counts:")
print(df["segment"].value_counts())

print("\nConversion rates:")
print(df.groupby("segment")["conversion"].mean())

print("\nColumns:")
print(df.columns.tolist())
