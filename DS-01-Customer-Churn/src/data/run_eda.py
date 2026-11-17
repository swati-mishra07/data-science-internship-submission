
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/telco_features.csv"
OUTPUT_DIR = ROOT / "reports/figures"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid")

    # 1. Churn distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
    plt.close()

    # 2. Tenure distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=30,
        kde=True
    )
    plt.title("Customer Tenure Distribution")
    plt.savefig(OUTPUT_DIR / "tenure_distribution.png", dpi=150)
    plt.close()

    # 3. Monthly charges
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges"
    )
    plt.title("Monthly Charges vs Churn")
    plt.savefig(OUTPUT_DIR / "monthly_charges.png", dpi=150)
    plt.close()

    # 4. Contract churn rates
    contract_churn = (
        df.groupby("Contract", observed=True)["ChurnFlag"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=contract_churn,
        x="Contract",
        y="ChurnFlag"
    )
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.savefig(OUTPUT_DIR / "contract_churn.png", dpi=150)
    plt.close()

    print("EDA completed. Charts saved in reports/figures.")


if __name__ == "__main__":
    main()
