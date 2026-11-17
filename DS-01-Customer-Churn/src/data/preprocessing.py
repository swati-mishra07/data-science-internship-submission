from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_cleaned.csv"


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data():
    """Load original telecom customer dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean telecom customer data."""

    df = df.copy()

    text_columns = df.select_dtypes(include="object").columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["customerID"].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected. " "Investigate before removing records."
        )

    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
    ]

    df["TotalCharges"] = df["TotalCharges"].replace("", pd.NA)

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["tenure"].isna().any():
        raise ValueError("Tenure contains invalid values.")

    zero_tenure = df["tenure"] == 0

    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0

    if df["TotalCharges"].isna().any():
        raise ValueError("Unexplained missing TotalCharges remain.")

    valid_churn_values = {"Yes", "No"}

    if not set(df["Churn"].dropna().unique()).issubset(valid_churn_values):
        raise ValueError("Unexpected Churn labels detected.")

    if df["Churn"].isna().any():
        raise ValueError("Churn target contains missing values.")

    if (df["tenure"] < 0).any():
        raise ValueError("Negative tenure detected.")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("Negative monthly charges detected.")

    if (df["TotalCharges"] < 0).any():
        raise ValueError("Negative total charges detected.")

    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1}).astype("int8")

    return df


def save_cleaned_data(df: pd.DataFrame):
    """Save validated customer dataset."""

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned data saved: {PROCESSED_DATA_PATH}")


def main():
    print("Loading original customer data...")

    raw_df = load_raw_data()

    print("Original shape:", raw_df.shape)

    print("\nCleaning and validating data...")

    cleaned_df = clean_data(raw_df)

    print("\nCleaning completed.")

    print("Final shape:", cleaned_df.shape)
    print("Unique customers:", cleaned_df["customerID"].nunique())
    print("Missing values:", cleaned_df.isnull().sum().sum())
    print("Duplicate rows:", cleaned_df.duplicated().sum())

    save_cleaned_data(cleaned_df)


if __name__ == "__main__":
    main()
