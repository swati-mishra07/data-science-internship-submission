
from pathlib import Path
import pandas as pd

from src.data.preprocessing import (
    clean_data,
    load_raw_data
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_features.csv"
)


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable customer behaviour features."""

    df = df.copy()

    for column in SERVICE_COLUMNS:
        df[column + "_flag"] = (
            df[column]
            .eq("Yes")
            .astype("int8")
        )

    flag_columns = [
        column + "_flag"
        for column in SERVICE_COLUMNS
    ]

    df["service_count"] = df[flag_columns].sum(axis=1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    df["is_month_to_month"] = (
        df["Contract"]
        .eq("Month-to-month")
        .astype("int8")
    )

    return df


def main():
    print("Loading and validating customer data...")

    raw_df = load_raw_data()

    clean_df = clean_data(raw_df)

    print("Building customer features...")

    feature_df = engineer_features(clean_df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFEATURE ENGINEERING COMPLETED")

    print("Dataset shape:", feature_df.shape)

    print("\nNew feature summary:")

    print(
        feature_df[
            [
                "service_count",
                "tenure_group",
                "is_month_to_month",
                "ChurnFlag"
            ]
        ].head()
    )

    print("\nCustomer service count distribution:")

    print(
        feature_df["service_count"].value_counts().sort_index()
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
