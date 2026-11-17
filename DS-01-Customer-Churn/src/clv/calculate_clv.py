
from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/telco_features.csv"
OUTPUT_PATH = ROOT / "data/processed/customer_intelligence.csv"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports/metrics"


def main():

    df = pd.read_csv(DATA_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Business assumption:
    # Evaluate potential revenue over the next 12 months.
    HORIZON_MONTHS = 12

    # Scenario-based value estimate, not observed true CLV.
    df["annualized_revenue"] = (
        df["MonthlyCharges"] * HORIZON_MONTHS
    )

    # Historical revenue proxy.
    df["historical_revenue"] = df["TotalCharges"]

    # Customer risk and value
    df["risk_category"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-1, 35, 70, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    # Segment using customer tenure, charges and service count
    clustering_features = [
        "tenure",
        "MonthlyCharges",
        "service_count"
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[clustering_features]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["customer_segment"] = kmeans.fit_predict(
        scaled_data
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "customer_scaler.joblib"
    )

    joblib.dump(
        kmeans,
        MODEL_DIR / "customer_kmeans.joblib"
    )

    # Segment interpretation
    segment_summary = (
        df.groupby("customer_segment", observed=True)
        .agg(
            customers=("customerID", "count"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_service_count=("service_count", "mean"),
            churn_rate=("ChurnFlag", "mean"),
            avg_historical_revenue=("TotalCharges", "mean")
        )
        .round(2)
    )

    segment_summary.to_csv(
        METRICS_DIR / "segment_summary.csv"
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("CUSTOMER SEGMENTATION COMPLETED")
    print("\nSegment summary:")
    print(segment_summary)

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
