from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/telco_features.csv"
MODEL_PATH = ROOT / "models/churn_pipeline.joblib"
FIGURE_DIR = ROOT / "reports/figures"
METRICS_DIR = ROOT / "reports/metrics"

TARGET = "ChurnFlag"
RANDOM_STATE = 42


def main():

    print("Loading customer data and trained model...")

    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    X = df.drop(columns=["customerID", "Churn", TARGET])
    y = df[TARGET].astype(int)

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_sample = X_test.sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    print("Transforming test data...")

    X_transformed = preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed, columns=feature_names, index=X_sample.index
    )

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Handle binary-class output across SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating SHAP summary plot...")

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Creating SHAP feature importance plot...")

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_absolute_shap": abs(shap_values).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(METRICS_DIR / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(
        importance.head(15)["feature"][::-1],
        importance.head(15)["mean_absolute_shap"][::-1],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top 15 Churn Prediction Drivers")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSHAP ANALYSIS COMPLETED")
    print("Summary:", FIGURE_DIR / "shap_summary.png")
    print("Importance:", FIGURE_DIR / "shap_importance.png")
    print("CSV:", METRICS_DIR / "shap_feature_importance.csv")


if __name__ == "__main__":
    main()
