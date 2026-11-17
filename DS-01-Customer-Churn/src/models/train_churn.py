from pathlib import Path
import json
import joblib

import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/telco_features.csv"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports/metrics"
FIGURE_DIR = ROOT / "reports/figures"

TARGET = "ChurnFlag"

RANDOM_STATE = 42


def create_preprocessor(X):
    """Create numerical and categorical transformations."""

    numeric_columns = X.select_dtypes(include="number").columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
    }


def main():

    for directory in [MODEL_DIR, METRICS_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError("Run feature engineering before model training.")

    print("Loading engineered customer data...")

    df = pd.read_csv(DATA_PATH)

    excluded = ["customerID", "Churn", TARGET]

    X = df.drop(columns=excluded)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining Logistic Regression baseline...")

    baseline = Pipeline(
        [
            ("preprocessor", create_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    baseline.fit(X_train, y_train)

    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    baseline_prediction = (baseline_probability >= 0.5).astype(int)

    baseline_metrics = calculate_metrics(
        y_test, baseline_prediction, baseline_probability
    )

    print("\nTraining LightGBM...")

    churn_model = Pipeline(
        [
            ("preprocessor", create_preprocessor(X_train)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=6,
                    num_leaves=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
            ),
        ]
    )

    churn_model.fit(X_train, y_train)

    probabilities = churn_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = calculate_metrics(y_test, predictions, probabilities)

    print("\nLOGISTIC REGRESSION BASELINE")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nLIGHTGBM RESULTS")
    print(json.dumps(metrics, indent=4))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save trained pipeline
    joblib.dump(churn_model, MODEL_DIR / "churn_pipeline.joblib")

    # Save baseline and model metrics
    results = {
        "model": "LightGBM",
        "baseline": baseline_metrics,
        "lightgbm": metrics,
        "threshold": 0.5,
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
    }

    with open(METRICS_DIR / "churn_metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("LightGBM Churn Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save model feature importance
    preprocessor = churn_model.named_steps["preprocessor"]
    model = churn_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    importance.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    print("\nMODEL SAVED SUCCESSFULLY")
    print("Model:", MODEL_DIR / "churn_pipeline.joblib")
    print("Metrics:", METRICS_DIR / "churn_metrics.json")
    print("Confusion matrix:", FIGURE_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
