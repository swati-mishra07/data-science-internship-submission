
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from causalml.inference.tree import CausalRandomForestRegressor
from causalml.metrics import auuc_score, qini_score, plot_qini


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "uplift" / "hillstrom.csv"
MODEL_DIR = ROOT / "models" / "uplift"
REPORT_DIR = ROOT / "reports" / "uplift"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("Loading Hillstrom dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hillstrom dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "conversion",
        "segment",
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. SELECT TREATMENT AND CONTROL GROUPS
    # --------------------------------------------------------

    print("\nPreparing treatment and control groups...")

    df = df[
        df["segment"].isin(
            ["Womens E-Mail", "No E-Mail"]
        )
    ].copy()

    # Treatment = 1
    # Control = 0

    df["treatment"] = (
        df["segment"] == "Womens E-Mail"
    ).astype(int)

    df["conversion"] = pd.to_numeric(
        df["conversion"],
        errors="raise",
    ).astype(int)

    if not set(df["conversion"].unique()).issubset({0, 1}):
        raise ValueError(
            "Conversion must contain only 0 and 1."
        )

    print("Experiment shape:", df.shape)

    print("\nTreatment distribution:")
    print(df["treatment"].value_counts())

    print("\nObserved conversion rates:")
    print(
        df.groupby("treatment")["conversion"].mean()
    )

    # --------------------------------------------------------
    # 3. PREPARE FEATURES
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            "conversion",
            "segment",
            "treatment",
        ]
    )

    treatment = df["treatment"].to_numpy()
    outcome = df["conversion"].to_numpy()

    categorical_columns = [
        "history_segment",
        "zip_code",
        "channel",
    ]

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\nSplitting experiment data...")

    stratification = (
        pd.Series(treatment).astype(str)
        + "_"
        + pd.Series(outcome).astype(str)
    )

    (
        X_train,
        X_test,
        treatment_train,
        treatment_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        treatment,
        outcome,
        test_size=0.25,
        random_state=42,
        stratify=stratification,
    )

    print("Training customers:", len(X_train))
    print("Testing customers:", len(X_test))

    # --------------------------------------------------------
    # 5. PREPROCESSING
    # --------------------------------------------------------

    print("\nPreprocessing features...")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # 6. TRAIN CAUSAL RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining CausalML model...")

    model = CausalRandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=200,
        min_samples_leaf=100,
        control_name=0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X=X_train_processed,
        treatment=treatment_train,
        y=y_train,
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # 7. PREDICT INDIVIDUAL TREATMENT EFFECTS
    # --------------------------------------------------------

    print("\nEstimating individual treatment effects...")

    uplift_predictions = np.asarray(
        model.predict(X_test_processed)
    ).reshape(len(X_test), -1)[:, 0]

    if not np.isfinite(uplift_predictions).all():
        raise ValueError(
            "Uplift predictions contain invalid values."
        )

    results = X_test.reset_index(drop=True).copy()

    results["actual_conversion"] = y_test
    results["treatment"] = treatment_test
    results["predicted_uplift"] = uplift_predictions

    results = results.sort_values(
        "predicted_uplift",
        ascending=False,
    )

    results.to_csv(
        REPORT_DIR / "uplift_customer_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. PREPARE EVALUATION DATA
    # --------------------------------------------------------

    evaluation_df = pd.DataFrame(
        {
            "conversion": y_test,
            "treatment": treatment_test,
            "predicted_uplift": uplift_predictions,
        }
    )

    # CausalML expects treatment labels and prediction
    # columns. Keep predicted_uplift in the dataframe.
    print("\nCalculating Qini score...")

    qini = qini_score(
        evaluation_df,
        outcome_col="conversion",
        treatment_col="treatment",
        normalize=True,
    )

    print("\nCalculating AUUC...")

    auuc = auuc_score(
        evaluation_df,
        outcome_col="conversion",
        treatment_col="treatment",
        normalize=True,
    )

    print("\nQini result:")
    print(qini)

    print("\nAUUC result:")
    print(auuc)

    # Extract our prediction column explicitly.
    if "predicted_uplift" not in qini.index:
        raise ValueError(
            "Qini output does not contain predicted_uplift. "
            f"Available values: {qini.to_dict()}"
        )

    if "predicted_uplift" not in auuc.index:
        raise ValueError(
            "AUUC output does not contain predicted_uplift. "
            f"Available values: {auuc.to_dict()}"
        )

    qini_value = float(qini["predicted_uplift"])
    auuc_value = float(auuc["predicted_uplift"])

    # --------------------------------------------------------
    # 9. EXPERIMENT METRICS
    # --------------------------------------------------------

    treatment_rate = float(
        evaluation_df.loc[
            evaluation_df["treatment"] == 1,
            "conversion",
        ].mean()
    )

    control_rate = float(
        evaluation_df.loc[
            evaluation_df["treatment"] == 0,
            "conversion",
        ].mean()
    )

    observed_test_ate = (
        treatment_rate - control_rate
    )

    mean_predicted_uplift = float(
        np.mean(uplift_predictions)
    )

    metrics = {
        "dataset": "Hillstrom MineThatData",
        "experiment": "Womens E-Mail vs No E-Mail",
        "total_experiment_customers": int(len(df)),
        "training_customers": int(len(X_train)),
        "test_customers": int(len(X_test)),
        "treatment_customers_test": int(
            (treatment_test == 1).sum()
        ),
        "control_customers_test": int(
            (treatment_test == 0).sum()
        ),
        "treatment_conversion_rate_test": treatment_rate,
        "control_conversion_rate_test": control_rate,
        "observed_test_ate": observed_test_ate,
        "mean_predicted_uplift": mean_predicted_uplift,
        "qini_score": qini_value,
        "auuc_score": auuc_value,
        "positive_uplift_customers": int(
            (uplift_predictions > 0).sum()
        ),
    }

    with open(
        REPORT_DIR / "uplift_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 10. SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_columns": X.columns.tolist(),
            "treatment": "Womens E-Mail",
            "control": "No E-Mail",
        },
        MODEL_DIR / "hillstrom_uplift_model.joblib",
    )

    # --------------------------------------------------------
    # 11. GENERATE QINI CURVE
    # --------------------------------------------------------

    print("\nGenerating Qini curve...")

    ax = plot_qini(
        evaluation_df,
        outcome_col="conversion",
        treatment_col="treatment",
        normalize=True,
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        REPORT_DIR / "qini_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(ax.figure)

    # --------------------------------------------------------
    # 12. SAVE AND DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("CAUSALML UPLIFT MODEL RESULTS")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nGenerated files:")

    print(
        "Model:",
        MODEL_DIR / "hillstrom_uplift_model.joblib",
    )

    print(
        "Metrics:",
        REPORT_DIR / "uplift_metrics.json",
    )

    print(
        "Customer predictions:",
        REPORT_DIR / "uplift_customer_predictions.csv",
    )

    print(
        "Qini curve:",
        REPORT_DIR / "qini_curve.png",
    )

    print("\nUplift modeling completed successfully.")


if __name__ == "__main__":
    main()
