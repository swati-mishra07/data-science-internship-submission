
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/telco_features.csv"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports/figures"
METRICS_DIR = ROOT / "reports/metrics"


def main():

    for folder in [MODEL_DIR, FIGURE_DIR, METRICS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Duration = observed customer tenure
    # Event = whether churn was observed
    durations = df["tenure"]
    events = df["ChurnFlag"]

    print("Fitting Kaplan-Meier estimator...")

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=durations,
        event_observed=events,
        label="Customer retention"
    )

    plt.figure(figsize=(9, 5))

    kmf.plot_survival_function()

    plt.title("Customer Survival Probability")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Probability of Remaining a Customer")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "kaplan_meier.png",
        dpi=150
    )

    plt.close()

    print("Median survival time:", kmf.median_survival_time_)

    # Cox model
    print("\nFitting Cox Proportional Hazards model...")

    columns = [
        "tenure",
        "ChurnFlag",
        "MonthlyCharges",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Contract",
        "InternetService",
        "PaperlessBilling"
    ]

    survival_df = df[columns].copy()

    categorical = [
        "Partner",
        "Dependents",
        "Contract",
        "InternetService",
        "PaperlessBilling"
    ]

    survival_df = pd.get_dummies(
        survival_df,
        columns=categorical,
        drop_first=True,
        dtype=int
    )

    survival_df = survival_df[
        survival_df["tenure"] > 0
    ].copy()

    cph = CoxPHFitter(penalizer=0.1)

    cph.fit(
        survival_df,
        duration_col="tenure",
        event_col="ChurnFlag"
    )

    print("\nCOX MODEL SUMMARY")
    print(cph.summary)

    cph.print_summary()

    cph.plot()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "cox_coefficients.png",
        dpi=150
    )

    plt.close()

    joblib.dump(
        cph,
        MODEL_DIR / "cox_survival_model.joblib"
    )

    cph.summary.to_csv(
        METRICS_DIR / "cox_summary.csv"
    )

    print("\nSurvival analysis completed.")


if __name__ == "__main__":
    main()
