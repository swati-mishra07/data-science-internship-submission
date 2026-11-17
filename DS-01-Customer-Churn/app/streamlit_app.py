
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "customer_intelligence.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "churn_metrics.json"

SHAP_PATH = ROOT / "reports" / "figures" / "shap_summary.png"
KM_PATH = ROOT / "reports" / "figures" / "kaplan_meier.png"
IMPORTANCE_PATH = ROOT / "reports" / "metrics" / "feature_importance.csv"

UPLIFT_METRICS_PATH = ROOT / "reports" / "uplift" / "uplift_metrics.json"
UPLIFT_PREDICTIONS_PATH = ROOT / "reports" / "uplift" / "uplift_customer_predictions.csv"
QINI_PATH = ROOT / "reports" / "uplift" / "qini_curve.png"


# ------------------------------------------------
# STREAMLIT CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------
# LOAD PROJECT DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if "customerID" not in data.columns:
        raise ValueError("customerID column is missing.")

    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


df = load_data()
model = load_model()
metrics = load_metrics()


# ------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ------------------------------------------------

required_columns = [
    "customerID",
    "Churn",
    "ChurnFlag",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "customer_segment",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# ------------------------------------------------
# MODEL FEATURE PREPARATION
# ------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
model_columns = list(preprocessor.feature_names_in_)


def predict_churn(data):
    """Return predicted churn probabilities."""

    missing = [
        column for column in model_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Model input columns missing: {missing}"
        )

    return model.predict_proba(
        data[model_columns]
    )[:, 1]


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("Customer Lifetime Value & Churn Intelligence")

st.caption(
    "Machine learning powered customer retention decision system"
)


# ------------------------------------------------
# TOP-LEVEL BUSINESS METRICS
# ------------------------------------------------

total_customers = len(df)
churned = int(df["ChurnFlag"].sum())

churn_rate = (
    churned / total_customers * 100
    if total_customers > 0
    else 0
)

monthly_revenue = df["MonthlyCharges"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned:,}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Monthly Revenue", f"${monthly_revenue:,.2f}")

st.divider()


# ------------------------------------------------
# APPLICATION TABS
# ------------------------------------------------

overview, customers, roi, explainability, uplift = st.tabs(
    [
        "Overview",
        "Customer Explorer",
        "Retention ROI",
        "Model Explainability",
        "Uplift Modeling",
    ]
)


# =================================================
# TAB 1: OVERVIEW
# =================================================

with overview:

    st.header("Business Overview")

    st.subheader("Customer Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .reindex(["No", "Yes"], fill_value=0)
        .rename_axis("Customer Status")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        churn_counts.set_index("Customer Status"),
        width="stretch"
    )

    st.divider()

    st.subheader("Customer Segments")

    segment_display = (
        df.groupby("customer_segment")
        .agg(
            customers=("customerID", "count"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            churn_rate=("ChurnFlag", "mean"),
            avg_historical_revenue=("TotalCharges", "mean"),
        )
        .reset_index()
    )

    segment_display["churn_rate"] *= 100

    numeric_columns = [
        "customers",
        "avg_tenure",
        "avg_monthly_charges",
        "churn_rate",
        "avg_historical_revenue",
    ]

    segment_display[numeric_columns] = (
        segment_display[numeric_columns].round(2)
    )

    st.dataframe(
        segment_display,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Customers by Segment")

    segment_chart = (
        segment_display
        .set_index("customer_segment")[["customers"]]
    )

    st.bar_chart(segment_chart, width="stretch")

    st.divider()

    st.subheader("Model Performance")

    model_metrics = metrics["lightgbm"]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{model_metrics['precision']:.3f}")
    m3.metric("Recall", f"{model_metrics['recall']:.3f}")
    m4.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")


# =================================================
# TAB 2: CUSTOMER EXPLORER
# =================================================

with customers:

    st.header("Individual Customer Intelligence")

    customer_ids = df["customerID"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    customer = df[
        df["customerID"].astype(str) == selected_id
    ].iloc[0]

    customer_features = pd.DataFrame(
        [customer[model_columns].to_dict()]
    )

    probability = predict_churn(customer_features)[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Churn Probability",
        f"{probability:.1%}"
    )

    c2.metric(
        "Monthly Charges",
        f"${customer['MonthlyCharges']:.2f}"
    )

    c3.metric(
        "Tenure",
        f"{int(customer['tenure'])} months"
    )

    if probability >= 0.5:
        st.error(
            "High predicted churn risk at the 0.5 threshold."
        )
    else:
        st.success(
            "Lower predicted churn risk at the 0.5 threshold."
        )

    st.subheader("Customer Profile")

    profile_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "customer_segment",
    ]

    available_profile_columns = [
        column for column in profile_columns
        if column in df.columns
    ]

    profile = customer[
        available_profile_columns
    ].to_frame(name="Customer Value")

    profile["Customer Value"] = (
        profile["Customer Value"].astype(str)
    )

    st.dataframe(profile, width="stretch")


# =================================================
# TAB 3: RETENTION ROI SIMULATOR
# =================================================

with roi:

    st.header("Retention Offer ROI Simulator")

    st.caption(
        "Estimate the potential financial impact of retention offers "
        "using predicted churn probabilities."
    )

    discount = st.slider(
        "Monthly discount per customer ($)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    effectiveness = (
        st.slider(
            "Assumed offer effectiveness (%)",
            min_value=0,
            max_value=100,
            value=30
        ) / 100
    )

    threshold = st.slider(
        "Churn risk threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5
    )

    simulation = df.copy()

    simulation["predicted_churn_probability"] = (
        predict_churn(simulation)
    )

    target = simulation[
        simulation["predicted_churn_probability"] >= threshold
    ].copy()

    target_count = len(target)

    expected_saved_revenue = (
        (
            target["predicted_churn_probability"]
            * target["MonthlyCharges"]
            * 12
        ).sum() * effectiveness
        if target_count > 0
        else 0
    )

    total_offer_cost = target_count * discount * 12

    net_benefit = expected_saved_revenue - total_offer_cost

    roi_value = (
        (net_benefit / total_offer_cost) * 100
        if total_offer_cost > 0
        else 0
    )

    r1, r2, r3 = st.columns(3)

    r1.metric("Target Customers", f"{target_count:,}")
    r2.metric("Expected Saved Revenue", f"${expected_saved_revenue:,.2f}")
    r3.metric("Annual Offer Cost", f"${total_offer_cost:,.2f}")

    st.divider()

    st.metric("Estimated Net Benefit", f"${net_benefit:,.2f}")
    st.metric("Scenario ROI", f"{roi_value:.2f}%")

    st.caption(
        "Important: This is a scenario-based estimate, not a causal "
        "uplift measurement. Offer effectiveness is an assumed input. "
        "Revenue is not profit, and this simplified calculation does "
        "not establish actual retention impact."
    )

    st.subheader("Customers Selected for Retention")

    if not target.empty:

        retention_segments = (
            target.groupby("customer_segment")
            .size()
            .reset_index(name="Customer Count")
        )

        retention_segments["customer_segment"] = (
            retention_segments["customer_segment"].astype(str)
        )

        retention_segments["Customer Count"] = (
            retention_segments["Customer Count"].astype(int)
        )

        st.bar_chart(
            retention_segments,
            x="customer_segment",
            y="Customer Count",
            x_label="Customer Segment",
            y_label="Number of Customers",
            width="stretch",
        )

    else:
        st.info("No customers match the selected retention criteria.")


# =================================================
# TAB 4: MODEL EXPLAINABILITY
# =================================================

with explainability:

    st.header("Model Explainability")

    st.subheader("Global SHAP Explanation")

    if SHAP_PATH.exists():

        st.image(
            str(SHAP_PATH),
            caption="SHAP global feature impact",
            width="stretch"
        )

    else:

        st.warning(
            "SHAP summary plot not found. "
            "Run the SHAP analysis module."
        )

    st.divider()

    st.subheader("Customer Survival Analysis")

    if KM_PATH.exists():

        st.image(
            str(KM_PATH),
            caption="Kaplan-Meier customer survival curve",
            width="stretch"
        )

    else:

        st.warning(
            "Kaplan-Meier plot not found. "
            "Run the survival analysis module."
        )

    st.divider()

    st.subheader("Feature Importance")

    if IMPORTANCE_PATH.exists():

        importance = pd.read_csv(IMPORTANCE_PATH)

        if {"feature", "importance"}.issubset(importance.columns):

            importance = importance.head(15)

            st.bar_chart(
                importance.set_index("feature")["importance"],
                width="stretch"
            )

        else:

            st.warning(
                "Feature importance file has an unexpected column format."
            )

    else:

        st.warning("Feature importance CSV not found.")


# =================================================
# TAB 5: UPLIFT MODELING
# =================================================

with uplift:

    st.header("Uplift Modeling")

    st.caption(
        "Hillstrom email campaign experiment | "
        "Women's Email vs No Email"
    )

    st.info(
        "This is a separate experiment using the Hillstrom dataset. "
        "It is not trained on the IBM Telco customer churn dataset. "
        "The results describe campaign-level test-group differences "
        "and model-based uplift rankings; they do not guarantee "
        "an individual response."
    )

    if not UPLIFT_METRICS_PATH.exists():

        st.warning(
            "Uplift metrics were not found. Run "
            "`python -m src.uplift.train_uplift` "
            "to generate the experiment outputs."
        )

    else:

        try:

            with open(
                UPLIFT_METRICS_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                uplift_metrics = json.load(file)

            # Actual keys from your uplift_metrics.json
            qini_score = uplift_metrics["qini_score"]
            auuc_score = uplift_metrics["auuc_score"]

            observed_ate = uplift_metrics["observed_test_ate"]

            predicted_uplift = uplift_metrics["mean_predicted_uplift"]

            treatment_rate = uplift_metrics[
                "treatment_conversion_rate_test"
            ]

            control_rate = uplift_metrics[
                "control_conversion_rate_test"
            ]

            treatment_count = uplift_metrics[
                "treatment_customers_test"
            ]

            control_count = uplift_metrics[
                "control_customers_test"
            ]

            test_customers = uplift_metrics["test_customers"]

            positive_customers = uplift_metrics[
                "positive_uplift_customers"
            ]

            # -----------------------------------------
            # MODEL PERFORMANCE
            # -----------------------------------------

            st.subheader("Held-out Test Set Performance")

            st.caption(
                f"{test_customers:,} customers | "
                f"{treatment_count:,} treated | "
                f"{control_count:,} control"
            )

            metric_columns = st.columns(4)

            metric_columns[0].metric(
                "Normalized Qini",
                f"{qini_score:.4f}"
            )

            metric_columns[1].metric(
                "Normalized AUUC",
                f"{auuc_score:.4f}"
            )

            metric_columns[2].metric(
                "Observed Treatment-Control Difference",
                f"{observed_ate * 100:.3f}%",
                help=(
                    "Absolute difference between test-group "
                    "conversion rates, expressed in percentage points."
                )
            )

            metric_columns[3].metric(
                "Mean Predicted Uplift",
                f"{predicted_uplift * 100:.3f}%"
            )

            st.divider()

            # -----------------------------------------
            # CONVERSION RATE COMPARISON
            # -----------------------------------------

            st.subheader("Test-Group Conversion Rates")

            conversion_columns = st.columns(2)

            conversion_columns[0].metric(
                "Women's Email",
                f"{treatment_rate * 100:.3f}%"
            )

            conversion_columns[1].metric(
                "No Email",
                f"{control_rate * 100:.3f}%"
            )

            conversion_comparison = pd.DataFrame({
                "Campaign Group": [
                    "Women's Email",
                    "No Email"
                ],
                "Conversion Rate (%)": [
                    treatment_rate * 100,
                    control_rate * 100
                ]
            })

            st.bar_chart(
                conversion_comparison,
                x="Campaign Group",
                y="Conversion Rate (%)",
                y_label="Conversion Rate (%)",
                width="stretch"
            )

            st.caption(
                "The observed difference is approximately "
                f"{observed_ate * 100:.3f} percentage points. "
                "This is a test-group difference, not an "
                "individual-level causal effect."
            )

            st.divider()

            # -----------------------------------------
            # POSITIVE UPLIFT SUMMARY
            # -----------------------------------------

            st.subheader("Uplift Targeting Summary")

            positive_percentage = (
                positive_customers / test_customers * 100
                if test_customers > 0
                else 0
            )

            u1, u2 = st.columns(2)

            u1.metric(
                "Positive Predicted Uplift Customers",
                f"{positive_customers:,}"
            )

            u2.metric(
                "Share of Test Customers",
                f"{positive_percentage:.2f}%"
            )

            st.caption(
                "Positive predicted uplift means the model estimates "
                "a positive treatment effect. It does not guarantee "
                "that each customer will respond."
            )

            st.divider()

            # -----------------------------------------
            # QINI CURVE
            # -----------------------------------------

            st.subheader("Uplift Model Evaluation")

            if QINI_PATH.exists():

                st.image(
                    str(QINI_PATH),
                    caption=(
                        "Qini curve on the held-out "
                        "Hillstrom test set"
                    ),
                    width="stretch"
                )

            else:

                st.warning(
                    "Qini curve not found at "
                    "reports/uplift/qini_curve.png."
                )

            st.divider()

            # -----------------------------------------
            # CUSTOMER-LEVEL PREDICTIONS
            # -----------------------------------------

            st.subheader("Customer-Level Uplift Predictions")

            if UPLIFT_PREDICTIONS_PATH.exists():

                predictions = pd.read_csv(
                    UPLIFT_PREDICTIONS_PATH
                )

                if predictions.empty:

                    st.info("The prediction CSV contains no records.")

                elif "predicted_uplift" not in predictions.columns:

                    st.error(
                        "The predicted_uplift column is missing "
                        "from the prediction CSV."
                    )

                else:

                    predictions["predicted_uplift"] = pd.to_numeric(
                        predictions["predicted_uplift"],
                        errors="coerce"
                    )

                    top_customers = (
                        predictions
                        .dropna(subset=["predicted_uplift"])
                        .sort_values(
                            "predicted_uplift",
                            ascending=False
                        )
                        .head(10)
                        .copy()
                    )

                    display_columns = [
                        "predicted_uplift",
                        "recency",
                        "history_segment",
                        "history",
                        "newbie",
                        "channel",
                        "treatment",
                        "actual_conversion"
                    ]

                    available_columns = [
                        column
                        for column in display_columns
                        if column in top_customers.columns
                    ]

                    display_table = top_customers[
                        available_columns
                    ].copy()

                    display_table["predicted_uplift"] = (
                        display_table["predicted_uplift"]
                        .map(lambda value: f"{value * 100:.3f}%")
                    )

                    display_table = display_table.rename(
                        columns={
                            "predicted_uplift": "Predicted Uplift (%)",
                            "recency": "Recency",
                            "history_segment": "History Segment",
                            "history": "Historical Spend",
                            "newbie": "New Customer",
                            "channel": "Channel",
                            "treatment": "Received Email",
                            "actual_conversion": "Actual Conversion"
                        }
                    )

                    st.caption(
                        "Top 10 customers ranked by predicted uplift. "
                        "These are model estimates, not guaranteed responders."
                    )

                    st.table(display_table)

                    st.caption(
                        "Received Email: 1 = Women's Email, "
                        "0 = No Email. Actual Conversion: "
                        "1 = converted, 0 = did not convert."
                    )

                    with st.expander("Prediction dataset information"):

                        st.write(
                            "Total test predictions:",
                            f"{len(predictions):,}"
                        )

                        st.write(
                            "Prediction columns:",
                            predictions.columns.tolist()
                        )

            else:

                st.warning(
                    "Customer uplift predictions CSV not found."
                )

            st.divider()

            # -----------------------------------------
            # METHODOLOGY
            # -----------------------------------------

            st.subheader("Experiment Methodology")

            st.markdown(
                """
                - **Dataset:** Hillstrom MineThatData
                - **Treatment:** Women's Email
                - **Control:** No Email
                - **Estimator:** CausalRandomForestRegressor
                - **Evaluation:** Held-out test set
                - **Metrics:** Normalized Qini and normalized AUUC

                Qini and AUUC evaluate uplift ranking, not
                classification accuracy.
                """
            )

            st.warning(
                "This experiment is separate from the IBM Telco "
                "churn model and the Telco retention ROI scenario. "
                "Predicted uplift is not a guaranteed individual "
                "causal effect."
            )

        except (
            KeyError,
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValueError
        ) as error:

            st.error(
                f"Unable to load uplift experiment results: {error}"
            )


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Customer Intelligence System | Data Science Internship Project"
)
