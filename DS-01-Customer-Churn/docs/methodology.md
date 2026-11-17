# Project Methodology

## 1. Project Overview

The Customer Lifetime Value and Churn Intelligence System is an end-to-end data science project designed to analyze customer behavior, identify customers at risk of churn, estimate customer value, understand customer retention patterns, and support data-driven retention decisions.

The system combines machine learning, survival analysis, customer segmentation, explainable AI, and uplift modeling.

The project uses two datasets for different analytical objectives:

- **IBM Telco Customer Churn dataset:** Used for churn prediction, survival analysis, customer lifetime value estimation, and segmentation.
- **Hillstrom MineThatData dataset:** Used separately for uplift modeling to estimate the incremental effect of a marketing treatment.

These datasets are not merged because they represent different customer populations and business experiments.

## 2. Data Collection and Storage

### Telco Customer Churn Dataset

The IBM Telco dataset contains 7,043 customer records and 21 original columns, including:

- Customer demographics
- Contract information
- Internet and phone services
- Tenure
- Monthly charges
- Total charges
- Churn status

The raw dataset is stored in `data/raw/`.

### Data Cleaning

The preprocessing pipeline performs the following operations:

1. Removes duplicate customer records where applicable.
2. Handles missing and blank values in `TotalCharges`.
3. Converts numerical columns into appropriate data types.
4. Encodes the churn target into binary values.
5. Preserves customer identifiers for customer-level analysis.

Customers with zero tenure and missing total charges are handled by assigning zero total charges where appropriate.

### PostgreSQL Integration

Cleaned and engineered customer data is loaded into PostgreSQL.

The database contains:

- `telco_cleaned`: 7,043 cleaned customer records.
- `telco_features`: 7,043 records containing engineered features.

The Streamlit dashboard currently reads the saved processed CSV files and model artifacts rather than querying PostgreSQL directly.

## 3. Exploratory Data Analysis and Feature Engineering

Exploratory Data Analysis is performed to understand customer distribution, service usage, contract patterns, charges, tenure, and churn behavior.

Visualizations include:

- Churn distribution
- Tenure distribution
- Monthly charges distribution
- Churn by contract type
- Customer service patterns

### Feature Engineering

Additional features are created to improve customer-level analysis:

- `service_count`: Number of subscribed services.
- `tenure_group`: Customer tenure category.
- `is_month_to_month`: Indicator for month-to-month contracts.
- Service-specific binary indicators.

These features help capture differences in customer engagement and contract behavior.

## 4. Churn Prediction

### Objective

Predict the probability that a customer will leave the service.

### Models

Two classification models are evaluated:

1. Logistic Regression — used as a baseline model.
2. LightGBM — selected as the primary churn prediction model.

### Training Methodology

- Train-test split: 80:20.
- Stratification: Applied to preserve the churn class distribution.
- Random state: 42.
- Numerical and categorical features are processed using a preprocessing pipeline.
- Missing numerical values are imputed.
- Categorical variables are imputed and one-hot encoded.

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Recall is particularly important because missing a customer who is likely to churn can result in lost business.

### Model Explainability

SHAP (SHapley Additive exPlanations) is used to understand how input features influence model predictions.

The project generates:

- SHAP summary plot
- SHAP feature importance plot

This helps make churn predictions more interpretable.

## 5. Customer Survival Analysis

### Objective

Estimate customer retention patterns and understand how different factors relate to the time until churn.

The project uses the `lifelines` library.

### Kaplan-Meier Estimator

The Kaplan-Meier estimator is used to estimate the probability that a customer remains active over time.

Customers who have not churned during the observed period are treated as censored observations.

Zero-tenure records are excluded from this survival analysis.

### Cox Proportional Hazards Model

The Cox model examines how customer characteristics are associated with the hazard of churn.

Hazard ratios are interpreted as follows:

- Hazard ratio greater than 1: Associated with a higher churn hazard.
- Hazard ratio less than 1: Associated with a lower churn hazard.
- Hazard ratio close to 1: Indicates a relatively small association.

Statistical significance is assessed using p-values.

The model's concordance index is reported as a training-data fit statistic and should not be interpreted as held-out validation performance.

## 6. Customer Lifetime Value and Segmentation

### Objective

Estimate customer value and identify groups with different behavioral and revenue characteristics.

### CLV Estimation

The project calculates a revenue-based customer value proxy using customer tenure and monthly charges.

This provides a simplified estimate of historical customer revenue.

**Important:** This is not a discounted profit-based lifetime value model. It does not directly account for acquisition costs, service costs, discount rates, or future retention probabilities.

### Customer Segmentation

K-Means clustering is applied using:

- Tenure
- Monthly charges
- Service count

Features are scaled before clustering.

The resulting customer segments are compared using:

- Customer count
- Average tenure
- Churn rate
- Average historical revenue

This helps identify higher-risk and higher-value customer groups.

## 7. Uplift Modeling

### Objective

Estimate which customers may respond positively to a marketing treatment, rather than simply predicting who is likely to purchase.

### Dataset

Uplift modeling uses the separate Hillstrom MineThatData campaign dataset.

The experiment compares:

- Women's Email: Treatment group.
- No Email: Control group.

The Men's Email group is excluded from this experiment.

The filtered experiment population contains 42,693 customers.

### Modeling Approach

A Causal Random Forest model from the CausalML library is trained using customer and campaign-related features.

The dataset is split into training and test sets using a 75:25 split with random state 42.

### Evaluation

The model is evaluated using:

- Qini score
- AUUC score
- Observed treatment-control conversion difference
- Mean predicted uplift

Qini and AUUC evaluate how effectively the model ranks customers according to estimated treatment responsiveness.

**Important limitations:**

- Uplift predictions are estimates, not guarantees of individual customer behavior.
- Qini and AUUC are ranking metrics, not classification accuracy.
- The observed treatment-control difference is a group-level experimental result.
- Hillstrom uplift results should not be interpreted as measured causal effects on the Telco dataset.

## 8. Retention ROI Simulation

The Streamlit dashboard includes a retention ROI simulator.

Users can adjust assumptions such as:

- Monthly discount amount
- Assumed retention effectiveness
- Churn probability threshold

The simulator estimates:

- Number of targeted customers
- Expected saved revenue
- Retention offer cost
- Net benefit
- Estimated ROI

The results are scenario-based estimates. They depend on user-defined assumptions and are not proof of actual causal retention impact.

## 9. Application Architecture

The project follows a modular architecture:

1. **Data layer:** Raw and processed customer datasets.
2. **Preprocessing layer:** Data cleaning and feature engineering.
3. **Storage layer:** PostgreSQL tables for cleaned and engineered data.
4. **Modeling layer:** Churn classification, survival analysis, CLV, clustering, and uplift modeling.
5. **Explainability layer:** SHAP-based model interpretation.
6. **Application layer:** Streamlit dashboard for customer exploration, model insights, and retention ROI analysis.

## 10. Tools and Technologies

| Component | Technology |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Database | PostgreSQL |
| Machine learning | Scikit-learn, LightGBM |
| Explainable AI | SHAP |
| Survival analysis | Lifelines |
| Uplift modeling | CausalML |
| Customer segmentation | K-Means |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Testing | Pytest |
| Version control | Git and GitHub |

## 11. Limitations and Future Improvements

Potential improvements include:

- Validating churn and survival models on additional datasets.
- Developing a more comprehensive predictive CLV model.
- Connecting the dashboard directly to PostgreSQL.
- Evaluating uplift models on additional randomized experiments.
- Adding model monitoring and retraining workflows.
- Deploying the application to a production environment.

## Conclusion

The methodology combines predictive, descriptive, and causal-analysis techniques to support customer retention decisions. By integrating churn prediction, survival analysis, customer value estimation, segmentation, explainability, and uplift modeling, the system provides multiple perspectives on customer behavior and retention opportunities.