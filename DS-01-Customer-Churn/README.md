# Customer Lifetime Value & Churn Intelligence System

An end-to-end data science project that combines customer churn prediction, survival analysis, customer lifetime value estimation, customer segmentation, explainable machine learning, and uplift modeling to support data-driven customer retention decisions.

## Project Overview

Customer churn directly affects recurring revenue and long-term customer relationships. This project builds a customer intelligence system to help businesses:

- Identify customers with a high probability of churning.
- Understand the factors associated with customer churn.
- Estimate customer value and identify valuable customer segments.
- Analyze customer retention patterns over time.
- Evaluate which customers may respond positively to retention campaigns.
- Estimate the potential financial impact of retention offers through an ROI simulator.

The project uses the IBM Telco Customer Churn dataset for churn, survival, and CLV analysis, and the Hillstrom campaign dataset for a separate uplift-modeling experiment.

## Key Features

### 1. Data Cleaning and Feature Engineering
- Processes the IBM Telco Customer Churn dataset.
- Handles missing and inconsistent values.
- Encodes the churn target.
- Creates customer service indicators and tenure-based features.
- Stores cleaned and engineered datasets.

### 2. Customer Churn Prediction
- Logistic Regression baseline.
- LightGBM classification pipeline.
- Stratified train-test split.
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- SHAP-based model explainability.

### 3. Survival Analysis
- Kaplan-Meier survival analysis.
- Cox Proportional Hazards model.
- Hazard-ratio analysis to understand factors associated with customer retention and churn.

### 4. Customer Lifetime Value and Segmentation
- Revenue-based CLV proxy estimation.
- K-Means customer segmentation.
- Customer profiles based on tenure, monthly charges, and service usage.
- Segment-level churn and revenue analysis.

### 5. Uplift Modeling
- Uses the Hillstrom MineThatData campaign dataset.
- Compares Women's Email treatment with the No Email control group.
- Uses a Causal Random Forest model.
- Evaluates uplift ranking with Qini and AUUC metrics.
- Estimates treatment effects on a held-out test set.

**Important:** Uplift modeling uses a separate campaign dataset. Its results are not trained on or directly transferable to the Telco dataset.

### 6. Interactive Streamlit Dashboard
The dashboard includes five sections:

- **Overview:** Customer and churn summary.
- **Customer Explorer:** Individual customer churn-risk predictions.
- **Retention ROI:** Scenario-based retention offer and financial impact simulation.
- **Model Explainability:** SHAP and survival-analysis visualizations.
- **Uplift Modeling:** Campaign uplift metrics and results.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, LightGBM |
| Explainable AI | SHAP |
| Survival analysis | Lifelines |
| Uplift modeling | CausalML, Scikit-uplift |
| Database | PostgreSQL, SQLAlchemy |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Testing | Pytest |
| Version control | Git, GitHub |

## Project Results

### Churn Model — LightGBM

Evaluation on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.7551 |
| Precision | 0.5263 |
| Recall | 0.7754 |
| F1-score | 0.6270 |
| ROC-AUC | 0.8413 |
| PR-AUC | 0.6537 |

The Logistic Regression baseline achieved a ROC-AUC of 0.8425. LightGBM achieved a higher recall and F1-score, while the baseline had a slightly higher ROC-AUC. Model selection should therefore consider the business cost of missed churners and unnecessary retention offers, rather than accuracy alone.

### Survival Analysis

- Customers analyzed after excluding zero-tenure records: 7,032.
- Observed churn events: 1,869.
- Censored observations: 5,163.
- Cox model concordance: approximately 0.82 on the fitted data.

The reported concordance is an in-sample result, not a held-out validation score. The Kaplan-Meier median survival estimate is infinite because the estimated survival curve did not fall below 50% during the observed follow-up period.

### Customer Segmentation

Four customer segments were identified using K-Means.

| Segment | Customers | Average Tenure | Churn Rate |
|---|---:|---:|---:|
| 0 | 1,557 | 10.27 | 24% |
| 1 | 2,144 | 56.78 | 16% |
| 2 | 2,315 | 14.79 | 47% |
| 3 | 1,027 | 54.56 | 5% |

Segment 2 has the highest observed churn rate, while Segment 3 has the lowest.

### Uplift Modeling — Hillstrom Experiment

Women's Email treatment versus No Email control, evaluated on a held-out test set of 10,674 customers:

| Metric | Result |
|---|---:|
| Normalized Qini score | 0.2366 |
| Normalized AUUC score | 0.7348 |
| Treatment conversion rate | 0.8790% |
| Control conversion rate | 0.5819% |
| Observed treatment-control difference | 0.297 percentage points |
| Mean predicted uplift | 0.319 percentage points |

Qini and AUUC evaluate uplift ranking, not ordinary classification accuracy. Predicted positive uplift does not guarantee that an individual customer will respond to a campaign.

### Retention ROI Simulator

The dashboard provides an adjustable scenario-based ROI calculation. One example scenario uses:

- Monthly discount: $10
- Assumed retention effectiveness: 30%
- Churn-risk threshold: 0.50

The displayed results are scenario estimates based on user-selected assumptions, not measured causal campaign impact.

## Dashboard Preview

The Streamlit application provides an interactive interface for exploring customer risk, model explanations, segmentation, uplift results, and retention scenarios.

Run the dashboard locally:

```bash
streamlit run app/streamlit_app.py
```

## Repository Structure

```text
clv-churn-intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── uplift/
│
├── docs/
│   ├── architecture.md
│   └── data_dictionary.md
│
├── models/
│   ├── uplift/
│   └── saved model artifacts
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── uplift/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── survival/
│   ├── clv/
│   └── uplift/
│
├── tests/
│   └── test_pipeline.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- PostgreSQL (required for the database-loading workflow)
- VS Code or another Python IDE

### 1. Clone the repository

```bash
git clone https://github.com/swati-mishra07/clv-churn-intelligence.git
cd clv-churn-intelligence
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and update the database connection string with your local PostgreSQL credentials.

Do not commit `.env` or expose database credentials.

### 5. Prepare the datasets

Download the IBM Telco Customer Churn dataset and place the CSV file at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The Hillstrom dataset is obtained through the project's uplift-data download script.

Raw datasets and generated customer-level prediction files are excluded from Git where appropriate. You may need to regenerate them locally.

## Running the Application

Once the required processed data and model artifacts are available, start the dashboard:

```bash
streamlit run app/streamlit_app.py
```

The current dashboard loads saved processed CSV files and model artifacts from the project directory. PostgreSQL is used in the separate data-loading workflow; the dashboard does not currently query PostgreSQL live.

## Database Workflow

The project includes PostgreSQL integration for storing cleaned and engineered customer data.

Database configuration is read from the `DATABASE_URL` environment variable.

The database-loading script is:

```bash
python -m src.data.load_to_postgres
```

The implemented database tables include:

- `telco_cleaned`
- `telco_features`

## Testing

Run the automated test suite:

```bash
python -m pytest -v
```

The current test suite contains five tests covering:

- Data-cleaning row-count and customer-ID integrity.
- Churn target encoding.
- Missing-value handling for total charges.
- Service-count feature boundaries.
- Feature-engineering output integrity.

Latest recorded test result: **5 tests passed**.

## Documentation

Additional project documentation:

- [Architecture](docs/architecture.md)
- [Data Dictionary](docs/data_dictionary.md)

## Limitations and Future Improvements

- The CLV calculation is a revenue-based proxy, not a discounted profit-based lifetime value model.
- The churn model's performance may change on new or real-world customer data.
- Survival-model concordance reported here is based on fitted data and requires independent validation.
- Uplift modeling uses the separate Hillstrom experimental dataset.
- Retention ROI results depend on scenario assumptions and should not be interpreted as proven campaign outcomes.
- The dashboard currently reads local saved artifacts rather than querying PostgreSQL in real time.

Potential future improvements include:

- Time-based and external validation.
- Profit-based CLV estimation.
- Model monitoring and drift detection.
- Live database integration.
- Deployment and automated model retraining.
- Validation of retention strategies through controlled experiments.


## Project Demo

Watch the recorded walkthrough of the Customer Lifetime Value and Churn Intelligence System:

[**Watch Project Demo**](https://drive.google.com/file/d/1PiJ29u2eYRAuuduipAquVYj6PpUyGovo/view?usp=sharing)

The demonstration covers the project overview, customer churn predictions, retention ROI, model explainability, survival analysis, and uplift modelling.



## Author

**Swati Mishra**

B.Tech — Information Technology  
GitHub: https://github.com/swati-mishra07

---

*Developed as a data science internship project focused on customer analytics, predictive modeling, explainable AI, survival analysis, and retention decision support.*