# Model Evaluation and Results

## 1. Overview

This document summarizes the evaluation results of the Customer Lifetime Value and Churn Intelligence System.

The project evaluates multiple analytical components:

1. Customer churn classification.
2. Customer survival analysis.
3. Customer lifetime value and segmentation.
4. Marketing uplift modeling.

Each component has a different objective and therefore uses different evaluation methods.

## 2. Churn Classification

### Dataset and Split

- Dataset: IBM Telco Customer Churn.
- Total observations: 7,043.
- Training/testing split: 80:20.
- Stratified split: Yes.
- Random state: 42.
- Models: Logistic Regression and LightGBM.

### Model Comparison

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| Accuracy | 0.7331 | 0.7551 |
| Precision | 0.4983 | 0.5263 |
| Recall | 0.7941 | 0.7754 |
| F1-score | 0.6124 | 0.6270 |
| ROC-AUC | 0.8425 | 0.8413 |
| PR-AUC | 0.6352 | 0.6537 |

### Interpretation

LightGBM achieved:

- Higher accuracy than Logistic Regression.
- Higher precision.
- A slightly higher F1-score.
- Higher PR-AUC.

Logistic Regression achieved slightly higher recall and ROC-AUC.

LightGBM was selected as the primary churn model because it provides a stronger overall balance across accuracy, precision, F1-score, and PR-AUC.

The recall difference should still be considered when selecting a business decision threshold. A retention team may prefer higher recall if identifying more potential churners is more important than minimizing false positives.

### SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to examine feature contributions to model predictions.

Generated outputs:

- `reports/figures/shap_summary.png`
- `reports/figures/shap_importance.png`

These visualizations help identify which features influence churn predictions.

## 3. Survival Analysis

### Kaplan-Meier Analysis

- Records used: 7,032.
- Observed churn events: 1,869.
- Censored observations: 5,163.

The Kaplan-Meier estimator provides an estimate of customer survival probability over observed tenure.

The estimated median survival time is reported as infinity because the estimated survival curve did not fall below 50% during the observed follow-up period.

This is a valid outcome and does not indicate a model execution error.

### Cox Proportional Hazards Model

The following table presents selected Cox model results:

| Feature | Hazard Ratio | P-value |
|---|---:|---:|
| MonthlyCharges | 0.9943 | 8.91e-10 |
| SeniorCitizen | 1.0706 | 0.1617 |
| Partner_Yes | 0.6180 | 1.02e-29 |
| Dependents_Yes | 0.8179 | 7.13e-05 |
| Contract_One year | 0.3495 | 9.15e-84 |
| Contract_Two year | 0.2126 | 1.78e-152 |
| InternetService_Fiber optic | 1.6681 | 2.83e-24 |
| InternetService_No | 0.6494 | 1.45e-10 |
| PaperlessBilling_Yes | 1.2252 | 5.19e-06 |

### Interpretation

- One-year and two-year contracts are associated with substantially lower churn hazards than the reference contract category.
- Partner_Yes and Dependents_Yes are associated with lower churn hazards.
- Fiber-optic internet service is associated with a higher churn hazard than the reference category.
- PaperlessBilling_Yes is associated with a higher churn hazard.
- SeniorCitizen has a hazard ratio above 1, but its p-value (0.1617) is greater than 0.05, so its coefficient is not statistically significant at the 5% level.
- MonthlyCharges has a statistically significant coefficient, although its hazard ratio is close to 1, indicating a small effect per unit increase.

The reported concordance index is approximately 0.82 on the fitted data. It is a training-data fit statistic, not an independent test-set performance result.

Hazard ratios describe associations under the Cox model; they should not automatically be interpreted as causal effects.

## 4. Customer Lifetime Value and Segmentation

### CLV Method

Customer value is estimated using a revenue-based proxy derived from tenure and monthly charges.

It represents simplified historical customer revenue rather than discounted future profit.

### Customer Segments

| Segment | Customers | Average Tenure | Churn Rate | Average Historical Revenue |
|---|---:|---:|---:|---:|
| 0 | 1,557 | 10.27 | 24% | 289.08 |
| 1 | 2,144 | 56.78 | 16% | 5,224.82 |
| 2 | 2,315 | 14.79 | 47% | 1,139.31 |
| 3 | 1,027 | 54.56 | 5% | 1,720.10 |

### Interpretation

- Segment 2 has the highest churn rate at approximately 47%.
- Segment 3 has the lowest churn rate at approximately 5%.
- Segment 1 has the highest average historical revenue.
- Segment 0 has the lowest average historical revenue.

These segments help prioritize customers based on behavioral and revenue characteristics.

## 5. Uplift Modeling

### Dataset and Experiment

Uplift modeling uses the separate Hillstrom MineThatData campaign dataset.

The experiment compares Women's Email treatment against No Email control. The Men's Email group is excluded.

- Experiment population: 42,693.
- Training observations: 32,019.
- Test observations: 10,674.
- Test treatment group: 5,347.
- Test control group: 5,327.

### Evaluation Results

| Metric | Result |
|---|---:|
| Normalized Qini score | 0.2366 |
| Normalized AUUC score | 0.7348 |
| Treatment conversion rate | 0.8790% |
| Control conversion rate | 0.5819% |
| Observed treatment-control difference | 0.297 percentage points |
| Mean predicted uplift | 0.319 percentage points |
| Customers with positive predicted uplift | 9,704 |
| Positive predicted uplift share | 90.91% |

### Interpretation

The treatment group had a higher observed conversion rate than the control group in the held-out test set.

The normalized Qini and AUUC scores evaluate the model's ability to rank customers by estimated treatment responsiveness.

The model predicts positive uplift for approximately 90.91% of test customers.

However, positive predicted uplift does not guarantee that an individual customer will respond to treatment. The observed conversion difference is a group-level result and should not be interpreted as an individual causal effect.

These uplift results belong to the Hillstrom experiment and are not measured treatment effects on the Telco customer dataset.

## 6. Retention ROI Scenario

The dashboard includes a configurable retention ROI simulator.

One illustrative scenario produced:

| Metric | Result |
|---|---:|
| Monthly discount | $10 |
| Assumed retention effectiveness | 30% |
| Churn probability threshold | 0.50 |
| Target customers | 2,736 |
| Expected saved revenue | $552,244.39 |
| Annual offer cost | $328,320 |
| Estimated net benefit | $223,924.39 |
| Estimated ROI | 68.20% |

These figures are scenario estimates based on dashboard assumptions, not measured campaign outcomes or validated causal effects.

## 7. Testing and Verification

The project includes automated tests covering:

1. Data cleaning and row-count preservation.
2. Unique customer identifiers.
3. Binary churn encoding.
4. Missing-value handling for total charges.
5. Service-count feature bounds.
6. Feature-engineering outputs.

The latest test execution completed successfully:

```text
5 passed in 2.64s
```

Additional checks performed:

- Streamlit application Python compilation.
- Source module compilation.
- Customer Explorer dashboard verification.
- Customer selection and prediction display.
- Customer profile table rendering and updates.

## 8. Evaluation Limitations

The results should be interpreted with the following limitations:

- Churn metrics depend on the selected train-test split and decision threshold.
- The survival concordance index reported here is from fitted data, not held-out validation.
- CLV is a revenue proxy, not a discounted profit-based lifetime value estimate.
- Uplift modeling uses a separate experimental dataset.
- The ROI simulator relies on assumptions rather than observed retention campaign outcomes.
- Further external validation and monitoring would be required before production deployment.

## 9. Conclusion

The evaluation demonstrates that the project integrates several complementary data science methods.

LightGBM provides the primary churn classification model, while SHAP supports interpretation. Survival analysis examines customer retention patterns, K-Means identifies customer segments, and the separate Hillstrom experiment demonstrates uplift modeling.

Together, these components provide an analytical foundation for understanding customer churn, value, and potential retention opportunities.