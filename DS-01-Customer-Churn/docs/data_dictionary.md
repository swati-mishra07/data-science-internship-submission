
# Data Dictionary

## 1. IBM Telco Customer Churn Dataset

The source dataset contains 7,043 customer records and 21 original columns.

| Column | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Customer gender category |
| SeniorCitizen | Senior citizen indicator |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months with the company |
| PhoneService | Phone service subscription |
| MultipleLines | Multiple telephone lines |
| InternetService | Internet service type |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Contract type |
| PaperlessBilling | Paperless billing indicator |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly customer charges |
| TotalCharges | Total customer charges |
| Churn | Whether the customer left the company |

## 2. Engineered Features

| Feature | Description |
|---|---|
| ChurnFlag | Numeric churn target (0 or 1) |
| service_count | Number of subscribed services |
| tenure_group | Customer tenure category |
| is_month_to_month | Indicator for month-to-month contract |
| Service flags | Binary indicators derived from service columns |

## 3. Hillstrom Uplift Dataset

The uplift experiment uses the Hillstrom MineThatData email campaign dataset.

| Feature | Description |
|---|---|
| recency | Days since the customer's last purchase |
| history_segment | Customer historical spending category |
| history | Historical purchase amount |
| mens | Previous men's merchandise purchase indicator |
| womens | Previous women's merchandise purchase indicator |
| zip_code | Customer geographic category |
| newbie | New customer indicator |
| channel | Customer acquisition channel |
| treatment | Email campaign assignment |
| conversion | Purchase conversion outcome |

The experiment compares Women's Email against No Email. Men's Email customers are excluded from this comparison.

## 4. Generated Outputs

- `telco_cleaned.csv`: cleaned Telco customer data.
- `telco_features.csv`: engineered Telco features.
- `customer_intelligence.csv`: customer-level CLV proxy and segmentation output.
- `churn_metrics.json`: churn model evaluation metrics.
- `cox_summary.csv`: Cox model coefficients and statistics.
- `uplift_metrics.json`: uplift experiment evaluation metrics.

## Data Handling

Datasets and customer-level prediction exports are excluded from the Git repository. Users should obtain the source datasets and run the documented pipeline to regenerate outputs.
