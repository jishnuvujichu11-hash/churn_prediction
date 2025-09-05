# Customer Churn Prediction

Predict and analyze customer churn in the telecom domain using machine learning. This project includes **data preprocessing, modeling, evaluation, feature importance, and churn prediction for new customers**.

---

## Repository Structure

```
customer-churn-project/
├── data/
│   ├── raw/            # Original dataset (Telco-Customer-Churn.csv)
│   └── processed/      # Cleaned/encoded datasets, scored customers
├── notebooks/
│   ├── churn_eda.ipynb
│   ├── preprocessing.ipynb
│   
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── artifacts/
│   ├── best_model.pkl
│   └── metrics.json
├── reports/
│   ├── figures/
│ 
└── README.md
```

---

## Environment Setup

**Using Conda:**

```bash
conda create -n churn python=3.10 -y
conda activate churn
pip install -r requirements.txt
```

**Using venv (Windows-friendly):**

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```


---

## Data Overview

- **Dataset:** Telco Customer Churn
- **Target:** `Churn` (Yes=1 / No=0)
- **Key columns:** `Contract`, `tenure`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `DeviceProtection`
- **Size:** 7,032 customers
- **Missing values:** Handled in `TotalCharges` (converted to numeric, blanks as NaN)

---

## Exploratory Data Analysis

- Month-to-month contracts have the highest churn (~43%).
- Fiber optic internet users churn more (~42%).
- Electronic check users churn highest (~45%).
- Short-tenure and high-paying customers are at higher churn risk.

Plots saved in `reports/figures/`.

---

## Data Preprocessing

- Numeric: Median imputation
- Categorical: Most frequent imputation + One-Hot Encoding
- Pipeline: `ColumnTransformer` combining numeric & categorical preprocessing
- Train/Test Split: 80/20 stratified

---

## Modeling

- Models: Logistic Regression, Random Forest (best)
- Hyperparameter tuning: RandomizedSearchCV for RF & LR
- Evaluation metric: ROC-AUC (target ≥ 0.80)

---

## Evaluation

- ROC-AUC: Random Forest best model
- Confusion Matrix saved in `reports/figures/roc_curve.png`
- Feature Importance:
  - Top drivers: `tenure`, `Contract_Month-to-month`, `InternetService_Fiber optic`, `OnlineSecurity_No`, `PaymentMethod_Electronic check`, `MonthlyCharges`

---

## Insights

- **High-risk segment:** Month-to-month + tenure < 6 months → ~46% churn
- **Pricing sensitivity:** Customers paying > ₹78/month have ~37% churn
- **Recommendations:**
  1. Offer discounts or add-ons for new customers.
  2. Encourage annual/2-year contracts for high-risk users.
  3. Proactively target top-value customers at risk.

---

## Predicting Churn for New Customers

```python
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 425.50
}])

churn_prob = best_model.predict_proba(new_customer)[:,1]
churn_label = best_model.predict(new_customer)
```

- `churn_prob` → probability of churn
- `churn_label` → 0 = No, 1 = Yes

---

## Artifacts

- `artifacts/best_model.pkl` → trained model
- `reports/figures/` → plots (churn rate, ROC, feature importance)

---

## Usage

1. Clone repo
2. Install dependencies
3. Run `preprocessing.ipynb` to train and evaluate
4. Use `best_model.pkl` to predict churn for new customers

