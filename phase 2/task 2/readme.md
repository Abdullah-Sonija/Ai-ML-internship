# Telco Customer Churn Prediction Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict customer churn using demographic, account, and services data. The pipeline handles data cleaning, automated preprocessing, model training, hyperparameter tuning, and serialization of the best-performing model.

---

##  Directory Structure

```text
phase 2/task 2/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw customer dataset
├── models/
│   └── churn_pipeline.pkl                     # Serialized best-performing pipeline
├── churn_pipeline.ipynb                        # Step-by-step Jupyter Notebook
└── README.md                                  # Project documentation
```

---

##  Dataset Overview

The dataset used is the popular **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing **7,043 rows** and **21 columns**.

### Target Variable
* **`Churn`**: Indicates whether the customer left within the last month (`Yes` / `No` mapped to `1` / `0`).

### Key Features
* **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Info**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

##  Pipeline Architecture & Workflow

### 1. Data Cleaning
* Drops unique identifier column `customerID`.
* Coerces `TotalCharges` to numeric values, automatically handling any empty strings by converting them to `NaN`.
* Maps the target column `Churn` (`Yes` -> `1`, `No` -> `0`).

### 2. Preprocessing & Feature Engineering
We utilize `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline` to prevent data leakage during cross-validation:
* **Numerical Pipeline**:
  * Imputation of missing values using the `median` strategy.
  * One-hot encoding values to capture high-cardinality non-linear patterns.
* **Categorical Pipeline**:
  * Imputation of missing values using the `most_frequent` strategy.
  * Robust `OneHotEncoder(handle_unknown="ignore")` to process new categories safely.

### 3. Model Training & Comparison
We split the data into **80% training** and **20% testing** sets, stratified by the `Churn` label to handle class imbalance. We then evaluate and fine-tune two distinct architectures using `GridSearchCV`:

#### A. Logistic Regression
* **Grid Parameters**: `C: [0.01, 0.1, 1, 10]`, `solver: ["liblinear"]`
* **Best Parameters**: `{'C': 0.1, 'solver': 'liblinear'}`
* **Performance**:
  * **Accuracy**: `79.91%`
  * **Precision (Class 1)**: `0.65`
  * **Recall (Class 1)**: `0.53`
  * **F1-Score (Class 1)**: `0.59`

#### B. Random Forest Classifier
* **Grid Parameters**: `n_estimators: [100, 200]`, `max_depth: [5, 10, None]`
* **Best Parameters**: `{'max_depth': None, 'n_estimators': 200}`
* **Performance**:
  * **Accuracy**: `78.57%`
  * **Precision (Class 1)**: `0.64`
  * **Recall (Class 1)**: `0.45`
  * **F1-Score (Class 1)**: `0.53`

---

##  Model Serialization & Deployment

The best-performing estimator from the **Random Forest** pipeline is serialized and saved as a production-ready artifact:
* **Output Path**: `models/churn_pipeline.pkl`
* **Technology**: `joblib`

This saved pipeline contains both the **preprocessing steps** and the **trained classifier**, allowing you to perform predictions directly on raw, unprocessed input data without manual pre-transformation.

### How to Load & Use the Pipeline

```python
import pandas as pd
import joblib

# 1. Load the serialized pipeline
pipeline = joblib.load("models/churn_pipeline.pkl")

# 2. Prepare new raw data (retaining exact original columns)
new_customer_data = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0
}])

# 3. Predict churn directly (returns 0 or 1)
prediction = pipeline.predict(new_customer_data)
probability = pipeline.predict_proba(new_customer_data)[:, 1]

print(f"Churn Prediction: {prediction[0]} (Probability: {probability[0]:.2%})")
```

---

##  Prerequisites & Setup

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scikit-learn joblib
```

To run the pipeline and explore the notebook:
```bash
jupyter notebook churn_pipeline.ipynb
```
