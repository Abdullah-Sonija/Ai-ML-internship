# Machine Learning and Data Analysis Project

This repository contains three distinct tasks encompassing Exploratory Data Analysis (EDA), Time-Series Regression, and Classification. Below is a summary of the objectives, datasets, models, and findings for each task.

---

##  Task 1: Exploratory Data Analysis (EDA)
**File:** `task1.ipynb`

* **Task Objective:** Perform initial data exploration, understand data structures, and visualize the pairwise relationships between different features.
* **Dataset Used:** [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (loaded directly via the Seaborn library).
* **Models Applied:** None (Purely Exploratory Data Analysis).
* **Key Results and Findings:**
  * Successfully loaded and inspected the dataset, revealing a shape of 150 rows and 5 columns (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species`).
  * Generated statistical summaries to understand feature distributions.
  * Created pairplots using Seaborn to visually identify class separability and correlations between sepal and petal measurements across different Iris species.

---

##  Task 2: Stock Price Prediction (Short-Term)
**File:** `task2.ipynb`

* **Task Objective:** Formulate a regression problem to predict the next day's closing price for Apple's stock using historical market data. 
* **Dataset Used:** Apple Inc. (AAPL) stock market history for the past 2 years, fetched dynamically using the `yfinance` API.
* **Models Applied:** * Linear Regression
  * Random Forest Regressor
* **Key Results and Findings:**
  * Extracted and engineered features including `Open`, `Close`, `High`, `Low`, and `Volume`.
  * Created a shifted target variable representing the `target_next_close` price.
  * Split the chronological data into training and testing sets (80/20 split without shuffling to preserve time-series integrity).
  * Generated visualizations comparing the Actual Next Day Close Price against the Predicted Next Day Close Price (specifically visualizing the Linear Regression model's performance), establishing a baseline for short-term forecasting.

---

##  Task 3: Heart Disease Classification
**File:** `task3.ipynb`

* **Task Objective:** Perform robust data cleaning, handle missing values, and build a predictive model to classify the presence or absence of heart disease in patients.
* **Dataset Used:** [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) (loaded directly via the Kaggle API).
* **Models Applied:** * Logistic Regression
  * Decision Tree Classifier
* **Key Results and Findings:**
  * Discovered and handled missing values (replaced '?' with `NaN`, followed by median imputation for numerical features like `trestbps`, `chol`, `thalch`, `oldpeak`, and mode imputation for categorical features like `fbs`, `exang`, `slope`, `thal`).
  * Engineered a binary classification `target` variable from the original `num` column (mapping >0 to 1 for heart disease, and 0 to 0 for no disease).
  * Set up the foundation to evaluate the models using robust metrics, including Accuracy Score, Confusion Matrix, and ROC-AUC curves.
