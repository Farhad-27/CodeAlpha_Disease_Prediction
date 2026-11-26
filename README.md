# Disease Prediction from Medical Data (Diabetes Classification)

**Task 4 — CodeAlpha Machine Learning Internship**

An end-to-end machine learning classification pipeline built to predict diabetes risk from structured clinical and diagnostic measurements. The project covers data cleaning, biologically informed missing-value handling, stratified train-test splitting, feature scaling, multi-model training, and evaluation using classification metrics and confusion matrix visualization.

---

## Project Overview

Early identification of diabetes helps support clinical screening and risk assessment. In this project, tabular patient health records are preprocessed and evaluated across multiple supervised classification algorithms:

* **Logistic Regression** (with standardized features)
* **Random Forest Classifier**
* **XGBoost Classifier**

---

## Dataset & Preprocessing

* **Dataset:** Pima Indians Diabetes Database (`diabetes.csv`) — structured medical diagnostic attributes (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) and binary target `Outcome` (`0` = Non-Diabetic, `1` = Diabetic).
* **Zero-Value Imputation:** Clinical measurements such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` cannot validly be `0` for living patients; zeros in these columns are treated as missing values (`NaN`) and imputed with the median.
* **Train / Test Split:** Stratified `80%` training and `20%` testing split (`random_state=42`) to preserve class balance.
* **Feature Scaling:** `StandardScaler` applied to training and test splits for scale-sensitive models.

---

## Evaluation Metrics

Models are compared using standard classification metrics suited for medical screening tasks:

* **Accuracy**
* **Precision**
* **Recall** (key for catching potential positive cases)
* **F1-Score**
* **ROC-AUC**
* **Confusion Matrix & Classification Report** for the top-performing model

---

## Repository Structure

```text
CodeAlpha_Disease_Prediction/
├── Disease_Prediction.ipynb   # Google Colab notebook (data setup, preprocessing, training, evaluation)
├── data/                      # Dataset files (if included locally)
└── README.md                  # Project documentation
