# Case Study 2: Credit Card Fraud Detection

## 📌 Objective
Detect fraudulent credit card transactions in a **heavily imbalanced** dataset, where fraud typically represents less than 1–2% of all transactions. Using XGBoost with SMOTE oversampling and tuned decision thresholds to build a practical, interpretable fraud detection model.

---

## 🧠 Approach

- **Model:** XGBoost Classifier

- **Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) applied **only to the training set** (never to the test set, to avoid data leakage and unrealistic evaluation).

- **Threshold Tuning:** Instead of using the default 0.5 threshold, the decision threshold is optimized using the Precision-Recall trade-off:
  - F1-maximizing threshold
  - Alternative: threshold constrained to guarantee a minimum recall (e.g. catch ≥90% of fraud)

- **Interpretation:**
  - XGBoost feature importance (weight, gain, cover)
  - SHAP (SHapley Additive exPlanations) for both global feature impact and individual prediction explanations

---

## 📂 Dataset

IEEE-CIS Fraud Detection dataset (Kaggle) or any similar labeled transactions dataset with a binary fraud label.

> Update the CSV path in the notebook (Cell 2) with your actual dataset location before running. The notebook includes both a Kaggle API download option and a direct local CSV upload option — use only one.

---

## 🔑 Key Concepts Demonstrated

- Handling severe class imbalance using SMOTE
- Why accuracy is misleading on imbalanced data — using PR-AUC and F1 instead
- Threshold tuning for business-driven precision/recall trade-offs
- XGBoost feature importance vs. SHAP for model interpretability
- Avoiding data leakage by applying SMOTE only within training data/folds

---

## 🛠️ Requirements

pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap

Notebook is designed to run on **Google Colab** with minimal setup.

---

## 🎯 Key Takeaway

In extreme class imbalance problems like fraud detection, accuracy is meaningless — a model that predicts "no fraud" every time can still be 99% accurate. This case study shows how to properly handle imbalance with SMOTE, evaluate with the right metrics (PR-AUC, F1), tune the decision threshold to match business priorities, and interpret the model's decisions using both feature importance and SHAP.

---

## 🧑‍💻 Author

Add your name/contact/LinkedIn here.
