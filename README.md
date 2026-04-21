# Fraud Detection System (Machine Learning)

<p align="center">
  <img src="assets/banner.png" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" />
</p>

---

<p align="center">
  <img src="assets/banner.png" width="100%">
</p>

<h1 align="center">Fraud Detection System (Machine Learning)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" />
</p>

<p align="center">
A machine learning project focused on identifying fraudulent credit card transactions using structured transactional data and business-oriented evaluation.
</p>

---

## About

This project explores how machine learning can be applied to fraud detection in a realistic setting. The focus is not only on building a predictive model, but on understanding how fraud behaves in imbalanced datasets and how modeling decisions translate into real-world impact.

---

## Objectives

* Detect fraudulent transactions with high sensitivity
* Handle class imbalance effectively
* Move beyond accuracy and focus on meaningful evaluation metrics
* Interpret results in a way that reflects real business trade-offs

---

## Workflow

1. Data loading and inspection
2. Data preprocessing and feature encoding
3. Handling imbalance using SMOTE
4. Model training using Random Forest
5. Threshold optimization
6. Evaluation and business impact analysis

---

## Approach

### Data Preparation

The dataset contains 10,000 transactions with both legitimate and fraudulent activity.
Categorical variables were encoded using one-hot encoding, and non-informative fields such as `transaction_id` were removed.

### Handling Imbalance

Fraud cases are rare relative to normal transactions.
SMOTE was applied to the training data to ensure the model could learn meaningful fraud patterns.

### Model

A Random Forest classifier was used due to its ability to handle structured data and capture non-linear relationships without heavy tuning.

### Evaluation Strategy

The model was evaluated using:

* Precision and recall
* F1-score
* ROC-AUC
* Confusion matrix
* A simple business cost framework

---

## Results

| Metric                | Value |
| --------------------- | ----- |
| Fraud Recall          | 70%   |
| Fraud Precision       | 41%   |
| ROC-AUC               | 0.978 |
| Estimated Cost Impact | 4800  |

---

## Interpretation

The model achieves strong recall, meaning most fraudulent transactions are correctly identified.

Precision is lower, indicating some false positives. This trade-off is expected in fraud detection, where failing to detect fraud is typically more costly than investigating flagged transactions.

---

## Model Performance

### Confusion Matrix

<p align="center">
  <img src="outputs/confusion_matrix.png" width="500"/>
</p>

### Precision-Recall Curve

<p align="center">
  <img src="outputs/pr_curve.png" width="500"/>
</p>

### Feature Importance

<p align="center">
  <img src="outputs/feature_importance.png" width="500"/>
</p>

---

## Key Insights

* Device trust score is the strongest predictor of fraud
* Transaction timing carries meaningful behavioral signals
* High transaction frequency within short periods often indicates risk

---

## Business Perspective

The model is intentionally tuned to prioritize fraud detection over strict precision. This reflects real-world systems where the cost of missing fraudulent activity outweighs the inconvenience of investigating false positives.

---

## Example Output

Running the model produces:

* Confusion matrix
* Classification report
* ROC-AUC score
* Threshold-based predictions
* Estimated business cost

---

## Dataset

## 📂 Dataset

Due to size constraints, the dataset is managed using Git LFS.
You can also use publicly available fraud datasets such as Kaggle.

---

## How to Run

```bash
pip install -r requirements.txt
python src/fraud_model.py
```

---

## Project Structure

```
fraud-detection-project/
│
├── data/          # Dataset (managed via Git LFS)
├── src/           # Model training code
├── outputs/       # Evaluation visuals
├── assets/        # Images and banner
├── notebooks/     # Exploratory analysis
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Compare performance with gradient boosting models such as XGBoost
* Deploy the model as an API
* Build an interactive dashboard for visualization
* Incorporate sequential or time-based features

---

## Author

Olamide Quadri
