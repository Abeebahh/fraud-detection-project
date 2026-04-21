# =========================
# FRAUD DETECTION PROJECT
# CLEAN WORKING VERSION
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    classification_report   # ✅ FIX ADDED HERE
)

from imblearn.over_sampling import SMOTE

# -------------------------
# STEP 1: LOAD DATA
# -------------------------
data = pd.read_csv("data/credit_card_fraud_10k.csv")

print("\n--- FIRST 5 ROWS ---")
print(data.head())

print("\n--- INFO ---")
print(data.info())

print("\n--- SUMMARY ---")
print(data.describe())

print("\n--- TARGET DISTRIBUTION ---")
print(data["is_fraud"].value_counts())


# -------------------------
# STEP 2: ONE-HOT ENCODING
# (Fixes 'Food' string error)
# -------------------------
data = pd.get_dummies(data, columns=["merchant_category"])

print("\n--- AFTER ENCODING ---")
print(data.dtypes)


# -------------------------
# STEP 3: SPLIT FEATURES & TARGET
# -------------------------
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split successful")


# -------------------------
# STEP 4: APPLY SMOTE (IMPORTANT: BEFORE SCALING)
# -------------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())


# -------------------------
# STEP 5: FEATURE SCALING (CLEAN + PROFESSIONAL)
# -------------------------
from sklearn.preprocessing import StandardScaler

# Save feature names BEFORE scaling
feature_names = X.columns

scaler = StandardScaler()

# Scale training data
X_train_resampled = scaler.fit_transform(X_train_resampled)

# Scale test data
X_test = scaler.transform(X_test)

# Convert back to DataFrame (keeps column names)
import pandas as pd

X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)


# -------------------------
# STEP 6: MODEL TRAINING
# -------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_resampled, y_train_resampled)

print("\nModel training completed")


# -------------------------
# STEP 7: PREDICTIONS
# -------------------------
y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.4
y_pred_threshold = (y_proba >= threshold).astype(int)


# =========================
# STEP 8: EVALUATION + PLOTS
# =========================

import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score
)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)


# -------------------------
# SAVE FUNCTION (IMPORTANT)
# -------------------------
def save_plot(filename):
    path = os.path.join("outputs", filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


# -------------------------
# 1. CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, y_pred_threshold)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

save_plot("confusion_matrix.png")


# -------------------------
# 2. PRECISION-RECALL CURVE
# -------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

save_plot("pr_curve.png")


# -------------------------
# 3. ROC CURVE
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

save_plot("roc_curve.png")


# -------------------------
# 4. FEATURE IMPORTANCE
# -------------------------
import numpy as np

importances = model.feature_importances_
features = X.columns

indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.title("Feature Importance")

save_plot("feature_importance.png")


print("\nAll plots saved to /outputs folder successfully")

# -------------------------
# STEP 9: OPTIMAL THRESHOLD (TOP 5% UPGRADE)
# -------------------------
from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Avoid division by zero
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\n--- OPTIMAL THRESHOLD ---")
print("Best Threshold:", best_threshold)


# -------------------------
# STEP 10: APPLY OPTIMAL THRESHOLD
# -------------------------
y_pred_opt = (y_proba >= best_threshold).astype(int)

print("\n--- CONFUSION MATRIX (OPTIMIZED) ---")
print(confusion_matrix(y_test, y_pred_opt))

print("\n--- CLASSIFICATION REPORT (OPTIMIZED) ---")
print(classification_report(y_test, y_pred_opt))

# -------------------------
# STEP 11: ROC-AUC SCORE
# -------------------------
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_proba)

print("\n--- ROC-AUC SCORE ---")
print("ROC-AUC:", auc)

# -------------------------
# STEP 12: PRECISION-RECALL CURVE
# -------------------------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# -------------------------
# STEP 13: BUSINESS COST SIMULATION
# -------------------------
false_neg_cost = 500   # missing fraud = expensive
false_pos_cost = 10    # false alarm = cheaper

cm = confusion_matrix(y_test, y_pred_opt)

tn, fp, fn, tp = cm.ravel()

total_cost = (fn * false_neg_cost) + (fp * false_pos_cost)

print("\n--- BUSINESS COST ---")
print("False Negatives:", fn)
print("False Positives:", fp)
print("Estimated Cost:", total_cost)

# -------------------------
# STEP 14: FEATURE IMPORTANCE
# -------------------------
import pandas as pd

# Get feature names after encoding
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance_df.head(10))