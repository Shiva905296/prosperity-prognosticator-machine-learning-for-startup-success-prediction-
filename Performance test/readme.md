"""
Performance Test Script
Prosperity Prognosticator: Startup Success Prediction

Tests:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Prediction Speed
"""

import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load model
model = joblib.load("model.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# -------------------------
# Prediction Time Test
# -------------------------

start_time = time.time()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

end_time = time.time()

prediction_time = end_time - start_time

# -------------------------
# Performance Metrics
# -------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# -------------------------
# Results
# -------------------------

print("\n📊 MODEL PERFORMANCE REPORT")
print("="*40)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

print("\n⚡ PERFORMANCE")

print(f"Prediction Time: {prediction_time:.4f} seconds")
print(f"Samples Tested : {len(X_test)}")
print(f"Time per Sample: {(prediction_time/len(X_test)):.6f} sec")

print("\n✅ Test Complete")
