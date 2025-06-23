#!/usr/bin/env python
# coding: utf-8

# --- Libraries ---
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings("ignore")
# --- Ensure folders ---
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data_enriched.csv", index_col='Date', parse_dates=True)
df = df.dropna()

# --- Create 3-day forward return & binary target ---
df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
df['Target_3d'] = (df['Return_3d'] > 0.01).astype(int)
df = df.dropna(subset=['Return_3d', 'Target_3d'])

# --- Feature set (10 enriched features) ---
features = [
    'Lag_7', 'Rolling_Return_5', 'Lag_8', 'RSI_14', 'MACD', 'MACD_Signal',
    'MA_Cross', 'Price_Above_MA_20', 'Volume_Diff', 'DayOfWeek'
]

X = df[features]
y = df['Target_3d']

# --- Train/test split (no shuffle) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Initialize and fit model ---
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# --- Evaluation ---
print("\n[XGB 3d] Evaluation (Threshold = 0.5):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model ---
joblib.dump(xgb_model, "models/xgb_model_enriched.pkl")

# --- Save predictions and evaluation ---
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# Save arrays
# np.save("results/preds_xgb.npy", y_pred)
np.save("results/y_true_xgb_enriched.npy", y_test.to_numpy())
np.save("results/probs_xgb_enriched.npy", y_probs)


# Save metrics
metrics = {
    "Model": "XGBoost (enriched)",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_probs)
}

df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv("results/xgb_enriched_results.csv", index=False)

