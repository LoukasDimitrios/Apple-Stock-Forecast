#!/usr/bin/env python
# coding: utf-8

# --- Libraries ---
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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

# --- Define full feature set ---
features = [
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7', 'Lag_8', 'Lag_9', 'Lag_10',
    'Rolling_Return_5', 'Rolling_Return_10',
    'Volatility_20', 'Volatility_50',
    'RSI_14', 'MACD', 'MACD_Signal',
    'MA_Cross', 'Price_Above_MA_20', 'Price_Above_MA_50',
    'Volume_Avg_20', 'Volume_Diff',
    'Month', 'DayOfWeek',
    'Intraday_Range'
]

# --- Version 1: 3-day target ---
df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
df['Target_3d'] = (df['Return_3d'] > 0.01).astype(int)
df_3d = df.dropna(subset=['Target_3d'])

X = df_3d[features]
y = df_3d['Target_3d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model_3d = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

xgb_model_3d.fit(X_train, y_train)
y_pred_3d = xgb_model_3d.predict(X_test)
y_prob_3d = xgb_model_3d.predict_proba(X_test)[:, 1]

print("\n[XGB - Target_3d] Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_3d))
print("Classification Report:\n", classification_report(y_test, y_pred_3d))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_3d))

# Save model and results
joblib.dump(xgb_model_3d, "models/xgb_model_3d_enriched_all_f_thresh05.pkl")
np.save("results/y_true_xgb_all_3d.npy", y_test.to_numpy())
# np.save("results/preds_xgb_all_3d.npy", y_pred_3d)
np.save("results/probs_xgb_all_3d.npy", y_prob_3d)

metrics_3d = {
    "Model": "XGBoost (all features, 3d)",
    "Accuracy": accuracy_score(y_test, y_pred_3d),
    "Precision": precision_score(y_test, y_pred_3d),
    "Recall": recall_score(y_test, y_pred_3d),
    "F1": f1_score(y_test, y_pred_3d),
    "AUC": roc_auc_score(y_test, y_prob_3d)
}

# --- Version 2: 5-day target ---
df['Return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
df['Target_5d'] = (df['Return_5d'] > 0.02).astype(int)
df_5d = df.dropna(subset=['Target_5d'])

X = df_5d[features]
y = df_5d['Target_5d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model_5d = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

xgb_model_5d.fit(X_train, y_train)
y_pred_5d = xgb_model_5d.predict(X_test)
y_prob_5d = xgb_model_5d.predict_proba(X_test)[:, 1]

print("\n[XGB - Target_5d] Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_5d))
print("Classification Report:\n", classification_report(y_test, y_pred_5d))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_5d))

# Save model and results
joblib.dump(xgb_model_5d, "models/xgb_model_5d_enriched_all_f_thresh05.pkl")
np.save("results/y_true_xgb_all_5d.npy", y_test.to_numpy())
# np.save("results/preds_xgb_all_5d.npy", y_pred_5d)
np.save("results/probs_xgb_all_5d.npy", y_prob_5d)

metrics_5d = {
    "Model": "XGBoost (all features, 5d)",
    "Accuracy": accuracy_score(y_test, y_pred_5d),
    "Precision": precision_score(y_test, y_pred_5d),
    "Recall": recall_score(y_test, y_pred_5d),
    "F1": f1_score(y_test, y_pred_5d),
    "AUC": roc_auc_score(y_test, y_prob_5d)
}

# --- Save all metrics ---
df_metrics = pd.DataFrame([metrics_3d, metrics_5d])
df_metrics.to_csv("results/xgb_all_features_results.csv", index=False)

