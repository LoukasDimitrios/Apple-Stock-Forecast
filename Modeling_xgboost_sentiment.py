# Modeling_xgboost_sentiment.py

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
# Ensure directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load data ---
df = pd.read_csv("data/apple_stock_data_enriched_with_sentiment.csv", index_col="Date", parse_dates=True)

# --- Define features ---
features = [
    'Lag_7', 'Rolling_Return_5', 'Lag_8', 'Lag_6', 'RSI_14', 'DayOfWeek',
    'Sentiment_Mean', 'News_Count'
]
target_col = 'Target_3d'

# --- Create target ---
df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
df['Target_3d'] = (df['Return_3d'] > 0.01).astype(int)
df = df.dropna(subset=features + [target_col])

# --- Train/Test split ---
split_idx = int(len(df) * 0.8)
df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

X_train = df_train[features]
y_train = df_train[target_col]
X_test = df_test[features]
y_test = df_test[target_col]

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler_xgb_sentiment.pkl")

# --- Fit model ---
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# --- Predict ---
y_pred = xgb_model.predict(X_test_scaled)
y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

# --- Evaluation ---
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model and results ---
joblib.dump(xgb_model, "models/xgb_model_sentiment.pkl")

np.save("results/y_true_xgb_sent.npy", y_test)
# np.save("results/preds_xgb_sent.npy", y_pred)
np.save("results/probs_xgb_sent.npy", y_prob)

metrics = {
    "Model": "XGBoost + Sentiment",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob)
}

pd.DataFrame([metrics]).to_csv("results/xgb_sent_results.csv", index=False)

