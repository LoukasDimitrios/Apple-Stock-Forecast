#!/usr/bin/env python
# coding: utf-8

# --- Libraries ---
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import warnings
warnings.filterwarnings("ignore")
# --- Ensure folders ---
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data_features.csv", index_col="Date", parse_dates=True)
df = df.dropna()

# --- Feature set ---
features = [
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
    'Rolling_Return_5', 'Rolling_Return_10',
    'Volatility_20', 'Volatility_50'
]

X = df[features].values
y = df['Target'].values

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- PyTorch Dataset Class ---
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Logistic Regression (PyTorch) ---
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logistic_model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(logistic_model.parameters(), lr=0.001)

n_epochs = 30
for epoch in range(n_epochs):
    logistic_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = logistic_model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"[Logistic] Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
# --- Evaluate Logistic Regression ---
logistic_model.eval()
logistic_preds, logistic_probs, y_true_log = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        batch_probs = logistic_model(X_batch)
        batch_preds = (batch_probs >= 0.5).float()
        logistic_preds.extend(batch_preds.cpu().numpy())
        logistic_probs.extend(batch_probs.cpu().numpy())
        y_true_log.extend(y_batch.cpu().numpy())

torch.save(logistic_model.state_dict(), 'models/logistic_model_baseline.pth')

# --- Random Forest ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
rf_best_model = grid_search.best_estimator_

y_pred_rf = rf_best_model.predict(X_test)
rf_probs = rf_best_model.predict_proba(X_test)[:, 1]
joblib.dump(rf_best_model, 'models/random_forest_baseline.pkl')

# --- MLP (PyTorch) ---
class MLPClassifierModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

mlp_model = MLPClassifierModel(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

n_epochs = 30
for epoch in range(n_epochs):
    mlp_model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = mlp_model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"[MLP] Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
# --- Evaluate MLP ---
mlp_model.eval()
mlp_preds, mlp_probs, y_true_mlp = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        batch_probs = mlp_model(X_batch)
        batch_preds = (batch_probs >= 0.5).float()
        mlp_probs.extend(batch_probs.cpu().numpy())
        mlp_preds.extend(batch_preds.cpu().numpy())
        y_true_mlp.extend(y_batch.cpu().numpy())

torch.save(mlp_model.state_dict(), 'models/mlp_model_baseline.pth')

# --- Save metrics and probabilities ---
results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_true_log, logistic_preds),
        "Precision": precision_score(y_true_log, logistic_preds),
        "Recall": recall_score(y_true_log, logistic_preds),
        "F1": f1_score(y_true_log, logistic_preds),
        "AUC": roc_auc_score(y_true_log, logistic_probs)
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "AUC": roc_auc_score(y_test, rf_probs)
    },
    {
        "Model": "MLP (baseline)",
        "Accuracy": accuracy_score(y_true_mlp, mlp_preds),
        "Precision": precision_score(y_true_mlp, mlp_preds),
        "Recall": recall_score(y_true_mlp, mlp_preds),
        "F1": f1_score(y_true_mlp, mlp_preds),
        "AUC": roc_auc_score(y_true_mlp, mlp_probs)
    }
]

# df_results = pd.DataFrame(results)
# df_results.to_csv("results/baseline_model_results.csv", index=False)

# --- Save individual model results with consistent naming ---

# Logistic Regression
np.save("results/y_true_logistic.npy", np.array(y_true_log).flatten())
np.save("results/probs_logistic.npy", np.array(logistic_probs).flatten())
pd.DataFrame([results[0]]).to_csv("results/logistic_results.csv", index=False)

# Random Forest
np.save("results/y_true_rf.npy", y_test)
np.save("results/probs_rf.npy", rf_probs)
pd.DataFrame([results[1]]).to_csv("results/rf_results.csv", index=False)

# MLP
np.save("results/y_true_mlp.npy", np.array(y_true_mlp).flatten())
np.save("results/probs_mlp.npy", np.array(mlp_probs).flatten())
pd.DataFrame([results[2]]).to_csv("results/mlp_results.csv", index=False)