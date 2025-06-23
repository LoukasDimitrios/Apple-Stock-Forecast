#!/usr/bin/env python
# coding: utf-8

# --- Libraries ---
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
# --- Ensure folders ---
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data_enriched.csv", index_col="Date", parse_dates=True)
df = df.dropna()

# --- PyTorch Dataset ---
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- MLP architecture ---
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.BCELoss()

# --- Version 1: SHAP-selected features (6) ---
features_v1 = ['Lag_7', 'Rolling_Return_5', 'Lag_8', 'Lag_6', 'RSI_14', 'DayOfWeek', 'Target']
df_v1 = df[features_v1].dropna()
X = df_v1.drop(columns=['Target']).values
y = df_v1['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

mlp_model_v1 = MLPClassifierModel(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(mlp_model_v1.parameters(), lr=0.01)

for epoch in range(30):
    mlp_model_v1.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = mlp_model_v1(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"[v1] Epoch {epoch+1}/30 - Loss: {epoch_loss / len(train_loader):.4f}")

# --- Evaluation (v1) ---
mlp_model_v1.eval()
preds, y_true, probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        batch_probs = mlp_model_v1(X_batch)
        batch_preds = (batch_probs >= 0.5).float()
        preds.extend(batch_preds.cpu().numpy())
        probs.extend(batch_probs.cpu().numpy())
        y_true.extend(y_batch.numpy())

# Save model and results (v1)
torch.save(mlp_model_v1.state_dict(), 'models/mlp_model_v1_shap6f_thresh05.pth')
np.save("results/probs_mlp_shap.npy", np.array(probs).flatten())
np.save("results/y_true_mlp_shap.npy", np.array(y_true).flatten())

metrics_v1 = {
    "Model": "MLP (baseline)",
    "Accuracy": accuracy_score(y_true, preds),
    "Precision": precision_score(y_true, preds),
    "Recall": recall_score(y_true, preds),
    "F1": f1_score(y_true, preds),
    "AUC": roc_auc_score(y_true, probs)
}
pd.DataFrame([metrics_v1]).to_csv("results/mlp_shap_results.csv", index=False)


# --- Version 2: enriched features (10) ---
features_v2 = [
    'Lag_7', 'Rolling_Return_5', 'Lag_8', 'RSI_14', 'MACD', 'MACD_Signal',
    'MA_Cross', 'Price_Above_MA_20', 'Volume_Diff', 'DayOfWeek', 'Target'
]
df_v2 = df[features_v2].dropna()
X = df_v2.drop(columns=['Target']).values
y = df_v2['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

mlp_model_v2 = MLPClassifierModel(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(mlp_model_v2.parameters(), lr=0.01)

for epoch in range(30):
    mlp_model_v2.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = mlp_model_v2(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"[v2] Epoch {epoch+1}/30 - Loss: {epoch_loss / len(train_loader):.4f}")

# --- Evaluation (v2) ---
mlp_model_v2.eval()
preds, y_true, probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        batch_probs = mlp_model_v2(X_batch)
        batch_preds = (batch_probs >= 0.5).float()
        preds.extend(batch_preds.cpu().numpy())
        probs.extend(batch_probs.cpu().numpy())
        y_true.extend(y_batch.numpy())

# Save model and results (v2)
torch.save(mlp_model_v2.state_dict(), 'models/mlp_model_v3_enriched11f_thresh05.pth')
np.save("results/probs_mlp_enriched.npy", np.array(probs).flatten())
np.save("results/y_true_mlp_enriched.npy", np.array(y_true).flatten())

metrics_v2 = {
    "Model": "MLP (enriched)",
    "Accuracy": accuracy_score(y_true, preds),
    "Precision": precision_score(y_true, preds),
    "Recall": recall_score(y_true, preds),
    "F1": f1_score(y_true, preds),
    "AUC": roc_auc_score(y_true, probs)
}
pd.DataFrame([metrics_v2]).to_csv("results/mlp_enriched_results.csv", index=False)


# # --- Combine and save all metrics ---
# df_metrics = pd.DataFrame([metrics_v1, metrics_v2])
# df_metrics.to_csv("results/mlp_model_results.csv", index=False)


