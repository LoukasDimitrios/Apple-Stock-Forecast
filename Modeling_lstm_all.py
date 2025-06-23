# Modeling LSTM with all features

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")
# --- Configuration ---
SEQ_LENGTH = 20
features = [
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7', 'Lag_8', 'Lag_9', 'Lag_10',
    'Rolling_Return_5', 'Rolling_Return_10', 'Volatility_20', 'Volatility_50',
    'RSI_14', 'MACD', 'MACD_Signal', 'MA_Cross', 'Price_Above_MA_20', 'Price_Above_MA_50',
    'Volume_Avg_20', 'Volume_Diff', 'Month', 'DayOfWeek', 'Intraday_Range'
]
target_col = 'Target_3d'

# Ensure directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load and prepare data ---
df = pd.read_csv("data/apple_stock_data_enriched.csv", index_col="Date", parse_dates=True)
df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
df['Target_3d'] = (df['Return_3d'] > 0.01).astype(int)
df = df.dropna(subset=features + [target_col])

# Create sequences
X, y = [], []
for i in range(SEQ_LENGTH, len(df)):
    X.append(df[features].iloc[i-SEQ_LENGTH:i].values)
    y.append(df.iloc[i][target_col])

X = np.array(X)
y = np.array(y)

# --- Train/Test split ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Scaling ---
X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d)
X_test_scaled = scaler.transform(X_test_2d)

X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)

joblib.dump(scaler, "models/scaler_lstm_best_grid_target3d_seq20_feats_all.pkl")

# --- Dataset class ---
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- LSTM model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

# --- Grid Search ---
param_grid = [
    {"hidden_dim": h, "dropout": d, "bidirectional": b, "num_layers": nl}
    for h in [64, 128, 256]
    for d in [0.0, 0.3, 0.5]
    for b in [True, False]
    for nl in [1, 2]
]

val_size = int(len(X_train) * 0.2)
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train_sub = X_train[:-val_size]
y_train_sub = y_train[:-val_size]

best_f1 = -1
best_config = None
best_model_state = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, params in enumerate(param_grid):
    print(f"\nRun {i+1}/{len(param_grid)} - Params: {params}")
    model = LSTMClassifier(
        input_dim=X_train.shape[2],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
        bidirectional=params["bidirectional"],
        num_layers=params["num_layers"]
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(StockDataset(X_train_sub, y_train_sub), batch_size=64, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=64, shuffle=False)

    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(model.fc.weight.device), y_batch.to(model.fc.weight.device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_preds, y_true_eval = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(model.fc.weight.device)
            probs = model(X_batch)
            preds = (probs >= 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            y_true_eval.extend(y_batch.numpy())


    f1 = f1_score(y_true_eval, val_preds)
    print(f"F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_config = params
        best_model_state = model.state_dict()

# --- Retrain best model ---
print("\nBest Config:", best_config)

model = LSTMClassifier(
    input_dim=X_train.shape[2],
    hidden_dim=best_config["hidden_dim"],
    dropout=best_config["dropout"],
    bidirectional=best_config["bidirectional"],
    num_layers=best_config["num_layers"]
).to(device)

model.load_state_dict(best_model_state)

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/30 - Loss: {total_loss / len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
preds, y_true, probs_all = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        batch_probs = model(X_batch)
        batch_preds = (batch_probs >= 0.5).float()
        preds.extend(batch_preds.cpu().numpy())
        probs_all.extend(batch_probs.cpu().numpy())
        y_true.extend(y_batch.numpy())

# Save model and results
torch.save(model.state_dict(), "models/lstm_best_grid_target3d_seq20_feats_all_thresh05.pt")

np.save("results/y_true_lstm_all.npy", y_true)
# np.save("results/preds_lstm_all.npy", preds)
np.save("results/probs_lstm_all.npy", np.array(probs_all).flatten())

metrics = {
    "Model": "LSTM (All Features)",
    "Accuracy": accuracy_score(y_true, preds),
    "Precision": precision_score(y_true, preds),
    "Recall": recall_score(y_true, preds),
    "F1": f1_score(y_true, preds),
    "AUC": roc_auc_score(y_true, np.array(probs_all).flatten())
}

df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv("results/lstm_all_results.csv", index=False)
