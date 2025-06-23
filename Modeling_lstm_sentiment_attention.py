# Modeling LSTM with Attention (Sentiment + SHAP features)

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
# --- Configuration ---
SEQ_LENGTH = 20
features = [
    'Lag_7', 'Rolling_Return_5', 'Lag_8', 'Lag_6', 'RSI_14', 'DayOfWeek',
    'Sentiment_Mean', 'News_Count'
]
target_col = 'Target_3d'

# Ensure directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load and prepare data ---
df = pd.read_csv("data/apple_stock_data_enriched_with_sentiment.csv", index_col="Date", parse_dates=True)

df['Return_3d'] = df['Close'].shift(-3) / df['Close'] - 1
df['Target_3d'] = (df['Return_3d'] > 0.01).astype(int)
df = df.dropna(subset=features + [target_col])

# --- Create sequences ---
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

joblib.dump(scaler, "models/scaler_lstm_sentiment_attn.pkl")

# --- Dataset class ---
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- LSTM with Attention ---
class LSTMAttentionClassifier(nn.Module):
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
        self.attn_linear = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq, hidden)
        attn_scores = torch.softmax(self.attn_linear(lstm_out), dim=1)  # (batch, seq, 1)
        context = torch.sum(attn_scores * lstm_out, dim=1)  # (batch, hidden)
        out = self.fc(context)
        return self.sigmoid(out)

# --- Train ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAttentionClassifier(input_dim=X_train.shape[2]).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

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

# --- Evaluate ---
model.eval()
preds, y_true, probs_all = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        probs = model(X_batch)
        batch_preds = (probs >= 0.5).float()
        preds.extend(batch_preds.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())
        y_true.extend(y_batch.numpy())

print("\nTest Accuracy:", accuracy_score(y_true, preds))
print("Classification Report:\n", classification_report(y_true, preds))
print("Confusion Matrix:\n", confusion_matrix(y_true, preds))

# --- Save ---
torch.save(model.state_dict(), "models/lstm_sentiment_attention.pt")
np.save("results/y_true_3d_seq.npy", y_true)
# np.save("results/preds_lstm_attn.npy", preds)
np.save("results/probs_lstm_attn.npy", np.array(probs_all).flatten())

metrics = {
    "Model": "LSTM + Sentiment + Attention",
    "Accuracy": accuracy_score(y_true, preds),
    "Precision": precision_score(y_true, preds),
    "Recall": recall_score(y_true, preds),
    "F1": f1_score(y_true, preds),
    "AUC": roc_auc_score(y_true, np.array(probs_all).flatten())
}

pd.DataFrame([metrics]).to_csv("results/lstm_attn_results.csv", index=False)
