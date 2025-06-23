#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Create folder to save plots
os.makedirs("figures/feature_enrichment", exist_ok=True)

# --- Load dataset with basic features ---
df = pd.read_csv("data/apple_stock_data_features.csv", index_col='Date', parse_dates=True)
# print(df.shape)
# --- Moving average cross (MA_20 > MA_50) ---
df['MA_Cross'] = (df['MA_20'] > df['MA_50']).astype(int)

# --- Price relative to moving averages ---
df['Price_Above_MA_20'] = (df['Close'] > df['MA_20']).astype(int)
df['Price_Above_MA_50'] = (df['Close'] > df['MA_50']).astype(int)

# --- Intraday range (High - Low) ---
df['Intraday_Range'] = df['High'] - df['Low']

# --- Volume average and deviation (20-day) ---
df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
df['Volume_Diff'] = df['Volume'] - df['Volume_Avg_20']

# --- Seasonality features ---
df['Month'] = df.index.month
df['DayOfWeek'] = df.index.dayofweek

# --- RSI (Relative Strength Index, 14-day) ---
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# --- MACD and signal line ---
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# --- Plot sample enriched features ---

# Plot: RSI
df['RSI_14'].plot(figsize=(14, 3), title="RSI (14-day)")
plt.axhline(70, color='red', linestyle='--', alpha=0.5)
plt.axhline(30, color='green', linestyle='--', alpha=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_enrichment/rsi14.png")
plt.close()

# Plot: MACD
df[['MACD', 'MACD_Signal']].plot(figsize=(14, 4), title="MACD and Signal Line")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_enrichment/macd_signal.png")
plt.close()

# --- Save enriched dataset ---
df.to_csv("data/apple_stock_data_enriched.csv")
# print(df.shape)
