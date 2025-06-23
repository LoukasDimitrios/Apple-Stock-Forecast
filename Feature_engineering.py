#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Create folder for saving plots
os.makedirs("figures/feature_engineering", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data_eda.csv", index_col='Date', parse_dates=True)
# print(df.shape)

# --- Feature creation ---
df['Daily_Return'] = df['Close'].pct_change()

# Lag features (up to 10 days back)
for lag in range(1, 11):
    df[f'Lag_{lag}'] = df['Daily_Return'].shift(lag)

# Rolling returns
df['Rolling_Return_5'] = df['Close'].pct_change(periods=5)
df['Rolling_Return_10'] = df['Close'].pct_change(periods=10)

# Volatility indicators
df['Volatility_20'] = df['Close'].rolling(window=20).std()
df['Volatility_50'] = df['Close'].rolling(window=50).std()

# Binary target: 1 if next day's return is positive
df['Target'] = (df['Daily_Return'].shift(-1) > 0).astype(int)

# --- Plot 1: Daily return over time ---
df['Daily_Return'].plot(figsize=(14, 4), color='orange')
plt.title("Daily Return of Apple Stock")
plt.ylabel("Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_engineering/daily_return.png")
plt.close()

# --- Plot 2: Lag_1 vs Daily_Return (scatter) ---
sns.scatterplot(x='Lag_1', y='Daily_Return', data=df)
plt.title("Lagged Return (t-1) vs Daily Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_engineering/lag1_vs_return.png")
plt.close()

# --- Plot 3: Lag_1 over time ---
df['Lag_1'].plot(figsize=(14, 4), title='Lag 1 Return Over Time', grid=True)
plt.ylabel('Lag_1')
plt.tight_layout()
plt.savefig("figures/feature_engineering/lag1_timeseries.png")
plt.close()

# --- Plot 4: Lag_1 distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(df['Lag_1'], bins=100, kde=True, color='teal')
plt.title("Distribution of Lag_1 Returns")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_engineering/lag1_distribution.png")
plt.close()

# --- Plot 5: Heatmap of lag correlations ---
corr_cols = ['Daily_Return'] + [f'Lag_{i}' for i in range(1, 11)] + ['Target']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Correlation Heatmap: Lag Features, Return, and Target")
plt.tight_layout()
plt.savefig("figures/feature_engineering/correlation_heatmap.png")
plt.close()

# --- Plot 6: Boxplot of Lag_1 by target ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Target', y='Lag_1')
plt.title("Boxplot of Lag_1 Return by Target (Up/Down)")
plt.xlabel("Target (0 = Down, 1 = Up)")
plt.ylabel("Lag_1")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_engineering/boxplot_lag1_by_target.png")
plt.close()

# --- Save final feature set ---
df.to_csv("data/apple_stock_data_features.csv")
# print(df.shape)

