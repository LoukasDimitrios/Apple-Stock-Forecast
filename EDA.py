#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder to save plots
os.makedirs("figures/eda", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data.csv", header=0, index_col=0, parse_dates=True)
# print(df.shape)
# Force conversion to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Close"])

# --- Basic info ---
# print("Data Info:")
# print(df.info())
# print("\nMissing values:\n", df.isnull().sum())
# print("\nStatistical Summary:\n", df.describe())
# print(f"\nDate Range: {df.index.min()} to {df.index.max()}")
# print("\nColumn types:")
# print(df.dtypes)

# print("\nSample values in 'Close':")
# print(df['Close'].head())

# --- Plot 1: Close price over time ---
plt.figure(figsize=(14, 6))
df['Close'].plot()
plt.title("Apple Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/closing_price.png")
plt.close()

# --- Plot 2: OHLC and Intraday Range ---
df['Intraday Range'] = df['High'] - df['Low']
df_zoom = df.loc['2022-06-01':'2022-06-30']

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)

df[['Open', 'High', 'Low', 'Close']].plot(ax=axes[0])
axes[0].set_title("Apple Stock Prices (OHLC) - Full Timeline")
axes[0].set_ylabel("Price (USD)")
axes[0].grid(True)

df_zoom[['Open', 'High', 'Low', 'Close']].plot(ax=axes[1])
axes[1].set_title("Apple OHLC - June 2022 (Zoom In)")
axes[1].set_ylabel("Price (USD)")
axes[1].grid(True)

df['Intraday Range'].plot(ax=axes[2], color='red')
axes[2].set_title("Daily Intraday Range (High - Low)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("USD")
axes[2].grid(True)



plt.tight_layout()
plt.savefig("figures/eda/ohlc_intraday_range.png")
plt.close()

# --- Plot 3: Volume over time ---
df['Volume'].plot(figsize=(14, 4), color='orange')
plt.title("Apple Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/volume_over_time.png")
plt.close()

# --- Plot 4: Mean close per year ---
df['Year'] = df.index.year
yearly_avg = df.groupby('Year')['Close'].mean()

yearly_avg.plot(kind='bar', figsize=(12, 6))
plt.title("Average Closing Price per Year")
plt.xlabel("Year")
plt.ylabel("Average Close Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/yearly_avg_close.png")
plt.close()

# --- Plot 5: Seasonality (boxplot by month) ---
df['Month'] = df.index.month
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Close', data=df)
plt.title("Distribution of Closing Price per Month")
plt.xlabel("Month")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/monthly_close_distribution.png")
plt.close()

# --- Plot 6: Correlation matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close', 'Volume']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Apple Stock Data")
plt.tight_layout()
plt.savefig("figures/eda/correlation_matrix.png")
plt.close()

# --- Moving averages ---
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

df[['Close', 'MA_20', 'MA_50']].plot(figsize=(14, 6))
plt.title("Apple Close Price with 20- and 50-day Moving Averages")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/moving_averages.png")
plt.close()

# --- Volatility plots ---
df['Volatility_20'] = df['Close'].rolling(window=20).std()
df['Volatility_50'] = df['Close'].rolling(window=50).std()

df['Volatility_20'].plot(figsize=(14, 4), color='purple')
plt.title("20-Day Rolling Volatility of Apple Stock")
plt.ylabel("Standard Deviation")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/volatility_20.png")
plt.close()

df['Volatility_50'].plot(figsize=(14, 4), color='purple')
plt.title("50-Day Rolling Volatility of Apple Stock")
plt.ylabel("Standard Deviation")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/eda/volatility_50.png")
plt.close()

# --- Save final dataframe (with Intraday Range, Year, Month, MA, Volatility) ---
df.to_csv("data/apple_stock_data_eda.csv", index=True)
# print(df.shape)