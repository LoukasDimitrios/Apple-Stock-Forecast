#!/usr/bin/env python
# coding: utf-8

# === Load and Save Apple Stock Data from Yahoo Finance ===

import yfinance as yf
import pandas as pd
import os

# --- Step 1: Download Apple stock data (single ticker) ---
apple_data = yf.download('AAPL', start='2010-01-01', end='2025-05-22')

# --- Step 2: Flatten MultiIndex (if exists) ---
if isinstance(apple_data.columns, pd.MultiIndex):
    apple_data.columns = apple_data.columns.get_level_values(0)

# --- Step 3: Ensure 'Date' is Index and save properly ---
apple_data.index.name = 'Date'  # for consistency with later scripts
os.makedirs("data", exist_ok=True)
apple_data.to_csv("data/apple_stock_data.csv", index=True)

# --- Step 4: Confirm shape and columns ---
# print("Saved to: data/apple_stock_data.csv")
# print("Date range:", apple_data.index.min(), "to", apple_data.index.max())
# print("Shape:", apple_data.shape)
# print("Columns:", apple_data.columns.tolist())
