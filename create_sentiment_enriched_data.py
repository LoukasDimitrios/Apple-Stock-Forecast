#!/usr/bin/env python
# coding: utf-8

# --- Libraries ---
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# --- Ensure folders exist ---
os.makedirs("data", exist_ok=True)

# --- Load Apple stock data (with technical features) ---
df_stock = pd.read_csv("data/apple_stock_data_enriched.csv", index_col='Date', parse_dates=True)
# print("🔹 Stock data shape (before filtering):", df_stock.shape)

# --- Load Apple news sentiment data ---
df_news = pd.read_csv("data/social_media/apple_news_data.csv", parse_dates=['date'])
df_news = df_news.dropna(subset=['sentiment_polarity'])
# print("🔹 News data shape (raw):", df_news.shape)

# --- Daily aggregation of sentiment + count ---
df_news_daily = df_news.groupby('date').agg({
    'sentiment_polarity': 'mean',
    'sentiment_pos': 'mean',
    'sentiment_neg': 'mean',
    'sentiment_neu': 'mean',
    'title': 'count'  # number of news per day
})
df_news_daily.columns = [
    'Sentiment_Mean',
    'Sentiment_Pos_Mean',
    'Sentiment_Neg_Mean',
    'Sentiment_Neu_Mean',
    'News_Count'
]

# --- Remove timezone info if present ---
df_news_daily.index = df_news_daily.index.tz_localize(None)
df_news_daily.index.name = 'Date'
# print("🔹 Daily sentiment shape:", df_news_daily.shape)

# --- Create Low Info Day flag ---
threshold = 3
df_news_daily['Low_Info_Day'] = (df_news_daily['News_Count'] < threshold).astype(int)

# --- Set sentiment scores to 0 for Low Info Days ---
sent_cols = [
    'Sentiment_Mean', 'Sentiment_Pos_Mean',
    'Sentiment_Neg_Mean', 'Sentiment_Neu_Mean'
]
df_news_daily.loc[df_news_daily['Low_Info_Day'] == 1, sent_cols] = 0

# --- Merge with stock data ---
df_merged = df_stock.merge(df_news_daily, how='left', left_index=True, right_index=True)
# print(" Merged stock + sentiment shape (with NaNs):", df_merged.shape)

# --- Fill missing values only in sentiment and News_Count ---
fill_cols = sent_cols + ['News_Count']
df_merged[fill_cols] = df_merged[fill_cols].fillna(0)

# For rows with no sentiment info at all, set Low_Info_Day = 1
df_merged['Low_Info_Day'] = df_merged['Low_Info_Day'].fillna(1)

# print(" Final shape after filling NaNs:", df_merged.shape)

# --- Save final enriched dataset ---
df_merged.to_csv("data/apple_stock_data_enriched_with_sentiment.csv")
# print(" Saved to: data/apple_stock_data_enriched_with_sentiment.csv")

# print(df_merged.columns)