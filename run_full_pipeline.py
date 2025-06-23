#!/usr/bin/env python
# coding: utf-8

import subprocess
import time
import os

def run_script(script_path, description):
    print("=" * 80)
    print(f"🔹 Starting: {description}")
    print("=" * 80)
    start = time.time()

    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script_path}: {e}")
        exit(1)

    end = time.time()
    print(f"✅ Completed: {description} in {end - start:.2f} seconds\n")


if __name__ == "__main__":
    print("\n🚀 Starting full project pipeline...\n")

    scripts = [
        ("Load_data.py", "Loading and preparing raw stock data"),
        ("EDA.py", "Generating exploratory plots (EDA)"),
        ("Feature_engineering.py", "Generating technical indicators and engineered features"),
        ("Feature_ernichment.py", "Merging sentiment scores into the dataset"),
        ("Feature_selection.py", "Running SHAP and selecting informative features"),
        ("create_sentiment_enriched_data.py", "Combining sentiment with stock data (final enrichment)"),
        ("Modeling_baselines.py", "Training Logistic Regression and Random Forest"),
        ("Modeling_mlp.py", "Training MLP with SHAP-selected and intermediate features"),
        ("Modeling_xgboost.py", "Training XGBoost with SHAP-selected features"),
        ("Modeling_xgboost_all_features.py", "Training XGBoost with all features (3d/5d targets)"),
        ("Modeling_xgboost_sentiment.py", "Training XGBoost with sentiment"),
        ("Modeling_lstm.py", "Training LSTM with SHAP features"),
        ("Modeling_lstm_all.py", "Training LSTM with all features"),
        ("Modeling_sentiment_lstm.py", "Training LSTM with sentiment"),
        ("Modeling_lstm_sentiment_attention.py", "Training LSTM with sentiment and attention"),
        ("Final_comparison.py", "Generating ROC curves and final results table"),
    ]

    for script_file, description in scripts:
        if not os.path.exists(script_file):
            print(f"⚠️  Script {script_file} not found, skipping.")
            continue
        run_script(script_file, description)

    print("🎉 Full pipeline execution complete!\n")
