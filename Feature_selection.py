#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Create folder for plots
os.makedirs("figures/feature_selection", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/apple_stock_data_enriched.csv", index_col='Date', parse_dates=True)
df = df.dropna()
# print(df.shape)
# --- Pearson Correlation with Target ---
numeric_cols = df.select_dtypes(include='number')
correlations = numeric_cols.corr()['Target'].drop('Target').sort_values(ascending=False)

plt.figure(figsize=(8, 12))
sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm', legend=False)
plt.title("Pearson Correlation with Target")
plt.xlabel("Correlation")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_selection/pearson_correlation.png")
plt.close()

# --- Random Forest Importance ---
X = df.drop(columns=['Target'])
y = df['Target']

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

top_n = 20
top_features = feature_names[indices][:top_n]
top_importances = importances[indices][:top_n]

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.title("Random Forest Feature Importance (Top 20)")
plt.xlabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/feature_selection/rf_importance.png")
plt.close()

# --- SHAP Analysis ---
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
shap_values_class1 = shap_values[:, :, 1]

# SHAP Summary Plot
shap.summary_plot(shap_values_class1, X, show=False)
plt.tight_layout()
plt.savefig("figures/feature_selection/shap_summary.png")
plt.close()



# --- SHAP force plot for a single instance ---
sample_idx = 0  # pick the first sample
shap_values_instance = shap_values[:, :, 1][sample_idx]  # SHAP values
features_instance = X.iloc[sample_idx]                   # actual feature values
expected_value = explainer.expected_value[1]             # base value

# Plot with matplotlib backend (for .png output)
plt.figure()
shap.plots.force(
    expected_value,
    shap_values_instance,
    features_instance,
    matplotlib=True,
    show=False  # <= Ensure nothing gets rendered interactively
)
plt.tight_layout()
plt.savefig("figures/feature_selection/shap_force_sample0.png")
plt.close()


# --- Compute mean absolute SHAP values ---
shap_mean_abs = np.abs(shap_values_class1).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': shap_mean_abs
}).sort_values(by='mean_abs_shap', ascending=False)

# --- Select features above threshold ---
selected_features = shap_df[shap_df['mean_abs_shap'] > 0.01]['feature'].tolist()

# print("\nSelected features (mean_abs_shap > 0.01):")
# print(selected_features)

# --- Save SHAP values ---
shap_df.to_csv("results/shap_feature_importance.csv", index=False)
with open("results/selected_features.txt", "w") as f:
    for feat in selected_features:
        f.write(f"{feat}\n")
# print(df.shape)