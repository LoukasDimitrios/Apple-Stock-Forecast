# Final_comparison.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from tabulate import tabulate

# --- Load all metrics CSVs ---
paths = [
    "results/logistic_results.csv",
    "results/rf_results.csv",
    "results/mlp_results.csv",
    "results/mlp_enriched_results.csv",
    "results/xgb_enriched_results.csv",
    "results/xgb_all_features_results.csv",
    "results/xgb_sent_results.csv",
    "results/lstm_shap_results.csv",
    "results/lstm_all_results.csv",
    "results/lstm_sent_results.csv",
    "results/lstm_attn_results.csv"
]

metrics_list = [pd.read_csv(p) for p in paths if os.path.exists(p)]
df_results_all = pd.concat(metrics_list, ignore_index=True).round(3)

# --- Plot ROC Curves ---
roc_data = [
    ("Logistic Regression", "results/probs_logistic.npy", "results/y_true_logistic.npy"),
    ("Random Forest", "results/probs_rf.npy", "results/y_true_rf.npy"),
    ("MLP (baseline)", "results/probs_mlp.npy", "results/y_true_mlp.npy"),
    ("MLP (enriched)", "results/probs_mlp_enriched.npy", "results/y_true_mlp_enriched.npy"),
    ("XGBoost (enriched)", "results/probs_xgb_enriched.npy", "results/y_true_xgb_enriched.npy"),
    ("XGBoost (all features, 3d)", "results/probs_xgb_all_3d.npy", "results/y_true_xgb_all_3d.npy"),
    ("XGBoost (all features, 5d)", "results/probs_xgb_all_5d.npy", "results/y_true_xgb_all_5d.npy"),
    ("XGBoost + Sentiment", "results/probs_xgb_sent.npy", "results/y_true_xgb_sent.npy"),
    ("LSTM (SHAP)", "results/probs_lstm_shap.npy", "results/y_true_lstm_shap.npy"),
    ("LSTM (All Features)", "results/probs_lstm_all.npy", "results/y_true_lstm_all.npy"),
    ("LSTM + Sentiment", "results/probs_lstm_sent.npy", "results/y_true_3d_seq.npy"),
    ("LSTM + Sentiment + Attention", "results/probs_lstm_attn.npy", "results/y_true_3d_seq.npy")
]

roc_auc_list = []
roc_curves = []

plt.figure(figsize=(10, 7))
for label, probs_path, y_path in roc_data:
    if os.path.exists(probs_path) and os.path.exists(y_path):
        probs = np.load(probs_path)
        y_true = np.load(y_path)
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append((label, roc_auc))
        roc_curves.append((label, fpr, tpr, roc_auc))

roc_auc_list.sort(key=lambda x: x[1], reverse=True)
top_3_labels = [label for label, _ in roc_auc_list[:3]]

for label, fpr, tpr, roc_auc in roc_curves:
    alpha = 1.0 if label in top_3_labels else 0.3
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})", alpha=alpha)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curve_comparison.png")
plt.show()

# --- Extended Table Summary ---
df_results_all["Harmonic_F1_AUC"] = (2 * df_results_all["F1"] * df_results_all["AUC"]) / (df_results_all["F1"] + df_results_all["AUC"])
sorted_results = df_results_all.sort_values(by="Harmonic_F1_AUC", ascending=False)
sorted_results.to_csv("results/model_comparison_summary.csv", index=False)

# --- Paper-style Table ---
columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Harmonic_F1_AUC"]
paper_table = sorted_results[columns].round(3)

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
table_data = paper_table.values
col_labels = paper_table.columns
mpl_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(9.5)
mpl_table.scale(1.2, 1.2)
plt.title("Figure: Model Performance Table (sorted by Harmonic Mean of F1 & AUC)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig("results/model_performance_table.png")
plt.show()

# Also print in console
print("\n Paper-style Model Performance Table (sorted by Harmonic Mean of F1 & AUC):")
print(tabulate(paper_table, headers='keys', tablefmt='grid', showindex=False))
