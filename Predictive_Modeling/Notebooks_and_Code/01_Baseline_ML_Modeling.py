"""
Stage 3 : Baseline ML Modeling

"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# -- Paths -----------------------------------------------------------------
BASE      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR  = os.path.join(BASE, "Data_Preparation", "Train-Test")
KMEANS    = os.path.join(BASE, "Clustering Analytics", "Kmeans Model", "kmeans_model.pkl")
MODEL_DIR = os.path.join(BASE, "Predictive_Modeling", "Models")
PROC_DIR  = os.path.join(BASE, "Predictive_Modeling", "Processed_Data")
VIZ_DIR   = os.path.join(BASE, "Predictive_Modeling", "Visualizations")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# -- 1. Load Data ----------------------------------------------------------
print("=" * 70)
print("STAGE 3 - SCRIPT 01: BASELINE ML MODELING")
print("=" * 70)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print(f"\nLoaded training set : {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"Loaded test set     : {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"Class distribution (train): No-Churn={np.sum(y_train==0)}, Churn={np.sum(y_train==1)}  "
      f"({np.mean(y_train==0)*100:.1f}% / {np.mean(y_train==1)*100:.1f}%)")

# -- 2. Add Cluster Feature ------------------------------------------------
print("\n-- Adding K-Means Cluster Labels --")
kmeans_model = joblib.load(KMEANS)
print(f"KMeans model loaded: {kmeans_model.n_clusters} clusters, "
      f"expects {kmeans_model.n_features_in_} input features (tenure, MonthlyCharges)")

cluster_cols = ["tenure", "MonthlyCharges"]
X_train["Cluster"] = kmeans_model.predict(X_train[cluster_cols])
X_test["Cluster"]  = kmeans_model.predict(X_test[cluster_cols])

print(f"Cluster labels appended. New feature count: {X_train.shape[1]}")
print(f"Cluster distribution (train):\n{X_train['Cluster'].value_counts().sort_index().to_string()}")

# -- 3. SMOTE (training set only) ------------------------------------------
print("\n-- Applying SMOTE to Training Set --")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {len(X_train)} samples  ->  After SMOTE: {len(X_train_sm)} samples")
print(f"Resampled distribution: No-Churn={np.sum(y_train_sm==0)}, Churn={np.sum(y_train_sm==1)}")

# Save SMOTE-balanced data for Script 02
joblib.dump(smote, os.path.join(MODEL_DIR, "smote_resampler.pkl"))

# -- 4. Train Baseline Models ----------------------------------------------
print("\n-- Training Logistic Regression --")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_model.fit(X_train_sm, y_train_sm)
lr_pred       = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

print("-- Training Random Forest --")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred       = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# -- 5. Evaluation ---------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_proba):
    """Print detailed metrics with TP/FP/TN/FN calculations."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)

    print(f"\n{'-' * 50}")
    print(f"  {name} - Test Set Evaluation")
    print(f"{'-' * 50}")
    print(f"  Confusion Matrix:")
    print(f"                 Predicted No   Predicted Yes")
    print(f"  Actual No      TN = {tn:<10d} FP = {fp}")
    print(f"  Actual Yes     FN = {fn:<10d} TP = {tp}")
    print(f"\n  Metric Calculations:")
    print(f"    Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn}) = {acc:.4f}")
    print(f"    Precision = TP/(TP+FP)             = {tp}/({tp}+{fp}) = {prec:.4f}")
    print(f"    Recall    = TP/(TP+FN)             = {tp}/({tp}+{fn}) = {rec:.4f}")
    print(f"    F1-Score  = 2*(P*R)/(P+R)          = 2*({prec:.4f}*{rec:.4f})/({prec:.4f}+{rec:.4f}) = {f1:.4f}")
    print(f"    AUC-ROC   = {auc:.4f}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC-ROC": auc,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred, lr_pred_proba)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_pred_proba)

# -- 6. Save Models --------------------------------------------------------
joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.pkl"))
# Also save the cluster-augmented data for Script 02
X_train_sm.to_csv(os.path.join(PROC_DIR, "X_train_smote.csv"), index=False)
pd.DataFrame(y_train_sm, columns=["Churn"]).to_csv(
    os.path.join(PROC_DIR, "y_train_smote.csv"), index=False
)
X_test.to_csv(os.path.join(PROC_DIR, "X_test_clustered.csv"), index=False)
pd.DataFrame(y_test, columns=["Churn"]).to_csv(
    os.path.join(PROC_DIR, "y_test.csv"), index=False
)

# Save feature names for later use
joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

print("\n Models saved to Models/")

# -- 7. Visualisations -----------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# 7a. Confusion Matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name, y_pred in zip(axes, ["Logistic Regression", "Random Forest"], [lr_pred, rf_pred]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "baseline_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved baseline_confusion_matrices.png")

# 7b. ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, y_proba, color in [("Logistic Regression", lr_pred_proba, "#2196F3"),
                              ("Random Forest", rf_pred_proba, "#4CAF50")]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2, color=color)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Baseline Models - ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "baseline_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved baseline_roc_curves.png")

# 7c. Metrics Comparison Bar Chart
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"] * 2,
    "Value":  [lr_metrics["Accuracy"], lr_metrics["Precision"], lr_metrics["Recall"],
               lr_metrics["F1"], lr_metrics["AUC-ROC"],
               rf_metrics["Accuracy"], rf_metrics["Precision"], rf_metrics["Recall"],
               rf_metrics["F1"], rf_metrics["AUC-ROC"]],
    "Model":  ["Logistic Regression"] * 5 + ["Random Forest"] * 5,
})
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = {"Logistic Regression": "#2196F3", "Random Forest": "#4CAF50"}
x = np.arange(5)
width = 0.35
for i, model in enumerate(["Logistic Regression", "Random Forest"]):
    vals = metrics_df[metrics_df["Model"] == model]["Value"].values
    bars = ax.bar(x + i * width, vals, width, label=model, color=bar_colors[model], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Baseline Model Comparison", fontsize=14, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "baseline_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved baseline_metrics_comparison.png")

# -- 8. Save metrics to JSON for report script -----------------------------
metrics_out = {"Logistic Regression": lr_metrics, "Random Forest": rf_metrics}
with open(os.path.join(MODEL_DIR, "baseline_metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)

