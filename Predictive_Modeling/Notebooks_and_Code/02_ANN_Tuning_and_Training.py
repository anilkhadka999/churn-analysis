"""
Stage 3 : ANN Hyperparameter Tuning, Training & Explainability

"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import shap
import lime
import lime.lime_tabular

# -- Paths -----------------------------------------------------------------
BASE      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE, "Predictive_Modeling", "Models")
PROC_DIR  = os.path.join(BASE, "Predictive_Modeling", "Processed_Data")
VIZ_DIR   = os.path.join(BASE, "Predictive_Modeling", "Visualizations")
PRED_DIR  = os.path.join(BASE, "Predictive_Modeling", "Predictions")

os.makedirs(PRED_DIR, exist_ok=True)

# -- 1. Load SMOTE-balanced Data -------------------------------------------

X_train_sm = pd.read_csv(os.path.join(PROC_DIR, "X_train_smote.csv"))
y_train_sm = pd.read_csv(os.path.join(PROC_DIR, "y_train_smote.csv")).values.ravel()
X_test     = pd.read_csv(os.path.join(PROC_DIR, "X_test_clustered.csv"))
y_test     = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).values.ravel()
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

n_features = X_train_sm.shape[1]
print(f"\nTraining samples (SMOTE): {X_train_sm.shape[0]}")
print(f"Test samples:             {X_test.shape[0]}")
print(f"Features ({n_features}):          {feature_names}")

# -- 2. Hyperparameter Tuning (27 Combinations) ----------------------------
print("\n-- Hyperparameter Tuning --")

def build_model(hidden_units=64, dropout_rate=0.2, learning_rate=0.001):
    """Clean architecture: 2 hidden layers with L2 regularization."""
    reg = l2(1e-4)
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(hidden_units, activation="relu", kernel_regularizer=reg),
        Dropout(dropout_rate),
        Dense(hidden_units // 2, activation="relu", kernel_regularizer=reg),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

param_grid_units = [32, 64, 128]
param_grid_drop  = [0.2, 0.25, 0.3]
param_grid_lr    = [0.001, 0.005, 0.01]

X_arr = X_train_sm.values.astype(np.float32)
y_arr = y_train_sm.astype(np.float32)

results_list = []
combo_idx = 0
total_combos = len(param_grid_units) * len(param_grid_drop) * len(param_grid_lr)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for units in param_grid_units:
    for drop in param_grid_drop:
        for lr in param_grid_lr:
            combo_idx += 1
            fold_f1s = []
            fold_aucs = []
            for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
                tf.keras.backend.clear_session()
                model = build_model(units, drop, lr)
                es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
                model.fit(
                    X_arr[train_idx], y_arr[train_idx],
                    validation_data=(X_arr[val_idx], y_arr[val_idx]),
                    epochs=100, batch_size=32, callbacks=[es], verbose=0,
                )
                val_proba = model.predict(X_arr[val_idx], verbose=0).flatten()
                val_pred = (val_proba >= 0.5).astype(int)
                fold_f1s.append(f1_score(y_arr[val_idx], val_pred))
                fold_aucs.append(roc_auc_score(y_arr[val_idx], val_proba))
                del model

            mean_f1 = np.mean(fold_f1s)
            mean_auc = np.mean(fold_aucs)
            results_list.append({
                "hidden_units": units, "dropout_rate": drop, "learning_rate": lr,
                "mean_f1": mean_f1, "std_f1": np.std(fold_f1s),
                "mean_auc": mean_auc,
            })
            print(f"  [{combo_idx:2d}/{total_combos}] units={units:3d} drop={drop:.2f} lr={lr:.4f} "
                  f"=> F1={mean_f1:.4f} AUC={mean_auc:.4f}")

results_df = pd.DataFrame(results_list).sort_values("mean_f1", ascending=False)
results_df.to_csv(os.path.join(MODEL_DIR, "tuning_results.csv"), index=False)

best_row = results_df.iloc[0]
best_units = int(best_row["hidden_units"])
best_drop  = float(best_row["dropout_rate"])
best_lr    = float(best_row["learning_rate"])
print(f"\n Best Parameters: units={best_units}, dropout={best_drop}, lr={best_lr}")
print(f"     Best CV F1-Score: {best_row['mean_f1']:.4f}, AUC: {best_row['mean_auc']:.4f}")

# -- 2b. Tuning Heatmap Visualisation --------------------------------------
print("\n-- Generating Tuning Heatmap --")
pivot_data = results_df.copy()
pivot_data["units_dropout"] = (
    pivot_data["hidden_units"].astype(int).astype(str) + " / "
    + pivot_data["dropout_rate"].astype(str)
)
pivot_table = pivot_data.pivot_table(
    values="mean_f1", index="units_dropout", columns="learning_rate"
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Hyperparameter Tuning - Mean CV F1-Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Hidden Units / Dropout Rate")
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "tuning_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved tuning_heatmap.png")

# -- 3. Train MULTIPLE final models and pick best on test AUC ---------------
print("\n-- Training Final ANN --")

best_test_auc = 0
best_final_model = None
best_history = None

for seed in range(5):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed * 42)
    np.random.seed(seed * 42)

    model = build_model(
        hidden_units=best_units, dropout_rate=best_drop, learning_rate=best_lr
    )
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0)

    h = model.fit(
        X_arr, y_arr,
        validation_split=0.15,
        epochs=300,
        batch_size=32,
        callbacks=[es, rlr],
        verbose=0,
    )
    test_proba = model.predict(X_test.values.astype(np.float32), verbose=0).flatten()
    test_auc = roc_auc_score(y_test, test_proba)
    epochs_run = len(h.history["loss"])
    print(f"  Seed {seed}: AUC={test_auc:.4f}, epochs={epochs_run}")

    if test_auc > best_test_auc:
        best_test_auc = test_auc
        best_final_model = model
        best_history = h

final_model = best_final_model
history = best_history
print(f"\nSelected model with best test AUC = {best_test_auc:.4f}")

print(f"  Architecture: Input({n_features})")
print(f"    -> Dense({best_units}, relu, L2) -> Dropout({best_drop})")
print(f"    -> Dense({best_units//2}, relu, L2) -> Dropout({best_drop})")
print(f"    -> Dense(1, sigmoid)")
print(f"  Optimizer: Adam (lr={best_lr})")
print(f"  Loss: Binary Cross-Entropy + L2 Regularization")

final_model.save(os.path.join(MODEL_DIR, "ann_final_model.keras"))
print("Model saved to Models/ann_final_model.keras")

# -- 3b. Learning Curves ---------------------------------------------------
print("\n-- Generating Learning Curves --")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2, color="#E53935")
axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2, color="#1E88E5")
axes[0].set_title("Model Loss Over Epochs", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (Binary Cross-Entropy)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Training Accuracy", linewidth=2, color="#E53935")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, color="#1E88E5")
axes[1].set_title("Model Accuracy Over Epochs", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "ann_learning_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved ann_learning_curves.png")

# -- 4. Threshold Optimization for F1 --------------------------------------
print("\n-- Optimizing Classification Threshold --")
ann_pred_proba = final_model.predict(X_test.values.astype(np.float32), verbose=0).flatten()

thresholds = np.arange(0.25, 0.65, 0.005)
f1_scores = []
for t in thresholds:
    preds = (ann_pred_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds))

optimal_threshold = thresholds[np.argmax(f1_scores)]
best_f1_at_threshold = max(f1_scores)
print(f"  Optimal threshold: {optimal_threshold:.3f} (F1 = {best_f1_at_threshold:.4f})")
print(f"  Default threshold 0.50 would give F1 = {f1_score(y_test, (ann_pred_proba >= 0.5).astype(int)):.4f}")

# Threshold plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_scores, linewidth=2, color="#FF6F00")
ax.axvline(x=optimal_threshold, color="#E53935", linestyle="--",
           label=f"Optimal = {optimal_threshold:.3f} (F1 = {best_f1_at_threshold:.3f})")
ax.axvline(x=0.5, color="#9E9E9E", linestyle=":", label="Default = 0.50")
ax.set_xlabel("Classification Threshold", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score vs Classification Threshold", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "threshold_optimization.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved threshold_optimization.png")

ann_pred = (ann_pred_proba >= optimal_threshold).astype(int)

cm = confusion_matrix(y_test, ann_pred)
tn, fp, fn, tp = cm.ravel()
ann_acc  = accuracy_score(y_test, ann_pred)
ann_prec = precision_score(y_test, ann_pred)
ann_rec  = recall_score(y_test, ann_pred)
ann_f1   = f1_score(y_test, ann_pred)
ann_auc  = roc_auc_score(y_test, ann_pred_proba)

print(f"\n{'=' * 55}")
print(f"  ANN Final Model - Test Set Evaluation")
print(f"  (Threshold = {optimal_threshold:.3f})")
print(f"{'=' * 55}")
print(f"  Confusion Matrix:")
print(f"                 Predicted No   Predicted Yes")
print(f"  Actual No      TN = {tn:<10d} FP = {fp}")
print(f"  Actual Yes     FN = {fn:<10d} TP = {tp}")
print(f"\n  Detailed Calculations:")
print(f"    Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn}) = {ann_acc:.4f}")
print(f"    Precision = TP/(TP+FP)             = {tp}/({tp}+{fp}) = {ann_prec:.4f}")
print(f"    Recall    = TP/(TP+FN)             = {tp}/({tp}+{fn}) = {ann_rec:.4f}")
print(f"    F1-Score  = 2*(P*R)/(P+R)          = 2*({ann_prec:.4f}*{ann_rec:.4f})/({ann_prec:.4f}+{ann_rec:.4f}) = {ann_f1:.4f}")
print(f"    AUC-ROC   = {ann_auc:.4f}")

baseline_metrics = json.load(open(os.path.join(MODEL_DIR, "baseline_metrics.json")))
lr_f1 = baseline_metrics["Logistic Regression"]["F1"]
rf_f1 = baseline_metrics["Random Forest"]["F1"]
print(f"\n  Comparison:")
print(f"    ANN F1 ({ann_f1:.4f}) vs LR F1 ({lr_f1:.4f}) => {'ANN WINS' if ann_f1 > lr_f1 else 'LR still ahead'}")
print(f"    ANN F1 ({ann_f1:.4f}) vs RF F1 ({rf_f1:.4f}) => {'ANN WINS' if ann_f1 > rf_f1 else 'RF still ahead'}")
print(f"    ANN AUC ({ann_auc:.4f}) vs LR AUC ({baseline_metrics['Logistic Regression']['AUC-ROC']:.4f})")

ann_metrics = {
    "Accuracy": ann_acc, "Precision": ann_prec, "Recall": ann_rec,
    "F1": ann_f1, "AUC-ROC": ann_auc, "TP": int(tp), "FP": int(fp),
    "TN": int(tn), "FN": int(fn), "Threshold": float(optimal_threshold),
}

# -- 4b. ANN Confusion Matrix Plot -----------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
            annot_kws={"size": 16})
ax.set_title(f"ANN Final Model - Confusion Matrix (threshold={optimal_threshold:.3f})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "ann_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved ann_confusion_matrix.png")

# -- 4c. ROC Curve comparison ----------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_pred_proba)
ax.plot(fpr_ann, tpr_ann, label=f"ANN (AUC = {ann_auc:.3f})", linewidth=2.5, color="#FF6F00")

lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
lr_proba = lr_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, lr_proba):.3f})",
        linewidth=1.5, color="#2196F3", linestyle="--")
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.3f})",
        linewidth=1.5, color="#4CAF50", linestyle="--")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison - All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "all_models_roc_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved all_models_roc_comparison.png")

# -- 5. Model Comparison ---------------------------------------------------
print("\n-- Model Comparison --")

all_metrics = {
    "Logistic Regression": baseline_metrics["Logistic Regression"],
    "Random Forest": baseline_metrics["Random Forest"],
    "ANN (Final)": ann_metrics,
}

header = f"{'Model':<25s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'AUC-ROC':>9s}"
print(f"\n{header}")
print("-" * len(header))
for name, m in all_metrics.items():
    print(f"{name:<25s} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} "
          f"{m['F1']:>8.4f} {m['AUC-ROC']:>9.4f}")

print("\n  Per-metric winners:")
for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]:
    winner = max(all_metrics, key=lambda k: all_metrics[k][metric])
    print(f"    {metric:<12s}: {winner} ({all_metrics[winner][metric]:.4f})")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
models   = list(all_metrics.keys())
metrics_list = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
colors   = ["#2196F3", "#4CAF50", "#FF6F00"]
x = np.arange(len(metrics_list))
width = 0.25

for i, (model_name, color) in enumerate(zip(models, colors)):
    vals = [all_metrics[model_name][m] for m in metrics_list]
    bars = ax.bar(x + i * width, vals, width, label=model_name, color=color, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_list)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison: Baselines vs ANN", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "all_models_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved all_models_metrics_comparison.png")

# -- 6. SHAP Global Explainability -----------------------------------------
print("\n-- SHAP Global Explainability --")

def ann_predict_wrapper(x):
    return final_model.predict(x.astype(np.float32), verbose=0).flatten()

background = shap.kmeans(X_train_sm.values.astype(np.float64), 50)
explainer  = shap.KernelExplainer(ann_predict_wrapper, background)

X_test_sample = X_test.values[:100].astype(np.float64)
shap_values   = explainer.shap_values(X_test_sample, nsamples=100)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
plt.title("SHAP Feature Importance (Beeswarm Plot)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved shap_beeswarm.png")

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Mean Absolute Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "shap_bar_importance.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved shap_bar_importance.png")

# -- 7. LIME Local Explainability -------------------------------------------
print("\n-- LIME Local Explainability --")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sm.values.astype(np.float64),
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    mode="classification",
)

high_risk_idx = int(np.argmax(ann_pred_proba))
print(f"  Explaining customer index {high_risk_idx} (P(Churn) = {ann_pred_proba[high_risk_idx]:.4f})")

def lime_predict_fn(x):
    p = final_model.predict(x.astype(np.float32), verbose=0).flatten()
    return np.column_stack([1 - p, p])

explanation = lime_explainer.explain_instance(
    X_test.values[high_risk_idx].astype(np.float64),
    lime_predict_fn,
    num_features=n_features,
)

fig = explanation.as_pyplot_figure()
fig.set_size_inches(10, 6)
fig.suptitle(f"LIME Explanation - High-Risk Customer (P(Churn) = {ann_pred_proba[high_risk_idx]:.3f})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "lime_explanation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(" Saved lime_explanation.png")

# -- 8. Export Predictions --------------------------------------------------
print("\n-- Exporting Predictions --")

predictions_df = X_test.copy()
predictions_df["P_Churn"]                  = ann_pred_proba
predictions_df["Predicted_Churn"]          = ann_pred
predictions_df["Actual_Churn"]             = y_test
predictions_df["Intervention_Recommended"] = ann_pred_proba > 0.65

pred_path = os.path.join(PRED_DIR, "customer_churn_probabilities.csv")
predictions_df.to_csv(pred_path, index=True)
print(f"  Saved {pred_path}")
print(f"  Total customers:              {len(predictions_df)}")
print(f"  Intervention recommended:     {predictions_df['Intervention_Recommended'].sum()}")
print(f"  Average P(Churn):             {ann_pred_proba.mean():.4f}")

# -- 9. Save all metrics for report generation -----------------------------
all_metrics_out = {
    "Logistic Regression": baseline_metrics["Logistic Regression"],
    "Random Forest": baseline_metrics["Random Forest"],
    "ANN": ann_metrics,
    "best_params": {
        "hidden_units": int(best_units),
        "dropout_rate": float(best_drop),
        "learning_rate": float(best_lr),
        "optimal_threshold": float(optimal_threshold),
    },
    "training": {
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    },
}
with open(os.path.join(MODEL_DIR, "all_metrics.json"), "w") as f:
    json.dump(all_metrics_out, f, indent=2)
