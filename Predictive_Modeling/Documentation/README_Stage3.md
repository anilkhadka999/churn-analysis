# Stage 3: Predictive Modeling (ANN & ML Baselines)

## Overview
This stage focuses on predicting customer churn using both baseline Machine Learning models (Logistic Regression, Random Forest) and an Artificial Neural Network (ANN). The process includes handling class imbalance via SMOTE, exhaustive hyperparameter tuning (27 combinations), model evaluation against key metrics (Precision, Recall, F1, AUC-ROC), and visual explainability using SHAP and LIME.

## Prerequisites
The environment requires specific libraries to run the modeling and explainability tasks.
Install them using:
```bash
pip install tensorflow scikit-learn imbalanced-learn shap lime fpdf2 scikeras pandas numpy matplotlib seaborn joblib
```

*Note: For Windows environments, a local virtual environment is recommended to avoid long-path errors during TensorFlow installation.*

## Directory Structure
- `Models/` - Saved `.pkl` (baseline) and `.keras` (ANN) models, plus metrics JSON files and feature names.
- `Processed_Data/` - SMOTE-balanced training data and cluster-augmented test data CSVs used by Script 02.
- `Visualizations/` - 12 generated plots: confusion matrices, ROC curves, learning curves, tuning heatmap, threshold optimization, metrics comparisons, SHAP, and LIME.
- `Notebooks_and_Code/` - Executable `.py` scripts for running the full pipeline.
- `Predictions/` - Final exported CSV `customer_churn_probabilities.csv` with P(Churn) scores and intervention flags.
- `Documentation/` - The final compiled `Stage3_ANN_Report.pdf` report.

## Execution Flow
Run the scripts in the following order from the project root:

1. **Baseline ML Modeling**
   ```bash
   python Predictive_Modeling_Stage3/Notebooks_and_Code/01_Baseline_ML_Modeling.py
   ```
   *Action:* Loads train/test data, appends KMeans cluster labels (4 clusters), applies SMOTE to training set only, trains LR & RF models, generates baseline confusion matrices, ROC curves, and metrics comparison plots.

2. **ANN Tuning & Training**
   ```bash
   python Predictive_Modeling_Stage3/Notebooks_and_Code/02_ANN_Tuning_and_Training.py
   ```
   *Action:* Performs GridSearchCV over 27 hyperparameter combinations, trains the best ANN with Early Stopping, generates learning curves, confusion matrix, ROC comparison across all 3 models, SHAP beeswarm/bar plots, LIME explanation, and exports the predictions CSV.

3. **Generate PDF Report**
   ```bash
   python Predictive_Modeling_Stage3/Notebooks_and_Code/generate_comprehensive_report.py
   ```
   *Action:* Compiles all visualisations and metrics into a single comprehensive PDF report with plain-English graph explanations.

## Key Decisions
- **SMOTE:** Applied *only* to the training set to prevent data leakage while addressing the 73.4% / 26.6% class imbalance.
- **Cluster Feature:** KMeans cluster labels (4 clusters, based on tenure & MonthlyCharges) appended as an 11th input feature.
- **Explainability:** `shap.KernelExplainer` with a k-means background for global feature importance; `lime.lime_tabular.LimeTabularExplainer` for local single-customer explanations.
- **Intervention Flag:** Added to the final predictions CSV for customers with `P(Churn) > 0.65`.
