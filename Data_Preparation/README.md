# Data Preparation Deliverables

This folder contains the data preparation outputs for the **Customer Churn Analysis** project. The work in this stage focused on preparing the dataset for clustering analysis and later predictive modeling.

## Contents

- `preprocessed_dataset.csv`  
  Final preprocessed dataset after cleaning, encoding, train-test split, and feature scaling.

- `X_train_scaled.csv`  
  Training feature set after preprocessing and scaling.

- `X_test_scaled.csv`  
  Testing feature set after preprocessing and scaling.

- `y_train.csv`  
  Training target labels.

- `y_test.csv`  
  Testing target labels.

- `Train_Test_Documentation.md`  
  Documentation of the train-test split, dataset sizes, and composition.

- `Scaling_Techniques_Documentation.md`  
  Documentation of the scaling approach used in preprocessing.

- `Scaling_Techniques_Documentation.pdf`  
  Documentation of the scaling approach used in preprocessing (PDF).

## Summary of preprocessing work

The original dataset contained **7043 rows and 10 columns**. Initial inspection showed that there were **no missing values**, but **302 duplicate rows** were present. After duplicate removal, the cleaned dataset contained **6741 rows**.

The preprocessing workflow included the following steps:

1. Dataset loading and inspection  
2. String cleanup and duplicate removal  
3. Target encoding for `Churn`  
4. One-hot encoding of categorical features  
5. Train-test split using stratification  
6. Standard scaling of numerical variables  
7. Export of the final processed datasets

The final outputs were prepared to support both Stage 2 clustering analysis and later predictive modeling work.
