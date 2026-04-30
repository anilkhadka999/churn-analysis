# Training and Testing Sets Documentation

## Overview

The cleaned and encoded dataset was divided into separate training and testing sets to support fair model development and evaluation. This step was included to satisfy the Stage 2 requirement to provide distinct training and testing sets with documentation on size and composition.

## Dataset preparation before split

Before splitting, the following preprocessing steps were completed:

- string values were cleaned using whitespace trimming
- duplicate rows were removed
- the target column `Churn` was encoded as binary
- categorical feature columns were one-hot encoded using `pd.get_dummies(..., drop_first=True)`

After encoding, the feature matrix contained **6741 rows and 10 columns**.

## Train-test split method

The dataset was split using `train_test_split()` from scikit-learn with the following settings:

- `test_size = 0.2`
- `random_state = 42`
- `stratify = y`

Stratification was used so that the class distribution of `Churn` remained balanced between the training and testing sets.

## Final sizes

The final dataset sizes were:

- `X_train`: **5392 rows, 10 columns**
- `X_test`: **1349 rows, 10 columns**
- `y_train`: **5392 rows**
- `y_test`: **1349 rows**

## Composition

The feature sets contain the processed customer attributes used for analysis. These include:

- scaled numerical columns
- encoded categorical columns

The target sets contain the binary churn label:

- `0` = No churn
- `1` = Churn

## Files included

- `X_train_scaled.csv`
- `X_test_scaled.csv`
- `y_train.csv`
- `y_test.csv`

## Notes

The training and testing sets were created after encoding and before model training so that the project could maintain a clear separation between model development data and evaluation data.
