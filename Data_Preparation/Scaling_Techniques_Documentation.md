# Scaling Techniques Documentation

## Overview

Feature scaling was applied to the numerical variables in the dataset to make them comparable in magnitude. This is important because variables with larger numeric ranges can dominate distance-based methods and affect later analysis quality.

## Numerical features scaled

Only the true numerical columns were scaled:

- `tenure`
- `MonthlyCharges`

Binary encoded columns were not scaled and were kept in encoded form. This keeps the dummy variables interpretable while standardizing the continuous numerical variables.

## Scaling method used

The scaling method used was **StandardScaler** from scikit-learn.

This technique transforms each selected numerical feature so that it has:

- mean close to 0
- standard deviation close to 1

## Why scaling was done after train-test split

The scaler was fitted only on the training set and then applied to the test set. This avoids data leakage because information from the test set is not used during training-time preprocessing.

## Implementation summary

The scaling workflow was:

1. create copies of `X_train` and `X_test`
2. fit `StandardScaler()` on the numerical columns of `X_train`
3. transform the same numerical columns in `X_test`
4. keep all encoded categorical columns unchanged

## Code snippet used

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

## Output files

The following files were generated after scaling:

- `X_train_scaled.csv`
- `X_test_scaled.csv`
- `preprocessed_dataset.csv`

## Notes

This scaling approach was chosen because the dataset contains both continuous numerical features and encoded categorical features. Scaling only the true numerical columns preserves the meaning of the binary encoded variables while still preparing the numerical features for later clustering and modeling.
