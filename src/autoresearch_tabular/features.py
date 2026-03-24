"""Feature engineering — edit this function to create new features."""
import numpy as np
import pandas as pd


def engineer_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # --- 1. credit_amount / duration (monthly payment proxy) ---
    for df in [X_train, X_val, X_test]:
        df["monthly_payment"] = df["credit_amount"] / df["duration"].replace(0, np.nan)

    # --- 2. age / duration (maturity-to-loan-length stability ratio) ---
    for df in [X_train, X_val, X_test]:
        df["age_per_duration"] = df["age"] / df["duration"].replace(0, np.nan)

    # --- 5. Drop 5 weakest unstable/low-SHAP features ---
    drop_cols = ["foreign_worker", "num_dependents", "job", "existing_credits", "housing"]
    for df in [X_train, X_val, X_test]:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return X_train, X_val, X_test
