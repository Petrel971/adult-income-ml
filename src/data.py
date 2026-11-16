"""Load the Adult / Census Income dataset and build the train / validation / test split.

Role of each set (never mixed):
  train       fit the pipelines, cross-validate, tune hyperparameters
  validation  compare models, choose thresholds, fit the calibrator
  test        touched once, at the very end, for the final numbers
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import CFG, ROOT, SEED


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_raw():
    """Return the raw features and the binary target (1 = income > 50K)."""
    d = CFG["data"]
    adult = fetch_openml(
        name=d["openml_name"], version=d["openml_version"], as_frame=True,
        data_home=str(ROOT / d["cache_dir"]),
    )
    X = adult.data.copy()
    y = (adult.target.astype(str).str.strip() == d["positive_label"]).astype(int)
    y.name = "income_gt_50k"
    # Categoricals as plain strings, missing values as real NaN (handled in the pipeline).
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = X[col].astype(object).where(X[col].notna(), None)
    return X, y


def load_clean():
    """Raw data minus the columns we decided not to use and minus exact duplicates."""
    X, y = load_raw()
    # Exact duplicates of a full raw row (sampling weight included) are the same record
    # entered twice: they could land on both sides of a split, so they are removed.
    # Duplicates are detected BEFORE dropping fnlwgt: two different people can share the
    # same age / job / education, and those must be kept.
    dup = pd.concat([X, y], axis=1).duplicated()
    X = X.drop(columns=CFG["data"]["drop_columns"])
    return X.loc[~dup].reset_index(drop=True), y.loc[~dup].reset_index(drop=True)


def get_splits():
    """Stratified 60 / 20 / 20 split. One row = one person, so no group leakage."""
    X, y = load_clean()
    test_size, val_size = CFG["data"]["test_size"], CFG["data"]["val_size"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=SEED
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    s = get_splits()
    for name in ("train", "val", "test"):
        X, y = getattr(s, f"X_{name}"), getattr(s, f"y_{name}")
        print(f"{name:5s}: {len(X):6d} rows, {X.shape[1]} columns, {y.mean():.1%} positives")
