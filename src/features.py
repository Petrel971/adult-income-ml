"""Feature engineering and preprocessing, all inside scikit-learn transformers.

Everything that learns from data (imputer, scaler, encoder) lives in the Pipeline, so it is
re-fitted inside every cross-validation fold: no leakage from validation rows.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}
ENGINEERED = ["capital_gain_log", "capital_loss_log", "has_capital", "is_married",
              "hours_band", "native_us"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add row-wise features. Stateless: nothing is learned, so it cannot leak.

    capital_gain_log / capital_loss_log
        capital-gain is zero for ~92% of people and reaches 99,999 for a few: log1p tames
        the tail so that a linear model is not driven by a handful of rows.
    has_capital
        1 if the person declares any capital gain or loss. Owning investments is a signal
        on its own, whatever the amount.
    is_married
        1 for a married person living with their spouse. marital-status has 7 levels but
        the income gap is essentially "married couple" versus everything else.
    hours_band
        part-time (<35h), full-time (35-40h), overtime (41-59h), extreme (60h+). The effect
        of hours on income is not linear: it plateaus, which bands can express.
    native_us
        1 if born in the United States. native-country has 41 levels, 90% of rows are US,
        and most other levels have a few dozen rows: one flag is more stable.
    """

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.add_features:
            return X
        X["capital_gain_log"] = np.log1p(X["capital-gain"].astype(float))
        X["capital_loss_log"] = np.log1p(X["capital-loss"].astype(float))
        X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
        X["is_married"] = X["marital-status"].isin(MARRIED).astype(int)
        X["hours_band"] = pd.cut(
            X["hours-per-week"].astype(float), bins=[0, 34, 40, 59, 200],
            labels=["part_time", "full_time", "overtime", "extreme"],
        ).astype(object)
        X["native_us"] = (X["native-country"] == "United-States").astype(int)
        # The raw amounts are replaced by their log version.
        return X.drop(columns=["capital-gain", "capital-loss"])


def build_preprocessor(scale=True):
    """Impute + scale the numeric columns, impute + one-hot encode the categorical ones.

    Missing values only exist in workclass, occupation and native-country. They are not
    missing at random (e.g. no occupation for people who never worked), so they get their
    own "Missing" category instead of being replaced by the most frequent value.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = Pipeline([
        # Missing categoricals are stored as Python None (see src/data.py), not NaN.
        ("impute", SimpleImputer(missing_values=None, strategy="constant",
                                 fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric), make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
