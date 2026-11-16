"""Checks on the feature engineering and on the Pipeline contract."""
import numpy as np
import pandas as pd
import pytest

from src.data import get_splits
from src.evaluate import best_f1_threshold, expected_calibration_error
from src.features import ENGINEERED, FeatureEngineer
from src.models import MODEL_NAMES, build_pipeline


@pytest.fixture(scope="module")
def splits():
    return get_splits()


def test_split_is_stratified_and_disjoint(splits):
    rates = [splits.y_train.mean(), splits.y_val.mean(), splits.y_test.mean()]
    assert max(rates) - min(rates) < 0.002
    idx = [set(splits.X_train.index), set(splits.X_val.index), set(splits.X_test.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])


def test_feature_engineer_adds_expected_columns(splits):
    out = FeatureEngineer().transform(splits.X_train.head(200))
    assert set(ENGINEERED) <= set(out.columns)
    assert "capital-gain" not in out.columns
    assert out["is_married"].isin([0, 1]).all()


def test_feature_engineer_is_stateless(splits):
    """Transforming a row alone or inside a batch gives the same result: nothing is learned."""
    batch = FeatureEngineer().transform(splits.X_train.head(50))
    alone = FeatureEngineer().transform(splits.X_train.head(1))
    pd.testing.assert_frame_equal(batch.head(1), alone)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_handles_missing_and_unseen_categories(splits, name):
    X, y = splits.X_train.head(1500), splits.y_train.head(1500)
    params = {"n_estimators": 20} if name == "random_forest" else None
    pipe = build_pipeline(name, params).fit(X, y)
    row = splits.X_val.head(1).copy()
    row["occupation"] = None
    row["native-country"] = "Atlantis"
    proba = pipe.predict_proba(row)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


def test_threshold_and_ece_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert 0.4 < best_f1_threshold(y, p) <= 0.6
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
