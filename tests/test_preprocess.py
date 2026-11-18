"""The preprocessed matrix is clean and has the same columns on every set."""
import numpy as np
import pytest

from src.data import get_splits
from src.features import FeatureEngineer, build_preprocessor


@pytest.fixture(scope="module")
def fitted():
    s = get_splits()
    pre = build_preprocessor()
    pre.fit(FeatureEngineer().transform(s.X_train))
    return s, pre


def test_no_missing_values_after_preprocessing(fitted):
    s, pre = fitted
    for X in (s.X_train, s.X_val, s.X_test):
        assert not np.isnan(np.asarray(pre.transform(FeatureEngineer().transform(X)),
                                       dtype=float)).any()


def test_same_columns_on_every_set(fitted):
    s, pre = fitted
    widths = {pre.transform(FeatureEngineer().transform(X)).shape[1]
              for X in (s.X_train, s.X_val, s.X_test)}
    assert len(widths) == 1


def test_missing_occupation_gets_its_own_column(fitted):
    # Missing values are kept as a separate level, never merged into the most frequent one.
    s, pre = fitted
    names = list(pre.get_feature_names_out())
    occ = [n for n in names if n.startswith("occupation_")]
    X = pre.transform(FeatureEngineer().transform(s.X_val))
    missing = s.X_val["occupation"].isna().to_numpy()
    hit = [n for n in occ if (X[missing, names.index(n)] == 1).all()]
    assert len(hit) == 1 and (X[~missing, names.index(hit[0])] == 0).all()
