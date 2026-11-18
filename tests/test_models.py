"""Model-level checks: probabilities are valid and regularisation does what it says."""
import numpy as np
import pytest

from src.data import get_splits
from src.models import MODEL_NAMES, build_pipeline


@pytest.fixture(scope="module")
def small():
    s = get_splits()
    return s.X_train.head(3000), s.y_train.head(3000), s.X_val.head(500)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_probabilities_are_valid(small, name):
    X, y, Xv = small
    params = {"n_estimators": 30} if name == "random_forest" else None
    proba = build_pipeline(name, params).fit(X, y).predict_proba(Xv)
    assert proba.shape == (len(Xv), 2)
    assert np.allclose(proba.sum(axis=1), 1) and (proba >= 0).all()


def test_lasso_sets_coefficients_to_zero(small):
    X, y, _ = small
    l1 = build_pipeline("logreg", {"C": 0.01, "l1_ratio": 1.0}).fit(X, y)
    l2 = build_pipeline("logreg", {"C": 0.01, "l1_ratio": 0.0}).fit(X, y)
    zeros = lambda p: int((np.abs(p.named_steps["model"].coef_) < 1e-8).sum())
    assert zeros(l1) > zeros(l2)
