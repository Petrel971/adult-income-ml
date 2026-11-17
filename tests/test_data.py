"""Checks on the cleaned dataset: the decisions of the data section hold in the code."""
import pytest

from src.config import CFG
from src.data import load_clean


@pytest.fixture(scope="module")
def clean():
    return load_clean()


def test_dropped_columns_are_gone(clean):
    X, _ = clean
    for col in CFG["data"]["drop_columns"]:
        assert col not in X.columns


def test_exact_duplicates_removed(clean):
    # 48,842 raw records minus 52 exact duplicates (detected before fnlwgt is dropped).
    X, y = clean
    assert len(X) == len(y) == 48_842 - 52


def test_target_is_binary_with_expected_prevalence(clean):
    _, y = clean
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.239) < 0.002


def test_missing_values_are_real_nan_not_question_marks(clean):
    X, _ = clean
    categorical = X.select_dtypes(exclude="number")
    assert not (categorical == "?").any().any()
    assert X["workclass"].isna().sum() > 0
