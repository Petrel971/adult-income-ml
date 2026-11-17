"""Checks on the fairness audit and on the group-threshold mitigation."""
import numpy as np
import pandas as pd

from src.fairness import fit_group_thresholds, gaps, group_table


def toy():
    rng = np.random.default_rng(1)
    groups = pd.Series(np.where(rng.random(3000) < 0.67, "Male", "Female"))
    y = (rng.random(3000) < np.where(groups == "Male", 0.31, 0.10)).astype(int)
    # Scores shifted down for women: same threshold gives them a lower TPR.
    proba = np.clip(0.2 + 0.5 * y - 0.15 * (groups == "Female") + rng.normal(0, 0.12, 3000), 0, 1)
    return y, proba, groups


def test_single_threshold_shows_equal_opportunity_gap():
    y, proba, groups = toy()
    table = group_table(y, proba, (proba >= 0.5).astype(int), groups)
    assert gaps(table)["equal_opportunity_diff"] > 0.1


def test_group_thresholds_close_the_gap():
    y, proba, groups = toy()
    t = fit_group_thresholds(y, proba, groups)
    assert t["Female"] < t["Male"]
    pred = np.where(groups == "Male", proba >= t["Male"], proba >= t["Female"]).astype(int)
    assert abs(gaps(group_table(y, proba, pred, groups))["equal_opportunity_diff"]) <= 0.02
