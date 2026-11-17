"""Checks on the shared metrics: what the imbalance section relies on."""
import numpy as np

from src.evaluate import best_f1_threshold, compute_metrics


def test_moving_the_threshold_keeps_ranking_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.24).astype(int)
    proba = np.clip(0.24 + 0.4 * (y - 0.24) + rng.normal(0, 0.2, 2000), 0, 1)
    low, high = compute_metrics(y, proba, 0.3), compute_metrics(y, proba, 0.6)
    assert low["pr_auc"] == high["pr_auc"] and low["roc_auc"] == high["roc_auc"]
    assert low["recall"] > high["recall"]


def test_always_no_has_high_accuracy_but_zero_f1():
    y = np.array([0] * 76 + [1] * 24)
    m = compute_metrics(y, np.zeros(100) + 0.1, threshold=0.5)
    assert m["accuracy"] == 0.76 and m["f1"] == 0.0


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    t = best_f1_threshold(y, proba)
    assert compute_metrics(y, proba, t)["f1"] == 1.0
