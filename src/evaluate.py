"""Metrics shared by every script, so that all models are reported the same way."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def expected_calibration_error(y_true, proba, n_bins=10):
    """ECE = sum over bins of (bin share) x |observed positive rate - mean confidence|."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def best_f1_threshold(y_true, proba):
    """Threshold maximising F1. Must be called on validation data, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_metrics(y_true, proba, threshold=0.5):
    """Ranking metrics (threshold-free), decision metrics (at threshold), probability metrics."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "threshold": threshold,
    }
