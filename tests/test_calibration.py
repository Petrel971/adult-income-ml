"""Calibration metrics behave as described in the report."""
import numpy as np

from src.calibrate import signed_gap
from src.evaluate import expected_calibration_error


def sample(shift=0.0, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p).astype(int)          # calibrated by construction
    return y, np.clip(p + shift, 0, 1)


def test_calibrated_scores_have_small_ece():
    y, p = sample()
    assert expected_calibration_error(y, p) < 0.02
    assert abs(signed_gap(y, p)) < 0.02


def test_signed_gap_detects_over_and_under_confidence():
    y, p = sample(shift=0.1)
    assert signed_gap(y, p) > 0.05
    y, p = sample(shift=-0.1)
    assert signed_gap(y, p) < -0.05
