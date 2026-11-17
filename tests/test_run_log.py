"""The run log is complete and consistent: one best trial per model, as reported."""
import pandas as pd

from src.config import RUN_LOG
from src.runlog_summary import summarise


def test_run_log_has_one_best_trial_per_model():
    log = pd.read_csv(RUN_LOG)
    assert len(log) == 110
    assert (log.groupby("model")["is_best"].sum() == 1).all()


def test_flagged_best_is_the_highest_cv_score():
    log = pd.read_csv(RUN_LOG)
    summary = summarise(log).set_index("model")
    flagged = log[log["is_best"] == 1].set_index("model")["trial"]
    assert (summary["best_trial"] == flagged.loc[summary.index]).all()
