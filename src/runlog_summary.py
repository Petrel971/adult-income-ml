"""Summary of the run log: one line per model, read from runs/run_log.csv.

Usage:  python -m src.runlog_summary
Writes reports/tables/run_log_summary.csv.
"""
import json

import pandas as pd

from src.config import RUN_LOG, TABLES_DIR


def summarise(log):
    rows = []
    for model, g in log.groupby("model", sort=False):
        best = g.loc[g["cv_pr_auc_mean"].idxmax()]
        rows.append({
            "model": model, "n_trials": len(g),
            "best_trial": int(best["trial"]),
            "best_cv_pr_auc": best["cv_pr_auc_mean"], "best_cv_std": best["cv_pr_auc_std"],
            "median_cv_pr_auc": g["cv_pr_auc_mean"].median(),
            "total_fit_minutes": round(g["fit_seconds"].sum() / 60, 1),
            "best_params": json.dumps(json.loads(best["params"])),
            "best_model_file": best["artifact_model"],
        })
    return pd.DataFrame(rows)


def main():
    table = summarise(pd.read_csv(RUN_LOG))
    table.to_csv(TABLES_DIR / "run_log_summary.csv", index=False)
    print(table.drop(columns="best_params").to_string(index=False))


if __name__ == "__main__":
    main()
