"""Hyperparameter tuning with Optuna (TPE) and stratified 5-fold cross-validation.

Only the training set is used. Every trial is appended to runs/run_log.csv with its
parameters, its cross-validated metrics and the artifacts it corresponds to: the fitted\nmodel file of the trial and the history plot of its study.

Usage:  python -m src.tune                (all three models)
        python -m src.tune hist_gb        (one model)
"""
import csv
import json
import shutil
import sys
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CFG, FIGURES_DIR, MODELS_DIR, ROOT, RUN_LOG, SEED, TABLES_DIR
from src.data import get_splits
from src.models import MODEL_NAMES, PRETTY, build_pipeline, suggest_params

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}
LOG_FIELDS = ["run_id", "timestamp", "model", "trial", "params", "cv_pr_auc_mean",
              "cv_pr_auc_std", "cv_roc_auc_mean", "cv_brier_mean", "fit_seconds",
              "is_best", "artifact_model", "artifact_plot"]


def cv_scores(pipe, X, y, n_splits=None, seed=SEED):
    cv = StratifiedKFold(n_splits or CFG["cv"]["n_splits"], shuffle=True, random_state=seed)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {
        "cv_pr_auc_mean": res["test_pr_auc"].mean(),
        "cv_pr_auc_std": res["test_pr_auc"].std(),
        "cv_roc_auc_mean": res["test_roc_auc"].mean(),
        "cv_brier_mean": -res["test_brier"].mean(),
    }


def make_objective(name, X, y, records):
    trials_dir = MODELS_DIR / "trials"
    trials_dir.mkdir(exist_ok=True)

    def objective(trial):
        params = suggest_params(name, trial)
        start = time.time()
        scores = cv_scores(build_pipeline(name, params), X, y)
        # Artifact of the trial: the same configuration refitted on the whole training set.
        model_path = trials_dir / f"{name}_trial{trial.number:03d}.joblib"
        joblib.dump(build_pipeline(name, params).fit(X, y), model_path, compress=3)
        records.append({"trial": trial.number, "params": params, **scores,
                        "fit_seconds": round(time.time() - start, 1),
                        "artifact_model": str(model_path.relative_to(ROOT))})
        return scores["cv_pr_auc_mean"]
    return objective


def plot_history(name, study):
    values = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(len(values)), values, color="#9AA5B1", label="trial")
    ax.plot(np.maximum.accumulate(values), color="#1F4E79", lw=2.5, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("5-fold CV PR-AUC")
    ax.set_title(f"Optuna history: {PRETTY[name]}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"optuna_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tune_model(name, X, y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    records = []
    study.optimize(make_objective(name, X, y, records), n_trials=CFG["tuning"]["n_trials"][name])

    # Artifacts of the study: its history plot, and a copy of the best trial's model.
    plot_path = plot_history(name, study)
    best_record = next(r for r in records if r["trial"] == study.best_trial.number)
    shutil.copy(ROOT / best_record["artifact_model"], MODELS_DIR / f"{name}_best.joblib")

    run_id = f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    rows = [{
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name, "trial": r["trial"], "params": json.dumps(r["params"]),
        "cv_pr_auc_mean": round(r["cv_pr_auc_mean"], 5),
        "cv_pr_auc_std": round(r["cv_pr_auc_std"], 5),
        "cv_roc_auc_mean": round(r["cv_roc_auc_mean"], 5),
        "cv_brier_mean": round(r["cv_brier_mean"], 5),
        "fit_seconds": r["fit_seconds"],
        "is_best": int(r["trial"] == study.best_trial.number),
        "artifact_model": r["artifact_model"],
        "artifact_plot": str(plot_path.relative_to(ROOT)),
    } for r in records]
    # A new study of a model replaces its previous lines; other models are kept.
    previous = []
    if RUN_LOG.exists():
        with open(RUN_LOG, newline="") as f:
            previous = [r for r in csv.DictReader(f) if r["model"] != name]
    with open(RUN_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(previous + rows)
    print(f"{name}: best CV PR-AUC {study.best_value:.4f} with {study.best_params}")
    return study.best_params, study.best_value


def main():
    names = sys.argv[1:] or MODEL_NAMES
    s = get_splits()
    path = TABLES_DIR / "best_params.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        params, value = tune_model(name, s.X_train, s.y_train)
        best[name] = {"params": params, "cv_pr_auc": value}
        path.write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
