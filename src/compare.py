"""Modelling progression: compare the three models with the same metrics, pick the final one.

  train       fit (default and tuned hyperparameters)
  validation  compare the models, choose each decision threshold
  train (CV)  nested cross-validation of the selected model: unbiased score of the whole
              "tune then fit" procedure
  test        final numbers, computed once, thresholds frozen from validation
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import StratifiedKFold

from src.config import CFG, FIGURES_DIR, MODELS_DIR, SEED, TABLES_DIR
from src.data import get_splits
from src.evaluate import best_f1_threshold, compute_metrics
from src.models import MODEL_NAMES, PRETTY, build_pipeline, suggest_params
from src.tune import cv_scores

COLORS = {"logreg": "#9AA5B1", "random_forest": "#E07A1F", "hist_gb": "#1F4E79"}


def nested_cv(name, X, y):
    """Outer loop evaluates, inner loop tunes. The outer folds never see the tuning."""
    n = CFG["tuning"]["nested"]
    outer = StratifiedKFold(n["outer_splits"], shuffle=True, random_state=SEED + 1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scores = []
    for train_idx, test_idx in outer.split(X, y):
        X_in, y_in = X.iloc[train_idx], y.iloc[train_idx]
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(
            lambda t: cv_scores(build_pipeline(name, suggest_params(name, t)), X_in, y_in,
                                n_splits=n["inner_splits"])["cv_pr_auc_mean"],
            n_trials=n["n_trials"])
        fitted = build_pipeline(name, study.best_params).fit(X_in, y_in)
        proba = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[test_idx], proba))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    s = get_splits()
    best = json.loads((TABLES_DIR / "best_params.json").read_text())

    val_rows, test_rows, fitted, thresholds = [], [], {}, {}

    dummy = build_pipeline("dummy").fit(s.X_train, s.y_train)
    val_rows.append({"model": PRETTY["dummy"], "version": "-", **compute_metrics(
        s.y_val, dummy.predict_proba(s.X_val)[:, 1])})
    test_rows.append({"model": PRETTY["dummy"], **compute_metrics(
        s.y_test, dummy.predict_proba(s.X_test)[:, 1])})

    for name in MODEL_NAMES:
        default = build_pipeline(name).fit(s.X_train, s.y_train)
        val_rows.append({"model": PRETTY[name], "version": "default, threshold 0.5",
                         **compute_metrics(s.y_val, default.predict_proba(s.X_val)[:, 1])})

        tuned = joblib.load(MODELS_DIR / f"{name}_best.joblib")
        p_val = tuned.predict_proba(s.X_val)[:, 1]
        val_rows.append({"model": PRETTY[name], "version": "tuned, threshold 0.5",
                         **compute_metrics(s.y_val, p_val)})
        t = best_f1_threshold(s.y_val, p_val)
        val_rows.append({"model": PRETTY[name], "version": "tuned, threshold from validation",
                         **compute_metrics(s.y_val, p_val, t)})
        fitted[name], thresholds[name] = tuned, t

    val = pd.DataFrame(val_rows)
    val.to_csv(TABLES_DIR / "model_comparison_val.csv", index=False)
    print(val.round(4).to_string(index=False))

    # Model selection on validation PR-AUC (threshold-free, honest under imbalance).
    val_pr = {n: average_precision_score(s.y_val, fitted[n].predict_proba(s.X_val)[:, 1])
              for n in MODEL_NAMES}
    final_name = max(val_pr, key=val_pr.get)
    print(f"\nselected on validation: {final_name}")

    nested_mean, nested_std = nested_cv(final_name, s.X_train, s.y_train)
    print(f"nested CV PR-AUC: {nested_mean:.4f} +/- {nested_std:.4f} "
          f"(best inner CV score was {best[final_name]['cv_pr_auc']:.4f})")

    # Test set: touched here for the first time.
    probas = {n: fitted[n].predict_proba(s.X_test)[:, 1] for n in MODEL_NAMES}
    for name in MODEL_NAMES:
        test_rows.append({"model": PRETTY[name],
                          **compute_metrics(s.y_test, probas[name], thresholds[name])})
    test = pd.DataFrame(test_rows)
    test.to_csv(TABLES_DIR / "model_comparison_test.csv", index=False)
    print("\n" + test.round(4).to_string(index=False))

    # PR and ROC curves on the test set.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name in MODEL_NAMES:
        prec, rec, _ = precision_recall_curve(s.y_test, probas[name])
        fpr, tpr, _ = roc_curve(s.y_test, probas[name])
        axes[0].plot(rec, prec, color=COLORS[name], lw=2.5, label=PRETTY[name])
        axes[1].plot(fpr, tpr, color=COLORS[name], lw=2.5, label=PRETTY[name])
    axes[0].axhline(s.y_test.mean(), color="black", ls="--", lw=1, label="no skill (prevalence)")
    axes[1].plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="no skill")
    axes[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test)")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_roc_curves.png", dpi=150)
    plt.close(fig)

    # Threshold choice of the final model, on validation.
    p_val = fitted[final_name].predict_proba(s.X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(s.y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thr, prec[:-1], color="#1F4E79", lw=2.5, label="precision")
    ax.plot(thr, rec[:-1], color="#E07A1F", lw=2.5, label="recall")
    ax.plot(thr, f1[:-1], color="#2E7D32", lw=2.5, label="F1")
    ax.axvline(thresholds[final_name], color="black", ls="--",
               label=f"chosen threshold {thresholds[final_name]:.2f}")
    ax.axvline(0.5, color="grey", ls=":", label="default 0.5")
    ax.set(xlabel="decision threshold", title="Threshold moving on validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold.png", dpi=150)
    plt.close(fig)

    # Confusion matrix of the final model on test.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        s.y_test, (probas[final_name] >= thresholds[final_name]).astype(int),
        display_labels=["<= 50K", "> 50K"], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{PRETTY[final_name]} on test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    joblib.dump({
        "pipeline": fitted[final_name], "model_name": final_name,
        "threshold": thresholds[final_name], "params": best[final_name]["params"],
        "columns": list(s.X_train.columns),
    }, MODELS_DIR / "final_model.joblib")
    (TABLES_DIR / "selection.json").write_text(json.dumps({
        "final_model": final_name, "validation_pr_auc": val_pr, "thresholds": thresholds,
        "inner_cv_pr_auc": best[final_name]["cv_pr_auc"],
        "nested_cv_pr_auc_mean": nested_mean, "nested_cv_pr_auc_std": nested_std,
    }, indent=2))


if __name__ == "__main__":
    main()
