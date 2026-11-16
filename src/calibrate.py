"""Calibration diagnostics of the final model: reliability curve, Brier score, ECE.

Three sets, three roles: the model is fitted on train, calibrators are fitted on validation,
everything is evaluated on the untouched test set.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.data import get_splits
from src.evaluate import compute_metrics
from src.models import build_pipeline


def signed_gap(y, p, n_bins=10):
    """Weighted mean of (confidence - observed rate). > 0: over-confident, < 0: under."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * (p[bins == b].mean() - y[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def main():
    s = get_splits()
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    pipe = final["pipeline"]
    p_val = pipe.predict_proba(s.X_val)[:, 1]
    p_test = pipe.predict_proba(s.X_test)[:, 1]

    # Post-hoc calibrators, fitted on validation only.
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, s.y_val)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression().fit(logit(p_val).reshape(-1, 1), s.y_val)

    # Contrast: the same model trained with class weights (the remedy we rejected).
    weighted = build_pipeline(final["model_name"], final["params"],
                              imbalance="class_weight").fit(s.X_train, s.y_train)

    variants = {
        "Final model, as is": p_test,
        "Final model + isotonic": iso.predict(p_test),
        "Final model + Platt": platt.predict_proba(logit(p_test).reshape(-1, 1))[:, 1],
        "Same model with class weights": weighted.predict_proba(s.X_test)[:, 1],
    }
    rows = []
    for label, p in variants.items():
        m = compute_metrics(s.y_test, p)
        rows.append({"variant": label, "brier": m["brier"], "ece": m["ece"],
                     "signed_gap": signed_gap(s.y_test, p), "roc_auc": m["roc_auc"],
                     "pr_auc": m["pr_auc"], "mean_predicted": float(np.mean(p)),
                     "observed_rate": float(s.y_test.mean())})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "calibration.csv", index=False)
    print(table.round(4).to_string(index=False))

    # Per-bin table of the final model (for the report).
    bins = np.minimum((p_test * 10).astype(int), 9)
    per_bin = pd.DataFrame({"bin": bins, "p": p_test, "y": s.y_test.to_numpy()}).groupby("bin").agg(
        n=("y", "size"), confidence=("p", "mean"), observed=("y", "mean")).reset_index()
    per_bin.to_csv(TABLES_DIR / "calibration_bins.csv", index=False)
    print(per_bin.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]})
    styles = {"Final model, as is": ("#1F4E79", "o-"),
              "Final model + Platt": ("#2E7D32", "s--"),
              "Same model with class weights": ("#E07A1F", "^-")}
    axes[0].plot([0, 1], [0, 1], color="black", ls=":", label="perfect calibration")
    for label, (color, style) in styles.items():
        obs, conf = calibration_curve(s.y_test, variants[label], n_bins=10)
        axes[0].plot(conf, obs, style, color=color, lw=2.2, ms=7, label=label)
    axes[0].set(xlabel="mean predicted probability (confidence)",
                ylabel="observed share of > 50K", title="Reliability curve (test set)")
    axes[0].text(0.62, 0.25, "below the diagonal:\nover-confident", fontsize=11)
    axes[0].text(0.08, 0.72, "above the diagonal:\nunder-confident", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.62))
    axes[1].hist(p_test, bins=20, color="#1F4E79")
    axes[1].set(xlabel="predicted probability", ylabel="number of people",
                title="Where the predictions fall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    # No calibrator is saved: the diagnostics show the raw probabilities can be served as is.


if __name__ == "__main__":
    main()
