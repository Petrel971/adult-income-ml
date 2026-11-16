"""Explainability of the final model: global importance and local SHAP explanations.

SHAP values are computed on the validation set, in the raw score (log-odds) of the boosted
trees: f(x) = base value + sum of contributions, probability = sigmoid(f(x)).
They describe the model, not the causes of income.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.config import FIGURES_DIR, MODELS_DIR, SEED, TABLES_DIR
from src.data import get_splits


def original_feature(name, raw_columns):
    """Map a one-hot column such as 'occupation_Sales' back to 'occupation'."""
    for col in sorted(raw_columns, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


def main():
    s = get_splits()
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    pipe, threshold = final["pipeline"], final["threshold"]
    model = pipe.named_steps["model"]

    # Matrix actually seen by the model, with readable column names.
    engineered = pipe.named_steps["features"].transform(s.X_val)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    Xt = pd.DataFrame(pipe[:-1].transform(s.X_val), columns=names, index=s.X_val.index)

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    if sv.values.ndim == 3:            # some models return one slice per class
        sv = sv[:, :, 1]

    # Global: beeswarm (direction + magnitude) and mean |SHAP| grouped by original feature.
    shap.plots.beeswarm(sv, max_display=14, show=False)
    plt.title("SHAP beeswarm (validation set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    groups = [original_feature(n, engineered.columns) for n in names]
    grouped = (pd.DataFrame(np.abs(sv.values), columns=names).T.groupby(groups).sum().T
               .mean().sort_values(ascending=False))
    grouped.rename("mean_abs_shap").to_csv(TABLES_DIR / "shap_importance.csv",
                                           index_label="feature")
    fig, ax = plt.subplots(figsize=(9, 6))
    grouped.iloc[::-1].plot.barh(ax=ax, color="#1F4E79")
    ax.set(xlabel="mean |SHAP| (log-odds), one-hot columns summed",
           title="Global importance by original feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_global.png", dpi=150)
    plt.close(fig)

    # Second opinion: permutation importance on raw columns (drop in validation PR-AUC).
    perm = permutation_importance(pipe, s.X_val, s.y_val, scoring="average_precision",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=s.X_val.columns).sort_values()
    perm_s.rename("pr_auc_drop").to_csv(TABLES_DIR / "permutation_importance.csv",
                                        index_label="feature")
    fig, ax = plt.subplots(figsize=(9, 6))
    perm_s.plot.barh(ax=ax, color="#E07A1F")
    ax.set(xlabel="drop in validation PR-AUC when the column is shuffled",
           title="Permutation importance (raw columns)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)

    # Shape of the effect: partial dependence (average) and ICE curves (individuals) of the
    # four main numeric features, computed on the whole Pipeline with raw columns.
    sample = s.X_val.sample(1000, random_state=SEED)
    numeric = sample.select_dtypes("number").columns
    sample = sample.astype({c: float for c in numeric})     # PDP grids need float columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    PartialDependenceDisplay.from_estimator(
        pipe, sample, ["age", "education-num", "hours-per-week", "capital-gain"],
        kind="both", subsample=80, random_state=SEED, ax=axes, grid_resolution=40,
        ice_lines_kw={"color": "#9AA5B1", "alpha": 0.25, "linewidth": 0.6},
        pd_line_kw={"color": "#E07A1F", "linewidth": 3, "label": "average (PDP)"})
    for ax in axes:
        ax.set_ylabel("")
    axes[0].set_ylabel("predicted probability of > 50K")
    fig.suptitle("Partial dependence (orange) and individual ICE curves (grey), validation set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pdp.png", dpi=150)
    plt.close(fig)

    # Local: one confident true positive, one confident true negative, and the two most
    # confident errors (a false positive and a false negative).
    proba = pipe.predict_proba(s.X_val)[:, 1]
    y = s.y_val.to_numpy()
    pred = (proba >= threshold).astype(int)
    idx = np.arange(len(y))
    cases = {
        "true_positive": idx[(y == 1) & (pred == 1)][np.argmax(proba[(y == 1) & (pred == 1)])],
        "true_negative": idx[(y == 0) & (pred == 0)][np.argmin(proba[(y == 0) & (pred == 0)])],
        "false_positive": idx[(y == 0) & (pred == 1)][np.argmax(proba[(y == 0) & (pred == 1)])],
        "false_negative": idx[(y == 1) & (pred == 0)][np.argmin(proba[(y == 1) & (pred == 0)])],
        "borderline": idx[np.argmin(np.abs(proba - threshold))],
    }
    local = {}
    for case, i in cases.items():
        shap.plots.waterfall(sv[i], max_display=9, show=False)
        plt.title(f"{case.replace('_', ' ')}: p = {proba[i]:.3f}, true label = {y[i]}")
        plt.savefig(FIGURES_DIR / f"shap_local_{case}.png", dpi=150, bbox_inches="tight")
        plt.close()
        order = np.argsort(-np.abs(sv.values[i]))[:5]
        local[case] = {
            "probability": float(proba[i]), "true_label": int(y[i]),
            "raw_row": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                        for k, v in s.X_val.iloc[i].items()},
            "top_contributions": [[names[j], float(sv.values[i][j])] for j in order],
        }
    (TABLES_DIR / "shap_local.json").write_text(json.dumps({
        "base_value_log_odds": float(np.ravel(sv.base_values)[0]), "cases": local}, indent=2))
    print(grouped.round(3).to_string())
    print(perm_s.round(4).to_string())
    print(json.dumps(local, indent=1)[:3000])


if __name__ == "__main__":
    main()
