"""Fairness and bias audit on the sensitive attribute (sex), with one mitigation.

Definitions (A = group, Y = true label, D = decision):
  demographic parity   P(D=1 | A=a) equal across groups          -> selection-rate gap
  equal opportunity    P(D=1 | Y=1, A=a) equal across groups     -> TPR gap
  equalized odds       equal TPR and equal FPR across groups     -> max(TPR gap, FPR gap)

Mitigation (post-processing): one decision threshold per group, chosen on the validation
set to equalise the true positive rates (equal opportunity) at the best possible F1.
Before / after are compared on the untouched test set.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.config import CFG, FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.data import get_splits
from src.evaluate import expected_calibration_error
from src.models import build_pipeline

ATTR = CFG["data"]["sensitive_attribute"]


def group_table(y, proba, pred, groups):
    """Per-group rates. Sizes are reported: a rate on a tiny group proves nothing."""
    rows = []
    for g in sorted(pd.unique(groups)):
        m = (groups == g).to_numpy()
        yt, pr = y[m], pred[m]
        rows.append({
            "group": g, "n": int(m.sum()), "base_rate": yt.mean(),
            "selection_rate": pr.mean(),
            "tpr": pr[yt == 1].mean() if (yt == 1).any() else np.nan,
            "fpr": pr[yt == 0].mean() if (yt == 0).any() else np.nan,
            "precision": precision_score(yt, pr, zero_division=0),
            "pr_auc": average_precision_score(yt, proba[m]),
            "ece": expected_calibration_error(yt, proba[m]),
        })
    return pd.DataFrame(rows)


def gaps(table, privileged="Male", unprivileged="Female"):
    a = table.set_index("group").loc[privileged]
    b = table.set_index("group").loc[unprivileged]
    return {
        "demographic_parity_diff": a["selection_rate"] - b["selection_rate"],
        "disparate_impact_ratio": b["selection_rate"] / a["selection_rate"],
        "equal_opportunity_diff": a["tpr"] - b["tpr"],
        "fpr_diff": a["fpr"] - b["fpr"],
        "equalized_odds_diff": max(abs(a["tpr"] - b["tpr"]), abs(a["fpr"] - b["fpr"])),
    }


def performance(y, pred):
    return {"f1": f1_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "accuracy": float((y == pred).mean())}


def fit_group_thresholds(y, proba, groups, tolerance=0.01):
    """Grid-search one threshold per group on validation data.

    Keep the pairs whose TPR gap is below `tolerance`, then take the pair with the best F1.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    male, female = (groups == "Male").to_numpy(), (groups == "Female").to_numpy()
    tpr = {g: np.array([(proba[m][y[m] == 1] >= t).mean() for t in grid])
           for g, m in (("Male", male), ("Female", female))}
    best, best_f1 = None, -1.0
    for i, t_m in enumerate(grid):
        for j, t_f in enumerate(grid):
            if abs(tpr["Male"][i] - tpr["Female"][j]) > tolerance:
                continue
            pred = np.where(male, proba >= t_m, proba >= t_f).astype(int)
            score = f1_score(y, pred)
            if score > best_f1:
                best, best_f1 = {"Male": float(t_m), "Female": float(t_f)}, score
    return best


def main():
    s = get_splits()
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    pipe, t = final["pipeline"], final["threshold"]
    y_val, y_test = s.y_val.to_numpy(), s.y_test.to_numpy()
    p_val = pipe.predict_proba(s.X_val)[:, 1]
    p_test = pipe.predict_proba(s.X_test)[:, 1]
    g_val, g_test = s.X_val[ATTR], s.X_test[ATTR]

    # 1. Audit of the final model with its single threshold.
    pred_before = (p_test >= t).astype(int)
    before = group_table(y_test, p_test, pred_before, g_test)

    # 2. Mitigation: group thresholds learned on validation, applied to test.
    group_t = fit_group_thresholds(y_val, p_val, g_val)
    pred_after = np.where((g_test == "Male").to_numpy(), p_test >= group_t["Male"],
                          p_test >= group_t["Female"]).astype(int)
    after = group_table(y_test, p_test, pred_after, g_test)

    # 3. Control experiment: "fairness through unawareness" (retrain without the attribute).
    blind = build_pipeline(final["model_name"], final["params"]).fit(
        s.X_train.assign(**{ATTR: "Hidden"}), s.y_train)
    p_blind = blind.predict_proba(s.X_test.assign(**{ATTR: "Hidden"}))[:, 1]
    pred_blind = (p_blind >= t).astype(int)
    unaware = group_table(y_test, p_blind, pred_blind, g_test)

    summary = pd.DataFrame([
        {"setting": "Before (single threshold)", **performance(y_test, pred_before), **gaps(before)},
        {"setting": "Sex removed from the features", **performance(y_test, pred_blind), **gaps(unaware)},
        {"setting": "After (group thresholds)", **performance(y_test, pred_after), **gaps(after)},
    ])
    before.assign(setting="before").to_csv(TABLES_DIR / "fairness_groups_before.csv", index=False)
    after.assign(setting="after").to_csv(TABLES_DIR / "fairness_groups_after.csv", index=False)
    summary.to_csv(TABLES_DIR / "fairness_summary.csv", index=False)
    race = group_table(y_test, p_test, pred_before, s.X_test["race"])
    race.to_csv(TABLES_DIR / "fairness_race.csv", index=False)
    (TABLES_DIR / "fairness_thresholds.json").write_text(json.dumps(
        {"single_threshold": t, "group_thresholds": group_t}, indent=2))

    print(before.round(3).to_string(index=False))
    print(after.round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    print(race.round(3).to_string(index=False))
    print(group_t)

    # Figure: rates by sex, before and after.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, col, title in zip(axes, ["selection_rate", "tpr", "fpr"],
                              ["Selection rate\n(demographic parity)",
                               "True positive rate\n(equal opportunity)",
                               "False positive rate\n(equalized odds, with TPR)"]):
        x = np.arange(2)
        for k, (tab, label, alpha) in enumerate([(before, "before", 0.45), (after, "after", 1.0)]):
            vals = tab.set_index("group").loc[["Female", "Male"], col].to_numpy()
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=["#E07A1F", "#1F4E79"],
                          alpha=alpha, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                        fontsize=10)
        ax.set_xticks(x, ["Female", "Male"])
        ax.set_title(title)
    axes[0].set_ylabel("rate on the test set")
    axes[0].text(0.02, 0.93, "light = before, solid = after", transform=axes[0].transAxes,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fairness.png", dpi=150)
    plt.close(fig)

    final["group_thresholds"] = group_t
    joblib.dump(final, MODELS_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
