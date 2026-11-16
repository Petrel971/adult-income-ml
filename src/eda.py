"""Exploratory data analysis: figures and summary numbers used in the report.

Run on the full cleaned dataset for descriptive statistics only; no modelling decision is
taken from the test rows.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CFG, FIGURES_DIR, TABLES_DIR
from src.data import get_splits, load_clean, load_raw

sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREY = "#1F4E79", "#E07A1F", "#9AA5B1"


def rate_bar(ax, df, col, title, order=None, top=None):
    """Horizontal bars of the share of >50K per category."""
    rates = df.groupby(col)["y"].mean().sort_values()
    if order is not None:
        rates = rates.reindex(order)
    if top:
        rates = rates.tail(top)
    ax.barh(rates.index.astype(str), rates.values * 100, color=BLUE)
    ax.axvline(df["y"].mean() * 100, color=ORANGE, ls="--", lw=2)
    ax.set_title(title)
    ax.set_xlabel("% earning > 50K")


def main():
    X_raw, y_raw = load_raw()
    X, y = load_clean()
    df = X.assign(y=y)
    splits = get_splits()

    summary = {
        "n_rows_raw": int(len(X_raw)),
        "n_duplicates_removed": int(len(X_raw) - len(X)),
        "n_rows": int(len(X)),
        "n_features_raw": int(X_raw.shape[1]),
        "positive_rate": float(y.mean()),
        "missing": {c: int(n) for c, n in X.isna().sum().items() if n > 0},
        "split_sizes": {k: int(len(getattr(splits, f"X_{k}"))) for k in ("train", "val", "test")},
        "split_positive_rate": {k: float(getattr(splits, f"y_{k}").mean())
                                for k in ("train", "val", "test")},
        "rate_by_sex": df.groupby("sex")["y"].mean().to_dict(),
        "share_by_sex": df["sex"].value_counts(normalize=True).to_dict(),
        "rate_by_race": df.groupby("race")["y"].mean().to_dict(),
        "capital_gain_zero_share": float((X["capital-gain"] == 0).mean()),
        "native_us_share": float((X["native-country"] == "United-States").mean()),
        "rate_married": float(df[df["marital-status"].isin(
            ["Married-civ-spouse", "Married-AF-spouse"])]["y"].mean()),
        "rate_not_married": float(df[~df["marital-status"].isin(
            ["Married-civ-spouse", "Married-AF-spouse"])]["y"].mean()),
        "no_skill_accuracy": float(1 - y.mean()),
    }
    (TABLES_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: target imbalance and the sensitive attribute.
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    counts = y.value_counts().sort_index()
    axes[0].bar(["<= 50K", "> 50K"], counts.values, color=[GREY, BLUE])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v / len(y):.1%}", ha="center", va="bottom")
    axes[0].set_title("Target: 1 positive for 3 negatives")
    rate_bar(axes[1], df, "sex", "Income rate by sex (sensitive attribute)")
    rate_bar(axes[2], df, "race", "Income rate by race")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_target.png", dpi=150)
    plt.close(fig)

    # Figure 2: the strongest drivers.
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    rate_bar(axes[0, 0], df, "marital-status", "Marital status")
    edu = df.groupby("education-num")["y"].mean() * 100
    axes[0, 1].plot(edu.index, edu.values, marker="o", color=BLUE, lw=3)
    axes[0, 1].set_title("Years of education")
    axes[0, 1].set_xlabel("education-num")
    axes[0, 1].set_ylabel("% earning > 50K")
    age = df.groupby(pd.cut(df["age"], range(15, 95, 5)), observed=True)["y"].mean() * 100
    axes[1, 0].plot([i.mid for i in age.index], age.values, marker="o", color=BLUE, lw=3)
    axes[1, 0].set_title("Age: rises, peaks around 50, falls")
    axes[1, 0].set_xlabel("age")
    axes[1, 0].set_ylabel("% earning > 50K")
    hours = df.groupby(pd.cut(df["hours-per-week"], [0, 20, 34, 40, 50, 60, 100]),
                       observed=True)["y"].mean() * 100
    axes[1, 1].bar([str(i) for i in hours.index], hours.values, color=BLUE)
    axes[1, 1].set_title("Hours per week: a plateau, not a line")
    axes[1, 1].set_xlabel("hours-per-week band")
    axes[1, 1].set_ylabel("% earning > 50K")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_drivers.png", dpi=150)
    plt.close(fig)

    # Figure 3: why capital-gain needs a log.
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    gains = X.loc[X["capital-gain"] > 0, "capital-gain"]
    axes[0].hist(gains, bins=60, color=BLUE)
    axes[0].set_title(f"capital-gain when > 0 ({(X['capital-gain'] > 0).mean():.0%} of rows)")
    axes[0].set_xlabel("dollars")
    import numpy as np
    axes[1].hist(np.log1p(gains), bins=60, color=ORANGE)
    axes[1].set_title("Same values after log1p")
    axes[1].set_xlabel("log(1 + dollars)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_capital.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"sensitive attribute: {CFG['data']['sensitive_attribute']}")


if __name__ == "__main__":
    main()
