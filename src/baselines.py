"""First experiments, all cross-validated on the training set only.

1. No-skill baseline: the floor every model must beat.
2. Feature-engineering ablation: same model with and without the engineered features.
3. Class-imbalance remedies: nothing / class weights / SMOTE / threshold moving.

Each configuration is scored on out-of-fold predictions: every training row is predicted
by a model that never saw it.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import CFG, SEED, TABLES_DIR
from src.data import get_splits
from src.evaluate import best_f1_threshold, compute_metrics
from src.models import build_pipeline

COLS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier", "ece",
        "threshold"]


def oof_proba(pipe, X, y):
    cv = StratifiedKFold(CFG["cv"]["n_splits"], shuffle=True, random_state=SEED)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def main():
    s = get_splits()
    X, y = s.X_train, s.y_train
    rows = []

    def add(experiment, model, variant, proba, threshold=0.5):
        rows.append({"experiment": experiment, "model": model, "variant": variant,
                     **compute_metrics(y, proba, threshold)})

    # 1. No-skill baseline.
    add("baseline", "dummy", "always predict the prior", oof_proba(build_pipeline("dummy"), X, y))

    # 2. Feature-engineering ablation.
    for model in ("logreg", "hist_gb"):
        for flag, label in ((False, "raw features"), (True, "engineered features")):
            proba = oof_proba(build_pipeline(model, add_features=flag), X, y)
            add("feature_engineering", model, label, proba)

    # 3. Imbalance remedies.
    for model in ("logreg", "hist_gb"):
        plain = oof_proba(build_pipeline(model), X, y)
        add("imbalance", model, "no remedy (threshold 0.5)", plain)
        add("imbalance", model, "class weights", oof_proba(
            build_pipeline(model, imbalance="class_weight"), X, y))
        add("imbalance", model, "SMOTE", oof_proba(
            build_pipeline(model, imbalance="smote"), X, y))
        # Threshold moving: same model as "no remedy", only the decision rule changes.
        # (Threshold picked on the same out-of-fold predictions: slightly optimistic here;
        # for the final models it is picked on the validation set.)
        add("imbalance", model, "threshold moving", plain, best_f1_threshold(y, plain))

    df = pd.DataFrame(rows)[["experiment", "model", "variant"] + COLS]
    df.to_csv(TABLES_DIR / "baselines.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
