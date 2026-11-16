"""The three models of the progression, their Optuna search spaces, and the single Pipeline.

Progression (each step is compared to the previous one with the same metrics):
  logreg         regularised logistic regression: linear, interpretable reference
  random_forest  bagging of deep trees: non-linearities and interactions, lower variance
  hist_gb        histogram gradient boosting: sequential shallow trees, lower bias
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.features import FeatureEngineer, build_preprocessor

MODEL_NAMES = ["logreg", "random_forest", "hist_gb"]
PRETTY = {"dummy": "No-skill baseline", "logreg": "Logistic regression",
          "random_forest": "Random forest", "hist_gb": "Gradient boosting"}


def make_estimator(name, params=None, class_weight=None):
    """Return an unfitted classifier."""
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED,
                                  class_weight=class_weight, **params)
    if name == "random_forest":
        params.setdefault("n_estimators", 300)
        return RandomForestClassifier(n_jobs=-1, random_state=SEED,
                                      class_weight=class_weight, **params)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=SEED, class_weight=class_weight,
                                              **params)
    raise ValueError(f"unknown model: {name}")


def build_pipeline(name, params=None, imbalance="none", add_features=True):
    """Raw DataFrame in, probability out: feature engineering + preprocessing + model.

    imbalance: "none", "class_weight" (loss re-weighting) or "smote" (synthetic positives,
    generated on the training fold only thanks to the imblearn Pipeline).
    """
    class_weight = "balanced" if imbalance == "class_weight" else None
    steps = [
        ("features", FeatureEngineer(add_features=add_features)),
        ("preprocess", build_preprocessor(scale=(name == "logreg"))),
    ]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", make_estimator(name, params, class_weight)))
    return ImbPipeline(steps) if imbalance == "smote" else Pipeline(steps)


def suggest_params(name, trial):
    """Optuna search space of each model."""
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            # l1_ratio = 0 is Ridge (L2), l1_ratio = 1 is Lasso (L1).
            "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 0.7),
        }
    if name == "hist_gb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    raise ValueError(f"unknown model: {name}")
