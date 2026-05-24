"""
Classification Model Training & Evaluation
===========================================
Trains 11 classifiers (base + ensemble) with GridSearchCV tuning and
cross-validation. Produces a ranked leaderboard and saves metrics to JSON.

Usage:
    from src.train import train_all_models, build_leaderboard

    results = train_all_models(X_train, y_train, X_test, y_test, cfg)
    leaderboard = build_leaderboard(results)
"""

from __future__ import annotations

import json
import time
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import pandas as pd


def _grid_search(estimator, param_grid: dict, X_train, y_train, cv: int = 5) -> Any:
    """Run GridSearchCV and return the best fitted estimator."""
    gs = GridSearchCV(
        estimator, param_grid,
        cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def _evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute accuracy, F1, and ROC-AUC for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "model":    name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg:     dict,
) -> Dict[str, dict]:
    """
    Train and evaluate all 11 classifiers defined in config.yaml.

    Base classifiers:
        Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree

    Ensemble classifiers:
        Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Stacking

    Each model (except Naive Bayes and Stacking) is tuned with GridSearchCV
    using the hyperparameter grids from config.yaml.

    Args:
        X_train, y_train: Scaled training data.
        X_test, y_test:   Scaled test data.
        cfg:              Full config dict.

    Returns:
        Dict mapping model name → {"model": fitted model, "metrics": dict}.
    """
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn_extra.cluster import KMedoids  # noqa: F401 — verifies install
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    c = cfg["classification"]
    results = {}

    def fit_and_eval(name, model, param_grid=None):
        print(f"  Training {name}...", end=" ", flush=True)
        t0 = time.time()
        fitted = _grid_search(model, param_grid, X_train, y_train, c["cv_folds"]) \
                 if param_grid else model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = _evaluate(name, fitted, X_test, y_test)
        print(f"ROC-AUC={metrics['roc_auc']:.3f} ({elapsed:.1f}s)")
        results[name] = {"model": fitted, "metrics": metrics}

    # ── Base classifiers ──────────────────────────────────────────────────────
    fit_and_eval("Logistic Regression",  LogisticRegression(max_iter=1000),
                 {k: v for k, v in c["logistic_regression"].items()})
    fit_and_eval("Naive Bayes",          GaussianNB())
    fit_and_eval("KNN",                  KNeighborsClassifier(), c["knn"])
    fit_and_eval("SVM",                  SVC(probability=True),   c["svm"])
    fit_and_eval("Decision Tree",        DecisionTreeClassifier(random_state=42),
                 c["decision_tree"])

    # ── Ensemble classifiers ──────────────────────────────────────────────────
    fit_and_eval("Random Forest",        RandomForestClassifier(random_state=42),
                 c["random_forest"])
    fit_and_eval("AdaBoost",             AdaBoostClassifier(random_state=42),
                 {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]})
    fit_and_eval("Gradient Boosting",    GradientBoostingClassifier(random_state=42),
                 c["gradient_boosting"])
    fit_and_eval("XGBoost",              XGBClassifier(random_state=42, verbosity=0,
                                                        eval_metric="logloss"),
                 c["xgboost"])
    fit_and_eval("LightGBM",             LGBMClassifier(random_state=42, verbose=-1),
                 c["lightgbm"])

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    base_models = [
        ("lr",  results["Logistic Regression"]["model"]),
        ("rf",  results["Random Forest"]["model"]),
        ("gb",  results["Gradient Boosting"]["model"]),
    ]
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=c["cv_folds"],
    )
    fit_and_eval("Stacking", stacking)

    return results


def build_leaderboard(results: Dict[str, dict]) -> pd.DataFrame:
    """
    Build a sorted leaderboard DataFrame from model results.

    Sorted by ROC-AUC descending. Columns: model, accuracy, f1, roc_auc.

    Args:
        results: Output of train_all_models().

    Returns:
        Sorted pandas DataFrame.
    """
    rows = [v["metrics"] for v in results.values()]
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def save_metrics(results: Dict[str, dict], path: str = "results/metrics.json") -> None:
    """
    Persist model metrics to a JSON file for reproducibility tracking.

    Args:
        results: Output of train_all_models().
        path:    Output file path.
    """
    metrics = {name: v["metrics"] for name, v in results.items()}
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
