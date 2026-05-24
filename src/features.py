"""
Feature Engineering & Preprocessing
=====================================
Derives the binary classification target (`is_developed`) from raw
socioeconomic indicators using a composite weighted development score,
then scales features for downstream models.

Usage:
    from src.features import load_and_engineer

    X_train, X_test, y_train, y_test, scaler = load_and_engineer(
        path="Country-data.csv", config=cfg
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def compute_development_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Compute a composite development score for each country.

    Normalises each indicator to [0, 1] then applies signed weights:
        child_mort  (negative indicator) → lower is better
        income      (positive indicator) → higher is better
        life_expec  (positive indicator) → higher is better
        gdpp        (positive indicator) → higher is better

    Args:
        df:  Raw DataFrame with country socioeconomic features.
        cfg: Feature engineering config dict (weights, threshold).

    Returns:
        Series of float scores, one per country.
    """
    fe = cfg["feature_engineering"]

    def norm(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    score = (
        fe["income_weight"]      * norm(df["income"])
        + fe["life_expec_weight"]  * norm(df["life_expec"])
        + fe["gdpp_weight"]        * norm(df["gdpp"])
        - fe["child_mort_weight"]  * norm(df["child_mort"])
    )
    return score


def derive_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Derive binary `is_developed` label from composite development score.

    Countries above the configured percentile threshold are labelled 1
    (developed), the rest are labelled 0.

    Args:
        df:  Raw DataFrame.
        cfg: Config with feature_engineering.development_threshold.

    Returns:
        Binary Series (0 / 1).
    """
    score = compute_development_score(df, cfg)
    threshold = score.quantile(cfg["feature_engineering"]["development_threshold"])
    return (score >= threshold).astype(int)


def load_and_engineer(
    path: str,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load raw data, engineer target, split, and scale features.

    Args:
        path: Path to Country-data.csv.
        cfg:  Full config dict (from load_config()).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, fitted_scaler).
    """
    df = pd.read_csv(path)

    feature_cols = [
        "child_mort", "exports", "health", "imports",
        "income", "inflation", "life_expec", "total_fer", "gdpp",
    ]
    X = df[feature_cols].values
    y = derive_target(df, cfg).values

    data_cfg = cfg["data"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Dataset: {len(df)} countries | "
          f"Developed: {y.sum()} ({y.mean():.1%}) | "
          f"Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler
