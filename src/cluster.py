"""
Clustering — Country Segmentation
===================================
Applies 4 clustering algorithms to segment countries by socioeconomic
and health patterns. Evaluates each with Silhouette, Davies-Bouldin,
and Calinski-Harabasz scores.

Usage:
    from src.cluster import run_all_clusters, evaluate_clusters

    cluster_results = run_all_clusters(X_scaled, cfg)
    eval_df = evaluate_clusters(X_scaled, cluster_results)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def run_kmeans(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Fit K-Means and return cluster labels.

    Args:
        X:   Scaled feature matrix (n_samples × n_features).
        cfg: Full config dict (clustering.kmeans section used).

    Returns:
        Integer label array of shape (n_samples,).
    """
    c = cfg["clustering"]["kmeans"]
    model = KMeans(
        n_clusters=c["n_clusters"],
        n_init=c["n_init"],
        max_iter=c["max_iter"],
        random_state=cfg["clustering"]["random_state"],
    )
    return model.fit_predict(X)


def run_kmedoids(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Fit K-Medoids (PAM) and return cluster labels.

    Requires: scikit-learn-extra

    Args:
        X:   Scaled feature matrix.
        cfg: Full config dict (clustering.kmedoids section used).

    Returns:
        Integer label array of shape (n_samples,).
    """
    from sklearn_extra.cluster import KMedoids
    c = cfg["clustering"]["kmedoids"]
    model = KMedoids(
        n_clusters=c["n_clusters"],
        method=c["method"],
        random_state=cfg["clustering"]["random_state"],
    )
    return model.fit_predict(X)


def run_dbscan(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Fit DBSCAN and return cluster labels.

    Label -1 denotes noise points (economic outliers). These are
    countries too dissimilar from any cluster to belong — useful
    insight for policy analysis.

    Args:
        X:   Scaled feature matrix.
        cfg: Full config dict (clustering.dbscan section used).

    Returns:
        Integer label array (includes -1 for outliers).
    """
    c = cfg["clustering"]["dbscan"]
    model = DBSCAN(
        eps=c["eps"],
        min_samples=c["min_samples"],
        metric=c["metric"],
    )
    return model.fit_predict(X)


def run_hierarchical(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Fit Agglomerative (Hierarchical) Clustering with Ward linkage.

    Args:
        X:   Scaled feature matrix.
        cfg: Full config dict (clustering.hierarchical section used).

    Returns:
        Integer label array of shape (n_samples,).
    """
    c = cfg["clustering"]["hierarchical"]
    model = AgglomerativeClustering(
        n_clusters=c["n_clusters"],
        linkage=c["linkage"],
    )
    return model.fit_predict(X)


def run_all_clusters(X: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """
    Run all 4 clustering algorithms and return their label arrays.

    Args:
        X:   Scaled feature matrix (all countries).
        cfg: Full config dict.

    Returns:
        Dict mapping algorithm name → label array.
    """
    print("Running clustering algorithms...")
    labels = {}
    for name, fn in [
        ("K-Means",      run_kmeans),
        ("K-Medoids",    run_kmedoids),
        ("DBSCAN",       run_dbscan),
        ("Hierarchical", run_hierarchical),
    ]:
        labels[name] = fn(X, cfg)
        n_clusters = len(set(labels[name]) - {-1})
        print(f"  {name}: {n_clusters} clusters")
    return labels


def evaluate_clusters(X: np.ndarray, labels: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Evaluate clustering quality with three internal metrics.

    Metrics:
        Silhouette Score       ↑ higher is better  (range: -1 to 1)
        Davies-Bouldin Score   ↓ lower is better   (range: 0 to ∞)
        Calinski-Harabasz Score ↑ higher is better (range: 0 to ∞)

    DBSCAN noise points (label == -1) are excluded from evaluation.

    Args:
        X:      Scaled feature matrix.
        labels: Output of run_all_clusters().

    Returns:
        DataFrame with one row per algorithm and three metric columns.
    """
    rows = []
    for name, lbls in labels.items():
        mask = lbls != -1
        X_valid, lbls_valid = X[mask], lbls[mask]
        n_unique = len(set(lbls_valid))
        if n_unique < 2:
            rows.append({"Algorithm": name, "Clusters": n_unique,
                         "Silhouette": None, "Davies-Bouldin": None,
                         "Calinski-Harabasz": None})
            continue
        rows.append({
            "Algorithm":          name,
            "Clusters":           n_unique,
            "Silhouette":         round(silhouette_score(X_valid, lbls_valid), 3),
            "Davies-Bouldin":     round(davies_bouldin_score(X_valid, lbls_valid), 3),
            "Calinski-Harabasz":  round(calinski_harabasz_score(X_valid, lbls_valid), 1),
        })
    return pd.DataFrame(rows)
