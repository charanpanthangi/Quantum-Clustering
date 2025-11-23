"""Classical k-means baseline using scikit-learn.

This module provides a thin wrapper around scikit-learn's implementation
so we can compare quantum and classical cluster assignments side by side.
"""
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(X: np.ndarray, n_clusters: int = 2, max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Run classical k-means and return labels and centers.

    Parameters
    ----------
    X: np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters: int
        Number of clusters to find.
    max_iters: int
        Maximum number of iterations for the optimizer.

    Returns
    -------
    labels: np.ndarray
        Assigned cluster index for each data point.
    centers: np.ndarray
        Learned cluster centers.
    """
    model = KMeans(n_clusters=n_clusters, n_init=5, max_iter=max_iters, random_state=0)
    model.fit(X)
    return model.labels_, model.cluster_centers_
