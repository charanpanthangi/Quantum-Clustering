"""Simple q-means style quantum clustering implementation.

The algorithm mirrors classical k-means but replaces Euclidean distance
with a quantum distance derived from fidelity. Cluster centers are kept
in classical space to keep the code lightweight, but similarity is
computed after encoding points and centers with a quantum feature map.
"""
from typing import Callable, Iterable, Tuple

import numpy as np

from .feature_map import encode_to_statevector
from .quantum_distance import quantum_distance


def _encode_points(feature_map, points: np.ndarray) -> np.ndarray:
    """Encode a batch of points into quantum statevectors."""
    return np.array([encode_to_statevector(feature_map, x) for x in points])


def _initialize_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Pick initial centers randomly from the data points."""
    indices = rng.choice(len(X), size=n_clusters, replace=False)
    return X[indices]


def quantum_kmeans(
    X: np.ndarray,
    n_clusters: int = 2,
    max_iters: int = 10,
    feature_map_builder: Callable[[], Iterable] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a quantum distance-based k-means loop.

    Parameters
    ----------
    X: np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters: int
        Number of clusters to find.
    max_iters: int
        Maximum iterations of assignment and update steps.
    feature_map_builder: Callable
        Function that returns a parameterized quantum circuit.
    random_state: int
        Seed for deterministic initialization.

    Returns
    -------
    labels: np.ndarray
        Assigned cluster index for each data point.
    centers: np.ndarray
        Final cluster centers in classical space.
    """
    rng = np.random.default_rng(random_state)
    # Build feature map once and reuse it for all encodings.
    feature_map = feature_map_builder() if feature_map_builder else None
    centers = _initialize_centers(X, n_clusters, rng)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iters):
        # Encode all data points once per iteration to keep things fast.
        data_states = _encode_points(feature_map, X)
        center_states = _encode_points(feature_map, centers)

        # Assignment step: pick the closest center using quantum distance.
        for i, state in enumerate(data_states):
            distances = [quantum_distance(state, c_state) for c_state in center_states]
            labels[i] = int(np.argmin(distances))

        # Update step: move centers to the mean of assigned points.
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                # If a cluster is empty, re-sample a point to keep progress.
                new_centers[k] = X[rng.integers(0, len(X))]
            else:
                new_centers[k] = cluster_points.mean(axis=0)

        if np.allclose(new_centers, centers):
            # Stop early if centers have stabilized.
            centers = new_centers
            break
        centers = new_centers

    return labels, centers
