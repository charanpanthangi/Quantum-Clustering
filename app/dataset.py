"""Dataset generation helpers for simple 2D clustering demos.

This module builds small synthetic datasets such as moons and circles.
Non-linear shapes make it easy to see how quantum feature maps reshape
the geometry of the data. The returned labels are only used for
visualization to color points; clustering is fully unsupervised.
"""
from typing import Tuple

import numpy as np
from sklearn.datasets import make_circles, make_moons


def make_moons_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create a two-moons dataset.

    Parameters
    ----------
    n_samples: int
        Number of points to draw.
    noise: float
        Noise level that controls how much the moons overlap.

    Returns
    -------
    X: np.ndarray
        Array of shape (n_samples, 2) with point coordinates.
    y_true: np.ndarray
        Ground-truth labels for visualization only.
    """
    # Moons are a classic non-linear dataset; Euclidean distance struggles here.
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y_true


def make_circles_data(n_samples: int = 200, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Create a concentric circles dataset.

    Parameters
    ----------
    n_samples: int
        Number of samples to draw.
    noise: float
        Noise applied to circle radius.

    Returns
    -------
    X: np.ndarray
        Array of shape (n_samples, 2) with point coordinates.
    y_true: np.ndarray
        Ground-truth labels for visualization only.
    """
    # Circles highlight how feature maps can separate points based on angle or radius.
    X, y_true = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    return X, y_true
