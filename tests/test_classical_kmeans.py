"""Tests for the classical k-means baseline."""
import numpy as np

from app.dataset import make_circles_data
from app.classical_kmeans import run_kmeans


def test_classical_kmeans_runs():
    X, _ = make_circles_data(n_samples=30, noise=0.02)
    labels, centers = run_kmeans(X, n_clusters=2, max_iters=5)
    assert labels.shape == (30,)
    assert centers.shape == (2, 2)
