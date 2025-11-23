"""Tests for the quantum clustering routine."""
import numpy as np

from app.dataset import make_moons_data
from app.feature_map import build_angle_encoding_map
from app.quantum_clustering import quantum_kmeans


def test_quantum_kmeans_runs():
    X, _ = make_moons_data(n_samples=20, noise=0.05)
    labels, centers = quantum_kmeans(
        X,
        n_clusters=2,
        max_iters=3,
        feature_map_builder=build_angle_encoding_map,
        random_state=0,
    )
    assert labels.shape == (20,)
    assert centers.shape == (2, 2)
