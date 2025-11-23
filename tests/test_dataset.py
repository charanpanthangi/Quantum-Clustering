"""Tests for dataset generation utilities."""
import numpy as np

from app.dataset import make_circles_data, make_moons_data


def test_make_moons_shape():
    X, y = make_moons_data(n_samples=50)
    assert X.shape == (50, 2)
    assert y.shape == (50,)


def test_make_circles_shape():
    X, y = make_circles_data(n_samples=60)
    assert X.shape == (60, 2)
    assert y.shape == (60,)
