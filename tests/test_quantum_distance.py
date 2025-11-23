"""Tests for quantum distance calculations."""
import numpy as np

from app.feature_map import build_angle_encoding_map, encode_to_statevector
from app.quantum_distance import fidelity, quantum_distance


def test_fidelity_self_is_one():
    fmap = build_angle_encoding_map()
    state = encode_to_statevector(fmap, [0.0, 0.0])
    assert np.isclose(fidelity(state, state), 1.0, atol=1e-6)


def test_quantum_distance_self_is_zero():
    fmap = build_angle_encoding_map()
    state = encode_to_statevector(fmap, [0.3, -0.1])
    assert np.isclose(quantum_distance(state, state), 0.0, atol=1e-6)
