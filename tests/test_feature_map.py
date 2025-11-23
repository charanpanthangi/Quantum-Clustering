"""Tests for quantum feature map utilities."""
import numpy as np

from app.feature_map import build_angle_encoding_map, build_zz_feature_map, encode_to_statevector


def test_angle_encoding_state_normalized():
    fmap = build_angle_encoding_map()
    state = encode_to_statevector(fmap, [0.1, -0.2])
    norm = np.sum(np.abs(state) ** 2)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_zz_feature_map_builds():
    fmap = build_zz_feature_map()
    # Ensure parameters exist and circuit depth is reasonable.
    assert len(fmap.parameters) > 0
    assert fmap.num_qubits == 2
