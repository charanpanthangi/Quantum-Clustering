"""Quantum similarity and distance utilities.

This module computes fidelity-based similarity between quantum states
and converts it into a distance for clustering. Fidelity is a natural
measure because it captures how likely two states are to overlap.
"""
import numpy as np


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute the fidelity between two quantum statevectors.

    Fidelity measures similarity and is the squared magnitude of the
    inner product between the states.

    Parameters
    ----------
    state1, state2: np.ndarray
        Complex statevector amplitudes.

    Returns
    -------
    float
        Value in [0, 1] where 1 means identical states.
    """
    overlap = np.vdot(state1, state2)
    return float(np.abs(overlap) ** 2)


def quantum_distance(state1: np.ndarray, state2: np.ndarray) -> float:
    """Convert fidelity into a distance value.

    The distance is defined as ``1 - fidelity`` so that identical states
    have distance 0 and orthogonal states approach 1.
    """
    return 1.0 - fidelity(state1, state2)


def compute_quantum_kernel(statevectors: np.ndarray) -> np.ndarray:
    """Build a kernel matrix using fidelity for each pair of states.

    Parameters
    ----------
    statevectors: np.ndarray
        Array of shape (n_samples, dim) where each row is a statevector.

    Returns
    -------
    np.ndarray
        Symmetric matrix K where K[i, j] = fidelity(state_i, state_j).
    """
    n_samples = statevectors.shape[0]
    kernel = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel[i, j] = fidelity(statevectors[i], statevectors[j])
    return kernel
