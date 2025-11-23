"""Quantum feature maps used for clustering experiments.

A feature map turns a classical point ``x`` into a quantum state by
encoding its values into a circuit. Different maps reshape the geometry
of the data, which changes the similarity between points.
"""
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


def build_angle_encoding_map(num_qubits: int = 2) -> QuantumCircuit:
    """Create a simple angle encoding circuit.

    The input vector is mapped to rotations around the Y and Z axes.
    This keeps circuits short so the examples run quickly.

    Parameters
    ----------
    num_qubits: int
        Number of qubits to use. For 2D data, two qubits suffice.

    Returns
    -------
    QuantumCircuit
        A parameterized circuit ready for binding real values.
    """
    params = ParameterVector("x", length=num_qubits)
    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(params):
        # Angle encoding: rotate each qubit based on the classical feature.
        circuit.ry(param, i)
        circuit.rz(param / 2, i)
    return circuit


def build_zz_feature_map(num_qubits: int = 2, reps: int = 2) -> QuantumCircuit:
    """Create a ZZFeatureMap circuit from Qiskit.

    The ZZFeatureMap entangles qubits based on pairwise products of input
    features. This often separates circular patterns more effectively.

    Parameters
    ----------
    num_qubits: int
        Number of qubits to use.
    reps: int
        How many times the entangling pattern is repeated.

    Returns
    -------
    QuantumCircuit
        A parameterized circuit provided by Qiskit.
    """
    return ZZFeatureMap(feature_dimension=num_qubits, reps=reps)


def encode_to_statevector(feature_map: QuantumCircuit, x: Iterable[float]) -> np.ndarray:
    """Bind data to a feature map and simulate its statevector.

    Parameters
    ----------
    feature_map: QuantumCircuit
        Circuit with free parameters representing the feature map.
    x: Iterable[float]
        Classical data point to encode.

    Returns
    -------
    np.ndarray
        Statevector amplitudes as a complex NumPy array.
    """
    # Bind values to parameters in the order they appear.
    bound_circuit = feature_map.bind_parameters({p: float(val) for p, val in zip(feature_map.parameters, x)})
    # Use the statevector simulator to obtain the final quantum state.
    state = Statevector.from_instruction(bound_circuit)
    return np.array(state.data)
