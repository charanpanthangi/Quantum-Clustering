"""Plotting utilities that always save figures as SVG files.

SVG output keeps the repository lightweight and review-friendly.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Ensure SVG backend is used when running scripts.
plt.switch_backend("Agg")


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories for an output file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_original_data(X: np.ndarray, output_path: str, y_true: Optional[np.ndarray] = None) -> None:
    """Scatter plot of the raw dataset.

    Parameters
    ----------
    X: np.ndarray
        Input coordinates.
    output_path: str
        File path for the SVG file.
    y_true: Optional[np.ndarray]
        Optional labels for coloring the points.
    """
    path = Path(output_path)
    _ensure_parent_dir(path)
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", edgecolor="k")
    plt.title("Original data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def plot_cluster_boundaries(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    """Visualize clustering result with decision boundaries.

    A dense grid is colored based on the assigned cluster labels. The
    colors show how the distance metric partitions the space.
    """
    path = Path(output_path)
    _ensure_parent_dir(path)

    # Build a mesh grid.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Simple nearest-center coloring using the provided labels/centers.
    # This mirrors the last assignment step used to generate labels.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    closest = np.argmin(((grid_points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    closest = closest.reshape(xx.shape)

    cmap_light = ListedColormap(["#F1C40F", "#3498DB", "#E74C3C", "#2ECC71"])
    cmap_bold = ["#F39C12", "#2980B9", "#C0392B", "#27AE60"]

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, closest, cmap=cmap_light, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=ListedColormap(cmap_bold), edgecolor="k")
    plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=80, label="Centers")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def plot_cluster_center_comparison(
    centers_quantum: np.ndarray, centers_classical: np.ndarray, output_path: str
) -> None:
    """Plot quantum vs classical cluster centers on the same axes."""
    path = Path(output_path)
    _ensure_parent_dir(path)
    plt.figure(figsize=(5, 4))
    plt.scatter(centers_quantum[:, 0], centers_quantum[:, 1], c="purple", marker="o", s=80, label="Quantum")
    plt.scatter(centers_classical[:, 0], centers_classical[:, 1], c="gray", marker="x", s=80, label="Classical")
    plt.title("Cluster centers: quantum vs classical")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()
