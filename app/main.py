"""Command-line entry point for quantum clustering demos.

This script generates a dataset, runs quantum clustering and classical
k-means, and saves SVG plots for easy viewing in GitHub.
"""
import argparse
from pathlib import Path

import numpy as np

from .classical_kmeans import run_kmeans
from .dataset import make_circles_data, make_moons_data
from .feature_map import build_angle_encoding_map, build_zz_feature_map
from .plots import (
    plot_cluster_boundaries,
    plot_cluster_center_comparison,
    plot_original_data,
)
from .quantum_clustering import quantum_kmeans


FEATURE_MAPS = {
    "angle": build_angle_encoding_map,
    "zz": build_zz_feature_map,
}


DATASETS = {
    "moons": make_moons_data,
    "circles": make_circles_data,
}


def parse_args() -> argparse.Namespace:
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantum vs classical clustering demo")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="moons", help="Dataset type")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters")
    parser.add_argument("--iters", type=int, default=10, help="Iterations for clustering")
    parser.add_argument(
        "--feature-map",
        choices=FEATURE_MAPS.keys(),
        default="angle",
        help="Quantum feature map to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples",
        help="Directory to save SVG plots",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full clustering pipeline and save visualizations."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate the chosen dataset.
    X, y_true = DATASETS[args.dataset]()

    # 2) Build the chosen quantum feature map.
    feature_map_builder = FEATURE_MAPS[args.feature_map]

    # 3) Run quantum clustering (q-means style).
    q_labels, q_centers = quantum_kmeans(
        X,
        n_clusters=args.clusters,
        max_iters=args.iters,
        feature_map_builder=feature_map_builder,
    )

    # 4) Run classical k-means baseline.
    c_labels, c_centers = run_kmeans(X, n_clusters=args.clusters, max_iters=args.iters)

    # 5) Create SVG plots for comparison.
    plot_original_data(X, output_dir / "original_data.svg", y_true=y_true)
    plot_cluster_boundaries(X, q_labels, q_centers, output_dir / "quantum_clustering_boundaries.svg", "Quantum clustering")
    plot_cluster_boundaries(
        X,
        c_labels,
        c_centers,
        output_dir / "classical_kmeans_boundaries.svg",
        "Classical k-means",
    )
    plot_cluster_center_comparison(
        q_centers, c_centers, output_dir / "cluster_centers_comparison.svg"
    )

    print("Quantum clustering completed.")
    print("Classical k-means completed.")
    print(f"SVG visualizations saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
