# Quantum Clustering – q-means–style Quantum Clustering

## What This Project Does
This project compares classical k-means clustering with a simple quantum-enhanced version. We map data to quantum states and use a fidelity-based quantum distance instead of Euclidean distance. A k-means–like loop then groups points based on that quantum similarity.

## Why Quantum Clustering Is Interesting
- Quantum feature maps can reshape how we measure distance.
- Points that are hard to separate classically might become easier to separate in quantum space.
- This idea connects to quantum kernel methods and wider quantum machine learning research.

## Why We Use SVG Instead of PNG
GitHub’s CODEX interface cannot preview binary image files like PNG or JPG and often shows “Binary files are not supported” in pull request views. To avoid this, all visualizations in this repository are saved as lightweight SVG (vector) images. SVGs are text-based, easy to diff, and render cleanly inside GitHub and CODEX.

## How It Works (Plain English)
1. Generate a 2D dataset (moons or circles).
2. Run classical k-means using Euclidean distance.
3. Run quantum clustering using a quantum feature map plus fidelity-based distance.
4. Visualize and compare both sets of labels and centers.

## Repository Structure
- `app/`: Source code for datasets, feature maps, clustering, plotting, and CLI.
- `notebooks/`: Jupyter notebook tutorial.
- `examples/`: SVG plots produced by the CLI.
- `tests/`: Lightweight pytest checks.
- `Dockerfile`, `requirements.txt`, `LICENSE`: Support files.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --dataset moons --clusters 2 --iters 10 --feature-map angle
```

For the notebook:
```bash
jupyter notebook notebooks/quantum_clustering_demo.ipynb
```

## What You Should See
- The original dataset.
- Cluster boundaries from quantum vs classical clustering.
- A plot comparing quantum cluster centers vs classical centers.

## Future Extensions
- More qubits and deeper feature maps.
- Different quantum distances.
- Hardware backends.
- Integration with quantum kernel methods.
