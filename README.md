# Fair Archetypal Analysis

Code, experiments, and results for the paper *Incorporating Fairness Constraints into Archetypal Analysis*.

## Repository structure

```
fair-archetypes/
├── src/                    # Shared algorithms and metrics
│   ├── fair_pca.py         # FairPCA baseline
│   └── metrics.py          # Evaluation metrics (MMD, ...)
├── data/
│   └── ansur/              # ANSUR anthropometric dataset
├── notebooks/              # Jupyter notebooks (exploratory)
│   ├── exemple.ipynb           # Basic AA on synthetic data
│   ├── exemple_multiple.ipynb  # Multiple synthetic scenarios
│   ├── exemple_moons.ipynb     # Moons synthetic dataset
│   ├── baselines.ipynb         # AA vs FairAA vs FairPCA+AA
│   └── ansur.ipynb             # Real-data experiment (ANSUR)
├── experiments/            # Runnable Python scripts
│   ├── baselines.py        # AA vs FairAA vs FairPCA+AA
│   ├── scaling.py          # Computation time vs n
│   └── scaling_nd.py       # Computation time vs n and d
└── figures/                # Generated figures (PDF)
```

## Algorithms

- **Archetypal Analysis (AA)**: decomposes data into convex combinations of extremal points (archetypes).
- **FairAA**: AA with fairness constraints to ensure fair representation across sensitive groups.
- **FairPCA + AA**: dimensionality reduction via Fair PCA followed by AA.

## Getting started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aleixalcacer/JA-fair-aa.git
   cd JA-fair-aa
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Run a notebook:**
   Open any `.ipynb` in `notebooks/` with JupyterLab or VS Code.

4. **Run an experiment script:**
   ```bash
   uv run python experiments/baselines.py
   ```

## Citation

```bibtex
@misc{alcacer2025incorporatingfairnessconstraintsarchetypal,
      title={Incorporating Fairness Constraints into Archetypal Analysis},
      author={Aleix Alcacer and Irene Epifanio},
      year={2025},
      eprint={2507.12021},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2507.12021},
}
```
