# Fair Archetypal Analysis

Code, experiments, and results for the paper *Incorporating Fairness Constraints into Archetypal Analysis* ([arXiv:2507.12021](https://arxiv.org/abs/2507.12021)).

## Repository structure

```
fair-archetypes/
├── src/
│   ├── methods/
│   │   ├── fair_aa_warm.py         # FairAA (linear constraint, warm-startable)
│   │   ├── fair_aa_3moment.py      # FairAA with 1st/2nd/3rd-moment alignment
│   │   ├── fair_aa_adversarial.py  # FairAA with adversarial probe
│   │   ├── fair_aa_mmd.py          # FairAA with MMD penalty
│   │   └── fair_pca_aa.py          # FairPCA + AA baseline
│   ├── metrics.py                  # MMD, linear/nonlinear separability,
│   │                               #   demographic parity, DP/EO/EOpp gaps,
│   │                               #   explained variance
│   └── visualization.py            # Pareto and bar-plot helpers
├── experiments/
│   ├── simple.py                   # Synthetic sweep (Section 4.1)
│   ├── simple_real.py              # Real-data sweep (Section 4.2)
│   ├── extensions.py               # Synthetic extensions: blobs, moons,
│   │                               #   multi-class, kernel variants
│   ├── datasets.py                 # Fairlearn loaders (Adult, Diabetes,
│   │                               #   Bank Marketing) + DatasetMeta
│   └── datasets_table.py           # LaTeX summary table of real datasets
├── data/                           # Cached sweep results (per dataset)
└── figures/                        # Generated PDFs / PNGs
```

## Environment

- **Python 3.12** (pinned in `.python-version`).
- All dependencies are declared in [`pyproject.toml`](pyproject.toml). Exact versions used in the paper are locked in [`uv.lock`](uv.lock), which records the resolved version, source, and hash for every transitive package.
- Key libraries: `numpy`, `scipy` (via `scikit-learn`), `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `pypalettes`, `tqdm`, `archetypes`, `fairlearn`, `ucimlrepo`.

```bash
git clone https://github.com/aleixalcacer/JA-fair-aa.git
cd JA-fair-aa
uv sync                       # creates .venv with the exact versions in uv.lock
```

To regenerate a verbatim copy of the environment elsewhere, run `uv sync --frozen` (refuses to update the lockfile) and report the contents of `uv.lock` alongside results.

## Datasets

All real-world datasets are fetched at runtime via [`fairlearn.datasets`](https://fairlearn.org/main/api_reference/generated/fairlearn.datasets.html), so no manual download or preprocessing is required. Loaders, target binarization, and one-hot encoding are centralized in [`experiments/datasets.py`](experiments/datasets.py):

| Key (`name`)         | Source              | Protected attribute       | Target                 |
|----------------------|---------------------|---------------------------|------------------------|
| `adult`              | `fetch_adult`              | `sex`                | income > \$50K         |
| `diabetes_hospital`  | `fetch_diabetes_hospital`  | `gender` (Female/Male; *Unknown/Invalid* dropped) | 30-day readmission |
| `bank_marketing`     | `fetch_bank_marketing`     | `V1` (age, binarized at 30) | term deposit subscription |

`datasets.load(name, n_max, seed)`:
- fetches the full frame,
- drops protected and label-leaking columns (`readmitted`, `readmit_binary` for Diabetes),
- one-hot encodes categorical columns with `pandas.get_dummies(drop_first=True)`,
- binarizes target and protected attribute to {0, 1},
- draws a stratified-by-(y, a) subsample of size `n_max` if needed.

`datasets.describe(name)` returns a `DatasetMeta` dataclass with `n_original`, continuous/categorical column lists, target description, protected label, and class balance.

The synthetic datasets in `experiments/simple.py` and `experiments/extensions.py` are generated with fixed seeds passed to `archetypes.datasets.make_archetypal_dataset`, `sklearn.datasets.make_blobs`, and `sklearn.datasets.make_moons` (a top-level `np.random.RandomState(42)` plus per-seed `random_state=i` inside loops).

## Algorithms

| Method            | Location                                  | Constraint type |
|-------------------|-------------------------------------------|---------------------|
| `AA`              | from `archetypes`                                  | unconstrained baseline |
| `FairAA` (`FairAAWarm`) | [src/methods/fair_aa_warm.py](src/methods/fair_aa_warm.py) | linear mean-alignment, warm-startable |
| `FairAA_3Moment`  | [src/methods/fair_aa_3moment.py](src/methods/fair_aa_3moment.py)      | 1st/2nd-moment alignment |
| `FairAA_Adv`      | [src/methods/fair_aa_adversarial.py](src/methods/fair_aa_adversarial.py)  | adversarial probe |
| `FairAA_MMD`      | [src/methods/fair_aa_mmd.py](src/methods/fair_aa_mmd.py)               | RBF-MMD penalty |
| `FairPCA_AA`      | [src/methods/fair_pca_aa.py](src/methods/fair_pca_aa.py)               | FairPCA preprocessing + AA |

## Evaluation metrics

Implemented in [`src/metrics.py`](src/metrics.py). Hyperparameters used in every experiment:

| Metric | Function | Hyperparameters |
|---|---|---|
| Explained variance | `explained_variance(X, X_hat)` | reconstruction-error / total-variance ratio |
| MMD (RBF) | `mmd_rbf(X, Y, sigmas` | **Multi-kernel sum** of RBF kernels with `sigmas` bandwiths. Implemented via `sklearn.metrics.pairwise.rbf_kernel`. |
| Linear separability | `linear_separability(H, z, n_splits=5, test_size=0.3, random_state=0)` | Probe: `LogisticRegression(max_iter=1000, random_state=seed)` (scikit-learn defaults otherwise: `penalty="l2"`, `C=1.0`, `solver="lbfgs"`). 5 stratified shuffle splits; score is mean balanced accuracy. |
| Nonlinear separability | `nonlinear_separability(H, z, n_splits=5, test_size=0.3, random_state=0)` | Probe: `RandomForestClassifier(n_estimators=200, random_state=seed)` (other parameters at scikit-learn defaults). 5 stratified shuffle splits; mean balanced accuracy. |
| Demographic parity (representation) | `demographic_parity(H, z)` | mean-difference of loadings between groups, averaged over archetypes |
| DP / EO / EOpp gap (classifier) | `dp_gap`, `eo_gap`, `eopp_gap` | prediction-level gaps in [0, 1] |

The downstream classifier in `simple_real.py` is `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)` fitted on the AA loadings. The MLP variant (toggled by `DOWNSTREAM = "mlp"`) is `MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=seed)`. All other parameters are scikit-learn defaults.

## Hyperparameters of the AA fits

Common AA solver settings (see [`AA_PARAMS` in simple_real.py](experiments/simple_real.py#L61) and [`BASE_PARAMS` in simple.py](experiments/simple.py#L61)):

| Parameter | Value |
|---|---|
| `init` | `"furthest_sum"` |
| `n_init` | 3 (synthetic) / 5 (real) restarts |
| `max_iter` | 1000 |
| `tol` | 1e-12 |
| `method` | `"pseudo_pgd"` |

Sweep settings:

| Experiment | `K` (archetypes) | Seeds | λ grid |
|---|---|---|---|
| `simple.py` (synthetic) | 4 (tetrahedron vertices) | `N_SEEDS = 20` | 20-point geometric grid per method, see `lambda_grids` in the script |
| `simple_real.py` (real) | chosen per-dataset by AA scree-elbow on `K ∈ [1, 10]` averaged over 3 inits | `N_SEEDS = 25` | `[0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]` |
| `extensions.py` | 3–4 depending on scenario | 5 inits | scenario-specific (`fair_const` in source) |


## Reproducing the paper

Each script is self-contained: open it, see the constants block at the top (`N_SEEDS`, `LAMBDAS`, `MAX_SAMPLES`, `K_MAX`, ...), run with `uv run python`. Outputs are written under `figures/` and `data/`. Results are cached as CSVs in `data/<dataset>/raw_results_simple.csv`; deleting the cache forces a re-run.

| Figure / Table in the paper | How to reproduce |
|---|---|
| Synthetic Pareto curves (Section 4.1) | `uv run python experiments/simple.py` → `figures/synthetic/*.pdf` |
| Real-data Pareto + bar plots (Section 4.2) | `uv run python experiments/simple_real.py` → `figures/{adult,diabetes_hospital,bank_marketing}/pareto_*.pdf` and `bars_*.pdf` |
| Extensions on blobs / moons / kernel AA (Section 4.3) | `uv run python experiments/extensions.py` → `figures/extensions/*.pdf` |
| Table of real-dataset summary | `uv run python experiments/datasets_table.py` → LaTeX printed to stdout |

Every random draw inside the scripts is seeded (`random_state=seed`, where `seed` iterates over `range(N_SEEDS)`), so reruns with the same `uv.lock` and Python 3.12 reproduce identical CSVs and figures.

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
