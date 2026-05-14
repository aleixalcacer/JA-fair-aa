import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from archetypes.datasets import make_archetypal_dataset
from tqdm import tqdm
import time
from src.methods import AA, FairAAWarm, FairAA_3Moment, FairAA_Adversarial, FairAA_MMD, FairPCA_AA
from src.metrics import (
    explained_variance,
    mmd_rbf,
    linear_separability,
    nonlinear_separability,
    demographic_parity,
)


# ── Data ──────────────────────────────────────────────────────────────────────

generator = np.random.RandomState(42)

s = 1 / (2 * math.sqrt(2))
tetra = [
    [s, s, s],
    [s, -s, -s],
    [-s, s, -s],
    [-s, -s, s],
]
archetypes = np.array(tetra)

# ── Hyperparameters and utilities for the sweep ──────────────────────────

FIGURES = Path(__file__).resolve().parent.parent / "figures/synthetic"

if not FIGURES.exists():
    print(f"Creating figures directory at {FIGURES}")
    FIGURES.mkdir(parents=True, exist_ok=True)

SAVE_FIG = True
SHOW_FIG = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data/synthetic"

if not DATA_DIR.exists():
    print(f"Creating data directory at {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

N_ARCHETYPES = archetypes.shape[0]


METHODS = ["AA", "FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
markers = {"AA": "s", "FairAA": "^", "FairAA_3Moment": "v",
           "FairAA_Adv": "D", "FairAA_MMD": "o", "FairPCA_AA": "P"}

N_SEEDS = 20

BASE_PARAMS = dict(n_archetypes=N_ARCHETYPES, init="furthest_sum", n_init=3,
                   max_iter=1000, tol=1e-14, method="pgd")

def fit_and_measure(name, lam, seed, base_params, S_init=None, C_init=None):
    """Fit one model at (method, lambda, seed) and return (metrics, model).

    S_init / C_init: warm-start arrays (A_, B_) from the previous λ, or None.
    """
    t0 = time.perf_counter()
    if name == "AA":
        model = AA(**base_params, random_state=seed)
        model.fit(X)
        t1 = time.perf_counter()
        S = model.transform(X)
        X_rec = S @ model.archetypes_
        n_iter = len(model.loss_)
    elif name == "FairAA":
        model = FairAAWarm(**base_params, fairness_const=lam, random_state=seed)
        model.fit(X, Z=Z, S_init=S_init, C_init=C_init)
        t1 = time.perf_counter()
        S = model.transform(X, Z)
        X_rec = S @ model.archetypes_
        n_iter = len(model.loss_)
    elif name == "FairAA_3Moment":
        model = FairAA_3Moment(**base_params, fairness_const=lam,
                               alpha_2=0.1, alpha_3=0.1, balance_orders=True,
                               random_state=seed)
        model.fit(X, Z=y.astype(float), S_init=S_init, C_init=C_init)
        t1 = time.perf_counter()
        S = model.transform(X)
        X_rec = S @ model.archetypes_
        n_iter = len(model.loss_)
    elif name == "FairAA_Adv":
        model = FairAA_Adversarial(**base_params, fairness_const=lam,
                                   n_adv_steps=1, lr_adv=1e-2, random_state=seed)
        model.fit(X, Z=y, S_init=S_init, C_init=C_init)
        t1 = time.perf_counter()
        S = model.transform(X, Z=y)
        X_rec = S @ model.archetypes_
        n_iter = len(model.loss_)
    elif name == "FairAA_MMD":
        model = FairAA_MMD(**base_params, fairness_const=lam, random_state=seed)
        model.fit(X, Z=y, S_init=S_init, C_init=C_init)
        t1 = time.perf_counter()
        S = model.transform(X, Z=y)
        X_rec = S @ model.archetypes_
        n_iter = len(model.loss_)
    elif name == "FairPCA_AA":
        model = FairPCA_AA(**base_params, target_dim=2, tradeoff_param=lam,
                           random_state=seed)
        model.fit(X, z=y.astype(int))
        t1 = time.perf_counter()
        S = model.transform(X)
        X_rec = model.fair_pca_.inverse_transform(S @ model.archetypes_)
        n_iter = len(model.loss_)
    else:
        raise ValueError(name)


    S0, S1 = S[y == 0], S[y == 1]
    metrics = {
        "ev":  explained_variance(X[:, :X_rec.shape[1]], X_rec),
        "mmd": mmd_rbf(S0, S1),
        "ls":  linear_separability(S, y, random_state=seed),
        "nls": nonlinear_separability(S, y, random_state=seed),
        "dp":  demographic_parity(S, y),
        "rtime": t1 - t0,
        "n_iter": n_iter,
    }
    return metrics, model


# ── Experiment 1: lambda sweep ───────────────────────────────────────────

# Lambda grids span from no/low penalty (near-AA behaviour) up to a regime
# where utility starts to collapse. FairPCA_AA's tradeoff_param is inverted
# (0 = fully fair, 1 = standard PCA), so its grid runs the opposite way.

n = 200
X, _, _ = make_archetypal_dataset(archetypes, (n,), alpha=0.1, noise=0.1, generator=generator)
y = (X[:, 0] > 0).astype(int)                  # sensitive attribute
Z = (y - y.mean()).reshape(-1, 1)              # centered, (n, 1) for FairAA


N_LAMBDAS = 20  # 20

lambda_grids = {
    "AA":             [None],
    "FairAA":         np.concatenate(([0.0], np.logspace(np.log10(5e-3),  np.log10(7.5), N_LAMBDAS - 1))),
    "FairAA_3Moment": np.concatenate(([0.0], np.logspace(np.log10(5e-3),  np.log10(7.5), N_LAMBDAS - 1))),
    "FairAA_Adv":     np.concatenate(([0.0], np.logspace(np.log10(5e-2),  np.log10(8.0), N_LAMBDAS - 1))),
    "FairAA_MMD":     np.concatenate(([0.0], np.logspace(np.log10(5e-1),  np.log10(500), N_LAMBDAS - 1))),
    "FairPCA_AA":     np.concatenate(([0.0], np.logspace(np.log10(1e-3),  0,             N_LAMBDAS - 1)[::-1])),  # reversed
}


LAMBDA_CSV = DATA_DIR / "synthetic_lambda_sweep.csv"

if LAMBDA_CSV.exists():
    print(f"Loading cached results from {LAMBDA_CSV} (delete the file to re-run the sweep).")
    df = pd.read_csv(LAMBDA_CSV)
else:
    rows = []
    total = sum(len(g) for g in lambda_grids.values()) * N_SEEDS
    with tqdm(total=total) as pbar:
        for name in METHODS:
            for seed in range(N_SEEDS):
                prev_model = None
                for lam in lambda_grids[name]:
                    S_init = prev_model.A_.copy() if prev_model is not None else None
                    C_init = prev_model.B_.copy() if prev_model is not None else None
                    m, prev_model = fit_and_measure(name, lam, seed, BASE_PARAMS,
                                                    S_init=S_init, C_init=C_init)
                    rows.append({"method": name, "lambda": lam, "seed": seed, **m})
                    pbar.update(1)
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LAMBDA_CSV, index=False)

# Mean and std across seeds for each (method, lambda)
_cols = ["ev", "mmd", "ls", "nls", "dp"]
agg_mean = (df.groupby(["method", "lambda"], dropna=False, sort=False)
              [_cols].mean())
agg_std = (df.groupby(["method", "lambda"], dropna=False, sort=False)
             [_cols].std().fillna(0.0))
agg = agg_mean.join(agg_std, rsuffix="_sd").reset_index()


# ── Pareto plots ──────────────────────────────────────────────────────────────

from src.visualization import plot_metric, plot_runtime


fig, axs = plt.subplots(1, 4, figsize=(12, 4))

for i, metric in enumerate(["mmd", "ls", "nls", "dp"]):
    plot_metric(agg, metric, METHODS, ev_col="ev", ax=axs[i], linewidth=2.0, alpha=1)

for ax in axs.flat:
    ax.legend().set_visible(False)

# Plot legend in the bottom of the figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(METHODS))
plt.tight_layout(rect=[0, 0.1, 1, 1])

if SAVE_FIG:
    plt.savefig(FIGURES / "synthetic_lambda_sweep.pdf")
if SHOW_FIG:
    plt.show()


# -- Experiment 2: fixed lambda, n sweep ─────────────────────────────────────────────────────────────

# This experiment is meant to be run after the lambda sweep, we will track the runtime of each method at a fixed lambda across different n.

N_MAX = 10
N_MIN = 5
N_NUM = N_MAX - N_MIN + 1

N_SEEDS = 5


BASE_PARAMS = dict(n_archetypes=N_ARCHETYPES, init="furthest_sum", n_init=3,
                   max_iter=500, tol=0, method="pgd")

N_SIZES = np.logspace(N_MIN, N_MAX, base=2, num=6, dtype=int)  # from 2^5=32 to 2^10=1024

LAMBDA = 1e-3  # a moderate fairness penalty for all methods (except AA which has no penalty)


N_CSV = DATA_DIR / "synthetic_n_sweep.csv"

if N_CSV.exists():
    print(f"Loading cached results from {N_CSV} (delete the file to re-run the sweep).")
    df = pd.read_csv(N_CSV)
else:
    rows = []
    total = len(N_SIZES) * N_SEEDS * len(METHODS)
    with tqdm(total=total) as pbar:
        for n in N_SIZES:
             # Regenerate the dataset with the new size
            X, _, _ = make_archetypal_dataset(archetypes, (n,), alpha=0.1, noise=0.1, generator=generator)
            y = (X[:, 0] > 0).astype(int)                  # sensitive attribute
            Z = (y - y.mean()).reshape(-1, 1)              # centered, (n, 1) for FairAA
            for name in METHODS:
                for seed in range(N_SEEDS):
                    prev_model = None
                    m, prev_model = fit_and_measure(name, LAMBDA, seed, BASE_PARAMS, S_init=None, C_init=None)
                    rows.append({"method": name, "n": n, "seed": seed, **m})
                    pbar.update(1)
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(N_CSV, index=False)

# Mean and std across seeds for each (method, n)
_cols = ["rtime"]
agg_mean = (df.groupby(["method", "n"], dropna=False, sort=False)
              [_cols].mean())
agg_std = (df.groupby(["method", "n"], dropna=False, sort=False)
             [_cols].std().fillna(0.0))
agg = agg_mean.join(agg_std, rsuffix="_sd").reset_index()


fig, ax = plt.subplots(figsize=(6, 4))

plot_runtime(agg, METHODS, ax=ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="best")

if SAVE_FIG:
    plt.savefig(FIGURES / "synthetic_n_sweep_runtime.pdf")
if SHOW_FIG:
    plt.show()

