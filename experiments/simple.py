import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from archetypes.datasets import make_archetypal_dataset
from tqdm import tqdm

from src.methods import AA, FairAA, FairAA_3Moment, FairAA_Adversarial, FairAA_MMD, FairPCA_AA
from src.metrics import (
    explained_variance,
    mmd_rbf,
    linear_separability,
    nonlinear_separability,
    demographic_parity,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
save_fig = False
plot_fig = True

# ── Data ──────────────────────────────────────────────────────────────────────

n = 100
generator = np.random.RandomState(42)

s = 1 / (2 * math.sqrt(2))
tetra = [
    [s, s, s],
    [s, -s, -s],
    [-s, s, -s],
    [-s, -s, s],
]
archetypes = np.array(tetra)
X, _, _ = make_archetypal_dataset(archetypes, (n,), alpha=0.1, noise=0.1, generator=generator)
y = (X[:, 0] > 0).astype(int)                  # sensitive attribute
Z = (y - y.mean()).reshape(-1, 1)              # centered, (n, 1) for FairAA

# 3D scatter of the data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="winter")
ax.set_title("Data")
if save_fig:
    plt.savefig(FIGURES / "example_data.pdf")

# ── Experiment: lambda sweep ──────────────────────────────────────────────────

n_archetypes = 4
base_params = dict(n_archetypes=n_archetypes, init="furthest_sum", n_init=3,
                   max_iter=1000, tol=1e-14, method="pgd")

METHODS = ["AA", "FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
markers = {"AA": "s", "FairAA": "^", "FairAA_3Moment": "v",
           "FairAA_Adv": "D", "FairAA_MMD": "o", "FairPCA_AA": "P"}
colors = dict(zip(METHODS, plt.cm.tab10.colors))

# Lambda grids span from no/low penalty (near-AA behaviour) up to a regime
# where utility starts to collapse. FairPCA_AA's tradeoff_param is inverted
# (0 = fully fair, 1 = standard PCA), so its grid runs the opposite way.
lambda_grids = {
    "AA":             [None],
    "FairAA":         np.concatenate(([0.0], np.logspace(np.log10(5e-3),  np.log10(7.5), 19))),
    "FairAA_3Moment": np.concatenate(([0.0], np.logspace(np.log10(5e-3),  np.log10(7.5), 19))),
    "FairAA_Adv":     np.concatenate(([0.0], np.logspace(np.log10(5e-2),  np.log10(8.0), 19))),
    "FairAA_MMD":     np.concatenate(([0.0], np.logspace(np.log10(5e-1),  np.log10(500), 19))),
    "FairPCA_AA":     np.concatenate(([0.0], np.logspace(np.log10(1e-3),  0,             9))),
}

N_SEEDS = 25


def fit_and_measure(name, lam, seed):
    """Fit one model at (method, lambda, seed) and compute all metrics."""
    if name == "AA":
        model = AA(**base_params, random_state=seed)
        model.fit(X)
        S = model.transform(X)
        X_rec = S @ model.archetypes_
    elif name == "FairAA":
        model = FairAA(**base_params, fairness_const=lam, random_state=seed)
        model.fit(X, Z=Z)
        S = model.transform(X, Z)
        X_rec = S @ model.archetypes_
    elif name == "FairAA_3Moment":
        model = FairAA_3Moment(**base_params, fairness_const=lam,
                               alpha_2=0.1, alpha_3=0.1, balance_orders=True,
                               random_state=seed)
        model.fit(X, Z=y.astype(float))
        S = model.transform(X)
        X_rec = S @ model.archetypes_
    elif name == "FairAA_Adv":
        model = FairAA_Adversarial(**base_params, fairness_const=lam,
                                   n_adv_steps=1, lr_adv=1e-2, random_state=seed)
        model.fit(X, Z=y)
        S = model.transform(X, Z=y)
        X_rec = S @ model.archetypes_
    elif name == "FairAA_MMD":
        model = FairAA_MMD(**base_params, fairness_const=lam, random_state=seed)
        model.fit(X, Z=y)
        S = model.transform(X, Z=y)
        X_rec = S @ model.archetypes_
    elif name == "FairPCA_AA":
        model = FairPCA_AA(**base_params, target_dim=2, tradeoff_param=lam,
                           random_state=seed)
        model.fit(X, z=y.astype(int))
        S = model.transform(X)
        X_rec = model.fair_pca_.inverse_transform(S @ model.archetypes_)
    else:
        raise ValueError(name)

    S0, S1 = S[y == 0], S[y == 1]
    return {
        "ev":  explained_variance(X[:, :X_rec.shape[1]], X_rec),
        "mmd": mmd_rbf(S0, S1),
        "ls":  linear_separability(S, y, random_state=seed),
        "nls": nonlinear_separability(S, y, random_state=seed),
        "dp":  demographic_parity(S, y),
    }


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV = DATA_DIR / "raw_results.csv"

if RAW_CSV.exists():
    print(f"Loading cached results from {RAW_CSV} (delete the file to re-run the sweep).")
    df = pd.read_csv(RAW_CSV)
else:
    rows = []
    total = sum(len(g) for g in lambda_grids.values()) * N_SEEDS
    with tqdm(total=total) as pbar:
        for name in METHODS:
            for lam in lambda_grids[name]:
                for seed in range(N_SEEDS):
                    m = fit_and_measure(name, lam, seed + 42)
                    rows.append({"method": name, "lambda": lam, "seed": seed, **m})
                    pbar.update(1)
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)

# Mean and std across seeds for each (method, lambda)
_cols = ["ev", "mmd", "ls", "nls", "dp"]
agg_mean = (df.groupby(["method", "lambda"], dropna=False, sort=False)
              [_cols].mean())
agg_std = (df.groupby(["method", "lambda"], dropna=False, sort=False)
             [_cols].std().fillna(0.0))
agg = agg_mean.join(agg_std, rsuffix="_sd").reset_index()

# ── Amplitude diagnostic ──────────────────────────────────────────────────────
# Per-method min/max of each metric, so we can see if the λ grid actually
# spans from near-AA behaviour to the fairness floor on each axis.

pd.set_option("display.precision", 3)
amp = (df.groupby("method", sort=False)
         [["ev", "mmd", "ls", "nls", "dp"]]
         .agg(["min", "max"]))
print("\nPer-method metric amplitudes (min / max across the λ sweep):")
print(amp.to_string())

# ── Pareto plots ──────────────────────────────────────────────────────────────

def pareto_front(sub, metric, ev_col="ev"):
    """Non-dominated rows: lower `metric` better, higher `ev_col` better."""
    pts = sub[[metric, ev_col]].values
    keep = []
    for i, (x_i, y_i) in enumerate(pts):
        dominated = any(
            (pts[j, 0] <= x_i and pts[j, 1] >= y_i) and
            (pts[j, 0] < x_i or pts[j, 1] > y_i)
            for j in range(len(pts)) if j != i
        )
        if not dominated:
            keep.append(i)
    return sub.iloc[keep].sort_values(metric)


metric_pairs = [
    ("mmd", "MMD"),
    ("ls",  "Linear Separability (BA)"),
    ("nls", "Non-linear (RF) Separability (BA)"),
    ("dp",  "Demographic Parity"),
]

fig, axes = plt.subplots(1, 4, figsize=(32, 8))
for ax, (metric, title) in zip(axes.flat, metric_pairs):
    for name in METHODS:
        sub = agg[agg["method"] == name].sort_values(metric)
        c = colors[name]
        # All sweep points as scatter
        # ax.scatter(sub[metric], sub["ev"], marker=".",
        #            color=c, s=50, alpha=0.1, label=name, zorder=3)
        # Pareto frontier as plain line (no markers)
        if len(sub) > 1:
            pf = pareto_front(sub, metric)
            ax.plot(pf[metric], pf["ev"], color=c,
                    linewidth=2.0, alpha=0.9, zorder=2, label=name)
            
    # Set 0 as y-axis limit, since that's the minimum explained variance (all methods should be above that)
    ax.set_ylim(bottom=0.2)
    ax.set_xlabel(title)
    ax.set_ylabel("Explained Variance")
    ax.grid(alpha=0.3)
for ax in axes.flat[len(metric_pairs):]:
    ax.set_visible(False)
axes.flat[0].legend(fontsize=9)
fig.suptitle("Utility–Fairness Pareto curves")
plt.tight_layout()
if save_fig:
    plt.savefig(FIGURES / "example_pareto.pdf")
if plot_fig:
    plt.show()
