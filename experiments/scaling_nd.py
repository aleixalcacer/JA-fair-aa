import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FIGURES = Path(__file__).resolve().parent.parent / "figures"

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from archetypes import AA, FairAA
from archetypes.datasets import make_archetypal_dataset
from tqdm import tqdm

# ── Visual setup ──────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF5722"]  # blue / deep-orange
sns.set_palette(COLORS)

# ── Benchmark config ──────────────────────────────────────────────────────────
N_SIZES = [100, 250, 500, 1000, 2500, 5000]
D_SIZES = [2, 10, 50]
N_RUNS  = 10

PARAMS = {
    "n_archetypes": 3,
    "init":         "furthest_sum",
    "max_iter":     100,
    "n_init":       1,
    "tol":          0,
    "method":       "pgd",
}

# ── Data generation ───────────────────────────────────────────────────────────
def make_dataset(n, d, rng):
    """Two groups of n/2 points each, generated from random archetypes in R^d."""
    n_archetypes = PARAMS["n_archetypes"]
    size_0 = n // 2
    size_1 = n - size_0

    arch_0 = rng.uniform(0, 1, size=(n_archetypes, d))
    arch_1 = arch_0 + rng.uniform(0.2, 0.5, size=d)  # shifted group

    X_0, _, _ = make_archetypal_dataset(arch_0, (size_0,), alpha=0.9, noise=0.1, generator=rng)
    X_1, _, _ = make_archetypal_dataset(arch_1, (size_1,), alpha=0.9, noise=0.1, generator=rng)

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([np.ones(size_0), np.zeros(size_1)])
    Z = (y - np.mean(y)).reshape(-1, 1)
    return X, Z

# ── Benchmark loop ────────────────────────────────────────────────────────────
records = []

for d in tqdm(D_SIZES, desc="dim"):
    for n in tqdm(N_SIZES, desc=f"  d={d}", leave=False):
        size_0 = n // 2
        size_1 = n - size_0

        for run in range(N_RUNS):
            rng = np.random.RandomState(run)
            X, Z = make_dataset(n, d, rng)

            # AA
            model_aa = AA(**PARAMS, random_state=run)
            t0 = time.perf_counter()
            model_aa.fit(X)
            t_aa = time.perf_counter() - t0

            # FairAA
            model_fair = FairAA(**PARAMS, fairness_const=1, random_state=run)
            t0 = time.perf_counter()
            model_fair.fit(X, Z=Z)
            t_fair = time.perf_counter() - t0

            records.append({"n": n, "d": d, "run": run, "method": "AA",     "time": t_aa})
            records.append({"n": n, "d": d, "run": run, "method": "FairAA", "time": t_fair})

df = pd.DataFrame(records)

# ── Plot: one subplot per d ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(D_SIZES), figsize=(5 * len(D_SIZES), 5), sharey=False)

for ax, d in zip(axes, D_SIZES):
    sub = df[df["d"] == d]
    sns.boxplot(data=sub, x="n", y="time", hue="method", ax=ax)
    ax.set_title(f"d = {d}")
    ax.set_xlabel("n")
    ax.set_ylabel("Time (s)" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=45)
    if ax is not axes[0]:
        ax.get_legend().remove()

axes[0].legend(title="Method")
fig.suptitle(
    f"AA vs FairAA — computation time ({PARAMS['max_iter']} iters, {N_RUNS} runs)",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES / "scaling_nd.pdf", bbox_inches="tight")
plt.show()
