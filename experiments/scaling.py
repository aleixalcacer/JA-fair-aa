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
N_SIZES = [100, 250, 500, 1000]
N_RUNS  = 10

# Identical params for both methods — tol=0 forces exactly max_iter iterations
PARAMS = {
    "n_archetypes": 3,
    "init":         "furthest_sum",
    "max_iter":     100,
    "n_init":       1,
    "tol":          0,
    "method":       "pgd",
}

archetypes_0 = np.array([[0, 0], [1, 2], [1, 0]])
archetypes_1 = np.array([[1, 0], [1, 2], [2, 0]])

# ── Benchmark loop ────────────────────────────────────────────────────────────
records = []

for n in tqdm(N_SIZES, desc="size"):
    size_0 = n // 2
    size_1 = n - size_0

    for run in tqdm(range(N_RUNS), desc=f"  n={n}", leave=False):
        rng = np.random.RandomState(run)

        X_0, _, _ = make_archetypal_dataset(
            archetypes_0, (size_0,), alpha=0.9, noise=0.1, generator=rng
        )
        X_1, _, _ = make_archetypal_dataset(
            archetypes_1, (size_1,), alpha=0.9, noise=0.1, generator=rng
        )

        X = np.concatenate([X_0, X_1])
        y = np.concatenate([np.ones(size_0), np.zeros(size_1)])
        Z = (y - np.mean(y)).reshape(-1, 1)

        # AA
        model_aa = AA(**PARAMS, random_state=run)
        t0 = time.perf_counter()
        model_aa.fit(X)
        t_aa = time.perf_counter() - t0

        # FairAA
        model_fair = FairAA(**PARAMS, fairness_const=1, random_state=run)

        t0 = time.perf_counter()
        _ = Z @ Z.T
        t_z = time.perf_counter() - t0

        t0 = time.perf_counter()
        model_fair.fit(X, Z=Z)
        t_fair = time.perf_counter() - t0

        t_fair -= t_z  # Subtract the time taken to compute the kernel matrix

        records.append({"n": n, "run": run, "method": "AA",     "time": t_aa})
        records.append({"n": n, "run": run, "method": "FairAA", "time": t_fair})

df = pd.DataFrame(records)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(data=df, x="n", y="time", hue="method", ax=ax)

ax.set_xlabel("Dataset size (n)")
ax.set_ylabel("Time (s)")
ax.set_title(f"AA vs FairAA — computation time ({PARAMS['max_iter']} iterations, {N_RUNS} runs)")
plt.tight_layout()
plt.savefig(FIGURES / "scaling.pdf")
plt.show()
