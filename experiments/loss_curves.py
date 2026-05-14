import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from archetypes.datasets import make_archetypal_dataset

from src.methods import FairAA, FairAA_3Moment, FairAA_Adversarial, FairAA_MMD

# ── Config ─────────────────────────────────────────────────────────────────────

N = 200
N_SEEDS = 20
N_ARCHETYPES = 4

FIXED_LAMBDAS = {
    "FairAA":         0.5,
    "FairAA_3Moment": 1e-1,
    "FairAA_Adv":     1,
    "FairAA_MMD":     100.,
}

BASE_PARAMS = dict(
    n_archetypes=N_ARCHETYPES,
    init="furthest_sum",
    n_init=3,
    max_iter=500,
    tol=1e-8,
    method="pgd",
)

_ALL_METHODS = ["AA", "FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
COLORS = dict(zip(_ALL_METHODS, plt.cm.tab10.colors))

# ── Data ───────────────────────────────────────────────────────────────────────

s = 1 / (2 * math.sqrt(2))
tetra = np.array([[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]])
generator = np.random.RandomState(42)
X, _, _ = make_archetypal_dataset(tetra, (N,), alpha=0.1, noise=0, generator=generator)
y = (X[:, 0] > 0).astype(int)
Z = (y - y.mean()).reshape(-1, 1)

# ── Fit & extract loss ─────────────────────────────────────────────────────────

def get_loss(method_name, seed):
    lam = FIXED_LAMBDAS[method_name]
    if method_name == "FairAA":
        model = FairAA(**BASE_PARAMS, fairness_const=lam, random_state=seed)
        model.fit(X, Z=Z)
        return list(model.loss_)
    elif method_name == "FairAA_3Moment":
        model = FairAA_3Moment(**BASE_PARAMS, fairness_const=lam,
                               alpha_2=0.1, alpha_3=0.1, balance_orders=True,
                               random_state=seed)
        model.fit(X, Z=y.astype(float))
        return list(model.loss_history_["total"])
    elif method_name == "FairAA_Adv":
        model = FairAA_Adversarial(**BASE_PARAMS, fairness_const=lam,
                                   n_adv_steps=2, lr_adv=1e-4, random_state=seed)
        model.fit(X, Z=y)
        return list(model.loss_history_["total"])
    elif method_name == "FairAA_MMD":
        model = FairAA_MMD(**BASE_PARAMS, fairness_const=lam, random_state=seed)
        model.fit(X, Z=y)
        return list(model.loss_history_["total"])
    else:
        raise ValueError(method_name)


METHODS = ["FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD"]

print(f"Fitting {len(METHODS)} methods × {N_SEEDS} seeds on n={N}…")
trajectories = {}
for m in METHODS:
    print(f"  {m}", end="", flush=True)
    runs = []
    for seed in range(N_SEEDS):
        try:
            runs.append(get_loss(m, seed + 42))
        except Exception as e:
            print(f" [seed {seed} failed: {e}]", end="")
    trajectories[m] = runs
    print(f" done ({len(runs)} runs)")

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(METHODS), figsize=(4.5 * len(METHODS), 4), sharey=False)

for ax, m in zip(axes, METHODS):
    runs = trajectories[m]
    if not runs:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(m)
        continue
    min_len = min(len(r) for r in runs)
    iters = np.arange(min_len)
    for r in runs:
        ax.plot(iters, r[:min_len], color=COLORS[m], alpha=0.4, linewidth=0.9)
    ax.set_title(m, fontsize=10)
    ax.set_xlabel("Iteration")
    if ax is axes[0]:
        ax.set_ylabel("Total loss")
    ax.grid(alpha=0.3)

fig.suptitle(f"Loss trajectories — all {N_SEEDS} seeds (n={N})")
fig.tight_layout()
plt.show()
