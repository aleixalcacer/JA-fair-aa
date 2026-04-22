import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from archetypes.datasets import make_archetypal_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from tqdm import tqdm

from src.methods import AA, FairAA, FairAA_Adversarial, FairAA_MMD, FairPCA_AA
from src.metrics import mmd_rbf

from archetypes.visualization import simplex

FIGURES = Path(__file__).resolve().parent.parent / "figures"
save_fig = False
plot_fig = True

# ── Data ──────────────────────────────────────────────────────────────────────

n = 200

generator = np.random.RandomState(42)

s = 1 / (2 * math.sqrt(2))

tetra = [
    [s, s, s],
    [s, -s, -s],
    [-s, s, -s],
    [-s, -s, s]
]

archetypes = np.array(tetra)
X, _, _ = make_archetypal_dataset(archetypes, (n,), alpha=0.1, noise=0.1, generator=generator)
y = (X[:, 0] > 0).astype(int)  # Sensitive attribute based on the first feature

X_0 = X[y == 0]
X_1 = X[y == 1]

y_0 = y[y == 0]
y_1 = y[y == 1]

# 3D scatter plot of the data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="winter")
ax.set_title("Data")
if save_fig:
    plt.savefig(FIGURES / "example_data.pdf")
# if plot_fig: plt.show()

# Sensitive attribute in the three formats required by the four methods
Z   = (y - y.mean()).reshape(-1, 1)          # FairAA:             centered, (n, 1)
Z_0 = (y_0 - y.mean()).reshape(-1, 1)
Z_1 = (y_1 - y.mean()).reshape(-1, 1)




# ── Experiment ────────────────────────────────────────────────────────────────

n_archetypes = 4

base_params = dict(n_archetypes=n_archetypes, init="furthest_sum", max_iter=500, tol=0, method="pgd")

METHODS = ["AA", "FairAA", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
markers = {"AA": "s", "FairAA": "^", "FairAA_Adv": "D", "FairAA_MMD": "o", "FairPCA_AA": "P"}

metrics = None
best = {m: None for m in METHODS}

for i in tqdm(range(5)):

    # AA ──────────────────────────────────────────────────────────────────────
    aa = AA(**base_params, random_state=i)
    t0 = time.perf_counter()
    aa.fit(X)
    t_aa = time.perf_counter() - t0
    S_aa   = aa.transform(X)
    S_0_aa = S_aa[y == 1]
    S_1_aa = S_aa[y == 0]
    X_aa   = S_aa @ aa.archetypes_

    # FairAA ──────────────────────────────────────────────────────────────────
    faa = FairAA(**base_params, fairness_const=0.5, random_state=i)
    t0 = time.perf_counter()
    faa.fit(X, Z=Z)
    t_faa = time.perf_counter() - t0
    S_faa   = faa.transform(X, Z)
    S_0_faa = S_faa[y == 1]
    S_1_faa = S_faa[y == 0]
    X_faa   = S_faa @ faa.archetypes_

    # FairAA_Adversarial ──────────────────────────────────────────────────────
    adv = FairAA_Adversarial(**base_params, fairness_const=2, n_adv_steps=1, lr_adv=1e-2, random_state=i)
    t0 = time.perf_counter()
    adv.fit(X, Z=y)
    t_adv = time.perf_counter() - t0
    S_adv   = adv.transform(X, Z=y)
    S_0_adv = S_adv[y == 1]
    S_1_adv = S_adv[y == 0]
    X_adv   = S_adv @ adv.archetypes_

    # FairAA_MMD ──────────────────────────────────────────────────────────────
    mmd_m = FairAA_MMD(**base_params, fairness_const=100, random_state=i)
    t0 = time.perf_counter()
    mmd_m.fit(X, Z=y)
    t_mmd = time.perf_counter() - t0
    S_mmd   = mmd_m.transform(X, Z=y)
    S_0_mmd = S_mmd[y == 1]
    S_1_mmd = S_mmd[y == 0]
    X_mmd   = S_mmd @ mmd_m.archetypes_

    # FairPCA_AA ──────────────────────────────────────────────────────────────
    fpca_aa = FairPCA_AA(**base_params, target_dim=2, tradeoff_param=0, random_state=i)
    t0 = time.perf_counter()
    fpca_aa.fit(X, z=y.astype(int))
    t_fpca = time.perf_counter() - t0
    S_fpca_aa   = fpca_aa.transform(X)
    S_0_fpca_aa = S_fpca_aa[y == 1]
    S_1_fpca_aa = S_fpca_aa[y == 0]
    X_fpca_aa   = fpca_aa.fair_pca_.inverse_transform(S_fpca_aa @ fpca_aa.archetypes_)

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_S  = [S_aa,   S_faa,   S_adv,   S_mmd,   S_fpca_aa]
    all_S0 = [S_0_aa, S_0_faa, S_0_adv, S_0_mmd, S_0_fpca_aa]
    all_S1 = [S_1_aa, S_1_faa, S_1_adv, S_1_mmd, S_1_fpca_aa]
    all_X  = [X_aa,   X_faa,   X_adv,   X_mmd,   X_fpca_aa]

    ev = pd.DataFrame({
        "metric": "explained_variance",
        "value":  [explained_variance_score(X[:, :Xr.shape[1]], Xr) for Xr in all_X],
        "method": METHODS,
        "run": i,
    })
    mmd_vals = pd.DataFrame({
        "metric": "mmd",
        "value":  [mmd_rbf(S0, S1) for S0, S1 in zip(all_S0, all_S1)],
        "method": METHODS,
        "run": i,
    })
    splits = [train_test_split(S, y, test_size=0.3, random_state=i) for S in all_S]
    ls = pd.DataFrame({
        "metric": "linear_separability",
        "value":  [LogisticRegression(random_state=i).fit(tr, y_tr).score(te, y_te)
                   for (tr, te, y_tr, y_te) in splits],
        "method": METHODS,
        "run": i,
    })
    rt = pd.DataFrame({
        "metric": "runtime",
        "value":  [t_aa, t_faa, t_adv, t_mmd, t_fpca],
        "method": METHODS,
        "run": i,
    })

    metrics = pd.concat([metrics, ev, mmd_vals, ls, rt])

    for model, name in [(aa, "AA"), (faa, "FairAA"), (adv, "FairAA_Adv"), (mmd_m, "FairAA_MMD"), (fpca_aa, "FairPCA_AA")]:
        if best[name] is None or model.rss_ < best[name].rss_:
            best[name] = model

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, metric, title in zip(axes,
                              ["explained_variance", "mmd", "linear_separability", "runtime"],
                              ["Explained Variance", "MMD", "Linear Separability", "Runtime (s)"]):
    sns.boxplot(data=metrics[metrics["metric"] == metric], x="method", y="value", ax=ax, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("")
plt.tight_layout()
if save_fig:
    plt.savefig(FIGURES / "example_metrics.pdf")
if plot_fig: plt.show()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*X.T, c=y, cmap="winter", alpha=0.2)
for name, model in best.items():
    if name == "FairPCA_AA":
        archs = model.fair_pca_.inverse_transform(model.archetypes_)
    else:
        archs = model.archetypes_
    ax.scatter(*archs.T, marker=markers[name], s=80, label=name)
ax.set_title("Archetypes")
ax.legend()
if save_fig:
    plt.savefig(FIGURES / "example_archetypes.pdf")
if plot_fig: plt.show()

# Make a plot of the best reconstructions for each method
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
last_S = {"AA": S_aa, "FairAA": S_faa, "FairAA_Adv": S_adv, "FairAA_MMD": S_mmd, "FairPCA_AA": S_fpca_aa}
for ax, name in zip(axes, METHODS):
    simplex(last_S[name], ax=ax, color=y, cmap="winter", alpha=0.2)
    ax.set_title(name)
plt.tight_layout()

if save_fig:
    plt.savefig(FIGURES / "example_reconstructions.pdf")
if plot_fig: plt.show()