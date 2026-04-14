import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from archetypes.datasets import make_archetypal_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from tqdm import tqdm

from src.methods import AA, FairAA, FairAA_Adversarial, FairAA_MMD
from src.metrics import mmd_rbf

from archetypes.visualization import simplex

FIGURES = Path(__file__).resolve().parent.parent / "figures"
save_fig = False

# ── Data ──────────────────────────────────────────────────────────────────────

generator = np.random.RandomState(42)

archetypes_0 = np.array([[0, 0], [1, 2], [1, 0]])
X_0, _, _ = make_archetypal_dataset(archetypes_0, (200,), alpha=0.9, noise=0.1, generator=generator)
y_0 = np.ones(200)

archetypes_1 = np.array([[1, 0], [1, 2], [2, 0]])
X_1, _, _ = make_archetypal_dataset(archetypes_1, (200,), alpha=0.9, noise=0.1, generator=generator)
y_1 = np.zeros(200)

X = np.concatenate([X_0, X_1])
y = np.concatenate([y_0, y_1])

# Sensitive attribute in the three formats required by the four methods
Z   = (y - y.mean()).reshape(-1, 1)          # FairAA:             centered, (n, 1)
Z_0 = (y_0 - y.mean()).reshape(-1, 1)
Z_1 = (y_1 - y.mean()).reshape(-1, 1)
z   = y.copy()                               # FairAA_Adversarial: float (n,)
z_0 = y_0.copy()
z_1 = y_1.copy()
z_i = y.astype(np.int32)                     # FairAA_MMD:         int   (n,)
z_0i = y_0.astype(np.int32)
z_1i = y_1.astype(np.int32)

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.scatter(X[:, 0], X[:, 1], c=y, cmap="winter")
# ax.set_title("Data")
# if save_fig:
#     plt.savefig(FIGURES / "example_data.pdf")
# plt.show()

# ── Experiment ────────────────────────────────────────────────────────────────

base_params = dict(n_archetypes=3, init="furthest_sum", max_iter=300, tol=1e-6, method="pgd")
# FairAA_MMD uses O(n²) kernel matrices per iteration — fewer iterations
mmd_params  = dict(n_archetypes=3, init="furthest_sum", max_iter=50,  tol=1e-6, method="pgd")

METHODS = ["AA", "FairAA", "FairAA_Adv"]
markers = {"AA": "s", "FairAA": "^", "FairAA_Adv": "D"}

metrics = None
best = {m: None for m in METHODS}

for i in tqdm(range(10)):

    # AA ──────────────────────────────────────────────────────────────────────
    aa = AA(**base_params, random_state=i)
    aa.fit(X)
    S_aa   = aa.transform(X)
    S_0_aa = aa.transform(X_0)
    S_1_aa = aa.transform(X_1)
    X_aa   = S_aa @ aa.archetypes_

    # FairAA ──────────────────────────────────────────────────────────────────
    faa = FairAA(**base_params, fairness_const=1, random_state=i)
    faa.fit(X, Z=Z)
    S_faa   = faa.transform(X, Z)
    S_0_faa = faa.transform(X_0, Z_0)
    S_1_faa = faa.transform(X_1, Z_1)
    X_faa   = S_faa @ faa.archetypes_

    # FairAA_Adversarial ──────────────────────────────────────────────────────
    adv = FairAA_Adversarial(**base_params, lambda_=10, random_state=i)
    adv.fit(X, z=y)
    S_adv   = adv.transform(X, y)
    S_0_adv = adv.transform(X_0, y_0)
    S_1_adv = adv.transform(X_1, y_1)
    X_adv   = S_adv @ adv.archetypes_

    # FairAA_MMD ──────────────────────────────────────────────────────────────
    # mmd_m = FairAA_MMD(**mmd_params, lambda_=1, random_state=i)
    # mmd_m.fit(X, z=z_i)
    # S_mmd   = mmd_m.transform(X, z_i)
    # S_0_mmd = mmd_m.transform(X_0, z_0i)
    # S_1_mmd = mmd_m.transform(X_1, z_1i)
    # X_mmd   = S_mmd @ mmd_m.archetypes_

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_S = [S_aa,   S_faa,   S_adv]
    all_S0 = [S_0_aa, S_0_faa, S_0_adv]
    all_S1 = [S_1_aa, S_1_faa, S_1_adv]
    all_X  = [X_aa,   X_faa,   X_adv]

    ev = pd.DataFrame({
        "metric": "explained_variance",
        "value":  [explained_variance_score(X, Xr) for Xr in all_X],
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

    metrics = pd.concat([metrics, ev, mmd_vals, ls])

    for model, name in [(aa, "AA"), (faa, "FairAA"), (adv, "FairAA_Adv")]:
        if best[name] is None or model.rss_ < best[name].rss_:
            best[name] = model

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric, title in zip(axes,
                              ["explained_variance", "mmd", "linear_separability"],
                              ["Explained Variance", "MMD", "Linear Separability"]):
    sns.boxplot(data=metrics[metrics["metric"] == metric], x="method", y="value", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("")
plt.tight_layout()
if save_fig:
    plt.savefig(FIGURES / "example_metrics.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", alpha=0.2)
for name, model in best.items():
    ax.scatter(*model.archetypes_.T, marker=markers[name], s=80, label=name)
ax.set_title("Archetypes")
ax.legend()
if save_fig:
    plt.savefig(FIGURES / "example_archetypes.pdf")
plt.show()

# Make a plot of the best reconstructions for each method
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, METHODS):
    model = best[name]
    if name == "AA":
        Xr = S_aa
    elif name == "FairAA":
        Xr = S_faa
    elif name == "FairAA_Adv":
        Xr = S_adv
    else:        raise ValueError(f"Unknown method: {name}")
    
    simplex(Xr, ax=ax, color=y, cmap="winter", alpha=0.2)
    ax.set_title(name)
plt.tight_layout()

if save_fig:
    plt.savefig(FIGURES / "example_reconstructions.pdf")
plt.show()