import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from archetypes.datasets import make_archetypal_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from tqdm import tqdm

from src.methods import AA, FairAA, FairAA_Adversarial, FairAA_MMD, FairPCA_AA
from src.metrics import mmd_rbf
from src.fair_pca import FairPCA

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
X, _, _ = make_archetypal_dataset(archetypes, (n,), alpha=0.9, noise=0.1, generator=generator)
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
if plot_fig: plt.show()

# Sensitive attribute in the three formats required by the four methods
Z   = (y - y.mean()).reshape(-1, 1)          # FairAA:             centered, (n, 1)
Z_0 = (y_0 - y.mean()).reshape(-1, 1)
Z_1 = (y_1 - y.mean()).reshape(-1, 1)




# ── Experiment ────────────────────────────────────────────────────────────────
n_archetypes = 4

base_params = dict(n_archetypes=n_archetypes, init="furthest_sum", max_iter=300, tol=1e-8, method="pgd")



# FairPCA_AA ──────────────────────────────────────────────────────────────


fair_pca = FairPCA(target_dim=2, tradeoff_param=0.5, standardize=True)
fair_pca.fit(X, prot_attribute=y)
S_fair_pca = fair_pca.transform(X)

fair_aa = AA(**base_params, random_state=42)
fair_aa.fit(S_fair_pca)

print(S_fair_pca.shape)

fair_archetypes = fair_aa.archetypes_

pca = PCA(n_components=2)
S_pca = pca.fit_transform(X)

print(S_pca.shape)
aa = AA(**base_params, random_state=42)
aa.fit(X)

archetypes = aa.archetypes_

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

print(y.shape)

sns.scatterplot(x=S_pca[:, 0], y=S_pca[:, 1], hue=y, palette="winter", ax=ax[0])
sns.scatterplot(x=archetypes[:, 0], y=archetypes[:, 1], ax=ax[0], marker="X", s=100, edgecolor="black", legend=False)
ax[0].set_title("PCA")

sns.scatterplot(x=S_fair_pca[:, 2], y=S_fair_pca[:, 3], hue=y, palette="winter", ax=ax[1])
sns.scatterplot(x=fair_archetypes[:, 0], y=fair_archetypes[:, 1], ax=ax[1], marker="X", s=100, edgecolor="black", legend=False)
ax[1].set_title("FairPCA")
if save_fig:
    plt.savefig(FIGURES / "example_fair_pca.pdf")
if plot_fig: plt.show() 