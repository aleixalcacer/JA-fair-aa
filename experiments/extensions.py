"""Synthetic-dataset extensions: archetypal pairs, blobs, moons, multi-class.

Two figures per dataset:
    1. dataset scatter with discovered archetypes overlaid
    2. simplex view of the loading coefficients
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pypalettes import load_cmap
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import OneHotEncoder

from archetypes import AA, FairAA, KernelAA, FairKernelAA
from archetypes.datasets import make_archetypal_dataset
from archetypes.visualization import simplex
from src.methods import FairAA_3Moment
from src.visualization import CMAP, CMAP_NAME
# ── Output ─────────────────────────────────────────────────────────────────────

FIGURES = Path(__file__).resolve().parent.parent / "figures/extensions"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────


try:
    mpl.colormaps.register(CMAP, name=CMAP_NAME, force=True)
except (ValueError, TypeError):
    pass
plt.set_cmap(CMAP_NAME)


CLASS_IDX       = [0, 2, 1, 3]
ARCH_COLOR      = "black"
ARCH_MARKER     = "s"           # default (used in simplex plot)
ARCH_MARKERS_2  = ("s", "D")    # square + diamond, one per method
POINT_SIZE      = 25            # data points (both plots)
ARCH_SIZE       = 50            # archetype markers — slightly larger than dots


def class_color(c):
    return CMAP(CLASS_IDX[c])


# ── Datasets ───────────────────────────────────────────────────────────────────


def dataset_archetypal():
    rng = np.random.RandomState(42)
    a0 = np.array([[0, 0], [1, 2], [1, 0]])
    a1 = np.array([[1, 0], [1, 2], [2, 0]])
    X0, *_ = make_archetypal_dataset(a0, (200,), alpha=0.9, noise=0.1, generator=rng)
    X1, *_ = make_archetypal_dataset(a1, (200,), alpha=0.9, noise=0.1, generator=rng)
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.ones(200), np.zeros(200)]).astype(int)
    Z = (y - y.mean()).reshape(-1, 1)
    return X, y, Z


def dataset_blobs():
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=(100, 200),
                      centers=[[0, 0], [0, 0]],
                      cluster_std=[1.0, 3.0], random_state=rng)
    Z = (y - y.mean()).reshape(-1, 1)
    return X, y, Z


def dataset_moons():
    rng = np.random.RandomState(42)
    X, y = make_moons(n_samples=(200, 100), noise=0.1, random_state=rng)
    Z = (y - y.mean()).reshape(-1, 1)
    return X, y, Z


def dataset_multi():
    rng = np.random.RandomState(2)
    X, y = make_blobs(centers=4, cluster_std=2.5, n_samples=200, random_state=rng)
    one_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    Z = one_hot / one_hot.mean(axis=0, keepdims=True)
    return X, y, Z


# Per-dataset config: which two methods to fit, hyper-parameters and titles.
# `pair` ∈ {"AA_FairAA", "FairAA_HO", "Kernel"}
DATASETS = {
    "archetypal": dict(make=dataset_archetypal, pair="AA_FairAA",
                       n_archetypes=3, fair_const=1.0,
                       titles=("AA", "FairAA")),
    "blobs":      dict(make=dataset_blobs, pair="FairAA_HO",
                       n_archetypes=3, fair_const=15.0, fair_const_2=65.0,
                       titles=("FairAA", "FairAA (High Orders)")),
    "moons":      dict(make=dataset_moons, pair="Kernel",
                       n_archetypes=8, fair_const=0.3,
                       titles=("KernelAA", "KernelFairAA")),
    "multi":      dict(make=dataset_multi, pair="AA_FairAA",
                       n_archetypes=3, fair_const=1.0,
                       titles=("AA", "FairAA")),
}


# ── Fit ────────────────────────────────────────────────────────────────────────


AA_PARAMS = dict(init="furthest_sum", max_iter=1_000, n_init=5, tol=1e-12, method="pgd")
KERNEL_PARAMS = dict(init="furthest_sum", max_iter=1_000, n_init=3, tol=1e-12,
                     method="pgd", kernel="rbf", kernel_params={"gamma": 1.0},
                     method_params={"n_iter": 10, "step_size": 1e-3, "beta": 0.8})


def _build_models(pair, cfg, i):
    """Return a (model_left, model_right) pair built with fixed hyper-params."""
    k = cfg["n_archetypes"]
    if pair == "AA_FairAA":
        return (AA(n_archetypes=k, **AA_PARAMS, random_state=i),
                FairAA(n_archetypes=k, **AA_PARAMS,
                       fairness_const=cfg["fair_const"], random_state=i))
    if pair == "FairAA_HO":
        return (FairAA(n_archetypes=k, **AA_PARAMS,
                       fairness_const=cfg["fair_const"], random_state=i),
                FairAA_3Moment(n_archetypes=k, **AA_PARAMS,
                               fairness_const=cfg["fair_const_2"],
                               random_state=i))
    if pair == "Kernel":
        return (KernelAA(n_archetypes=k, **KERNEL_PARAMS, random_state=i),
                FairKernelAA(n_archetypes=k, **KERNEL_PARAMS,
                             fairness_const=cfg["fair_const"], random_state=i))
    raise ValueError(pair)


def _fit_pair(model_left, model_right, X, Z, pair):
    """Dispatch the fairness signal correctly to each model variant."""
    if pair == "AA_FairAA":
        model_left.fit(X)
        model_right.fit(X, Z=Z)
    elif pair == "FairAA_HO":
        model_left.fit(X, Z=Z)
        # FairAA_3Moment expects a 1-D float vector
        model_right.fit(X, Z=Z.reshape(-1).astype(float))
    elif pair == "Kernel":
        model_left.fit(X)
        model_right.fit(X, Z=Z)


def fit_best(X, Z, cfg, n_runs=10):
    """Fit each model in the pair `n_runs` times; keep the lowest-loss run."""
    pair = cfg["pair"]
    best_l, best_r = None, None
    for i in range(n_runs):
        m_l, m_r = _build_models(pair, cfg, i)
        _fit_pair(m_l, m_r, X, Z, pair)
        if best_l is None or m_l.loss_[-1] < best_l.loss_[-1]:
            best_l = m_l
        if best_r is None or m_r.loss_[-1] < best_r.loss_[-1]:
            best_r = m_r
    return best_l, best_r


def coefs(model):
    return getattr(model, "coefficients_",
                   getattr(model, "similarity_degree_", None))


def has_2d_archetypes(model):
    """True if archetypes can be plotted in input space (i.e. not kernel-space)."""
    archs = getattr(model, "archetypes_", None)
    return archs is not None and not isinstance(model, (KernelAA, FairKernelAA))


# ── Plot 1: data + archetypes ──────────────────────────────────────────────────


def plot_data_archetypes(X, y, model_l, model_r, titles, name):
    """Single-panel scatter; both methods' archetypes overlaid on the same
    axes, each method using a different marker."""
    classes = np.unique(y)
    fig, ax = plt.subplots(figsize=(5, 4))

    for c in classes:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=class_color(c), s=POINT_SIZE, alpha=0.7)

    method_handles = []
    for model, title, marker in zip([model_l, model_r], titles, ARCH_MARKERS_2):
        archs = np.asarray(model.archetypes_)
        ax.scatter(archs[:, 0], archs[:, 1],
                   color=ARCH_COLOR, marker=marker,
                   s=ARCH_SIZE*2, edgecolor="white", linewidth=0.5, zorder=5)
        method_handles.append(
            Line2D([], [], marker=marker, linestyle="", color=ARCH_COLOR,
                   markersize=8, markeredgecolor="white", markeredgewidth=0.5,
                   label=title))

    # ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    # ax.set_aspect("equal", "datalim")
    ax.axis("off")

    handles = [Line2D([], [], marker="o", linestyle="", color=class_color(c),
                      markersize=6, label=f"Class {int(c)}")
               for c in classes] + method_handles
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(FIGURES / f"extensions_{name}_archetypes.pdf",
                bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: simplex of loadings ────────────────────────────────────────────────


def plot_simplex(model_l, model_r, y, titles, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    point_colors = np.array([class_color(c) for c in y])
    classes = np.unique(y)

    for ax, m, title, marker in zip(axes, [model_l, model_r], titles,
                                    ARCH_MARKERS_2):
        simplex(coefs(m), ax=ax, color=point_colors, show_vertices=True,
                s=POINT_SIZE, alpha=0.7,
                vertices_params={"color": ARCH_COLOR, "marker": marker,
                                 "s": ARCH_SIZE,
                                 "linewidth": 0.5, "zorder": 5},
                                 axis_params={
                                     "color": "lightgrey",
                                 })
        ax.set_title(title)

    handles = [Line2D([], [], marker="o", linestyle="",
                      color=class_color(c), markersize=6,
                      label=f"Class {int(c)}")
               for c in classes]
    handles.append(Line2D([], [], marker=ARCH_MARKERS_2[0], linestyle="",
                          color=ARCH_COLOR, markersize=8,
                          label="Archetypes"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(FIGURES / f"extensions_{name}_simplex.pdf",
                bbox_inches="tight")
    plt.close(fig)



# ── Main ───────────────────────────────────────────────────────────────────────


def run(name):
    print(f"\n=== {name} ===")
    cfg = DATASETS[name]
    X, y, Z = cfg["make"]()
    model_l, model_r = fit_best(X, Z, cfg)
    plot_data_archetypes(X, y, model_l, model_r, cfg["titles"], name)
    plot_simplex(model_l, model_r, y, cfg["titles"], name)
    
    print(f"saved figures/extensions_{name}_archetypes.pdf "
          f"and figures/extensions_{name}_simplex.pdf")


if __name__ == "__main__":
    for ds_name in DATASETS:
        run(ds_name)
