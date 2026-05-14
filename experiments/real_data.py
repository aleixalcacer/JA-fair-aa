"""Real-data downstream experiment.

Compares Raw X / Standard AA / FairAA on downstream utility (AUC, Acc, F1)
and group-fairness gaps (DP, EO, EOpp) on a tabular dataset registered in
``experiments/datasets.py``. Spec: ``real_data_experiment.md``.

Usage::

    python experiments/real_data.py --dataset german_credit \\
        --K 10 --lambdas 0,0.01,0.1,0.5,1,5,10 --n_seeds 5

Cached artefacts live in ``data/``; figures in ``figures/``. Delete the
``raw_results_realdata_<dataset>.csv`` file to re-run the sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# allow `python experiments/real_data.py` to find ``src`` and siblings
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.methods import (AA, FairAAWarm, FairAA_3Moment,
                         FairAA_Adversarial, FairAA_MMD, FairPCA_AA)
from src.metrics import (dp_gap, eo_gap, eopp_gap,
                         explained_variance, mmd_rbf,
                         linear_separability, nonlinear_separability,
                         demographic_parity)

import datasets as dataset_registry

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
FIGURES = ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def data_dir(dataset: str):
    p = DATA_DIR / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def fig_dir(dataset: str):
    p = FIGURES / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── AA hyper-parameters (shared across seeds and λ) ───────────────────────────

AA_PARAMS = dict(init="furthest_sum", n_init=5, max_iter=1000,
                 tol=1e-12, method="pseudo_pgd")


# ── Data preparation ─────────────────────────────────────────────────────────

def split_and_scale(X, y, a, seed, test_size=0.25, val_size=0.0):
    """Joint-stratify by (y, a); fit StandardScaler on train only."""
    strat = np.char.add(y.astype(str), "_")
    strat = np.char.add(strat, a.astype(str))

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                           stratify=strat, random_state=seed)
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size / (1 - test_size),
            stratify=strat[train_idx], random_state=seed)
    else:
        val_idx = np.empty(0, dtype=int)

    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X)
    return {
        "X_train": Xs[train_idx], "y_train": y[train_idx], "a_train": a[train_idx],
        "X_val":   Xs[val_idx],   "y_val":   y[val_idx],   "a_val":   a[val_idx],
        "X_test":  Xs[test_idx],  "y_test":  y[test_idx],  "a_test":  a[test_idx],
        "scaler": scaler,
        "train_idx": train_idx, "test_idx": test_idx, "val_idx": val_idx,
    }


# ── Representation learning ───────────────────────────────────────────────────


def _elbow_index(xs, ys):
    """Knee/elbow of an increasing concave curve via max-distance-to-chord
    (the standard kneedle heuristic). Returns the index in `xs`."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return len(xs) // 2
    x_n = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-12)
    y_n = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    # signed distance from each (xi, yi) to the chord from (x0,y0) to (xn,yn)
    return int(np.argmax(y_n - x_n))


def find_best_K(X_fit, K_values, n_seeds, log=None):
    """Sweep AA over `K_values`, fit `n_seeds` times each, return the K at
    the elbow of mean train-EV(K) plus the per-K stats.

    EV is computed on the fit set (the same subset AA actually optimises);
    AA reconstruction is monotone in K, so the question is only where the
    diminishing returns begin — train-EV is the right summary."""
    rows = []
    for K in K_values:
        evs = []
        for seed in range(n_seeds):
            m = AA(n_archetypes=K, random_state=seed, **AA_PARAMS)
            m.fit(X_fit)
            R = m.transform(X_fit)
            evs.append(explained_variance(X_fit, R @ m.archetypes_))
        rows.append({"K": K, "ev_mean": float(np.mean(evs)),
                     "ev_std": float(np.std(evs))})
        if log is not None:
            log.info("scree  K=%2d  EV=%.4f ± %.4f",
                     K, rows[-1]["ev_mean"], rows[-1]["ev_std"])
    df = pd.DataFrame(rows)
    idx = _elbow_index(df["K"], df["ev_mean"])
    return int(df["K"].iloc[idx]), df


def plot_scree(scree_df, best_K, dataset):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(scree_df["K"], scree_df["ev_mean"], yerr=scree_df["ev_std"],
                fmt="o-", color="#1f77b4", capsize=3)
    ax.axvline(best_K, color="#d62728", linestyle="--",
               label=f"elbow K*={best_K}")
    ax.set_xlabel("K (number of archetypes)")
    ax.set_ylabel("Explained variance (train)")
    ax.set_title(f"Scree plot: {dataset}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir(dataset) / f"scree.{ext}", bbox_inches="tight")
    plt.close(fig)


def fit_aa(X_train, K, seed):
    m = AA(n_archetypes=K, random_state=seed, **AA_PARAMS)
    m.fit(X_train)
    return m


def fit_fairaa(X_train, a_train, K, lam, seed, S_init=None, C_init=None):
    Z = (a_train - a_train.mean()).reshape(-1, 1).astype(float)
    m = FairAAWarm(n_archetypes=K, fairness_const=float(lam), random_state=seed,
                   **AA_PARAMS)
    m.fit(X_train, Z=Z, S_init=S_init, C_init=C_init)
    return m


def transform_aa(model, X):
    return model.transform(X)


def transform_fairaa(model, X, a):
    Z = (a - a.mean()).reshape(-1, 1).astype(float)
    return model.transform(X, Z)


def fit_fairaa_3m(X_train, a_train, K, lam, seed, S_init=None, C_init=None):
    m = FairAA_3Moment(n_archetypes=K, fairness_const=float(lam),
                       alpha_2=0.1, alpha_3=0.1, balance_orders=True,
                       random_state=seed, **AA_PARAMS)
    m.fit(X_train, Z=a_train.astype(float),
          S_init=S_init, C_init=C_init)
    return m


def transform_fairaa_3m(model, X, a):
    """Inductive transform: re-derive z_c / zzT from the new sample's
    protected attribute, then run the same moment_transform routine that
    fit/transform uses internally. Bypasses the fit-time Z stored on the
    model, which would otherwise mismatch the test-set size."""
    from src.methods.fair_aa_3moment import moment_transform
    z = np.asarray(a, dtype=np.float64)
    z_c = z - z.mean()
    zzT = None if model.rank1_zzT else z_c[:, None] * z_c[None, :]
    a2, a3 = model._resolve_alphas()
    method_params = {} if model.method_params is None else model.method_params
    return moment_transform(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)),
        z_c, zzT, model.archetypes_,
        fairness_const=model.fairness_const,
        alpha_2=a2, alpha_3=a3, rank1_zzT=model.rank1_zzT,
        max_iter=model.max_iter, tol=model.tol, **method_params,
    )


def fit_fairaa_adv(X_train, a_train, K, lam, seed, S_init=None, C_init=None):
    m = FairAA_Adversarial(n_archetypes=K, fairness_const=float(lam),
                           n_adv_steps=3, lr_adv=1e-2,
                           random_state=seed, **AA_PARAMS)
    m.fit(X_train, Z=a_train, S_init=S_init, C_init=C_init)
    return m


def transform_fairaa_adv(model, X, a):
    return model.transform(X, Z=a)


def fit_fairaa_mmd(X_train, a_train, K, lam, seed, S_init=None, C_init=None):
    m = FairAA_MMD(n_archetypes=K, fairness_const=float(lam),
                   random_state=seed, **AA_PARAMS)
    m.fit(X_train, Z=a_train, S_init=S_init, C_init=C_init)
    return m


def transform_fairaa_mmd(model, X, a):
    return model.transform(X, Z=a)


# FairPCA + AA — two-stage method. λ → tradeoff_param ∈ [0,1] is mapped via
# `tp = lam / (1 + lam)` so that "higher our_lam = more fairness" matches
# the rest of the registry (and λ=0 ≡ standard PCA + AA).
#
# `_PCA_TARGET_DIM` is computed once per dataset by main() (variance-retention
# rule) and read by `fit_fairpca_aa`. If unset, falls back to min(K, d-1).
_PCA_TARGET_DIM: int | None = None


def _pca_target_dim(X_fit, var_threshold: float) -> int:
    """Smallest d such that the first d PCs retain ≥ var_threshold of the
    total variance. Clipped to [2, n_features-1] (FairPCA constraint)."""
    from sklearn.decomposition import PCA
    pca = PCA().fit(X_fit)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d_var = int(np.searchsorted(cumvar, var_threshold) + 1)
    return max(2, min(d_var, X_fit.shape[1] - 1))


def fit_fairpca_aa(X_train, a_train, K, lam, seed, S_init=None, C_init=None):
    target_dim = (_PCA_TARGET_DIM if _PCA_TARGET_DIM is not None
                  else max(2, min(K, X_train.shape[1] - 1)))
    tp = float(lam) / (1.0 + float(lam))
    # FairPCA_AA doesn't accept S_init/C_init; ignore.
    m = FairPCA_AA(n_archetypes=K, target_dim=target_dim,
                   tradeoff_param=tp, random_state=seed,
                   init=AA_PARAMS["init"], n_init=AA_PARAMS["n_init"],
                   max_iter=AA_PARAMS["max_iter"], tol=AA_PARAMS["tol"],
                   method=AA_PARAMS["method"])
    m.fit(X_train, z=a_train.astype(int))
    return m


def transform_fairpca_aa(model, X, a):
    return model.transform(X)


# Reconstruction in original X-space, used to compute EV on the test set.

def _reconstruct_default(model, R):
    return R @ model.archetypes_


def _reconstruct_fairpca(model, R):
    # archetypes_ live in PCA-reduced space; lift back via inverse_transform.
    return model.fair_pca_.inverse_transform(R @ model.archetypes_)


# Registry of fair methods:
#   (name, fit_fn, transform_fn, reconstruct_fn, display label, color)
FAIR_METHODS = [
    ("FairAA",     fit_fairaa,       transform_fairaa,      _reconstruct_default, "FairAA",     "#1f77b4"),
    ("FairAA_3M",  fit_fairaa_3m,    transform_fairaa_3m,   _reconstruct_default, "FairAA-3M",  "#9467bd"),
    ("FairAA_Adv", fit_fairaa_adv,   transform_fairaa_adv,  _reconstruct_default, "FairAA-Adv", "#ff7f0e"),
    # ("FairAA_MMD", fit_fairaa_mmd,  transform_fairaa_mmd, _reconstruct_default, "FairAA-MMD", "#8c564b"),
    ("FairPCA_AA", fit_fairpca_aa,   transform_fairpca_aa,  _reconstruct_fairpca, "FairPCA-AA", "#17becf"),
]


# ── Downstream classifier ─────────────────────────────────────────────────────


def make_classifier(kind: str, seed: int):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=seed)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=400,
                             random_state=seed)
    raise ValueError(f"Unknown downstream model: {kind}")


REPR_NAN = {"ev": np.nan, "mmd_alpha": np.nan,
            "linsep_alpha": np.nan, "nonlinsep_alpha": np.nan,
            "dp_alpha": np.nan}


def evaluate_classifier(clf, R_test, y_test, a_test):
    """Six classifier-level metrics on the test set."""
    y_prob = clf.predict_proba(R_test)[:, 1]
    y_pred = clf.predict(R_test)
    return {
        "auc":      float(roc_auc_score(y_test, y_prob)),
        "acc":      float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred)),
        "dp_gap":   dp_gap(y_test, y_pred, a_test),
        "eo_gap":   eo_gap(y_test, y_pred, a_test),
        "eopp_gap": eopp_gap(y_test, y_pred, a_test),
    }


def evaluate_representation(R_test, a_test, X_test, X_hat_test, seed):
    """Five representation-level metrics on the test loadings.

    `X_hat_test=None` ⇒ Raw row (no projection); returns NaN for all five."""
    if X_hat_test is None or R_test is None:
        return dict(REPR_NAN)
    a_arr = np.asarray(a_test)
    R0, R1 = R_test[a_arr == 0], R_test[a_arr == 1]
    out = {
        "ev":              explained_variance(X_test, X_hat_test),
        "mmd_alpha":       mmd_rbf(R0, R1) if len(R0) and len(R1) else np.nan,
        "linsep_alpha":    linear_separability(R_test, a_arr, random_state=seed),
        "nonlinsep_alpha": nonlinear_separability(R_test, a_arr, random_state=seed),
        "dp_alpha":        demographic_parity(R_test, a_arr),
    }
    return out


def evaluate_all(clf, R_test, y_test, a_test, X_test, X_hat_test, seed):
    return {**evaluate_classifier(clf, R_test, y_test, a_test),
            **evaluate_representation(R_test, a_test, X_test, X_hat_test, seed)}


# ── Sweep ─────────────────────────────────────────────────────────────────────


def fit_subset(sp, fit_n_samples: int | None, seed: int):
    """Indices into the train split for the (Y, A)-stratified subset that
    AA / FairAA fit on. Transforms still apply to the full train+test."""
    if fit_n_samples is None or len(sp["X_train"]) <= fit_n_samples:
        return np.arange(len(sp["X_train"]))
    return dataset_registry.stratified_subsample_idx(
        sp["y_train"], sp["a_train"], n_max=fit_n_samples, seed=seed)


def _bootstrap_clf_eval(clf, R_test, y_test, a_test, B, rng):
    """Run B bootstrap samples of the test set; for each, compute the six
    classifier-level metrics. Returns a list of B metric dicts."""
    n = len(y_test)
    rows = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        rows.append(evaluate_classifier(clf, R_test[idx],
                                        y_test[idx], a_test[idx]))
    return rows


def run_sweep_bootstrap(X, y, a, lambdas, K, B, downstream, fit_n_samples,
                        seed, log):
    """Bootstrap variant of `run_sweep`.

    Single train/test split (`seed`); single AA / FairAA fit per (method, λ)
    using AA's internal `n_init` to pick the best initialisation; single
    classifier fit on top. The variance comes only from B bootstrap samples
    of the test set, so the error bars reflect generalisation noise without
    the model-fitting noise that the n_seeds-based sweep entangles.

    Representation-level metrics are computed once on the full test (same
    value broadcast to all B rows); classifier metrics vary per bootstrap."""
    rows = []
    archetypes_cache = {}
    rng = np.random.RandomState(seed * 1000 + 7)

    sp = split_and_scale(X, y, a, seed=seed)
    sub = fit_subset(sp, fit_n_samples, seed=seed)
    Xfit, afit = sp["X_train"][sub], sp["a_train"][sub]
    log.info("bootstrap mode: train=%d  fit=%d  test=%d  B=%d",
             len(sp["X_train"]), len(sub), len(sp["X_test"]), B)

    n_models = 1 + 1 + len(FAIR_METHODS) * len(lambdas)
    pbar = tqdm(total=n_models, desc="real-data sweep (bootstrap)")

    def _emit(method, lam, repr_metrics, clf, R_test):
        boot = _bootstrap_clf_eval(clf, R_test, sp["y_test"], sp["a_test"],
                                   B, rng)
        for b, clf_m in enumerate(boot):
            rows.append({"method": method, "lambda": lam, "seed": b,
                         **clf_m, **repr_metrics})

    # ─ Raw ─
    clf = make_classifier(downstream, seed).fit(sp["X_train"], sp["y_train"])
    _emit("Raw", np.nan, dict(REPR_NAN), clf, sp["X_test"])
    pbar.update(1)

    # ─ AA ─
    aa_model = fit_aa(Xfit, K, seed)
    R_train_aa = transform_aa(aa_model, sp["X_train"])
    R_test_aa  = transform_aa(aa_model, sp["X_test"])
    X_hat_aa   = _reconstruct_default(aa_model, R_test_aa)
    clf = make_classifier(downstream, seed).fit(R_train_aa, sp["y_train"])
    repr_aa = evaluate_representation(R_test_aa, sp["a_test"], sp["X_test"],
                                      X_hat_aa, seed)
    _emit("AA", np.nan, repr_aa, clf, R_test_aa)
    archetypes_cache[("AA", None, 0)] = {
        "archetypes_x": np.asarray(aa_model.archetypes_).copy(),
        "alpha_train": R_train_aa,
        "split": sp,
    }
    pbar.update(1)

    # ─ Fair methods ─
    for fam, fit_fn, tr_fn, recon_fn, _label, _color in FAIR_METHODS:
        prev = None
        for lam in lambdas:
            S_init = (prev.A_.copy()
                      if prev is not None and hasattr(prev, "A_") else None)
            C_init = (prev.B_.copy()
                      if prev is not None and hasattr(prev, "B_") else None)
            model = fit_fn(Xfit, afit, K, lam, seed,
                           S_init=S_init, C_init=C_init)
            R_train_f = tr_fn(model, sp["X_train"], sp["a_train"])
            R_test_f  = tr_fn(model, sp["X_test"],  sp["a_test"])
            X_hat_f   = recon_fn(model, R_test_f)
            clf = make_classifier(downstream, seed).fit(R_train_f, sp["y_train"])
            repr_m = evaluate_representation(R_test_f, sp["a_test"],
                                             sp["X_test"], X_hat_f, seed)
            _emit(fam, float(lam), repr_m, clf, R_test_f)
            archetypes_cache[(fam, float(lam), 0)] = {
                "archetypes_x": recon_fn(model, np.eye(K)),
                "alpha_train": R_train_f,
                "split": sp,
            }
            prev = model
            pbar.update(1)

    pbar.close()
    log.info("sweep finished (bootstrap): %d rows", len(rows))
    return pd.DataFrame(rows), archetypes_cache


def run_sweep(X, y, a, lambdas, K, n_seeds, downstream, fit_n_samples, log):
    rows = []
    archetypes_cache = {}   # (method, lam, seed) → (archetypes_train, alpha_train)

    # raw + AA + |λ| × (#fair methods)
    total = n_seeds * (1 + 1 + len(FAIR_METHODS) * len(lambdas))
    pbar = tqdm(total=total, desc="real-data sweep")
    for seed in range(n_seeds):
        sp = split_and_scale(X, y, a, seed=seed)
        sub = fit_subset(sp, fit_n_samples, seed=seed)
        Xfit, afit = sp["X_train"][sub], sp["a_train"][sub]
        log.info("seed=%d  train=%d  fit=%d  test=%d", seed,
                 len(sp["X_train"]), len(sub), len(sp["X_test"]))

        # ─ Raw X ─ (no AA fit; classifier on full train); repr metrics NaN
        clf = make_classifier(downstream, seed).fit(sp["X_train"], sp["y_train"])
        m = evaluate_all(clf, sp["X_test"], sp["y_test"], sp["a_test"],
                         X_test=None, X_hat_test=None, seed=seed)
        rows.append({"method": "Raw", "lambda": np.nan, "seed": seed, **m})
        pbar.update(1)

        # ─ Standard AA ─ fit on subset, transform on FULL set
        aa_model = fit_aa(Xfit, K, seed)
        R_train_aa = transform_aa(aa_model, sp["X_train"])
        R_test_aa  = transform_aa(aa_model, sp["X_test"])
        X_hat_aa   = _reconstruct_default(aa_model, R_test_aa)
        clf = make_classifier(downstream, seed).fit(R_train_aa, sp["y_train"])
        m = evaluate_all(clf, R_test_aa, sp["y_test"], sp["a_test"],
                         X_test=sp["X_test"],
                         X_hat_test=X_hat_aa,
                         seed=seed)
        rows.append({"method": "AA", "lambda": np.nan, "seed": seed, **m})
        archetypes_cache[("AA", None, seed)] = {
            "archetypes_x": np.asarray(aa_model.archetypes_).copy(),
            "alpha_train": R_train_aa,
            "split": sp,
        }
        pbar.update(1)

        # ─ Fair-method λ-ladders (warm-start chain) ─ fit on subset
        for fam, fit_fn, tr_fn, recon_fn, _label, _color in FAIR_METHODS:
            prev = None
            for lam in lambdas:
                # FairPCA-AA doesn't expose A_ / B_; warm-start only when both exist.
                S_init = (prev.A_.copy()
                          if prev is not None and hasattr(prev, "A_") else None)
                C_init = (prev.B_.copy()
                          if prev is not None and hasattr(prev, "B_") else None)
                model = fit_fn(Xfit, afit, K, lam, seed,
                               S_init=S_init, C_init=C_init)
                R_train_f = tr_fn(model, sp["X_train"], sp["a_train"])
                R_test_f  = tr_fn(model, sp["X_test"],  sp["a_test"])
                X_hat_f   = recon_fn(model, R_test_f)
                clf = make_classifier(downstream, seed).fit(R_train_f, sp["y_train"])
                m = evaluate_all(clf, R_test_f, sp["y_test"], sp["a_test"],
                                 X_test=sp["X_test"],
                                 X_hat_test=X_hat_f,
                                 seed=seed)
                rows.append({"method": fam, "lambda": float(lam),
                             "seed": seed, **m})
                archetypes_cache[(fam, float(lam), seed)] = {
                    # Archetype centroids in standardised X-space (lifted via
                    # recon_fn so FairPCA-AA's PCA-space archetypes are
                    # comparable to the X-space ones from AA / FairAA).
                    "archetypes_x": recon_fn(model, np.eye(K)),
                    "alpha_train": R_train_f,
                    "split": sp,
                }
                prev = model
                pbar.update(1)

    pbar.close()
    log.info("sweep finished: %d rows", len(rows))
    return pd.DataFrame(rows), archetypes_cache


# ── Aggregation + paired stats ────────────────────────────────────────────────


CLF_METRICS  = ["auc", "acc", "f1", "dp_gap", "eo_gap", "eopp_gap"]
REPR_METRICS = ["ev", "mmd_alpha", "linsep_alpha", "nonlinsep_alpha", "dp_alpha"]
METRIC_COLS  = CLF_METRICS + REPR_METRICS
FAIRNESS_TEST_COLS = ["dp_gap", "eo_gap", "eopp_gap",
                      "mmd_alpha", "linsep_alpha", "nonlinsep_alpha",
                      "dp_alpha"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "lambda"], dropna=False, sort=False)
    mean = g[METRIC_COLS].mean()
    std  = g[METRIC_COLS].std().fillna(0.0)
    return mean.join(std, rsuffix="_sd").reset_index()


def pick_representative_lambda(agg: pd.DataFrame, method: str = "FairAA") -> float:
    """λ at the Pareto knee in (DP-gap, 1−AUC) space.

    Both axes are min-max normalised over `method`'s λ sweep, then we
    keep the non-dominated points (lower-better on both axes) and pick
    the one whose Euclidean distance to the utopia corner (DP=0, AUC=1)
    is smallest — the standard "knee" in two-objective Pareto fronts.

    λ=0 is excluded: at λ=0 the regulariser is off and FairAA degenerates
    to AA, which would always sit at the high-AUC/high-DP corner of the
    front and bias the knee selection toward "no regularisation"."""
    fair = agg[(agg["method"] == method) & (agg["lambda"] > 0)].copy()
    if fair.empty:
        return float("nan")
    if len(fair) == 1:
        return float(fair["lambda"].iloc[0])

    pts = np.column_stack([fair["dp_gap"].to_numpy(),
                           1.0 - fair["auc"].to_numpy()])

    # Pareto front: keep points not dominated on both axes
    keep = []
    for i, (xi, yi) in enumerate(pts):
        dominated = any(
            (xj <= xi and yj <= yi) and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(pts) if j != i
        )
        if not dominated:
            keep.append(i)
    front = fair.iloc[keep]
    front_pts = pts[keep]

    # Normalise each axis to [0, 1] using the front's own range, then
    # pick the closest point to the origin (= utopia after normalisation)
    span = front_pts.max(axis=0) - front_pts.min(axis=0)
    span[span == 0] = 1.0
    norm = (front_pts - front_pts.min(axis=0)) / span
    dist = np.linalg.norm(norm, axis=1)
    return float(front["lambda"].iloc[int(np.argmin(dist))])


def paired_tests(df: pd.DataFrame, lam_star: float,
                 method: str = "FairAA") -> dict:
    """Wilcoxon between AA and `method`(λ*) on each fairness metric."""
    out = {}
    aa = df[df["method"] == "AA"].sort_values("seed")
    fa = df[(df["method"] == method) &
            (df["lambda"] == lam_star)].sort_values("seed")
    for col in FAIRNESS_TEST_COLS:
        a_vals = aa[col].to_numpy()
        f_vals = fa[col].to_numpy()
        if len(a_vals) < 2 or len(f_vals) < 2 or len(a_vals) != len(f_vals):
            out[f"{col}_p"] = np.nan
            continue
        try:
            res = stats.wilcoxon(a_vals, f_vals, zero_method="zsplit")
            out[f"{col}_p"] = float(res.pvalue)
        except ValueError:
            out[f"{col}_p"] = np.nan
    out["lambda_star"] = lam_star
    return out


# ── Step 6: archetype analysis ────────────────────────────────────────────────


def archetype_profiles(method, alpha_train, archetypes, a_train, scaler,
                       feature_names, top_k=5):
    """Return a DataFrame with: archetype k, top features (destandardized),
    demographic composition (proportion of A=0 vs A=1 among points with
    argmax_k α = k)."""
    archetypes = np.asarray(archetypes)
    raw_centroids = scaler.inverse_transform(archetypes)
    dom = alpha_train.argmax(axis=1)

    rows = []
    for k in range(archetypes.shape[0]):
        order = np.argsort(-np.abs(raw_centroids[k]))[:top_k]
        top = "; ".join(f"{feature_names[i]}={raw_centroids[k, i]:.3f}"
                        for i in order)
        mask = dom == k
        n_k = int(mask.sum())
        if n_k > 0:
            p_a0 = float((a_train[mask] == 0).mean())
            p_a1 = float((a_train[mask] == 1).mean())
        else:
            p_a0 = p_a1 = float("nan")
        rows.append({"method": method, "archetype": k,
                     "top_features": top, "n_dominant": n_k,
                     "frac_a0": p_a0, "frac_a1": p_a1})
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _pareto_front(points):
    """Lower x is better, higher y is better. Returns sorted non-dominated set."""
    pts = np.asarray(points)
    keep = []
    for i, (x_i, y_i) in enumerate(pts):
        dominated = False
        for j in range(len(pts)):
            if j == i:
                continue
            x_j, y_j = pts[j]
            if (x_j <= x_i and y_j >= y_i) and (x_j < x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return pts[sorted(keep, key=lambda k: pts[k, 0])]


# Color palette used across all figures (built from FAIR_METHODS)
PAL = {"Raw": "#2ca02c", "AA": "#d62728"}
for _fam, _, _, _, _label, _color in FAIR_METHODS:
    PAL[_fam] = _color
    PAL[_label] = _color


def _pareto_front_indices(x, y, *, x_lower_better=True, y_higher_better=True):
    """Indices of non-dominated points, sorted by x ascending.
    `x_lower_better=True`: smaller x is preferred. Same convention via flag for y."""
    x = np.asarray(x); y = np.asarray(y)
    sign_x = 1 if x_lower_better else -1
    sign_y = -1 if y_higher_better else 1
    xs, ys = sign_x * x, sign_y * y
    keep = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if j == i:
                continue
            if (xs[j] <= xs[i] and ys[j] <= ys[i]) and (xs[j] < xs[i] or ys[j] < ys[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda k: x[k])


def _plot_pareto(agg, dataset, *, x_key, y_key, x_label, y_label,
                 y_higher_better, suffix, title):
    """Pareto plot showing only the front (no individual sweep points).
    x is always lower-better (fairness); error bars are drawn only on the
    fairness axis. y direction is controlled by `y_higher_better`."""
    aa  = agg[agg["method"] == "AA"]
    raw = agg[agg["method"] == "Raw"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for fam, _f, _t, _r, label, color in FAIR_METHODS:
        # exclude λ=0 from the front (= AA in disguise)
        sub = agg[(agg["method"] == fam) &
                  (agg["lambda"] > 0)].copy()
        if sub.empty:
            continue
        idx = _pareto_front_indices(sub[x_key].to_numpy(),
                                    sub[y_key].to_numpy(),
                                    y_higher_better=y_higher_better)
        front = sub.iloc[idx]
        ax.errorbar(front[x_key], front[y_key],
                    xerr=front[f"{x_key}_sd"],
                    fmt="o-", color=color, label=f"{label} Pareto",
                    capsize=2, linewidth=1.6, markersize=5)
    if not aa.empty:
        ax.errorbar(aa[x_key], aa[y_key],
                    xerr=aa[f"{x_key}_sd"],
                    fmt="s", color=PAL["AA"], markersize=10,
                    capsize=3, label="AA")
    if not raw.empty and not raw[x_key].isna().all():
        ax.errorbar(raw[x_key], raw[y_key],
                    xerr=raw[f"{x_key}_sd"],
                    fmt="^", color=PAL["Raw"], markersize=10,
                    capsize=3, label="Raw X")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir(dataset) / f"pareto_{suffix}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_pareto_clf(agg, dataset):
    _plot_pareto(agg, dataset,
                 x_key="dp_gap", y_key="auc",
                 x_label="Demographic-parity gap (predictions)",
                 y_label="AUC",
                 y_higher_better=True, suffix="clf",
                 title=f"Classifier utility–fairness trade-off: {dataset}")


def plot_pareto_repr(agg, dataset):
    _plot_pareto(agg, dataset,
                 x_key="mmd_alpha", y_key="ev",
                 x_label="MMD between α | A=0 and α | A=1",
                 y_label="Explained variance",
                 y_higher_better=True, suffix="repr",
                 title=f"Representation reconstruction–fairness trade-off: {dataset}")


def _plot_lambda_sweep(agg, dataset, metrics, *, suffix, title):
    """metrics: list of (key, ylabel, lower_better)."""
    aa  = agg[agg["method"] == "AA"]
    raw = agg[agg["method"] == "Raw"]
    fair_methods = [(fam, label, color) for fam, _f, _t, _r, label, color
                    in FAIR_METHODS if not agg[agg["method"] == fam].empty]
    if not fair_methods:
        return

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.5 * len(metrics), 3.2))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (key, ylabel, _lower_better) in zip(axes, metrics):
        for fam, label, color in fair_methods:
            sub = agg[agg["method"] == fam].sort_values("lambda")
            lam_plot = sub["lambda"].copy()
            lam_min_pos = (lam_plot[lam_plot > 0].min()
                           if (lam_plot > 0).any() else 1e-3)
            lam_plot = lam_plot.replace(0.0, lam_min_pos / 2)
            ax.errorbar(lam_plot, sub[key], yerr=sub[f"{key}_sd"],
                        fmt="o-", color=color, capsize=2,
                        linewidth=1.4, label=label)
        if not aa.empty and not pd.isna(aa[key].iloc[0]):
            ax.axhline(float(aa[key].iloc[0]), color=PAL["AA"],
                       linestyle="--", linewidth=1.2, label="AA")
        if not raw.empty and not pd.isna(raw[key].iloc[0]):
            ax.axhline(float(raw[key].iloc[0]), color=PAL["Raw"],
                       linestyle=":", linewidth=1.2, label="Raw X")
        ax.set_xscale("log")
        ax.set_xlabel("λ"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir(dataset) / f"lambda_sweep_{suffix}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_lambda_sweep_clf(agg, dataset):
    _plot_lambda_sweep(agg, dataset, [
        ("auc",      "AUC",                    False),
        ("dp_gap",   "Demographic-parity gap", True),
        ("eo_gap",   "Equalized-odds gap",     True),
        ("eopp_gap", "Equal-opportunity gap",  True),
    ], suffix="clf", title=f"Classifier metrics vs λ: {dataset}")


def plot_lambda_sweep_repr(agg, dataset):
    _plot_lambda_sweep(agg, dataset, [
        ("ev",              "Explained variance",            False),
        ("mmd_alpha",       "MMD(α | A=0, α | A=1)",         True),
        ("linsep_alpha",    "Linear separability of A",      True),
        ("nonlinsep_alpha", "Non-linear separability of A",  True),
        ("dp_alpha",        "DP of α (loadings)",            True),
    ], suffix="repr", title=f"Representation metrics vs λ: {dataset}")


def _plot_method_bars(agg, lam_stars, dataset, metrics, *, suffix, title):
    """metrics: list of (key, ylabel)."""
    rows = [("Raw", agg[agg["method"] == "Raw"]),
            ("AA",  agg[agg["method"] == "AA"])]
    for fam, _f, _t, _r, label, _color in FAIR_METHODS:
        lam = lam_stars.get(fam)
        if lam is None or np.isnan(lam):
            continue
        rows.append((f"{label}(λ={lam:g})",
                     agg[(agg["method"] == fam) & (agg["lambda"] == lam)]))
    rows = [(name, sub.iloc[0]) for name, sub in rows if not sub.empty]

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.7 * len(metrics), 3.4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (key, ylabel) in zip(axes, metrics):
        names, vals, errs, colors = [], [], [], []
        for n, r in rows:
            if pd.isna(r[key]):
                continue
            names.append(n)
            vals.append(float(r[key]))
            errs.append(float(r[f"{key}_sd"]))
            colors.append(PAL.get(n.split("(")[0], "#888"))
        ax.bar(names, vals, yerr=errs, color=colors, capsize=4, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir(dataset) / f"method_bars_{suffix}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_method_bars_clf(agg, lam_stars, dataset):
    _plot_method_bars(agg, lam_stars, dataset, [
        ("auc", "AUC"), ("dp_gap", "DP gap"),
        ("eo_gap", "EO gap"), ("eopp_gap", "EOpp gap"),
    ], suffix="clf", title=f"Classifier metrics @ λ*: {dataset}")


def plot_method_bars_repr(agg, lam_stars, dataset):
    _plot_method_bars(agg, lam_stars, dataset, [
        ("ev", "EV"), ("mmd_alpha", "MMD(α | A)"),
        ("linsep_alpha", "LinSep of A"), ("nonlinsep_alpha", "NonLinSep of A"),
        ("dp_alpha", "DP of α"),
    ], suffix="repr", title=f"Representation metrics @ λ*: {dataset}")


def plot_demographic_composition(profiles: dict, dataset: str):
    """Stacked bars per archetype — A=0 vs A=1 share of points whose dominant
    weight is k. One panel per (method, profile_df) pair in `profiles`."""
    items = [(t, df_) for t, df_ in profiles.items()
             if df_ is not None and not df_.empty]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items),
                             figsize=(5.5 * len(items), 3.6), sharey=True,
                             squeeze=False)
    for ax, (title, df_) in zip(axes[0], items):
        df_ = df_.sort_values("archetype")
        x = np.arange(len(df_))
        ax.bar(x, df_["frac_a0"], color="#9ecae1", label="A=0")
        ax.bar(x, df_["frac_a1"], bottom=df_["frac_a0"],
               color="#fdae6b", label="A=1")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}\n(n={int(n)})"
                            for k, n in zip(df_["archetype"], df_["n_dominant"])],
                           fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Demographic composition per archetype: {dataset}")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir(dataset) / f"archetype_composition.{ext}",
                    bbox_inches="tight")
    plt.close(fig)





# ── Logging + config ──────────────────────────────────────────────────────────


def setup_logger(dataset: str) -> logging.Logger:
    log_path = data_dir(dataset) / "run.log"
    logger = logging.getLogger(f"real_data.{dataset}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info("logging to %s", log_path)
    return logger


def write_config(args, dataset: str):
    import sklearn, scipy, numpy
    cfg = {
        "dataset": dataset,
        "K": args.K,
        "lambdas": args.lambdas,
        "n_seeds": args.n_seeds,
        "downstream_model": args.downstream_model,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed_offset": args.seed,
        "versions": {
            "numpy":   numpy.__version__,
            "scipy":   scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas":  pd.__version__,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = data_dir(dataset) / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return cfg, path


# ── Data-prep summary ─────────────────────────────────────────────────────────


def report_distributions(y, a, log):
    log.info("class prevalence  P(Y=1) = %.3f", float(y.mean()))
    log.info("group split       P(A=1) = %.3f", float(a.mean()))
    for yv in (0, 1):
        for av in (0, 1):
            n = int(((y == yv) & (a == av)).sum())
            log.info("  P(Y=%d, A=%d) = %.3f  (n=%d)",
                     yv, av, n / len(y), n)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_lambdas(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german_credit",
                   choices=dataset_registry.available_datasets())
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--lambdas", type=parse_lambdas,
                   default=str(sorted([0.0] + list(np.logspace(-2, 2, 10)))))
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--downstream_model", choices=["logreg", "mlp"], default="logreg")
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--val_size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit_n_samples", type=int, default=1000,
                   help="Stratified subset size used to FIT AA / FairAA. "
                        "Transforms still apply to the full train+test "
                        "(0 or negative = no cap, fit on the full train).")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="Stratified cap on the total dataset size at load "
                        "time (transform + classifier eval). 0 or negative "
                        "= no cap.")
    p.add_argument("--K_max", type=int, default=None,
                   help="If set, ignore --K and auto-pick K ∈ [1, K_max] at "
                        "the elbow of the AA explained-variance scree curve "
                        "(unsupervised, AA only).")
    p.add_argument("--K_seeds", type=int, default=3,
                   help="How many seeds per K when running the scree sweep.")
    p.add_argument("--pca_var", type=float, default=0.95,
                   help="FairPCA-AA target_dim is the smallest d whose "
                        "cumulative PCA variance ratio ≥ pca_var (default "
                        "0.95). Computed once on the seed-0 fit subset.")
    p.add_argument("--bootstrap_clf", type=int, default=0,
                   help="If > 0, switch to bootstrap mode: 1 train/test "
                        "split, 1 AA fit per (method, λ) (best of n_init), "
                        "1 classifier fit; classifier metrics from B "
                        "bootstrap samples of the test set. Reports "
                        "generalisation noise without conflating model "
                        "non-determinism. Overrides --n_seeds.")
    args = p.parse_args(argv)
    if args.fit_n_samples is not None and args.fit_n_samples <= 0:
        args.fit_n_samples = None
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    ds = args.dataset
    log = setup_logger(ds)
    cfg, cfg_path = write_config(args, ds)
    log.info("config written to %s", cfg_path)

    log.info("loading dataset %s (max_samples=%s)", ds, args.max_samples)
    X, y, a, feature_names = dataset_registry.load(
        ds, n_max=args.max_samples, seed=args.seed)
    log.info("X.shape=%s, n_features=%d", X.shape, len(feature_names))
    report_distributions(y, a, log)

    # Seed-0 fit subset is shared by auto-K, PCA target_dim, and (later) the
    # archetype-analysis re-fit when loading from cache.
    sp0 = split_and_scale(X, y, a, seed=0)
    sub0 = fit_subset(sp0, args.fit_n_samples, seed=0)
    Xfit0 = sp0["X_train"][sub0]

    # Auto-K via scree elbow on AA train EV (unsupervised; uses Xfit0)
    if args.K_max is not None and args.K_max > 0:
        log.info("auto-K: scree sweep K ∈ [1, %d] on AA (n_seeds=%d)",
                 args.K_max, args.K_seeds)
        best_K, scree_df = find_best_K(Xfit0, range(1, args.K_max + 1),
                                       n_seeds=args.K_seeds, log=log)
        scree_df.to_csv(data_dir(ds) / "scree.csv", index=False)
        plot_scree(scree_df, best_K, ds)
        log.info("scree elbow → K* = %d  (overrides --K=%d)", best_K, args.K)
        args.K = best_K

    # FairPCA-AA target_dim: variance retention rule (computed once)
    global _PCA_TARGET_DIM
    _PCA_TARGET_DIM = _pca_target_dim(Xfit0, var_threshold=args.pca_var)
    log.info("FairPCA-AA target_dim=%d  (PCA cumvar ≥ %.2f, n_features=%d)",
             _PCA_TARGET_DIM, args.pca_var, X.shape[1])

    raw_csv = data_dir(ds) / "raw_results.csv"
    if raw_csv.exists():
        log.info("loading cached results from %s (delete to re-run)", raw_csv)
        df = pd.read_csv(raw_csv)
        # need to re-fit the AA + FairAA for archetype analysis (alphas not cached)
        log.info("re-fitting representations on seed 0 for archetype analysis")
        afit0 = sp0["a_train"][sub0]
        aa0 = fit_aa(Xfit0, args.K, seed=0)
        archetypes_cache = {
            ("AA", None, 0): {
                "archetypes_x": np.asarray(aa0.archetypes_).copy(),
                "alpha_train": transform_aa(aa0, sp0["X_train"]),
                "split": sp0,
            }
        }
        # Each fair method's λ-ladder at seed 0 (warm-start chain)
        for fam, fit_fn, tr_fn, recon_fn, _label, _color in FAIR_METHODS:
            prev = None
            for lam in args.lambdas:
                S_init = (prev.A_.copy()
                          if prev is not None and hasattr(prev, "A_") else None)
                C_init = (prev.B_.copy()
                          if prev is not None and hasattr(prev, "B_") else None)
                model = fit_fn(Xfit0, afit0, args.K, lam,
                               seed=0, S_init=S_init, C_init=C_init)
                archetypes_cache[(fam, float(lam), 0)] = {
                    "archetypes_x": recon_fn(model, np.eye(args.K)),
                    "alpha_train": tr_fn(model, sp0["X_train"], sp0["a_train"]),
                    "split": sp0,
                }
                prev = model
    else:
        if args.bootstrap_clf > 0:
            df, archetypes_cache = run_sweep_bootstrap(
                X, y, a, lambdas=args.lambdas, K=args.K,
                B=args.bootstrap_clf,
                downstream=args.downstream_model,
                fit_n_samples=args.fit_n_samples,
                seed=args.seed, log=log)
        else:
            df, archetypes_cache = run_sweep(
                X, y, a, lambdas=args.lambdas, K=args.K,
                n_seeds=args.n_seeds,
                downstream=args.downstream_model,
                fit_n_samples=args.fit_n_samples, log=log)
        df.to_csv(raw_csv, index=False)
        log.info("cached raw results to %s", raw_csv)

    # results_table.csv: same as raw (one row per (method, lambda, seed))
    df.to_csv(data_dir(ds) / "results_table.csv", index=False)

    # Aggregate + Pareto figure
    agg = aggregate(df)
    lam_stars = {fam: pick_representative_lambda(agg, method=fam)
                 for fam, _f, _t, _r, _l, _c in FAIR_METHODS}
    log.info("representative λ* per method: %s",
             {k: f"{v:g}" for k, v in lam_stars.items()})

    summary = agg.copy()
    for fam, lam in lam_stars.items():
        if np.isnan(lam):
            continue
        pvals = paired_tests(df, lam, method=fam)
        for k, v in pvals.items():
            if k == "lambda_star":
                summary[f"{fam}_lambda_star"] = v
            else:
                summary[f"{fam}_{k}"] = v
    summary.to_csv(data_dir(ds) / "results_summary.csv", index=False)

    plot_pareto_clf(agg, ds)
    plot_pareto_repr(agg, ds)
    plot_lambda_sweep_clf(agg, ds)
    plot_lambda_sweep_repr(agg, ds)
    plot_method_bars_clf(agg, lam_stars, ds)
    plot_method_bars_repr(agg, lam_stars, ds)

    from src.visualization import plot_metric_fixed_lambda

    fig, axs = plt.subplots(1, 5, figsize=(5 * 5, 4))

    for i, ax in enumerate(axs):
        metric = FAIRNESS_TEST_COLS[i]
        print(f"Plotting {metric} @ λ*...")



    log.info("scalar-metric figures saved "
             "(pareto/lambda_sweep/method_bars × {clf, repr})")

    # Step 6 — archetype analysis at seed 0 (interpretability)
    seed0_aa = archetypes_cache.get(("AA", None, 0))
    if seed0_aa is None:
        log.warning("seed-0 AA cache missing; skipping archetype analysis")
    else:
        sp = seed0_aa["split"]
        prof_aa = archetype_profiles(
            "AA", seed0_aa["alpha_train"], seed0_aa["archetypes_x"],
            sp["a_train"], sp["scaler"], feature_names)
        prof_aa.to_csv(data_dir(ds) / "archetype_profiles_aa.csv", index=False)
        composition = {"AA": prof_aa}

        for fam, _f, _t, _r, label, _color in FAIR_METHODS:
            lam = lam_stars[fam]
            seed0 = (archetypes_cache.get((fam, float(lam), 0))
                     if not np.isnan(lam) else None)
            if seed0 is None:
                continue
            prof = archetype_profiles(
                fam, seed0["alpha_train"], seed0["archetypes_x"],
                sp["a_train"], sp["scaler"], feature_names)
            suffix = fam.lower().replace("-", "_")
            prof.to_csv(data_dir(ds) / f"archetype_profiles_{suffix}.csv", index=False)
            composition[f"{label}(λ={lam:g})"] = prof
        log.info("archetype profiles written to %s", data_dir(ds))

        plot_demographic_composition(composition, ds)
        log.info("archetype figure saved (composition)")

    log.info("done")


if __name__ == "__main__":
    main()
