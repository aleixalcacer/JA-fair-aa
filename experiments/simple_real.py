"""Simplified real-data experiment — flat-script style, no CLI flags.

Mirrors `experiments/simple.py`: hard-coded constants up top, top-level
loops, no argparse. Loops over the three real tabular datasets registered
in `experiments/datasets.py` and produces four figures per dataset:

    figures/<ds>/pareto_clf.pdf    — DP gap vs AUC
    figures/<ds>/pareto_repr.pdf   — MMD(α | A) vs explained variance
    figures/<ds>/bars_clf.pdf      — AUC + DP gap @ λ*
    figures/<ds>/bars_repr.pdf     — EV + MMD @ λ*

All plot rendering is delegated to `src.visualization`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.methods import (AA, FairAAWarm, FairAA_3Moment,
                         FairAA_Adversarial, FairPCA_AA)
from src.methods.fair_aa_3moment import moment_transform
from src.metrics import (dp_gap, eo_gap, eopp_gap,
                         explained_variance, mmd_rbf,
                         linear_separability, nonlinear_separability,
                         demographic_parity)
from src.visualization import plot_metric, plot_metric_fixed_lambda

import datasets as dataset_registry


# ── Hard-coded experiment parameters ─────────────────────────────────────────

DATASETS      = ["adult", "diabetes_hospital", "bank_marketing"]
N_SEEDS       = 25
MAX_SAMPLES   = 2000
FIT_N_SAMPLES = 500
K_MAX         = 10
K_SEEDS       = 3
PCA_VAR       = 0.8
DOWNSTREAM    = "logreg"
TEST_SIZE     = 0.25
# LAMBDAS       = [0.0, 0.1, 1.0, 10.0]
LAMBDAS       = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

AA_PARAMS = dict(init="furthest_sum", n_init=5, max_iter=1000,
                 tol=1e-12, method="pseudo_pgd")

DATA_ROOT = ROOT / "data"
FIG_ROOT  = ROOT / "figures"


# ── Method registry ──────────────────────────────────────────────────────────

# Mutated per-dataset before the FairPCA-AA fit runs.
_PCA_TARGET_DIM: int | None = None


def fit_aa(X, a, K, lam, seed, S_init=None, C_init=None):
    m = AA(n_archetypes=K, random_state=seed, **AA_PARAMS); m.fit(X); return m

def tr_aa(m, X, a):
    return m.transform(X)

def fit_fairaa(X, a, K, lam, seed, S_init=None, C_init=None):
    Z = (a - a.mean()).reshape(-1, 1).astype(float)
    m = FairAAWarm(n_archetypes=K, fairness_const=float(lam),
                   random_state=seed, **AA_PARAMS)
    m.fit(X, Z=Z, S_init=S_init, C_init=C_init); return m

def tr_fairaa(m, X, a):
    Z = (a - a.mean()).reshape(-1, 1).astype(float)
    return m.transform(X, Z)

def fit_fairaa_3m(X, a, K, lam, seed, S_init=None, C_init=None):
    m = FairAA_3Moment(n_archetypes=K, fairness_const=float(lam),
                       alpha_2=0.1, alpha_3=0.1, balance_orders=True,
                       random_state=seed, **AA_PARAMS)
    m.fit(X, Z=a.astype(float), S_init=S_init, C_init=C_init); return m

def tr_fairaa_3m(m, X, a):
    # Inductive transform: re-derive z_c / zzT for the new sample so the
    # fit-time Z stored on the model doesn't have to match the test size.
    z = np.asarray(a, dtype=np.float64); z_c = z - z.mean()
    zzT = None if m.rank1_zzT else z_c[:, None] * z_c[None, :]
    a2, a3 = m._resolve_alphas()
    method_params = {} if m.method_params is None else m.method_params
    return moment_transform(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64)),
        z_c, zzT, m.archetypes_,
        fairness_const=m.fairness_const,
        alpha_2=a2, alpha_3=a3, rank1_zzT=m.rank1_zzT,
        max_iter=m.max_iter, tol=m.tol, **method_params,
    )

def fit_fairaa_adv(X, a, K, lam, seed, S_init=None, C_init=None):
    m = FairAA_Adversarial(n_archetypes=K, fairness_const=float(lam),
                           n_adv_steps=1, lr_adv=1e-2,
                           random_state=seed, **AA_PARAMS)
    m.fit(X, Z=a, S_init=S_init, C_init=C_init); return m

def tr_fairaa_adv(m, X, a):
    return m.transform(X, Z=a)

def fit_fairpca_aa(X, a, K, lam, seed, S_init=None, C_init=None):
    target_dim = (_PCA_TARGET_DIM if _PCA_TARGET_DIM is not None
                  else max(2, min(K, X.shape[1] - 1)))
    # λ → tradeoff_param ∈ [0, 1] so higher λ = more fairness (and λ=0
    # ≡ standard PCA + AA), matching the rest of the registry.
    tp = float(lam) / (1.0 + float(lam))
    m = FairPCA_AA(n_archetypes=K, target_dim=target_dim, tradeoff_param=tp,
                   random_state=seed,
                   init=AA_PARAMS["init"], n_init=AA_PARAMS["n_init"],
                   max_iter=AA_PARAMS["max_iter"], tol=AA_PARAMS["tol"],
                   method=AA_PARAMS["method"])
    m.fit(X, z=a.astype(int)); return m

def tr_fairpca_aa(m, X, a):
    return m.transform(X)


def rec_default(m, R):
    return R @ m.archetypes_

def rec_fairpca(m, R):
    return m.fair_pca_.inverse_transform(R @ m.archetypes_)


# (name, fit, transform, reconstruct)
FAIR_METHODS = [
    ("FairAA",         fit_fairaa,     tr_fairaa,     rec_default),
    ("FairAA_3Moment", fit_fairaa_3m,  tr_fairaa_3m,  rec_default),
    ("FairAA_Adv",     fit_fairaa_adv, tr_fairaa_adv, rec_default),
    ("FairPCA_AA",     fit_fairpca_aa, tr_fairpca_aa, rec_fairpca),
]
ALL_METHODS = ["Raw", "AA"] + [m[0] for m in FAIR_METHODS]


# ── Helpers ──────────────────────────────────────────────────────────────────

def split_and_scale(X, y, a, seed):
    """Joint-stratify by (y, a); fit StandardScaler on train only."""
    strat = np.char.add(y.astype(str), "_")
    strat = np.char.add(strat, a.astype(str))
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE,
                              stratify=strat, random_state=seed)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    return Xs[tr], y[tr], a[tr], Xs[te], y[te], a[te]


def fit_subset(y_tr, a_tr, seed):
    if FIT_N_SAMPLES is None or len(y_tr) <= FIT_N_SAMPLES:
        return np.arange(len(y_tr))
    return dataset_registry.stratified_subsample_idx(
        y_tr, a_tr, n_max=FIT_N_SAMPLES, seed=seed)


def make_clf(seed):
    if DOWNSTREAM == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=seed)
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=400,
                         random_state=seed)


REPR_NAN = {"ev": np.nan, "mmd_alpha": np.nan, "linsep_alpha": np.nan,
            "nonlinsep_alpha": np.nan, "dp_alpha": np.nan}


def evaluate(clf, R_te, y_te, a_te, X_te=None, X_hat_te=None, seed=0):
    """Six classifier metrics + five representation metrics (NaN for Raw)."""
    y_prob = clf.predict_proba(R_te)[:, 1]
    y_pred = clf.predict(R_te)
    out = {
        "auc": float(roc_auc_score(y_te, y_prob)),
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1":  float(f1_score(y_te, y_pred)),
        "dp_gap":   dp_gap(y_te, y_pred, a_te),
        "eo_gap":   eo_gap(y_te, y_pred, a_te),
        "eopp_gap": eopp_gap(y_te, y_pred, a_te),
    }
    if X_hat_te is None:
        return {**out, **REPR_NAN}
    a_arr = np.asarray(a_te)
    R0, R1 = R_te[a_arr == 0], R_te[a_arr == 1]
    out.update({
        "ev":              explained_variance(X_te, X_hat_te),
        "mmd_alpha":       mmd_rbf(R0, R1) if len(R0) and len(R1) else np.nan,
        "linsep_alpha":    linear_separability(R_te, a_arr, random_state=seed),
        "nonlinsep_alpha": nonlinear_separability(R_te, a_arr, random_state=seed),
        "dp_alpha":        demographic_parity(R_te, a_arr),
    })
    return out


def pick_K(X_fit):
    """Scree elbow on AA train EV over K ∈ [1, K_MAX] (kneedle heuristic)."""
    Ks = np.arange(1, K_MAX + 1)
    evs = []
    for K in Ks:
        es = []
        for s in range(K_SEEDS):
            m = AA(n_archetypes=int(K), random_state=s, **AA_PARAMS).fit(X_fit)
            R = m.transform(X_fit)
            es.append(explained_variance(X_fit, R @ m.archetypes_))
        evs.append(np.mean(es))
    xs = Ks.astype(float); ys = np.asarray(evs)
    xn = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-12)
    yn = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    return int(Ks[int(np.argmax(yn - xn))])


def pick_pca_target_dim(X_fit, n_features):
    cum = np.cumsum(PCA().fit(X_fit).explained_variance_ratio_)
    d = int(np.searchsorted(cum, PCA_VAR) + 1)
    return max(2, min(d, n_features - 1))


def pick_lambda_star(agg, method):
    """Knee in (dp_gap, 1−auc) on `method`'s λ>0 Pareto front; closest to
    utopia after per-axis min-max normalisation."""
    sub = agg[(agg["method"] == method) & (agg["lambda"] > 0)]
    if sub.empty:
        return float("nan")
    if len(sub) == 1:
        return float(sub["lambda"].iloc[0])
    pts = np.column_stack([sub["dp_gap"].to_numpy(),
                           1.0 - sub["auc"].to_numpy()])
    keep = []
    for i, (xi, yi) in enumerate(pts):
        dominated = any(
            (pts[j, 0] <= xi and pts[j, 1] <= yi) and
            (pts[j, 0] < xi or pts[j, 1] < yi)
            for j in range(len(pts)) if j != i
        )
        if not dominated:
            keep.append(i)
    front = sub.iloc[keep]; fp = pts[keep]
    span = fp.max(axis=0) - fp.min(axis=0); span[span == 0] = 1.0
    norm = (fp - fp.min(axis=0)) / span
    return float(front["lambda"].iloc[int(np.argmin(np.linalg.norm(norm, axis=1)))])


# ── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(X, y, a, K):
    rows = []
    total = N_SEEDS * (2 + len(FAIR_METHODS) * len(LAMBDAS))
    pbar = tqdm(total=total, desc="sweep", leave=False)
    t_sweep = time.perf_counter()
    for seed in range(N_SEEDS):
        t_seed = time.perf_counter()
        X_tr, y_tr, a_tr, X_te, y_te, a_te = split_and_scale(X, y, a, seed)
        sub = fit_subset(y_tr, a_tr, seed)
        X_fit, a_fit = X_tr[sub], a_tr[sub]
        pbar.set_postfix_str(f"seed={seed} step=Raw")

        # Raw — classifier on standardised X, no AA
        clf = make_clf(seed).fit(X_tr, y_tr)
        rows.append({"method": "Raw", "lambda": np.nan, "seed": seed,
                     **evaluate(clf, X_te, y_te, a_te, seed=seed)})
        pbar.update(1)

        # AA baseline
        pbar.set_postfix_str(f"seed={seed} step=AA")
        aa = fit_aa(X_fit, a_fit, K, 0.0, seed)
        R_tr = aa.transform(X_tr); R_te = aa.transform(X_te)
        clf = make_clf(seed).fit(R_tr, y_tr)
        rows.append({"method": "AA", "lambda": np.nan, "seed": seed,
                     **evaluate(clf, R_te, y_te, a_te, X_te,
                                rec_default(aa, R_te), seed=seed)})
        pbar.update(1)

        # Fair methods, warm-started along the λ ladder
        for name, fit_fn, tr_fn, rec_fn in FAIR_METHODS:
            t_meth = time.perf_counter()
            prev = None
            for lam in LAMBDAS:
                pbar.set_postfix_str(f"seed={seed} {name} λ={lam:g}")
                S_init = (prev.A_.copy()
                          if prev is not None and hasattr(prev, "A_") else None)
                C_init = (prev.B_.copy()
                          if prev is not None and hasattr(prev, "B_") else None)
                model = fit_fn(X_fit, a_fit, K, lam, seed,
                               S_init=S_init, C_init=C_init)
                R_tr = tr_fn(model, X_tr, a_tr)
                R_te = tr_fn(model, X_te, a_te)
                clf = make_clf(seed).fit(R_tr, y_tr)
                rows.append({"method": name, "lambda": float(lam),
                             "seed": seed,
                             **evaluate(clf, R_te, y_te, a_te, X_te,
                                        rec_fn(model, R_te), seed=seed)})
                prev = model
                pbar.update(1)
            tqdm.write(f"    seed={seed:2d}  {name:<15s} "
                       f"({len(LAMBDAS)} λ) → {time.perf_counter()-t_meth:5.1f}s")
        tqdm.write(f"  seed={seed:2d} done in "
                   f"{time.perf_counter()-t_seed:5.1f}s "
                   f"(elapsed {time.perf_counter()-t_sweep:.1f}s)")
    pbar.close()
    print(f"  sweep total: {time.perf_counter()-t_sweep:.1f}s, "
          f"{len(rows)} rows")
    return pd.DataFrame(rows)


METRIC_COLS = ["auc", "acc", "f1", "dp_gap", "eo_gap", "eopp_gap",
               "ev", "mmd_alpha", "linsep_alpha", "nonlinsep_alpha", "dp_alpha"]


def aggregate(df):
    g = df.groupby(["method", "lambda"], dropna=False, sort=False)
    mean = g[METRIC_COLS].mean()
    std  = g[METRIC_COLS].std().fillna(0.0)
    return mean.join(std, rsuffix="_sd").reset_index()


# ── Main loop ────────────────────────────────────────────────────────────────

_t0_all = time.perf_counter()

for ds_idx, ds in enumerate(DATASETS, 1):
    out_data = DATA_ROOT / ds; out_data.mkdir(parents=True, exist_ok=True)
    out_fig  = FIG_ROOT  / ds; out_fig.mkdir(parents=True, exist_ok=True)
    print(f"\n═══════════ [{ds_idx}/{len(DATASETS)}] {ds} ═══════════")
    _t0_ds = time.perf_counter()

    print(f"[{ds}] loading (max_samples={MAX_SAMPLES})…")
    X, y, a, meta = dataset_registry.load(ds, n_max=MAX_SAMPLES, seed=0)
    print(f"[{ds}] X={X.shape}, P(Y=1)={y.mean():.3f}, P(A=1)={a.mean():.3f}")

    csv_path = out_data / "raw_results_simple.csv"
    if csv_path.exists():
        print(f"[{ds}] cache HIT  → {csv_path} (delete to re-run)")
        df = pd.read_csv(csv_path)
    else:
        print(f"[{ds}] cache MISS → running sweep")
        # K and PCA target_dim are only needed for the sweep itself —
        # skip them entirely on a cache hit.
        print(f"[{ds}] picking K via AA scree (K ∈ [1, {K_MAX}], "
              f"{K_SEEDS} seeds each)…")
        _t = time.perf_counter()
        X_tr0, y_tr0, a_tr0, *_ = split_and_scale(X, y, a, seed=0)
        sub0 = fit_subset(y_tr0, a_tr0, seed=0)
        X_fit0 = X_tr0[sub0]
        K = pick_K(X_fit0)
        _PCA_TARGET_DIM = pick_pca_target_dim(X_fit0, X.shape[1])
        print(f"[{ds}] K*={K}  FairPCA target_dim={_PCA_TARGET_DIM} "
              f"(cumvar ≥ {PCA_VAR})  [{time.perf_counter()-_t:.1f}s]")
        n_models = N_SEEDS * (2 + len(FAIR_METHODS) * len(LAMBDAS))
        print(f"[{ds}] starting sweep: {N_SEEDS} seeds × "
              f"(2 baselines + {len(FAIR_METHODS)} fair × {len(LAMBDAS)} λ) "
              f"= {n_models} models")
        df = run_sweep(X, y, a, K)
        df.to_csv(csv_path, index=False)
        print(f"[{ds}] cached → {csv_path}")

    print(f"[{ds}] aggregating + picking λ*…")
    agg = aggregate(df)
    agg.to_csv(out_data / "results_summary.csv", index=False)

    lambda_star = {name: pick_lambda_star(agg, name)
                   for name, *_ in FAIR_METHODS}
    print(f"[{ds}] λ*: " + ", ".join(f"{k}={v:g}"
                                     for k, v in lambda_star.items()))

    print(f"[{ds}] rendering 4 figures…")
    # ── 2 Pareto plots ────────────────────────────────────────────────────

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    for i, metric in enumerate(["dp_alpha", "mmd_alpha", "linsep_alpha", "nonlinsep_alpha"]):
        plot_metric(agg, metric, ALL_METHODS, ev_col="ev", ax=axs[i], linewidth=2.0, alpha=1)
        lo, hi = axs[i].get_ylim()
        if lo < 0:
            axs[i].set_ylim(0, hi)

    for ax in axs.flat:
        ax.legend().set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALL_METHODS))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
   
    for ext in ("pdf", "png"):
        fig.savefig(out_fig / f"pareto_rep.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    for i, metric in enumerate(["dp_gap", "eo_gap", "eopp_gap"]):
        plot_metric(agg, metric, ALL_METHODS, ev_col="auc", ax=axs[i], linewidth=2.0, alpha=1)

    for ax in axs.flat:
        ax.legend().set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALL_METHODS))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
   
    for ext in ("pdf", "png"):
        fig.savefig(out_fig / f"pareto_clf.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ── 2 Bar plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, metric in zip(axes, ["auc", "dp_gap", "eo_gap", "eopp_gap"]):
        plot_metric_fixed_lambda(agg, lambda_star=lambda_star,
                                 metric=metric, methods=ALL_METHODS, ax=ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_fig / f"bars_clf.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, metric in zip(axes, ["ev", "dp_alpha", "mmd_alpha", "linsep_alpha", "nonlinsep_alpha"]):
        plot_metric_fixed_lambda(agg, lambda_star=lambda_star,
                                 metric=metric, methods=ALL_METHODS, ax=ax)
        if metric == "ev":
            lo, hi = ax.get_ylim()
            if lo < 0:
                ax.set_ylim(0, hi)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_fig / f"bars_repr.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"[{ds}] DONE → {out_fig}  "
          f"({time.perf_counter()-_t0_ds:.1f}s)")

print(f"\nAll {len(DATASETS)} datasets done "
      f"in {time.perf_counter()-_t0_all:.1f}s.")
