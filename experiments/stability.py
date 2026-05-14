import sys
import math
import time
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from tqdm import tqdm

from archetypes.datasets import make_archetypal_dataset
from src.methods import FairAA, FairAA_3Moment, FairAA_Adversarial, FairAA_MMD
from src.metrics import demographic_parity

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────

METHODS = [
    "FairAA", 
    "FairAA_3Moment",
    "FairAA_Adv",
    "FairAA_MMD"
    ]
N_SIZES = np.logspace(5, 10, base=2, num=6, dtype=int)  # from 2^5=32 to 2^10=1024
N_SEEDS = 20
N_ARCHETYPES = 4
EPS = 1e-4

# Fixed λ per method — adjust here after calibration
FIXED_LAMBDAS = {
    "FairAA":         1E-3,
    "FairAA_3Moment": 1E-3,
    "FairAA_Adv":     1E-3,
    "FairAA_MMD":     1E-3,
}

_METHOD_NAMES_ALL = ["AA", "FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
COLORS = dict(zip(_METHOD_NAMES_ALL, plt.cm.tab10.colors))

BASE_PARAMS = dict(
    n_archetypes=N_ARCHETYPES,
    init="furthest_sum",
    n_init=3,
    max_iter=500,
    tol=0,        # run all iterations so we capture the full trajectory
    method="pgd",
)

# ── Data ───────────────────────────────────────────────────────────────────────


def make_data(n, generator):
    """Generate the same tetrahedral dataset as the Pareto experiment."""
    s = 1 / (2 * math.sqrt(2))
    tetra = np.array([
        [s,  s,  s],
        [s, -s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
    ])
    X, _, _ = make_archetypal_dataset(tetra, (n,), alpha=0.1, noise=0,
                                      generator=generator)
    y = (X[:, 0] > 0).astype(int)
    Z = (y - y.mean()).reshape(-1, 1)
    return X, y, Z, tetra


# ── Single-run fitting ─────────────────────────────────────────────────────────


def _ev_from_rec(rec_list, TSS):
    return [1.0 - r / TSS for r in rec_list]


def _iters_to_converge(ev_list, eps):
    for t in range(1, len(ev_list)):
        if abs(ev_list[t] - ev_list[t - 1]) < eps:
            return t, True
    return len(ev_list) - 1, False


def fit_one(method_name, lam, X, y, Z, seed):
    """Fit one model and return per-seed metrics dict."""
    TSS = float(np.sum((X - X.mean(axis=0)) ** 2))

    try:
        t0 = time.perf_counter()

        if method_name == "FairAA":
            model = FairAA(**BASE_PARAMS, fairness_const=lam, random_state=seed)
            model.fit(X, Z=Z)
            S = model.transform(X, Z)
            loss_per_iter = list(model.loss_)
            ev_per_iter = _ev_from_rec(model.loss_, TSS)

        elif method_name == "FairAA_3Moment":
            model = FairAA_3Moment(**BASE_PARAMS, fairness_const=lam,
                                   alpha_2=0.1, alpha_3=0.1,
                                   balance_orders=True, random_state=seed)
            model.fit(X, Z=y.astype(float))
            S = model.transform(X)
            loss_per_iter = list(model.loss_history_["total"])
            ev_per_iter = _ev_from_rec(model.loss_history_["reconstruction"], TSS)

        elif method_name == "FairAA_Adv":
            model = FairAA_Adversarial(**BASE_PARAMS, fairness_const=lam,
                                       n_adv_steps=2, lr_adv=1e-4,
                                       random_state=seed)
            model.fit(X, Z=y)
            S = model.transform(X, Z=y)
            loss_per_iter = list(model.loss_history_["total"])
            ev_per_iter = _ev_from_rec(model.loss_history_["reconstruction"], TSS)

        elif method_name == "FairAA_MMD":
            model = FairAA_MMD(**BASE_PARAMS, fairness_const=lam,
                               random_state=seed)
            model.fit(X, Z=y)
            S = model.transform(X, Z=y)
            loss_per_iter = list(model.loss_history_["total"])
            ev_per_iter = _ev_from_rec(model.loss_history_["reconstruction"], TSS)

        else:
            raise ValueError(f"Unknown method: {method_name}")

        total_runtime = time.perf_counter() - t0
        n_iters = len(ev_per_iter)
        iters_to_conv, converged = _iters_to_converge(ev_per_iter, EPS)

        return {
            "seed": seed,
            "loss_per_iter": loss_per_iter,
            "ev_per_iter": ev_per_iter,
            "iters_to_converge": iters_to_conv,
            "converged": converged,
            "final_ev": ev_per_iter[-1],
            "total_runtime": total_runtime,
            "runtime_per_iter": total_runtime / n_iters if n_iters > 0 else float("nan"),
            "archetypes": model.archetypes_.copy(),
            "fairness": demographic_parity(S, y),
        }

    except Exception as exc:
        return {
            "seed": seed,
            "loss_per_iter": [],
            "ev_per_iter": [],
            "iters_to_converge": None,
            "converged": False,
            "final_ev": float("nan"),
            "total_runtime": float("nan"),
            "runtime_per_iter": float("nan"),
            "archetypes": None,
            "fairness": float("nan"),
            "error": str(exc),
        }


# ── Aggregation ────────────────────────────────────────────────────────────────


def _align_archetypes(reference, archetypes_s):
    """Reorder rows of archetypes_s to best match reference via cosine distance."""
    D = cosine_distances(reference, archetypes_s)
    _, col_ind = linear_sum_assignment(D)
    return archetypes_s[col_ind]


def aggregate_seeds(per_seed, true_archetypes):
    """Compute aggregated statistics from a list of per-seed result dicts."""
    valid = [r for r in per_seed if r["archetypes"] is not None]
    conv = [r for r in valid if r["converged"]]
    n_total = len(per_seed)
    n_valid = len(valid)

    def _mean_std(vals):
        a = np.array(vals, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    # ── Convergence ──────────────────────────────────────────────────────────
    mean_iters, std_iters = (_mean_std([r["iters_to_converge"] for r in conv])
                             if conv else (float("nan"), float("nan")))
    pct_converged = len(conv) / n_total if n_total > 0 else float("nan")
    mean_final_ev, std_final_ev = _mean_std([r["final_ev"] for r in valid])

    # ── Runtime ──────────────────────────────────────────────────────────────
    mean_runtime, std_runtime = _mean_std([r["total_runtime"] for r in valid])
    mean_rpi, std_rpi = _mean_std([r["runtime_per_iter"] for r in valid])

    # ── Archetype alignment + variance ───────────────────────────────────────
    global_variance = float("nan")
    std_global_variance = float("nan")
    max_variance = float("nan")
    per_archetype_variance = None
    mean_pairwise_similarity = float("nan")
    std_pairwise_similarity = float("nan")

    if n_valid >= 1:
        # Align every seed's archetypes to the ground-truth reference
        reference = np.asarray(true_archetypes)
        aligned = [_align_archetypes(reference, r["archetypes"]) for r in valid]
        stack = np.stack(aligned, axis=0)  # (n_valid, k, d)

        # per_archetype_variance: mean over d of variance over seeds → (k,)
        per_archetype_variance = np.var(stack, axis=0).mean(axis=1)
        global_variance = float(per_archetype_variance.mean())
        std_global_variance = float(per_archetype_variance.std())
        max_variance = float(per_archetype_variance.max())

        # mean pairwise cosine similarity between aligned archetypes across seeds
        if n_valid >= 2:
            sims = []
            for (i, j) in itertools.combinations(range(n_valid), 2):
                sim_matrix = cosine_similarity(aligned[i], aligned[j])
                sims.append(float(np.diag(sim_matrix).mean()))
            mean_pairwise_similarity = float(np.mean(sims))
            std_pairwise_similarity = float(np.std(sims))

    # ── Fairness ─────────────────────────────────────────────────────────────
    mean_fairness, std_fairness = _mean_std([r["fairness"] for r in valid])

    return {
        "mean_iters": mean_iters,
        "std_iters": std_iters,
        "pct_converged": pct_converged,
        "mean_final_ev": mean_final_ev,
        "std_final_ev": std_final_ev,
        "mean_runtime": mean_runtime,
        "std_runtime": std_runtime,
        "mean_runtime_per_iter": mean_rpi,
        "std_runtime_per_iter": std_rpi,
        "global_variance": global_variance,
        "std_global_variance": std_global_variance,
        "max_variance": max_variance,
        "per_archetype_variance": per_archetype_variance,
        "mean_pairwise_similarity": mean_pairwise_similarity,
        "std_pairwise_similarity": std_pairwise_similarity,
        "mean_fairness": mean_fairness,
        "std_fairness": std_fairness,
    }


# ── Persistence ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV  = DATA_DIR / "raw_results_stability.csv"
RAW_NPZ  = DATA_DIR / "raw_results_stability_archetypes.npz"


def _arch_key(method, n, seed):
    return f"{method}__{n}__{seed}"


def save_results(rows, arch_dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW_CSV, index=False)
    np.savez_compressed(RAW_NPZ, **arch_dict)


def load_results(true_arch_by_n):
    df = pd.read_csv(RAW_CSV)
    arch_dict = dict(np.load(RAW_NPZ))
    results = {m: {} for m in METHODS}
    plot_n = N_SIZES[-1]
    plot_data = {}

    for method_name in METHODS:
        for n in N_SIZES:
            sub = df[(df["method"] == method_name) & (df["n"] == n)]
            per_seed = []
            for _, row in sub.iterrows():
                key = _arch_key(method_name, n, int(row["seed"]))
                arch = arch_dict[key] if key in arch_dict else None
                per_seed.append({
                    "seed":             int(row["seed"]),
                    "ev_per_iter":      [],
                    "iters_to_converge": row["iters_to_converge"],
                    "converged":        bool(row["converged"]),
                    "final_ev":         row["final_ev"],
                    "total_runtime":    row["total_runtime"],
                    "runtime_per_iter": row["runtime_per_iter"],
                    "archetypes":       arch,
                    "fairness":         row["fairness"],
                })
            true_archetypes = true_arch_by_n[n]
            agg = aggregate_seeds(per_seed, true_archetypes)
            results[method_name][n] = {"per_seed": per_seed, "aggregated": agg}
            if n == plot_n and not plot_data:
                plot_data = {"X": None, "y": None, "true_archetypes": true_archetypes}

    return results, plot_data


# ── Main sweep ─────────────────────────────────────────────────────────────────


def run_experiment():
    # ── load from cache if available ─────────────────────────────────────────
    if RAW_CSV.exists() and RAW_NPZ.exists():
        print(f"Loading cached results from {RAW_CSV} (delete to re-run).")
        true_arch_by_n = {}
        for n in N_SIZES:
            _, _, _, tetra = make_data(n, np.random.RandomState(42))
            true_arch_by_n[n] = tetra
        return load_results(true_arch_by_n)

    # ── run sweep ─────────────────────────────────────────────────────────────
    results  = {m: {} for m in METHODS}
    plot_n   = N_SIZES[-1]
    plot_data = {}
    rows     = []
    arch_dict = {}

    total = len(METHODS) * len(N_SIZES) * N_SEEDS
    with tqdm(total=total, desc="stability sweep") as pbar:
        for method_name in METHODS:
            lam = FIXED_LAMBDAS[method_name]
            for n in N_SIZES:
                generator = np.random.RandomState(42)
                X, y, Z, true_archetypes = make_data(n, generator)
                if n == plot_n and not plot_data:
                    plot_data = {"X": X, "y": y, "true_archetypes": true_archetypes}
                per_seed = []
                for seed in range(N_SEEDS):
                    r = fit_one(method_name, lam, X, y, Z, seed + 42)
                    per_seed.append(r)
                    rows.append({
                        "method":           method_name,
                        "n":                n,
                        "seed":             seed + 42,
                        "iters_to_converge": r["iters_to_converge"],
                        "converged":        r["converged"],
                        "final_ev":         r["final_ev"],
                        "total_runtime":    r["total_runtime"],
                        "runtime_per_iter": r["runtime_per_iter"],
                        "fairness":         r["fairness"],
                    })
                    if r["archetypes"] is not None:
                        arch_dict[_arch_key(method_name, n, seed + 42)] = r["archetypes"]
                    pbar.update(1)
                agg = aggregate_seeds(per_seed, true_archetypes)
                results[method_name][n] = {"per_seed": per_seed, "aggregated": agg}

    save_results(rows, arch_dict)
    return results, plot_data


# ── Plotting ───────────────────────────────────────────────────────────────────

# Display names for legends / panel labels
METHOD_LABELS = {
    "FairAA":         "FairAA",
    "FairAA_3Moment": "FairAA-3M",
    "FairAA_Adv":     "FairAA-Adv",
    "FairAA_MMD":     "FairAA-MMD",
}

RC = {
    "font.family":       "serif",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.4,
    "lines.markersize":  4,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth":    0.5,
    "grid.color":        "0.85",
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
}


def _clean_ax(ax, logx=False, which="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which=which, alpha=1)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))


def make_plots(results, plot_data):
    ns = N_SIZES

    with plt.rc_context(RC):

        # ── Panel figure: 4 scalar metrics (2 × 2) ───────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))

        panels = [
            (axes[0, 0], "mean_final_ev",          "std_final_ev",          "Explained variance",       False, False, None),
            (axes[0, 1], "mean_runtime",            "std_runtime",           "Wall time (s)",            True,  True,  "both"),
            (axes[1, 0], "mean_fairness",           "std_fairness",          "Demographic parity",       True,  False, "both"),
            (axes[1, 1], "mean_pairwise_similarity","std_pairwise_similarity","Archetype cosine similarity", False, False, None),
        ]

        for ax, mean_key, std_key, ylabel, logx, logy, grid_which in panels:
            for m in METHODS:
                agg   = [results[m][n]["aggregated"] for n in ns]
                means = [a[mean_key] for a in agg]
                stds  = [a[std_key]  for a in agg]
                ax.errorbar(ns, means, yerr=stds, color=COLORS[m],
                            label=METHOD_LABELS[m], marker="o", capsize=2,
                            linewidth=1.4, elinewidth=0.8, capthick=0.8)
            ax.set_xlabel("$n$")
            ax.set_ylabel(ylabel)
            _clean_ax(ax, logx=logx, which=grid_which or "both")
            if logy:
                ax.set_yscale("log")
            if mean_key == "mean_pairwise_similarity":
                ax.set_ylim(0, 1.05)

        axes[0, 0].legend(loc="best")
        fig.tight_layout(h_pad=2.5, w_pad=2.0)
        fig.savefig(FIGURES / "stability_metrics.pdf")
        plt.close(fig)

        # ── Loss trajectories ─────────────────────────────────────────────────
        TRAJ_N = N_SIZES[len(N_SIZES) // 2]
        fig, axes = plt.subplots(1, len(METHODS),
                                 figsize=(2.8 * len(METHODS), 2.6))
        for ax, m in zip(axes, METHODS):
            trajs = [r["loss_per_iter"] for r in results[m][TRAJ_N]["per_seed"]
                     if len(r["loss_per_iter"]) > 0]
            if not trajs:
                ax.text(0.5, 0.5, "no data\n(cached)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.set_title(METHOD_LABELS[m])
                continue
            min_len = min(len(t) for t in trajs)
            iters = np.arange(min_len)
            for r in trajs:
                ax.plot(iters, r[:min_len], color=COLORS[m],
                        alpha=0.35, linewidth=0.8)
            ax.set_title(METHOD_LABELS[m])
            ax.set_xlabel("Iteration")
            if ax is axes[0]:
                ax.set_ylabel("Total loss")
            _clean_ax(ax)
        fig.tight_layout(w_pad=1.5)
        fig.savefig(FIGURES / "stability_trajectories.pdf")
        plt.close(fig)

    # ── 3D archetype comparison (interactive — no RC override needed) ─────────
    true_arch = np.asarray(plot_data["true_archetypes"])
    n_cmap    = plt.cm.plasma(np.linspace(0.15, 0.85, len(ns)))

    fig = plt.figure(figsize=(4.2 * len(METHODS), 4))
    for idx, m in enumerate(METHODS):
        ax = fig.add_subplot(1, len(METHODS), idx + 1, projection="3d")
        for n, color in zip(ns, n_cmap):
            valid_arch = [r["archetypes"] for r in results[m][n]["per_seed"]
                          if r["archetypes"] is not None]
            if not valid_arch:
                continue
            aligned   = [_align_archetypes(true_arch, a) for a in valid_arch]
            mean_arch = np.stack(aligned).mean(axis=0)
            ax.scatter(*mean_arch.T, color=color, marker="^", s=60,
                       edgecolors="black", linewidths=0.3, zorder=4, label=f"$n={n}$")
        ax.scatter(*true_arch.T, color="black", marker="*", s=150, zorder=5, label="True")
        ax.set_title(METHOD_LABELS[m], fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=7); ax.set_ylabel("$x_2$", fontsize=7)
        ax.set_zlabel("$x_3$", fontsize=7)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    plt.show()

    print(f"Plots saved to {FIGURES}/")


# ── Entry point ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    results, plot_data = run_experiment()
    make_plots(results, plot_data)
