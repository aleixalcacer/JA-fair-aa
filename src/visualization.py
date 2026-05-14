import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypalettes import load_cmap

CMAP_NAME = "Lupi"
CMAP = load_cmap(CMAP_NAME)


methods = ["Raw", "AA", "FairAA", "FairAA_3Moment", "FairAA_Adv", "FairAA_MMD", "FairPCA_AA"]
methods_colors = {method: CMAP(i) for i, method in enumerate(methods)}
methods_labels = {
    "Raw": "Raw",
    "AA": "AA",
    "FairAA": "FairAA",
    "FairAA_3Moment": "FairAA (2nd Order)",
    "FairAA_Adv": "FairAA (Adversarial)",
    "FairAA_MMD": "FairAA (MMD)",
    "FairPCA_AA": "FairPCA + AA",
}

metrics_labels = {
    "ev": "Explained Variance",
    "mmd": "MMD",
    "dp": "Demographic Parity",
    "ls": "Linear Separability",
    "nls": "Nonlinear Separability",
    "auc": "AUC",
    "dp_gap": "Demographic Parity",
    "eo_gap": "Equal Opportunity",
    "eopp_gap": "Equalized Odds",
    "mmd_alpha": "MMD",
    "linsep_alpha": "Linear separability",
    "nonlinsep_alpha": "Non-linear separability",
    "dp_alpha": "Demographic parity",
}


def pareto_front(sub, metric, ev_col="ev"):
    """Non-dominated rows: lower `metric` better, higher `ev_col` better."""
    pts = sub[[metric, ev_col]].values
    keep = []
    for i, (x_i, y_i) in enumerate(pts):
        dominated = any(
            (pts[j, 0] <= x_i and pts[j, 1] >= y_i) and
            (pts[j, 0] < x_i or pts[j, 1] > y_i)
            for j in range(len(pts)) if j != i
        )
        if not dominated:
            keep.append(i)
    return sub.iloc[keep].sort_values(metric)


def plot_metric(data: pd.DataFrame, metric: str, methods: list,
                ev_col: str = "ev", ax=None, frontier_only: bool = False,
                uniform_marker: str | None = None, **kwargs):
    """Pareto plot: scatter all sweep points + plain frontier line per
    method. Singleton methods (Raw / AA) render as a distinct marker.

    Lower `metric` is better; higher `ev_col` is better.
    Rows with NaN in either axis are dropped silently.

    If `frontier_only`, skip the all-points scatter. If `uniform_marker`
    is set, use the same marker for every method (overriding the
    per-method marker map for singletons and adding markers on the
    frontier line)."""
    if ax is None:
        _, ax = plt.subplots()

    for method in methods:
        sub = data[data["method"] == method]
        sub = sub.dropna(subset=[metric, ev_col])
        if sub.empty:
            continue
        c = methods_colors[method]
        label = methods_labels.get(method, method)
       
        pf = pareto_front(sub, metric, ev_col)
        line_kwargs = dict(kwargs)
        ax.plot(pf[metric], pf[ev_col], color=c, marker=".", label=label, **line_kwargs)
        
    ax.set_xlabel(metrics_labels.get(metric, metric))
    ax.set_ylabel(metrics_labels.get(ev_col, ev_col))
    ax.grid(alpha=0.3)
    return ax


def plot_metric_fixed_lambda(data: pd.DataFrame, lambda_star: dict,
                             metric: str, methods: list, ax=None, **kwargs):
    """Bar chart at a per-method λ\\* (one bar per method).

    Expects `data` to be the aggregated DataFrame produced by grouping the
    raw sweep on `(method, lambda)`: columns `<metric>` and `<metric>_sd`.
    For methods not in `lambda_star` (e.g. Raw / AA), pick the row whose
    `lambda` is NaN."""
    if ax is None:
        _, ax = plt.subplots()

    names, vals, errs, colors, lam_labels = [], [], [], [], []
    for method in methods:
        if method in lambda_star:
            lam = lambda_star[method]
            if pd.isna(lam):
                continue
            sub = data[(data["method"] == method) &
                       (data["lambda"] == lam)]
            label = methods_labels.get(method, method)
            lam_labels.append(f"$\lambda^*={lam:g}$")
        else:
            sub = data[(data["method"] == method) &
                       data["lambda"].isna()]
            label = methods_labels.get(method, method)
            lam_labels.append("")
        if sub.empty or pd.isna(sub[metric].iloc[0]):
            lam_labels.pop()
            continue
        names.append(label)
        vals.append(float(sub[metric].iloc[0]))
        sd_col = f"{metric}_sd"
        errs.append(float(sub[sd_col].iloc[0]) if sd_col in sub else 0.0)
        colors.append(methods_colors[method])

    x = np.arange(len(names))
    bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85, capsize=4, **kwargs)
    for rect, err, txt in zip(bars, errs, lam_labels):
        if not txt:
            continue
        top = rect.get_height() + (err if err else 0.0)
        ax.annotate(txt, xy=(rect.get_x() + rect.get_width() / 2, top),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metrics_labels.get(metric, metric))
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_runtime(data: pd.DataFrame, methods: list, ax=None, **kwargs):
    """Plot the runtime for each method and each n using line plot with error bars."""
    if ax is None:
        fig, ax = plt.subplots()

    for method in methods:
        sub = data[data["method"] == method]
        c = methods_colors[method]

        if len(sub) > 0:
            ax.plot(sub["n"], sub["rtime"], color=c, label=methods_labels.get(method, method), **kwargs)
            ax.errorbar(sub["n"], sub["rtime"], yerr=sub["rtime_sd"], fmt="none", color=c, capsize=2)
    ax.set_xlabel("Number of Samples (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(alpha=0.3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax
