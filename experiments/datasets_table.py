"""Print a LaTeX summary table of the real-world datasets used in the
downstream experiment, reading metadata from ``datasets.describe``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import datasets as dataset_registry  # noqa: E402


CAPTION = (
    r"\rev{Summary of the three real-world datasets used in the downstream "
    r"evaluation. All categorical features are one-hot encoded before fitting, "
    r"which expands the effective feature dimension. Original sizes refer to "
    r"the full datasets; in our experiments we draw a stratified subsample of "
    r"$2{,}000$ instances per run.}"
)


def _fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def render(metas) -> str:
    body = []
    for m in metas:
        d_raw = len(m.continuous) + len(m.categorical)
        body.append(
            f"{m.display_name} & {_fmt_n(m.n_original)} & {d_raw} & "
            f"{len(m.continuous)} & {len(m.categorical)} & {m.target} & "
            f"{m.class_balance[0]} / {m.class_balance[1]} & {m.protected} \\\\"
        )
    return (
        "\\begin{table}[!ht]\n"
        "\\centering\n"
        f"\\caption{{{CAPTION}}}\n"
        "\\label{tab:datasets}\n"
        "\\rev{\n"
        "\\begin{tabular}{lccccccc}\n"
        "\\toprule\n"
        "Dataset & $n_{\\text{original}}$ & $d$ & Cont. & Cat. & Target & "
        "Class balance & Protected attr. \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


if __name__ == "__main__":
    metas = [dataset_registry.describe(name)
             for name in dataset_registry.available_datasets()]
    print(render(metas))
