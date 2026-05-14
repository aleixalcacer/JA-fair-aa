"""Print a LaTeX summary table of the three real-world datasets used in the
downstream experiment. Counts are computed on the raw (pre one-hot) frames
fetched via fairlearn, mirroring the loaders in ``datasets.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.datasets import (fetch_adult, fetch_bank_marketing,
                                fetch_diabetes_hospital)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _is_continuous(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
    cont = sum(_is_continuous(df[c]) for c in df.columns)
    cat = df.shape[1] - cont
    return df.shape[1], cont, cat


def _class_balance(y: pd.Series, positive) -> str:
    pos = (y == positive).mean()
    neg = 1.0 - pos
    return f"{round(neg * 100)} / {round(pos * 100)}"


def _fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def summarise_adult():
    b = fetch_adult(as_frame=True)
    df = b.data.copy()
    y_pos = ">50K"
    a_name = "sex"
    df = df.drop(columns=[a_name])
    d, cont, cat = _counts(df)
    y = b.target.astype(str).str.strip()
    return {
        "name": "Adult",
        "n": len(y), "d": d, "cont": cont, "cat": cat,
        "target": r"income $>$ \$50K",
        "balance": _class_balance(y, y_pos),
        "protected": "sex",
    }


def summarise_diabetes():
    b = fetch_diabetes_hospital(as_frame=True)
    df = b.data.copy()
    y = pd.Series(np.asarray(b.target).astype(int))
    df = df.drop(columns=["readmitted", "readmit_binary"], errors="ignore")
    mask = df["gender"].isin(["Female", "Male"]).to_numpy()
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    df = df.drop(columns=["gender"])
    d, cont, cat = _counts(df)
    return {
        "name": "Diabetes 130-Hospitals",
        "n": len(y), "d": d, "cont": cont, "cat": cat,
        "target": "30-day readmission",
        "balance": _class_balance(y, 1),
        "protected": "sex",
    }


def summarise_bank():
    b = fetch_bank_marketing(as_frame=True)
    df = b.data.copy()
    df = df.drop(columns=["V1"])
    d, cont, cat = _counts(df)
    y = b.target.astype(str).str.strip()
    return {
        "name": "Bank Marketing",
        "n": len(y), "d": d, "cont": cont, "cat": cat,
        "target": "term deposit",
        "balance": _class_balance(y, "2"),
        "protected": "age",
    }


CAPTION = (
    r"\rev{Summary of the three real-world datasets used in the downstream "
    r"evaluation. All categorical features are one-hot encoded before fitting, "
    r"which expands the effective feature dimension. Original sizes refer to "
    r"the full datasets; in our experiments we draw a stratified subsample of "
    r"$2{,}000$ instances per run.}"
)


def render(rows: list[dict]) -> str:
    body = []
    for r in rows:
        body.append(
            f"{r['name']} & {_fmt_n(r['n'])} & {r['d']} & {r['cont']} & "
            f"{r['cat']} & {r['target']} & {r['balance']} & {r['protected']} \\\\"
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
    rows = [summarise_adult(), summarise_diabetes(), summarise_bank()]
    print(render(rows))
