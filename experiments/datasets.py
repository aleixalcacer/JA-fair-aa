"""Loaders for the three real-world datasets used in the downstream
experiment: Adult, Diabetes 130-Hospitals, and Bank Marketing. All three
are fetched via ``fairlearn.datasets``.

Public API:
    ``load(name, n_max=None, seed=0) -> (X, y, a, meta)``
        One-hot encoded ndarray + binary labels + protected attribute +
        :class:`DatasetMeta`.
    ``describe(name) -> DatasetMeta``
        Metadata only — used to render the LaTeX summary table.
    ``available_datasets()`` and ``stratified_subsample_idx(...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from fairlearn.datasets import (fetch_adult, fetch_bank_marketing,
                                fetch_diabetes_hospital)
from sklearn.model_selection import train_test_split


# ── Public dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetMeta:
    name: str                        # registry key
    display_name: str                # "Adult"
    n_original: int                  # full row count pre-subsample
    continuous: list[str]            # raw column names (pre one-hot)
    categorical: list[str]
    target: str                      # e.g. "income > $50K"
    protected: str                   # human label, e.g. "sex"
    class_balance: tuple[int, int]   # (neg %, pos %) rounded to integers


# ── Internal helpers ────────────────────────────────────────────────────────

def _normalize_binary(values: np.ndarray) -> np.ndarray:
    """Map any 2-valued array to {0, 1} (smaller / first-sorted → 0)."""
    values = np.asarray(values).ravel()
    uniq = np.unique(values)
    if uniq.size != 2:
        raise ValueError(f"Expected binary array, got {uniq.size} unique values")
    return (values == uniq[1]).astype(int)


def _split_cont_cat(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cont = [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in df.columns if c not in cont]
    return cont, cat


def _finalize(df_features: pd.DataFrame, y_raw, a_raw, spec: "_Spec"):
    """Shared path: derive meta from the raw frame, then one-hot encode."""
    cont, cat = _split_cont_cat(df_features)
    y = _normalize_binary(np.asarray(y_raw))
    a = _normalize_binary(np.asarray(a_raw))
    pos = float(y.mean())
    balance = (int(round((1.0 - pos) * 100)), int(round(pos * 100)))
    meta = DatasetMeta(
        name=spec.name,
        display_name=spec.display_name,
        n_original=len(y),
        continuous=cont,
        categorical=cat,
        target=spec.target,
        protected=spec.protected,
        class_balance=balance,
    )
    X_df = pd.get_dummies(df_features, drop_first=True)
    X = X_df.to_numpy(dtype=float)
    return X, y, a, meta


# ── Dataset specs ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Spec:
    name: str
    display_name: str
    target: str
    protected: str
    prepare: Callable[[], tuple[pd.DataFrame, np.ndarray, np.ndarray]]
    # `prepare` returns (df_features_without_protected, y_raw, a_raw)


def _prepare_adult():
    """Adult / UCI Census Income. Protected: sex."""
    bunch = fetch_adult(as_frame=True)
    df = bunch.data.copy()
    a_raw = df.pop("sex")
    y_raw = (bunch.target.astype(str).str.strip() == ">50K").to_numpy()
    return df, y_raw, a_raw.to_numpy()


def _prepare_diabetes_hospital():
    """Diabetes hospital readmission. Drops two label-leaking columns
    (``readmitted``, ``readmit_binary``) and rows with ``gender ==
    Unknown/Invalid``. Protected: gender."""
    bunch = fetch_diabetes_hospital(as_frame=True)
    df = bunch.data.copy()
    y = np.asarray(bunch.target).astype(int)
    df = df.drop(columns=["readmitted", "readmit_binary"], errors="ignore")
    mask = df["gender"].isin(["Female", "Male"]).to_numpy()
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    a_raw = df.pop("gender").to_numpy()
    return df, y, a_raw


def _prepare_bank_marketing():
    """Bank Marketing (Moro et al.) via OpenML id 1461. Columns are
    V1..V16. Protected: V1 (age), binarized at 30 (younger = 1)."""
    bunch = fetch_bank_marketing(as_frame=True)
    df = bunch.data.copy()
    a_raw = (df.pop("V1") < 30).to_numpy().astype(int)
    y_raw = (bunch.target.astype(str).str.strip() == "2").to_numpy()
    return df, y_raw, a_raw


_SPECS: dict[str, _Spec] = {
    "adult": _Spec(
        name="adult", display_name="Adult",
        target=r"income $>$ \$50K", protected="sex",
        prepare=_prepare_adult,
    ),
    "diabetes_hospital": _Spec(
        name="diabetes_hospital", display_name="Diabetes 130-Hospitals",
        target="30-day readmission", protected="sex",
        prepare=_prepare_diabetes_hospital,
    ),
    "bank_marketing": _Spec(
        name="bank_marketing", display_name="Bank Marketing",
        target="term deposit", protected="age",
        prepare=_prepare_bank_marketing,
    ),
}


# ── Public API ──────────────────────────────────────────────────────────────

def available_datasets() -> list[str]:
    return list(_SPECS)


def stratified_subsample_idx(y, a, n_max, seed=0):
    """Indices of a stratified subsample by (Y, A). Returns all indices
    when ``n_max`` is None / ≤ 0 / ≥ len(y)."""
    n = len(y)
    if n_max is None or n_max <= 0 or n <= n_max:
        return np.arange(n)
    strat = np.char.add(np.asarray(y).astype(str), "_")
    strat = np.char.add(strat, np.asarray(a).astype(str))
    idx, _ = train_test_split(np.arange(n), train_size=n_max,
                              stratify=strat, random_state=seed)
    return idx


def _get_spec(name: str) -> _Spec:
    if name not in _SPECS:
        raise KeyError(f"Unknown dataset {name!r}. "
                       f"Available: {available_datasets()}")
    return _SPECS[name]


def load(name: str, n_max: int | None = None, seed: int = 0):
    """Load a dataset. Encodes on the full frame, then subsamples by (Y, A)."""
    spec = _get_spec(name)
    df_features, y_raw, a_raw = spec.prepare()
    X, y, a, meta = _finalize(df_features, y_raw, a_raw, spec)
    if n_max is not None and len(y) > n_max:
        idx = stratified_subsample_idx(y, a, n_max=n_max, seed=seed)
        X, y, a = X[idx], y[idx], a[idx]
    return X, y, a, meta


def describe(name: str) -> DatasetMeta:
    """Build metadata for the summary table. Still touches the raw frame
    (column dtypes drive the cont/cat split), but skips the one-hot encode."""
    spec = _get_spec(name)
    df_features, y_raw, a_raw = spec.prepare()
    cont, cat = _split_cont_cat(df_features)
    y = _normalize_binary(np.asarray(y_raw))
    pos = float(y.mean())
    balance = (int(round((1.0 - pos) * 100)), int(round(pos * 100)))
    return DatasetMeta(
        name=spec.name,
        display_name=spec.display_name,
        n_original=len(y),
        continuous=cont,
        categorical=cat,
        target=spec.target,
        protected=spec.protected,
        class_balance=balance,
    )
