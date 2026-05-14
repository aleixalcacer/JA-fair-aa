"""Per-dataset loaders for the real-data downstream experiment.

Each loader returns ``(X, y, a, feature_names)``:
    X              : ndarray of shape (n_samples, n_features), already one-hot
                     encoded. *Not* yet standardized.
    y              : ndarray of shape (n_samples,), int with values {0, 1}.
    a              : ndarray of shape (n_samples,), int with values {0, 1}.
    feature_names  : list[str], length n_features.

To add a new dataset, write a private loader and register it in ``LOADERS``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# allow `python experiments/real_data.py …` (script-style) to import siblings
sys.path.insert(0, str(Path(__file__).resolve().parent))

from german_credit import get_german_credit_data, get_healthcare_data, get_adult_data  # noqa: E402
from fairlearn.datasets import (fetch_adult, fetch_diabetes_hospital,  # noqa: E402
                                fetch_bank_marketing)

MAX_SAMPLES = None  # default: load full dataset (orchestrator subsamples for fit only)


def _normalize_binary(values: np.ndarray) -> np.ndarray:
    """Map any 2-valued integer/categorical array to {0, 1} (smaller → 0)."""
    values = np.asarray(values).ravel()
    uniq = np.unique(values)
    if uniq.size != 2:
        raise ValueError(f"Expected binary array, got {uniq.size} unique values")
    return (values == uniq[1]).astype(int)


def _load_german_credit():
    X_df, y_df, a_df = get_german_credit_data()
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    y = _normalize_binary(y_df.to_numpy())
    a = _normalize_binary(a_df.to_numpy())
    return X, y, a, feature_names


def _load_healthcare():
    X_df, y_df, a_df = get_healthcare_data()
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    y = _normalize_binary(y_df.to_numpy())
    a = _normalize_binary(a_df.to_numpy())
    return X, y, a, feature_names

def _load_adult():
    X_df, y_df, a_df = get_adult_data()
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    y = _normalize_binary(y_df.to_numpy())
    a = _normalize_binary(a_df.to_numpy())
    return X, y, a, feature_names


def _load_adult_fairlearn():
    """Adult / UCI Census Income via fairlearn. Protected: sex."""
    bunch = fetch_adult(as_frame=True)
    df = bunch.data.copy()
    a_raw = df.pop("sex")
    y_raw = (bunch.target.astype(str).str.strip() == ">50K")
    df = pd.get_dummies(df, drop_first=True)
    return (df.to_numpy(dtype=float),
            _normalize_binary(y_raw.to_numpy()),
            _normalize_binary(a_raw.to_numpy()),
            list(df.columns))


def _load_diabetes_hospital():
    """Diabetes hospital readmission via fairlearn. Protected: gender
    (rows with `Unknown/Invalid` gender are dropped).

    The fairlearn frame ships two label-leaking columns:
        - ``readmitted`` ∈ {NO, >30, <30}; '<30' is exactly target=1
        - ``readmit_binary`` (Spearman ~0.4 with target by construction)
    Both must be removed from the feature matrix."""
    bunch = fetch_diabetes_hospital(as_frame=True)
    df = bunch.data.copy()
    y = np.asarray(bunch.target).astype(int)
    df = df.drop(columns=["readmitted", "readmit_binary"], errors="ignore")
    mask = df["gender"].isin(["Female", "Male"]).to_numpy()
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    a_raw = df.pop("gender")
    df = pd.get_dummies(df, drop_first=True)
    return (df.to_numpy(dtype=float),
            _normalize_binary(y),
            _normalize_binary(a_raw.to_numpy()),
            list(df.columns))


def _load_bank_marketing():
    """Bank Marketing (Moro et al.) via fairlearn / OpenML id 1461.
    Columns are V1..V16 in the OpenML version. Protected: V1 (age),
    binarized at 30 (younger=1, older=0)."""
    bunch = fetch_bank_marketing(as_frame=True)
    df = bunch.data.copy()
    a_raw = (df.pop("V1") < 30).astype(int)
    y_raw = (bunch.target.astype(str).str.strip() == "2")
    df = pd.get_dummies(df, drop_first=True)
    return (df.to_numpy(dtype=float),
            _normalize_binary(y_raw.to_numpy()),
            _normalize_binary(a_raw.to_numpy()),
            list(df.columns))


LOADERS = {
    "german_credit":     _load_german_credit,
    "healthcare":        _load_healthcare,
    "adult":             _load_adult_fairlearn,
    "diabetes_hospital": _load_diabetes_hospital,
    "bank_marketing":    _load_bank_marketing,
}


def available_datasets():
    return sorted(LOADERS)


def stratified_subsample_idx(y, a, n_max, seed=0):
    """Return indices of a stratified subsample by (Y, A).

    If n_max is None or n_max ≥ len(y), returns all indices.
    Useful from the orchestrator to pick the rows that AA/FairAA fits on
    while transforming the full set."""
    n = len(y)
    if n_max is None or n_max <= 0 or n <= n_max:
        return np.arange(n)
    strat = np.char.add(np.asarray(y).astype(str), "_")
    strat = np.char.add(strat, np.asarray(a).astype(str))
    idx, _ = train_test_split(np.arange(n), train_size=n_max,
                              stratify=strat, random_state=seed)
    return idx


def load(name: str, n_max: int | None = MAX_SAMPLES, seed: int = 0):
    """Load a registered dataset. By default returns the full dataset; pass
    `n_max` to apply a stratified (Y, A) cap at load time."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {available_datasets()}")
    X, y, a, feature_names = LOADERS[name]()
    if n_max is not None and len(y) > n_max:
        idx = stratified_subsample_idx(y, a, n_max=n_max, seed=seed)
        X, y, a = X[idx], y[idx], a[idx]
    return X, y, a, feature_names
