from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

from archetypes.numpy._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from archetypes.numpy._projection import l1_normalize_proj, unit_simplex_proj
from archetypes.numpy._fair_aa import (
    pgd_transform,
    pgd_fit_transform,
    pseudo_pgd_transform,
    pseudo_pgd_fit_transform,
    verbose_print_rss,
)


class FairAAWarm(TransformerMixin, BaseEstimator):
    """
    Fair Archetypal Analysis with warm-start support.

    Identical to archetypes.FairAA but fit() and fit_transform() accept
    optional S_init / C_init arrays to seed the optimiser from a previous
    solution, bypassing the random initialisation.

    Parameters
    ----------
    n_archetypes : int
    fairness_const : float, default=200.0
    max_iter : int, default=300
    tol : float, default=1e-4
    init : str, default='uniform'
    n_init : int, default=1
    init_params : dict or None, default=None
    save_init : bool, default=False
    method : {'pgd', 'pseudo_pgd'}, default='pgd'
    method_params : dict or None, default=None
    verbose : bool, default=False
    random_state : int or None, default=None

    Attributes
    ----------
    archetypes_ : ndarray of shape (n_archetypes, n_features)
    coefficients_, A_ : ndarray of shape (n_samples, n_archetypes)
    arch_coefficients_, B_ : ndarray of shape (n_archetypes, n_samples)
    loss_ : list
    rss_, reconstruction_error_ : float
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "init": [
            StrOptions({"uniform", "furthest_sum", "furthest_first", "coreset", "aa_plus_plus"}),
            None,
        ],
        "init_params": [dict, None],
        "save_init": [bool],
        "method": [StrOptions({"pgd", "pseudo_pgd"})],
        "method_params": [dict, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_archetypes,
        *,
        fairness_const=200.0,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        n_init=1,
        init_params=None,
        save_init=False,
        method="pgd",
        method_params=None,
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.fairness_const = fairness_const
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.init_params = init_params
        self.save_init = save_init
        self.method = method
        self.method_params = method_params
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

    def _init_archetypes(self, X, rng):
        n_samples, _ = X.shape

        if self.init == "uniform":
            init_archetype_func = uniform
        elif self.init == "furthest_sum":
            init_archetype_func = furthest_sum
        elif self.init == "furthest_first":
            init_archetype_func = furthest_first
        elif self.init == "aa_plus_plus":
            init_archetype_func = aa_plus_plus

        init_params = {} if self.init_params is None else self.init_params
        B = np.zeros((self.n_archetypes, n_samples), dtype=X.dtype)
        ind = init_archetype_func(X, self.n_archetypes, random_state=rng, **init_params)
        for i, j in enumerate(ind):
            B[i, j] = 1

        archetypes = X[ind]

        A = np.zeros((n_samples, self.n_archetypes), dtype=X.dtype)
        ind = rng.choice(self.n_archetypes, n_samples, replace=True)
        for i, j in enumerate(ind):
            A[i, j] = 1

        return A, B, archetypes

    def fit(self, X, y=None, Z=None, *, S_init=None, C_init=None):
        self.fit_transform(X, y, Z, S_init=S_init, C_init=C_init)
        return self

    def transform(self, X, Z):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.ascontiguousarray(X)
        archetypes = self.archetypes_

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        if self.method == "pgd":
            transform_func = pgd_transform
        elif self.method == "pseudo_pgd":
            transform_func = pseudo_pgd_transform

        method_params = {} if self.method_params is None else self.method_params
        return transform_func(
            X, Z, archetypes,
            fairness_const=self.fairness_const,
            max_iter=self.max_iter,
            tol=self.tol,
            **method_params,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, Z=None, S_init=None, C_init=None, **params):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)

        if self.n_archetypes == 1:
            n_samples = X.shape[0]
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            best_rss = float(
                squared_norm(X - archetypes_) + self.fairness_const * squared_norm(Z.T @ A_)
            )
            n_iter_ = 0
            loss_ = [best_rss]

        else:
            if self.method == "pgd":
                fit_transform_func = pgd_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = pseudo_pgd_fit_transform

            method_params = {} if self.method_params is None else self.method_params
            rng = check_random_state(self.random_state)

            best_rss = np.inf

            if S_init is not None:
                # Warm start: use provided A and B, skip random init
                A = np.array(S_init, dtype=X.dtype)
                B = np.array(C_init, dtype=X.dtype)
                archetypes = B @ X
                A, B, archetypes, n_iter, loss, _ = fit_transform_func(
                    X, A, B, Z, archetypes,
                    fairness_const=self.fairness_const,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    verbose=self.verbose,
                    **method_params,
                )
                best_rss = loss[-1]
                A_ = A; B_ = B; archetypes_ = archetypes
                n_iter_ = n_iter; loss_ = loss
            else:
                for i in range(self.n_init):
                    A, B, archetypes = self._init_archetypes(X, rng)

                    if self.save_init:
                        self.B_init_ = B.copy()
                        self.archetypes_init_ = archetypes.copy()

                    A, B, archetypes, n_iter, loss, _ = fit_transform_func(
                        X, A, B, Z, archetypes,
                        fairness_const=self.fairness_const,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        verbose=self.verbose,
                        **method_params,
                    )

                    rss = loss[-1]
                    if i == 0 or rss < best_rss:
                        best_rss = rss
                        A_ = A; B_ = B; archetypes_ = archetypes
                        n_iter_ = n_iter; loss_ = loss

        self.A_ = A_
        self.B_ = B_
        self.archetypes_ = archetypes_
        self.n_iter_ = n_iter_
        self.loss_ = loss_
        self.rss_ = best_rss

        self.coefficients_ = self.A_
        self.arch_coefficients_ = self.B_
        self.n_archetypes_ = self.B_.shape[0]
        self.labels_ = np.argmax(self.A_, axis=1)
        self.reconstruction_error_ = self.rss_

        return self.A_
