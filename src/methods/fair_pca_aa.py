from numbers import Integral, Real

import numpy as np
from archetypes import AA
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from src.fair_pca import FairPCA


class FairPCA_AA(TransformerMixin, BaseEstimator):
    """
    FairPCA + AA baseline.

    Applies FairPCA dimensionality reduction to obtain a fair representation,
    then runs standard Archetypal Analysis on the projected data::

        X_reduced = FairPCA(X, z)
        min_{A,B}  ||X_reduced - A B X_reduced||_F^2

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes.
    target_dim : int
        Target dimension for FairPCA.
    tradeoff_param : float in [0, 1], default=0
        FairPCA fairness-accuracy tradeoff; 0 = fully fair, 1 = standard PCA.
    standardize : bool, default=False
        Whether FairPCA should standardize the data.
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
    fair_pca_ : FairPCA
        Fitted FairPCA object. Use ``fair_pca_.inverse_transform(A_ @ archetypes_)``
        to reconstruct in original feature space.
    aa_ : AA
        Fitted AA object (operates in the FairPCA-reduced space).
    archetypes_ : ndarray of shape (n_archetypes, reduced_dim)
        Archetypes in FairPCA space.
    coefficients_, A_ : ndarray of shape (n_samples, n_archetypes)
    arch_coefficients_, B_ : ndarray of shape (n_archetypes, n_samples)
    loss_ : list
        Reconstruction loss history in reduced space.
    rss_ : float
        Final reconstruction loss in reduced space.
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left")],
        "target_dim": [Interval(Integral, 1, None, closed="left")],
        "tradeoff_param": [Interval(Real, 0, 1, closed="both")],
        "standardize": [bool],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "init": [
            StrOptions({"uniform", "furthest_sum", "furthest_first", "coreset", "aa_plus_plus"}),
            None,
        ],
        "n_init": [Interval(Integral, 1, None, closed="left")],
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
        target_dim,
        tradeoff_param=0,
        standardize=False,
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
        self.target_dim = target_dim
        self.tradeoff_param = tradeoff_param
        self.standardize = standardize
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
        if self.target_dim >= X.shape[1]:
            raise ValueError(
                f"target_dim={self.target_dim} must be < n_features={X.shape[1]}."
            )

    def fit(self, X, y=None, z=None):
        self.fit_transform(X, y, z)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X_reduced = self.fair_pca_.transform(X)
        return self.aa_.transform(X_reduced)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, z=None, **params):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        z = np.asarray(z, dtype=int)

        # Step 1: Fair dimensionality reduction
        self.fair_pca_ = FairPCA(
            target_dim=self.target_dim,
            standardize=self.standardize,
            tradeoff_param=self.tradeoff_param,
        )
        self.fair_pca_.fit(X, z)
        X_reduced = self.fair_pca_.transform(X)

        # Step 2: Archetypal Analysis on reduced data
        method_params = {} if self.method_params is None else self.method_params
        aa = AA(
            n_archetypes=self.n_archetypes,
            max_iter=self.max_iter,
            tol=self.tol,
            init=self.init,
            n_init=self.n_init,
            init_params=self.init_params,
            save_init=self.save_init,
            method=self.method,
            method_params=method_params if method_params else None,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        A = aa.fit_transform(X_reduced)

        self.aa_ = aa
        self.A_ = A
        self.B_ = aa.B_
        self.archetypes_ = aa.archetypes_
        self.n_iter_ = aa.n_iter_
        self.loss_ = aa.loss_
        self.rss_ = aa.rss_

        self.coefficients_ = self.A_
        self.arch_coefficients_ = self.B_
        self.n_archetypes_ = self.B_.shape[0]
        self.labels_ = np.argmax(self.A_, axis=1)
        self.reconstruction_error_ = self.rss_

        return self.A_
