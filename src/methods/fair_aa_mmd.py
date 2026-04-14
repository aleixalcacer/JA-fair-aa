from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

from archetypes.numpy._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from archetypes.numpy._projection import l1_normalize_proj, unit_simplex_proj


class FairAA_MMD(TransformerMixin, BaseEstimator):
    """
    MMD Fair Archetypal Analysis.

    Solves::

        min_{A,B}  ||X - A B X||_F^2  +  lambda_ * MMD^2(A[z=0], A[z=1])

    where MMD is computed with a mixture of RBF kernels::

        k(s_i, s_j) = sum_l exp(-||s_i - s_j||^2 / (2 * sigma_l^2))

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes.
    lambda_ : float, default=1.0
        Fairness regularisation weight.
    sigmas : list of float, default=[0.1, 1.0, 10.0]
        RBF kernel bandwidths.
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
    loss_history_ : dict with keys 'total', 'reconstruction', 'mmd'
    loss_ : list  – alias for loss_history_['total']
    rss_ : float  – final total loss
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left")],
        "lambda_": [Interval(Real, 0, None, closed="left")],
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
        lambda_=1.0,
        sigmas=None,
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
        self.lambda_ = lambda_
        self.sigmas = sigmas
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

    def fit(self, X, y=None, z=None):
        self.fit_transform(X, y, z)
        return self

    def transform(self, X, z):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.ascontiguousarray(X)
        z = np.asarray(z, dtype=np.int32)
        archetypes = self.archetypes_
        sigmas = self.sigmas if self.sigmas is not None else [0.1, 1.0, 10.0]

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        if self.method == "pgd":
            transform_func = mmd_transform
        elif self.method == "pseudo_pgd":
            transform_func = mmd_pseudo_transform

        method_params = {} if self.method_params is None else self.method_params
        A = transform_func(
            X,
            z,
            archetypes,
            lambda_=self.lambda_,
            sigmas=sigmas,
            max_iter=self.max_iter,
            tol=self.tol,
            **method_params,
        )
        return A

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, z=None, **params):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)
        z = np.asarray(z, dtype=np.int32)
        sigmas = self.sigmas if self.sigmas is not None else [0.1, 1.0, 10.0]

        if self.n_archetypes == 1:
            n_samples = X.shape[0]
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            rss = float(squared_norm(A_ @ archetypes_ - X))
            mmd = float(self.lambda_ * _mmd_squared(A_[z == 0], A_[z == 1], sigmas))
            n_iter_ = 0
            loss_history_ = {
                "total": [rss + mmd],
                "reconstruction": [rss],
                "mmd": [mmd],
            }

        else:
            if self.method == "pgd":
                fit_transform_func = mmd_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = mmd_pseudo_fit_transform

            method_params = {} if self.method_params is None else self.method_params

            rng = check_random_state(self.random_state)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, archetypes = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = B.copy()
                    self.archetypes_init_ = archetypes.copy()

                A, B, archetypes, n_iter, loss_history, _ = fit_transform_func(
                    X,
                    A,
                    B,
                    z,
                    archetypes,
                    lambda_=self.lambda_,
                    sigmas=sigmas,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    verbose=self.verbose,
                    **method_params,
                )

                rss = loss_history["total"][-1]
                if i == 0 or rss < best_rss:
                    best_rss = rss
                    A_ = A
                    B_ = B
                    archetypes_ = archetypes
                    n_iter_ = n_iter
                    loss_history_ = loss_history

        self.A_ = A_
        self.B_ = B_
        self.archetypes_ = archetypes_
        self.n_iter_ = n_iter_
        self.loss_history_ = loss_history_
        self.loss_ = loss_history_["total"]
        self.rss_ = loss_history_["total"][-1]

        self.coefficients_ = self.A_
        self.arch_coefficients_ = self.B_
        self.n_archetypes_ = self.B_.shape[0]
        self.labels_ = np.argmax(self.A_, axis=1)
        self.reconstruction_error_ = self.rss_

        return self.A_


# ── Module-level helpers ───────────────────────────────────────────────────────


def rbf_kernel_matrix(A, B, sigma):
    """Compute RBF kernel matrix between rows of A (n, k) and B (m, k) -> (n, m)."""
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)
    dist2 = np.maximum(sq_A - 2 * A @ B.T + sq_B.T, 0.0)
    return np.exp(-dist2 / (2 * sigma ** 2))


def _mmd_squared(S0, S1, sigmas):
    """MMD^2 between two sample sets using a sum of RBF kernels."""
    n0, n1 = len(S0), len(S1)
    if n0 == 0 or n1 == 0:
        return 0.0
    mmd2 = 0.0
    for sigma in sigmas:
        K00 = rbf_kernel_matrix(S0, S0, sigma)
        K11 = rbf_kernel_matrix(S1, S1, sigma)
        K01 = rbf_kernel_matrix(S0, S1, sigma)
        mmd2 += K00.sum() / n0 ** 2 - 2 * K01.sum() / (n0 * n1) + K11.sum() / n1 ** 2
    return mmd2


def _mmd_loss(A, z, lambda_, sigmas):
    mask0 = z == 0
    return float(lambda_ * _mmd_squared(A[mask0], A[~mask0], sigmas))


def _mmd_grad_A(A, z, sigmas):
    """Gradient of MMD^2 w.r.t. A (vectorized over samples within each group)."""
    mask0 = z == 0
    S0, S1 = A[mask0], A[~mask0]
    n0, n1 = len(S0), len(S1)
    if n0 == 0 or n1 == 0:
        return np.zeros_like(A)
    idx0 = np.where(mask0)[0]
    idx1 = np.where(~mask0)[0]

    grad0 = np.zeros_like(S0)
    grad1 = np.zeros_like(S1)

    for sigma in sigmas:
        K00 = rbf_kernel_matrix(S0, S0, sigma)  # (n0, n0)
        K11 = rbf_kernel_matrix(S1, S1, sigma)  # (n1, n1)
        K01 = rbf_kernel_matrix(S0, S1, sigma)  # (n0, n1)
        s2 = sigma ** 2

        # Gradient for group-0 samples
        grad0 += (-1.0 / s2) * (
            (2.0 / n0 ** 2) * (K00.sum(axis=1, keepdims=True) * S0 - K00 @ S0)
            - (2.0 / (n0 * n1)) * (K01.sum(axis=1, keepdims=True) * S0 - K01 @ S1)
        )

        # Gradient for group-1 samples (K10 = K01.T)
        K10 = K01.T  # (n1, n0)
        grad1 += (-1.0 / s2) * (
            (2.0 / n1 ** 2) * (K11.sum(axis=1, keepdims=True) * S1 - K11 @ S1)
            - (2.0 / (n0 * n1)) * (K10.sum(axis=1, keepdims=True) * S1 - K10 @ S0)
        )

    grad = np.zeros_like(A)
    grad[idx0] = grad0
    grad[idx1] = grad1
    return grad


# ── Fit-transform entry points (mirrors pgd_fit_transform / pseudo_pgd_fit_transform) ──


def mmd_transform(X, z, archetypes, *, lambda_, sigmas, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    unit_simplex_proj(A)
    A, _, _, _, _, _ = _mmd_optimize_aa(
        X, A, None, z, archetypes,
        lambda_=lambda_, sigmas=sigmas,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=False, **params,
    )
    return A


def mmd_pseudo_transform(X, z, archetypes, *, lambda_, sigmas, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    l1_normalize_proj(A)
    A, _, _, _, _, _ = _mmd_optimize_aa(
        X, A, None, z, archetypes,
        lambda_=lambda_, sigmas=sigmas,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=True, **params,
    )
    return A


def mmd_fit_transform(X, A, B, z, archetypes, *, lambda_, sigmas, max_iter, tol, verbose, **params):
    return _mmd_optimize_aa(
        X, A, B, z, archetypes,
        lambda_=lambda_, sigmas=sigmas,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=False, **params,
    )


def mmd_pseudo_fit_transform(X, A, B, z, archetypes, *, lambda_, sigmas, max_iter, tol, verbose, **params):
    return _mmd_optimize_aa(
        X, A, B, z, archetypes,
        lambda_=lambda_, sigmas=sigmas,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=True, **params,
    )


# ── Core optimizer (mirrors _pgd_like_optimize_aa) ────────────────────────────


def _mmd_optimize_aa(
    X,
    A,
    B,
    z,
    archetypes,
    *,
    lambda_,
    sigmas,
    max_iter,
    tol,
    verbose=False,
    update_B=True,
    pseudo_pgd=False,
    step_size=1.0,
    max_iter_optimizer=10,
    beta=0.5,
    **params,
):
    # precomputing and memory allocation
    BX = archetypes
    XXt = X @ X.T
    ABX = A @ BX
    ABX -= X
    XXtBt = X @ BX.T
    BXXtBt = BX @ BX.T
    AtXXt = A.T @ XXt

    A_grad = np.empty_like(A)
    A_new = np.empty_like(A)
    B_grad = np.empty_like(B) if B is not None else None
    B_new = np.empty_like(B) if B is not None else None

    rec0 = float(squared_norm(ABX))
    mmd0 = _mmd_loss(A, z, lambda_, sigmas)
    rss = rec0 + mmd0

    loss_history = {
        "total": [rss],
        "reconstruction": [rec0],
        "mmd": [mmd0],
    }

    step_size_A = step_size
    step_size_B = step_size

    for i in range(1, max_iter + 1):

        # ── Update A ──────────────────────────────────────────────────────────
        rss, step_size_A = _mmd_update_A_inplace(
            X, A, z, BX, ABX, XXtBt, BXXtBt,
            A_grad, A_new, pseudo_pgd, step_size_A, lambda_, sigmas,
            max_iter_optimizer, beta, rss,
        )

        # ── Update B ──────────────────────────────────────────────────────────
        if update_B:
            rss, step_size_B = _mmd_update_B_inplace(
                X, A, B, z, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
                B_grad, B_new, pseudo_pgd, step_size_B, lambda_, sigmas,
                max_iter_optimizer, beta, rss,
            )

        convergence = abs(loss_history["total"][-1] - rss) < tol
        rec = float(squared_norm(A @ BX - X))
        mmd = _mmd_loss(A, z, lambda_, sigmas)
        loss_history["total"].append(rss)
        loss_history["reconstruction"].append(rec)
        loss_history["mmd"].append(mmd)

        if verbose and i % 10 == 0:
            _verbose_print(max_iter, rss, i)
        if convergence:
            break

    return A, B, archetypes, i, loss_history, convergence


# ── Inplace update functions (mirror _pgd_like_update_A_inplace / _B_inplace) ─


def _mmd_update_A_inplace(
    X, A, z, BX, ABX, XXtBt, BXXtBt,
    A_grad, A_new, pseudo_pgd, step_size_A, lambda_, sigmas,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient (identical to FairAA)
    A_grad = np.matmul(A, BXXtBt, out=A_grad)
    A_grad -= XXtBt
    # MMD gradient: kernel matrices recomputed each call (S changes every iteration)
    A_grad += lambda_ * _mmd_grad_A(A, z, sigmas)

    if pseudo_pgd:
        A_grad -= np.expand_dims(np.einsum("ij,ij->i", A, A_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    improved = False
    for _ in range(max_iter_optimizer):
        A_new = np.multiply(-step_size_A, A_grad, out=A_new)
        A_new += A
        project(A_new)
        ABX = np.matmul(A_new, BX, out=ABX)
        ABX -= X
        rss_new = float(squared_norm(ABX)) + _mmd_loss(A_new, z, lambda_, sigmas)
        improved = rss_new < rss
        if improved:
            step_size_A /= beta
            break
        step_size_A *= beta

    if improved:
        np.copyto(A, A_new)
        rss = rss_new

    return rss, step_size_A


def _mmd_update_B_inplace(
    X, A, B, z, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
    B_grad, B_new, pseudo_pgd, step_size_B, lambda_, sigmas,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient only — MMD does not depend on B
    ABX += X  # now ABX holds A @ BX
    B_grad = np.linalg.multi_dot([A.T, ABX, X.T], out=B_grad)
    B_grad -= np.matmul(A.T, XXt, out=AtXXt)

    if pseudo_pgd:
        B_grad -= np.expand_dims(np.einsum("ij,ij->i", B, B_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    # Precompute MMD term (constant during B update — A is fixed)
    mmd_term = _mmd_loss(A, z, lambda_, sigmas)

    improved = False
    for _ in range(max_iter_optimizer):
        B_new = np.multiply(-step_size_B, B_grad, out=B_new)
        B_new += B
        project(B_new)
        ABX = np.linalg.multi_dot([A, B_new, X], out=ABX)
        ABX -= X
        rss_new = float(squared_norm(ABX)) + mmd_term
        improved = rss_new < rss
        if improved:
            step_size_B /= beta
            break
        step_size_B *= beta

    if improved:
        np.copyto(B, B_new)
        BX = np.matmul(B, X, out=BX)
        XXtBt = np.matmul(X, BX.T, out=XXtBt)
        BXXtBt = np.matmul(B, XXtBt, out=BXXtBt)
        rss = rss_new

    return rss, step_size_B


def _verbose_print(max_iter, loss, i):
    print(f"Iteration {i}/{max_iter}: loss = {loss}")
