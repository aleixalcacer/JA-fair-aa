from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

from archetypes.numpy._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from archetypes.numpy._projection import l1_normalize_proj, unit_simplex_proj


class FairAA_3Moment(TransformerMixin, BaseEstimator):
    """
    Moment-matching Fair Archetypal Analysis.

    Solves::

        min_{A,B}  ||X - A B X||_F^2  +  fairness_const * R(A; z)

    where the regularizer equates group-conditional moments of the loadings::

        R(A; z) = ||z_c^T A||_F^2
                + alpha_2 * ||z_c^T (A * A)||_F^2
                + alpha_3 * ||z_c^T A^(2)||_F^2

    with z_c = z - mean(z) the centered sensitive attribute, A*A the
    elementwise square, and A^(2) the matrix of pairwise archetype products.

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes.
    fairness_const : float, default=1.0
        Overall fairness regularisation weight.
    alpha_2 : float, default=1.0
        Weight for order-2 per-archetype variance term. 0 disables it.
    alpha_3 : float, default=1.0
        Weight for order-2 cross-archetype covariance term. 0 disables it.
    balance_orders : bool, default=False
        If True, rescale alpha_2 and alpha_3 by k^2 to balance magnitudes.
    rank1_zzT : bool, default=False
        If True, never materialise zzT; use rank-1 products for large n.
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
    loss_history_ : dict
    loss_ : list – alias for loss_history_['total']
    rss_ : float – final total loss
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left")],
        "fairness_const": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "alpha_3": [Interval(Real, 0, None, closed="left")],
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
        fairness_const=1.0,
        alpha_2=1.0,
        alpha_3=1.0,
        balance_orders=False,
        rank1_zzT=False,
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
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.balance_orders = balance_orders
        self.rank1_zzT = rank1_zzT
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

    def _resolve_alphas(self):
        k = self.n_archetypes
        a2 = self.alpha_2
        a3 = self.alpha_3
        if self.balance_orders:
            a2 = a2 * k ** 2
            a3 = a3 * k ** 2
        return a2, a3

    def fit(self, X, y=None, Z=None):
        self.fit_transform(X, y, Z)
        return self

    def transform(self, X, Z=None):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.ascontiguousarray(X)

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        archetypes = self.archetypes_
        a2, a3 = self._resolve_alphas()

        if self.method == "pgd":
            transform_func = moment_transform
        elif self.method == "pseudo_pgd":
            transform_func = moment_pseudo_transform

        method_params = {} if self.method_params is None else self.method_params
        A = transform_func(
            X,
            self.z_c_,
            self.zzT_,
            archetypes,
            fairness_const=self.fairness_const,
            alpha_2=a2,
            alpha_3=a3,
            rank1_zzT=self.rank1_zzT,
            max_iter=self.max_iter,
            tol=self.tol,
            **method_params,
        )
        return A

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, Z=None, **params):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)
        z = np.asarray(Z, dtype=np.float64)

        # Precompute centered z and zzT
        z_c = z - z.mean()
        if self.rank1_zzT:
            zzT = None
        else:
            zzT = z_c[:, None] * z_c[None, :]

        a2, a3 = self._resolve_alphas()

        if self.n_archetypes == 1:
            n_samples = X.shape[0]
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            rss = float(squared_norm(A_ @ archetypes_ - X))
            fair = _moment_loss(A_, z_c, zzT, self.fairness_const, a2, a3, self.rank1_zzT)
            n_iter_ = 0
            loss_history_ = {
                "total": [rss + fair],
                "reconstruction": [rss],
                "fair_total": [fair],
            }
        else:
            if self.method == "pgd":
                fit_transform_func = moment_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = moment_pseudo_fit_transform

            method_params = {} if self.method_params is None else self.method_params
            rng = check_random_state(self.random_state)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, archetypes = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = B.copy()
                    self.archetypes_init_ = archetypes.copy()

                A, B, archetypes, n_iter, loss_history, _ = fit_transform_func(
                    X, A, B, z_c, zzT, archetypes,
                    fairness_const=self.fairness_const,
                    alpha_2=a2,
                    alpha_3=a3,
                    rank1_zzT=self.rank1_zzT,
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

        # Store for transform
        self.z_c_ = z_c
        self.zzT_ = zzT

        return self.A_


# ── Module-level helpers ───────────────────────────────────────────────────────


def _zzT_matmul(z_c, zzT, M, rank1):
    """Compute zzT @ M, either materialised or via rank-1 factorisation."""
    if rank1:
        # z_c: (n,), M: (n, cols)
        # zzT @ M = z_c[:, None] * (z_c @ M)
        return z_c[:, None] * (z_c @ M)[None, :]
    else:
        return zzT @ M


def _moment_regularizer(A, z_c, zzT, alpha_2, alpha_3, rank1):
    """Compute the three moment-matching terms (before weighting by fairness_const)."""
    # Order 1: ||z_c^T A||_F^2
    zTA = z_c @ A  # (k,)
    order1 = float(np.dot(zTA, zTA))

    order2 = 0.0
    if alpha_2 > 0:
        S2 = A * A
        zTS2 = z_c @ S2  # (k,)
        order2 = float(np.dot(zTS2, zTS2))

    order3 = 0.0
    if alpha_3 > 0:
        k = A.shape[1]
        j_idx, jp_idx = np.triu_indices(k, k=1)
        P = A[:, j_idx] * A[:, jp_idx]  # (n, m)
        zTP = z_c @ P  # (m,)
        order3 = float(np.dot(zTP, zTP))

    return order1, order2, order3


def _moment_loss(A, z_c, zzT, fairness_const, alpha_2, alpha_3, rank1):
    """Scalar fairness loss: fairness_const * (order1 + alpha_2*order2 + alpha_3*order3)."""
    o1, o2, o3 = _moment_regularizer(A, z_c, zzT, alpha_2, alpha_3, rank1)
    return fairness_const * (o1 + alpha_2 * o2 + alpha_3 * o3)


def _moment_grad_A(A, z_c, zzT, alpha_2, alpha_3, rank1):
    """Gradient of R(A; z) w.r.t. A (shape (n, k))."""
    # Order 1: 2 * zzT @ A
    grad = 2.0 * _zzT_matmul(z_c, zzT, A, rank1)

    # Order 2: 4 * alpha_2 * (zzT @ (A*A)) * A
    if alpha_2 > 0:
        S2 = A * A
        grad += alpha_2 * 4.0 * _zzT_matmul(z_c, zzT, S2, rank1) * A

    # Order 3: scatter-add for cross-archetype pairs
    if alpha_3 > 0:
        k = A.shape[1]
        j_idx, jp_idx = np.triu_indices(k, k=1)
        P = A[:, j_idx] * A[:, jp_idx]  # (n, m)
        zzT_P = _zzT_matmul(z_c, zzT, P, rank1)  # (n, m)

        grad3 = np.zeros_like(A)
        np.add.at(grad3.T, j_idx, (2.0 * zzT_P * A[:, jp_idx]).T)
        np.add.at(grad3.T, jp_idx, (2.0 * zzT_P * A[:, j_idx]).T)
        grad += alpha_3 * grad3

    return grad


# ── Fit-transform entry points ────────────────────────────────────────────────


def moment_transform(X, z_c, zzT, archetypes, *, fairness_const, alpha_2, alpha_3,
                     rank1_zzT, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    unit_simplex_proj(A)
    A, _, _, _, _, _ = _moment_optimize_aa(
        X, A, None, z_c, zzT, archetypes,
        fairness_const=fairness_const, alpha_2=alpha_2, alpha_3=alpha_3,
        rank1_zzT=rank1_zzT,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=False, **params,
    )
    return A


def moment_pseudo_transform(X, z_c, zzT, archetypes, *, fairness_const, alpha_2, alpha_3,
                            rank1_zzT, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    l1_normalize_proj(A)
    A, _, _, _, _, _ = _moment_optimize_aa(
        X, A, None, z_c, zzT, archetypes,
        fairness_const=fairness_const, alpha_2=alpha_2, alpha_3=alpha_3,
        rank1_zzT=rank1_zzT,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=True, **params,
    )
    return A


def moment_fit_transform(X, A, B, z_c, zzT, archetypes, *, fairness_const, alpha_2, alpha_3,
                         rank1_zzT, max_iter, tol, verbose, **params):
    return _moment_optimize_aa(
        X, A, B, z_c, zzT, archetypes,
        fairness_const=fairness_const, alpha_2=alpha_2, alpha_3=alpha_3,
        rank1_zzT=rank1_zzT,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=False, **params,
    )


def moment_pseudo_fit_transform(X, A, B, z_c, zzT, archetypes, *, fairness_const, alpha_2, alpha_3,
                                rank1_zzT, max_iter, tol, verbose, **params):
    return _moment_optimize_aa(
        X, A, B, z_c, zzT, archetypes,
        fairness_const=fairness_const, alpha_2=alpha_2, alpha_3=alpha_3,
        rank1_zzT=rank1_zzT,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=True, **params,
    )


# ── Core optimizer ────────────────────────────────────────────────────────────


def _moment_optimize_aa(
    X,
    A,
    B,
    z_c,
    zzT,
    archetypes,
    *,
    fairness_const,
    alpha_2,
    alpha_3,
    rank1_zzT,
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
    fair0 = _moment_loss(A, z_c, zzT, fairness_const, alpha_2, alpha_3, rank1_zzT)
    rss = rec0 + fair0

    loss_history = {
        "total": [rss],
        "reconstruction": [rec0],
        "fair_total": [fair0],
    }

    step_size_A = step_size
    step_size_B = step_size

    for i in range(1, max_iter + 1):

        # ── Update A ──────────────────────────────────────────────────────
        rss, step_size_A = _moment_update_A_inplace(
            X, A, z_c, zzT, BX, ABX, XXtBt, BXXtBt,
            A_grad, A_new, pseudo_pgd, step_size_A,
            fairness_const, alpha_2, alpha_3, rank1_zzT,
            max_iter_optimizer, beta, rss,
        )

        # ── Update B ──────────────────────────────────────────────────────
        if update_B:
            rss, step_size_B = _moment_update_B_inplace(
                X, A, B, z_c, zzT, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
                B_grad, B_new, pseudo_pgd, step_size_B,
                fairness_const, alpha_2, alpha_3, rank1_zzT,
                max_iter_optimizer, beta, rss,
            )

        convergence = abs(loss_history["total"][-1] - rss) < tol
        rec = float(squared_norm(A @ BX - X))
        fair = _moment_loss(A, z_c, zzT, fairness_const, alpha_2, alpha_3, rank1_zzT)
        loss_history["total"].append(rss)
        loss_history["reconstruction"].append(rec)
        loss_history["fair_total"].append(fair)

        if verbose and i % 10 == 0:
            print(f"Iteration {i}/{max_iter}: loss = {rss}")
        if convergence:
            break

    return A, B, archetypes, i, loss_history, convergence


# ── Inplace update functions ──────────────────────────────────────────────────


def _moment_update_A_inplace(
    X, A, z_c, zzT, BX, ABX, XXtBt, BXXtBt,
    A_grad, A_new, pseudo_pgd, step_size_A,
    fairness_const, alpha_2, alpha_3, rank1_zzT,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient
    A_grad = np.matmul(A, BXXtBt, out=A_grad)
    A_grad -= XXtBt
    # Moment-matching gradient
    A_grad += fairness_const * _moment_grad_A(A, z_c, zzT, alpha_2, alpha_3, rank1_zzT)

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
        rss_new = float(squared_norm(ABX)) + _moment_loss(
            A_new, z_c, zzT, fairness_const, alpha_2, alpha_3, rank1_zzT
        )
        improved = rss_new < rss
        if improved:
            step_size_A = min(step_size_A / beta, 1e6)
            break
        step_size_A *= beta

    if improved:
        np.copyto(A, A_new)
        rss = rss_new

    return rss, step_size_A


def _moment_update_B_inplace(
    X, A, B, z_c, zzT, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
    B_grad, B_new, pseudo_pgd, step_size_B,
    fairness_const, alpha_2, alpha_3, rank1_zzT,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient only — moment regularizer does not depend on B
    ABX += X  # now ABX holds A @ BX
    B_grad = np.linalg.multi_dot([A.T, ABX, X.T], out=B_grad)
    B_grad -= np.matmul(A.T, XXt, out=AtXXt)

    if pseudo_pgd:
        B_grad -= np.expand_dims(np.einsum("ij,ij->i", B, B_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    # Moment term is constant during B update (A is fixed)
    moment_term = _moment_loss(A, z_c, zzT, fairness_const, alpha_2, alpha_3, rank1_zzT)

    improved = False
    for _ in range(max_iter_optimizer):
        B_new = np.multiply(-step_size_B, B_grad, out=B_new)
        B_new += B
        project(B_new)
        ABX = np.linalg.multi_dot([A, B_new, X], out=ABX)
        ABX -= X
        rss_new = float(squared_norm(ABX)) + moment_term
        improved = rss_new < rss
        if improved:
            step_size_B = min(step_size_B / beta, 1e6)
            break
        step_size_B *= beta

    if improved:
        np.copyto(B, B_new)
        BX = np.matmul(B, X, out=BX)
        XXtBt = np.matmul(X, BX.T, out=XXtBt)
        BXXtBt = np.matmul(B, XXtBt, out=BXXtBt)
        rss = rss_new

    return rss, step_size_B
