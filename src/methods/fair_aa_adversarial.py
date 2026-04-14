from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

from archetypes.numpy._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from archetypes.numpy._projection import l1_normalize_proj, unit_simplex_proj


class FairAA_Adversarial(TransformerMixin, BaseEstimator):
    """
    Adversarial Fair Archetypal Analysis.

    Solves the minimax problem::

        min_{A,B} max_{W,b}  ||X - A B X||_F^2  +  lambda_ * sum_i log p(z_i | W @ a_i + b)

    where p(z_i | ...) is the sigmoid (logistic) function.

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes.
    lambda_ : float, default=1.0
        Fairness regularisation weight.
    n_adv_steps : int, default=5
        Adversary gradient-ascent steps per main iteration.
    lr_adv : float, default=0.01
        Adversary learning rate.
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
    W_ : ndarray of shape (1, n_archetypes)  – learned adversary weights
    b_ : float  – learned adversary bias
    loss_history_ : dict with keys 'total', 'reconstruction', 'adversary'
    loss_ : list  – alias for loss_history_['total']
    rss_ : float  – final total loss
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left")],
        "lambda_": [Interval(Real, 0, None, closed="left")],
        "n_adv_steps": [Interval(Integral, 1, None, closed="left")],
        "lr_adv": [Interval(Real, 0, None, closed="neither")],
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
        n_adv_steps=5,
        lr_adv=0.01,
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
        self.n_adv_steps = n_adv_steps
        self.lr_adv = lr_adv
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
        z = np.asarray(z, dtype=X.dtype)
        archetypes = self.archetypes_

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        if self.method == "pgd":
            transform_func = adv_transform
        elif self.method == "pseudo_pgd":
            transform_func = adv_pseudo_transform

        method_params = {} if self.method_params is None else self.method_params
        A = transform_func(
            X,
            z,
            archetypes,
            lambda_=self.lambda_,
            n_adv_steps=self.n_adv_steps,
            lr_adv=self.lr_adv,
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
        z = np.asarray(z, dtype=X.dtype)

        if self.n_archetypes == 1:
            n_samples = X.shape[0]
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            W_ = np.zeros((1, self.n_archetypes), dtype=X.dtype)
            b_ = 0.0
            rss = float(squared_norm(A_ @ archetypes_ - X))
            n_iter_ = 0
            loss_history_ = {"total": [rss], "reconstruction": [rss], "adversary": [0.0]}

        else:
            if self.method == "pgd":
                fit_transform_func = adv_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = adv_pseudo_fit_transform

            method_params = {} if self.method_params is None else self.method_params

            rng = check_random_state(self.random_state)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, archetypes = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = B.copy()
                    self.archetypes_init_ = archetypes.copy()

                A, B, archetypes, W, b, n_iter, loss_history, _ = fit_transform_func(
                    X,
                    A,
                    B,
                    z,
                    archetypes,
                    lambda_=self.lambda_,
                    n_adv_steps=self.n_adv_steps,
                    lr_adv=self.lr_adv,
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
                    W_ = W
                    b_ = b
                    n_iter_ = n_iter
                    loss_history_ = loss_history

        self.A_ = A_
        self.B_ = B_
        self.archetypes_ = archetypes_
        self.W_ = W_
        self.b_ = b_
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


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _log_likelihood(z, A, W, b):
    """Binary log-likelihood of z given A, W, b (clipped for stability)."""
    logits = (A @ W.T + b).flatten()
    sigma = np.clip(_sigmoid(logits), 1e-12, 1 - 1e-12)
    return float(np.sum(z * np.log(sigma) + (1 - z) * np.log(1 - sigma)))


# ── Fit-transform entry points (mirrors pgd_fit_transform / pseudo_pgd_fit_transform) ──


def adv_transform(X, z, archetypes, *, lambda_, n_adv_steps, lr_adv, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    unit_simplex_proj(A)
    A, _, _, _, _, _, _, _ = _adv_optimize_aa(
        X, A, None, z, archetypes,
        lambda_=lambda_, n_adv_steps=n_adv_steps, lr_adv=lr_adv,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=False, **params,
    )
    return A


def adv_pseudo_transform(X, z, archetypes, *, lambda_, n_adv_steps, lr_adv, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    l1_normalize_proj(A)
    A, _, _, _, _, _, _, _ = _adv_optimize_aa(
        X, A, None, z, archetypes,
        lambda_=lambda_, n_adv_steps=n_adv_steps, lr_adv=lr_adv,
        max_iter=max_iter, tol=tol, verbose=False,
        update_B=False, pseudo_pgd=True, **params,
    )
    return A


def adv_fit_transform(X, A, B, z, archetypes, *, lambda_, n_adv_steps, lr_adv, max_iter, tol, verbose, **params):
    return _adv_optimize_aa(
        X, A, B, z, archetypes,
        lambda_=lambda_, n_adv_steps=n_adv_steps, lr_adv=lr_adv,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=False, **params,
    )


def adv_pseudo_fit_transform(X, A, B, z, archetypes, *, lambda_, n_adv_steps, lr_adv, max_iter, tol, verbose, **params):
    return _adv_optimize_aa(
        X, A, B, z, archetypes,
        lambda_=lambda_, n_adv_steps=n_adv_steps, lr_adv=lr_adv,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_B=True, pseudo_pgd=True, **params,
    )


# ── Core optimizer (mirrors _pgd_like_optimize_aa) ────────────────────────────


def _adv_optimize_aa(
    X,
    A,
    B,
    z,
    archetypes,
    *,
    lambda_,
    n_adv_steps,
    lr_adv,
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
    n, k = A.shape

    # precomputing and memory allocation
    BX = archetypes
    XXt = X @ X.T
    ABX = A @ BX
    ABX -= X
    XXtBt = X @ BX.T
    BXXtBt = BX @ BX.T
    AtXXt = A.T @ XXt

    W = np.zeros((1, k), dtype=X.dtype)
    b = 0.0

    A_grad = np.empty_like(A)
    A_new = np.empty_like(A)
    B_grad = np.empty_like(B) if B is not None else None
    B_new = np.empty_like(B) if B is not None else None

    rec0 = float(squared_norm(ABX))
    adv0 = lambda_ * _log_likelihood(z, A, W, b)
    rss = rec0 + adv0

    loss_history = {
        "total": [rss],
        "reconstruction": [rec0],
        "adversary": [adv0],
    }

    step_size_A = step_size
    step_size_B = step_size

    for i in range(1, max_iter + 1):

        # ── Adversary update (gradient ascent on log-likelihood) ──────────────
        for _ in range(n_adv_steps):
            logits = A @ W.T + b                          # (n, 1)
            sigma = _sigmoid(logits)                      # (n, 1)
            grad_W = (z.reshape(-1, 1) - sigma).T @ A    # (1, k)
            grad_b = float((z.reshape(-1, 1) - sigma).sum())
            W = W + lr_adv * grad_W
            b = b + lr_adv * grad_b

        # ── Update A ──────────────────────────────────────────────────────────
        rss, step_size_A = _adv_update_A_inplace(
            X, A, z, W, b, BX, ABX, XXtBt, BXXtBt,
            A_grad, A_new, pseudo_pgd, step_size_A, lambda_,
            max_iter_optimizer, beta, rss,
        )

        # ── Update B ──────────────────────────────────────────────────────────
        if update_B:
            rss, step_size_B = _adv_update_B_inplace(
                X, A, B, z, W, b, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
                B_grad, B_new, pseudo_pgd, step_size_B, lambda_,
                max_iter_optimizer, beta, rss,
            )

        convergence = abs(loss_history["total"][-1] - rss) < tol
        rec = float(squared_norm(A @ BX - X))
        adv = lambda_ * _log_likelihood(z, A, W, b)
        loss_history["total"].append(rss)
        loss_history["reconstruction"].append(rec)
        loss_history["adversary"].append(adv)

        if verbose and i % 10 == 0:
            _verbose_print(max_iter, rss, i)
        if convergence:
            break

    return A, B, archetypes, W, b, i, loss_history, convergence


# ── Inplace update functions (mirror _pgd_like_update_A_inplace / _B_inplace) ─


def _adv_update_A_inplace(
    X, A, z, W, b, BX, ABX, XXtBt, BXXtBt,
    A_grad, A_new, pseudo_pgd, step_size_A, lambda_,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient (identical to FairAA)
    A_grad = np.matmul(A, BXXtBt, out=A_grad)
    A_grad -= XXtBt
    # Adversarial gradient: d/dA [+lambda_ * sum log p(z | W a + b)]
    logits = A @ W.T + b      # (n, 1)
    sigma = _sigmoid(logits)  # (n, 1)
    A_grad += lambda_ * np.outer(z - sigma.flatten(), W.flatten())

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
        rss_new = float(squared_norm(ABX)) + lambda_ * _log_likelihood(z, A_new, W, b)
        improved = rss_new < rss
        if improved:
            step_size_A /= beta
            break
        step_size_A *= beta

    if improved:
        np.copyto(A, A_new)
        rss = rss_new

    return rss, step_size_A


def _adv_update_B_inplace(
    X, A, B, z, W, b, BX, XXt, ABX, AtXXt, XXtBt, BXXtBt,
    B_grad, B_new, pseudo_pgd, step_size_B, lambda_,
    max_iter_optimizer, beta, rss,
):
    # Reconstruction gradient only — adversary does not depend on B
    ABX += X  # now ABX holds A @ BX
    B_grad = np.linalg.multi_dot([A.T, ABX, X.T], out=B_grad)
    B_grad -= np.matmul(A.T, XXt, out=AtXXt)

    if pseudo_pgd:
        B_grad -= np.expand_dims(np.einsum("ij,ij->i", B, B_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    # Precompute adversary term (constant during B update — A is fixed)
    adv_term = lambda_ * _log_likelihood(z, A, W, b)

    improved = False
    for _ in range(max_iter_optimizer):
        B_new = np.multiply(-step_size_B, B_grad, out=B_new)
        B_new += B
        project(B_new)
        ABX = np.linalg.multi_dot([A, B_new, X], out=ABX)
        ABX -= X
        rss_new = float(squared_norm(ABX)) + adv_term
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
