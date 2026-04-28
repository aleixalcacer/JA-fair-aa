import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score


def explained_variance(X, X_hat):
    """
    Explained Variance of the reconstruction.

    EV = 1 - ||X - X_hat||_F^2 / ||X - mean(X)||_F^2

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Original data matrix.
    X_hat : array, shape (n_samples, n_features)
        Reconstructed data matrix.

    Returns
    -------
    ev : float
        Explained variance in (-inf, 1]; 1 is perfect reconstruction,
        0 matches the global-mean baseline.
    """
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    rss = np.sum((X - X_hat) ** 2)
    tss = np.sum((X - X.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - rss / tss)


def mmd_rbf(X, Y, sigmas=None):
    """
    Squared empirical MMD between X and Y with a (sum of) RBF kernel(s).

    Parameters
    ----------
    X : array, shape (n_samples_X, n_features)
        Loadings of group 0.
    Y : array, shape (n_samples_Y, n_features)
        Loadings of group 1.
    sigmas : None, "median", "median-single", or list of float
        - "median" (default): multi-kernel with {0.5 sigma, sigma, 2 sigma},
          where sigma is the median heuristic on the pooled sample.
        - "median-single": single RBF with sigma from the median heuristic.
        - list: sum of RBF kernels with the given bandwidths.

    Returns
    -------
    mmd2 : float
        Squared MMD; zero iff the two group distributions are
        indistinguishable in the induced RKHS.
    """
    if sigmas is None:
        sigmas = "median"

    if sigmas in ("median", "median-single"):
        pooled = np.vstack([X, Y])
        dists = metrics.pairwise.euclidean_distances(pooled, pooled)
        mask = dists > 0
        sigma_med = float(np.median(dists[mask])) if mask.any() else 1.0
        if sigmas == "median-single":
            sigmas = [sigma_med]
        else:
            sigmas = [0.5 * sigma_med, sigma_med, 2.0 * sigma_med]

    mmd2 = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma ** 2)
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        mmd2 += XX.mean() + YY.mean() - 2 * XY.mean()
    return float(mmd2)


def linear_separability(H, z, n_splits=5, test_size=0.3, random_state=0):
    """
    Recoverability of the sensitive attribute from the loadings under a
    logistic-regression probe, averaged over n_splits stratified splits.

    Parameters
    ----------
    H : array, shape (n_samples, n_archetypes)
        Archetype loadings.
    z : array, shape (n_samples,)
        Binary sensitive attribute.

    Returns
    -------
    balanced_acc : float
        Balanced accuracy on held-out test sets, in [0.5, 1].
    """
    z = np.asarray(z)
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                      random_state=random_state)
    ba_scores = []
    for train_idx, test_idx in splitter.split(H, z):
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(H[train_idx], z[train_idx])
        ba_scores.append(balanced_accuracy_score(
            z[test_idx], clf.predict(H[test_idx])))
    return float(np.mean(ba_scores))


def nonlinear_separability(H, z, n_splits=5, test_size=0.3, random_state=0):
    """
    Recoverability of the sensitive attribute from the loadings under a
    Random Forest (200 trees, scikit-learn defaults) probe, averaged over
    n_splits stratified splits.

    Parameters
    ----------
    H : array, shape (n_samples, n_archetypes)
        Archetype loadings.
    z : array, shape (n_samples,)
        Binary sensitive attribute.

    Returns
    -------
    balanced_acc : float
        Balanced accuracy on held-out test sets, in [0.5, 1].
    """
    z = np.asarray(z)
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                      random_state=random_state)
    ba_scores = []
    for train_idx, test_idx in splitter.split(H, z):
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(H[train_idx], z[train_idx])
        ba_scores.append(balanced_accuracy_score(
            z[test_idx], rf.predict(H[test_idx])))
    return float(np.mean(ba_scores))


def demographic_parity(H, z):
    """
    Mean per-archetype absolute difference of group-wise mean loadings:

    DP = (1/k) * sum_j | bar_s_j^(0) - bar_s_j^(1) |

    Returns
    -------
    dp : float
        Demographic parity in [0, 1]; 0 is perfect parity.
    """
    z = np.asarray(z)
    mean_0 = H[z == 0].mean(axis=0)
    mean_1 = H[z == 1].mean(axis=0)
    return float(np.mean(np.abs(mean_0 - mean_1)))
 