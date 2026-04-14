from sklearn import metrics


def mmd_rbf(X, Y, sigmas=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples
    using a sum of RBF kernels (matches the kernel used in FairAA_MMD).

    Parameters
    ----------
    X : array, shape (n_samples_X, n_features)
        First set of samples.
    Y : array, shape (n_samples_Y, n_features)
        Second set of samples.
    sigmas : list of float, default=[0.1, 1.0, 10.0]
        RBF kernel bandwidths. gamma = 1 / (2 * sigma^2).

    Returns
    -------
    mmd : float
        The MMD^2 between X and Y.
    """
    if sigmas is None:
        sigmas = [0.1, 1.0, 10.0]

    mmd2 = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma ** 2)
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        mmd2 += XX.mean() + YY.mean() - 2 * XY.mean()

    return mmd2
