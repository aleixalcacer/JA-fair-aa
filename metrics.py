from sklearn import metrics

def mmd_rbf(X, Y, gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Parameters
    ----------
    
    X : array, shape (n_samples_X, n_features)
        First set of samples.
    Y : array, shape (n_samples_Y, n_features)
        Second set of samples.
    gamma : float, default=1.0
        The gamma parameter of the RBF kernel.

    Returns
    -------

    mmd : float
        The MMD between X and Y.
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    
    return XX.mean() + YY.mean() - 2 * XY.mean()