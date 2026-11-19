"""OLS with state-clustered (CR1) or HC3 standard errors."""

import numpy as np


def ols(y, X, se_type="hc3", groups=None):
    """Return (coefficients, standard errors)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    B = np.linalg.pinv(X.T @ X)
    b = B @ X.T @ y
    e = y - X @ b
    if se_type == "cluster":
        G = np.unique(groups)
        S = [X[groups == g].T @ e[groups == g] for g in G]
        V = B @ sum(np.outer(s, s) for s in S) @ B * (len(G) / (len(G) - 1) * (n - 1) / (n - k))
    else:
        h = np.einsum("ij,jk,ik->i", X, B, X)
        V = B @ (X.T * (e / np.clip(1 - h, 1e-8, None)) ** 2) @ X @ B
    return b, np.sqrt(np.clip(np.diag(V), 0, None))
