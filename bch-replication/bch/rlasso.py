"""Rigorous Lasso (BCH 2014, eq. 2.12-2.13 and Appendix A). Used only to select controls."""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Lasso


def penalty_level(n, p, c=1.1, gamma=0.05):
    return 2 * c * np.sqrt(n) * norm.ppf(1 - gamma / (2 * p))


def _lasso(X, y, lam, psi):
    # BCH objective rewritten for sklearn: rescale columns by the loadings.
    psi = np.maximum(psi, 1e-12)
    fit = Lasso(alpha=lam / (2 * len(y)), fit_intercept=False, max_iter=100_000, tol=1e-8)
    return fit.fit(X / psi, y).coef_ / psi


def _loadings(X, e):
    return np.sqrt(np.mean(X**2 * e[:, None] ** 2, axis=0))


def rlasso(X, y, c=1.1, gamma=0.05, max_iter=15, tol=1e-5, n_init=5, center=True):
    """Return the indices of the selected columns of X."""
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    if center:
        X, y = X - X.mean(0), y - y.mean()
    n, p = X.shape
    lam = penalty_level(n, p, c, gamma)

    # Start from residuals on the n_init columns most correlated with y (as in hdm).
    sd = X.std(0)
    top = np.argsort(np.abs(X.T @ y) / np.where(sd > 0, sd, np.inf))[::-1][:n_init]
    e = y - X[:, top] @ np.linalg.lstsq(X[:, top], y, rcond=None)[0]
    psi = _loadings(X, e)

    for _ in range(max_iter):
        sel = np.flatnonzero(_lasso(X, y, lam, psi))
        e = y - X[:, sel] @ np.linalg.lstsq(X[:, sel], y, rcond=None)[0] if sel.size else y
        new = _loadings(X, e) * np.sqrt(n / max(n - sel.size, 1))
        done = np.max(np.abs(new - psi)) <= tol
        psi = new
        if done:
            break
    return np.flatnonzero(_lasso(X, y, lam, psi))
