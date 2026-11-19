"""Estimators from BCH (2014). Each returns a dict with alpha, se and the controls used."""

import numpy as np

from .ols import ols
from .rlasso import rlasso


def _fit(y, d, X, cols, intercept, **se):
    Z = [np.asarray(d, float)[:, None]] + [np.ones((len(y), 1))] * intercept + [X[:, cols]]
    b, s = ols(y, np.hstack(Z), **se)
    return {"alpha": b[0], "se": s[0], "selected": np.asarray(cols, int)}


def no_controls(y, d, intercept=True, **se):
    return _fit(y, d, np.empty((len(y), 0)), [], intercept, **se)


def all_controls(y, d, X, intercept=True, **se):
    X = np.asarray(X, float)
    return _fit(y, d, X, np.arange(X.shape[1]), intercept, **se)


def single_selection(y, d, X, intercept=True, c=1.1, gamma=0.05, **se):
    """Lasso of y on (d, X) with d unpenalized: partial d out, then select (Frisch-Waugh)."""
    y, d, X = (np.asarray(a, float) for a in (y, d, X))
    W = np.column_stack([np.ones_like(d), d]) if intercept else d[:, None]
    P = W @ np.linalg.pinv(W)
    sel = rlasso(X - P @ X, y - P @ y, c, gamma, center=False)
    return _fit(y, d, X, sel, intercept, **se)


def double_selection(y, d, X, intercept=True, c=1.1, gamma=0.05, force=(), **se):
    """Post-double-selection; `force` adds controls to the union (the paper's PDS+)."""
    y, d, X = (np.asarray(a, float) for a in (y, d, X))
    I_y = rlasso(X, y, c, gamma, center=intercept)
    I_d = rlasso(X, d, c, gamma, center=intercept)
    out = _fit(y, d, X, np.union1d(np.union1d(I_y, I_d), np.asarray(force, int)), intercept, **se)
    return {**out, "I_y": I_y, "I_d": I_d}
