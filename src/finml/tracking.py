"""Sparse index replication with the LASSO."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, lasso_path

TRADING_DAYS = 252


def lasso_weights(stocks: pd.DataFrame, index: pd.Series, alpha: float) -> pd.Series:
    """Long-only LASSO regression of index returns on stock returns, rescaled to sum to 1."""
    m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=50_000).fit(stocks.values, index.values)
    w = pd.Series(m.coef_, index=stocks.columns)
    return w / w.sum() if w.sum() > 0 else w


def select_names(stocks, index, n_names: int, alphas=None) -> tuple[list, float]:
    """Walk the LASSO path from sparse to dense; return the first ``n_names`` stocks to enter."""
    alphas = np.logspace(-7, -2.5, 300)[::-1] if alphas is None else alphas
    _, coefs, _ = lasso_path(stocks.values, index.values, alphas=alphas, positive=True, max_iter=50_000)
    order, seen = [], set()
    for j in range(coefs.shape[1]):
        for i in np.flatnonzero(coefs[:, j] > 1e-12):
            if i not in seen:
                seen.add(i); order.append(i)
        if len(order) >= n_names:
            return list(stocks.columns[order[:n_names]]), float(alphas[j])
    return list(stocks.columns[order]), float(alphas[-1])


def min_te_weights(stocks: pd.DataFrame, index: pd.Series) -> pd.Series:
    """Long-only, fully invested weights that minimise tracking variance (constrained least squares)."""
    X, y = stocks.values, index.values
    n = X.shape[1]
    obj = lambda w: np.mean((X @ w - y) ** 2)
    grad = lambda w: 2 * X.T @ (X @ w - y) / len(y)
    res = minimize(obj, np.full(n, 1 / n), jac=grad, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-14, "maxiter": 1000})
    return pd.Series(res.x, index=stocks.columns).clip(lower=0) / res.x.clip(min=0).sum()


def sparse_tracking_portfolio(stocks, index, n_names: int) -> tuple[pd.Series, float]:
    """Two-stage tracker: LASSO picks ``n_names`` stocks, then weights are refit by minimum tracking error.

    The refit removes the L1 shrinkage bias that distorts LASSO coefficients.
    """
    names, alpha = select_names(stocks, index, n_names)
    return min_te_weights(stocks[names], index), alpha


def weights_for_n_names(stocks, index, n_names: int, alphas=None) -> tuple[pd.Series, float]:
    """Plain LASSO weights (rescaled to sum to 1) at the sparsest alpha selecting >= ``n_names``."""
    names, alpha = select_names(stocks, index, n_names, alphas)
    return lasso_weights(stocks, index, alpha), alpha


def tracking_error(portfolio: pd.Series, index: pd.Series) -> float:
    """Annualised standard deviation of active returns."""
    return float((portfolio - index).std() * np.sqrt(TRADING_DAYS))


def evaluate(weights: pd.Series, stocks: pd.DataFrame, index: pd.Series) -> dict:
    port = stocks[weights.index] @ weights
    active = port - index
    return {"names": int((weights > 0).sum()), "tracking error": tracking_error(port, index),
            "correlation": float(np.corrcoef(port, index)[0, 1]),
            "active return (ann.)": float(active.mean() * TRADING_DAYS),
            "beta": float(np.cov(port, index)[0, 1] / index.var())}
