"""Out-of-sample return forecasting and factor timing.

Forecasts are made with an expanding window: a model is refit every `refit`
periods on all data available up to that point and predicts the next period
only. Performance is judged with the Campbell-Thompson (2008) out-of-sample R²
against the expanding historical mean.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def expanding_forecast(X: pd.DataFrame, y: pd.Series, make_model, start: int, refit: int = 12) -> pd.Series:
    """Return one-step-ahead forecasts of y[t] using X[t] (features known at t-1) for t >= start."""
    X, y = X.loc[y.index], y
    preds = pd.Series(np.nan, index=y.index)
    model = None
    for t in range(start, len(y)):
        if model is None or (t - start) % refit == 0:
            model = make_model(); model.fit(X.iloc[:t].to_numpy(), y.iloc[:t].to_numpy())
        preds.iloc[t] = float(model.predict(X.iloc[[t]].to_numpy())[0])
    return preds


def historical_mean_forecast(y: pd.Series, start: int) -> pd.Series:
    f = y.expanding().mean().shift(1)
    f.iloc[:start] = np.nan
    return f


def oos_r2(y: pd.Series, forecast: pd.Series, benchmark: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R²: 1 − SSE(model) / SSE(benchmark), over periods where both exist."""
    m = forecast.notna() & benchmark.notna()
    return 1 - ((y[m] - forecast[m]) ** 2).sum() / ((y[m] - benchmark[m]) ** 2).sum()


def timing_weights(forecast: pd.Series, var_forecast: pd.Series, risk_aversion: float = 3.0, cap: float = 2.0) -> pd.Series:
    """Mean-variance exposure w = forecast / (γ·variance), clipped to [0, cap]."""
    return (forecast / (risk_aversion * var_forecast)).clip(0, cap)


def vol_managed(returns: pd.Series, realised_var: pd.Series) -> pd.Series:
    """Moreira-Muir (2017) volatility-managed return: scale by 1/lagged variance, normalised to the unmanaged volatility (full-sample constant)."""
    raw = returns / realised_var.shift(1)
    c = returns.std() / raw.std()
    return c * raw
