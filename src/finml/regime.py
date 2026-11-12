"""Monthly market/macro features and walk-forward classification of next-month equity returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score


def monthly_features(px: pd.DataFrame) -> pd.DataFrame:
    """Features known at each month-end, from daily prices of SPY, ^VIX, ^TNX, ^IRX, HYG, IEF, GLD, UUP."""
    m = px.resample("ME").last()
    d = px["SPY"].pct_change()
    f = pd.DataFrame(index=m.index)
    for k in (1, 3, 6, 12):
        f[f"spy_mom_{k}m"] = m["SPY"].pct_change(k)
    f["spy_vol_1m"] = d.rolling(21).std().resample("ME").last() * np.sqrt(252)
    f["spy_dd_12m"] = m["SPY"] / m["SPY"].rolling(12).max() - 1
    f["vix"] = m["^VIX"]
    f["vix_chg_1m"] = m["^VIX"].diff()
    f["vix_minus_rv"] = m["^VIX"] / 100 - f["spy_vol_1m"]          # variance risk premium proxy
    f["term_spread"] = (m["^TNX"] - m["^IRX"]) / 100
    f["t10y_chg_3m"] = m["^TNX"].diff(3) / 100
    f["credit_mom_3m"] = (m["HYG"] / m["IEF"]).pct_change(3)        # high yield vs Treasuries
    f["gold_mom_3m"] = m["GLD"].pct_change(3)
    f["usd_mom_3m"] = m["UUP"].pct_change(3)
    return f


def target_next_month(px: pd.DataFrame, asset="SPY") -> pd.Series:
    """1 if next month's return on ``asset`` is positive (label aligned to the decision month-end)."""
    m = px[asset].resample("ME").last()
    return (m.pct_change().shift(-1) > 0).astype(float).where(m.pct_change().shift(-1).notna())


def expanding_walk_forward(model, X: pd.DataFrame, y: pd.Series, min_train=60, step=12) -> pd.Series:
    """Refit every ``step`` months on all data so far; predict P(y=1) for the next block."""
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    Xv, yv = data.drop(columns="y"), data["y"]
    out = pd.Series(np.nan, index=data.index)
    for start in range(min_train, len(data), step):
        m = clone(model).fit(Xv.iloc[:start], yv.iloc[:start])
        out.iloc[start:start + step] = m.predict_proba(Xv.iloc[start:start + step])[:, 1]
    return out.dropna()


def scores(prob: pd.Series, y: pd.Series) -> dict:
    yt = y.loc[prob.index]
    return {"accuracy": accuracy_score(yt, prob > 0.5), "AUC": roc_auc_score(yt, prob),
            "base rate": yt.mean(), "share predicted up": (prob > 0.5).mean()}
