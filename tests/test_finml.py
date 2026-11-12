import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from finml import factors, regime, tracking

rng = np.random.default_rng(0)
T, N = 800, 10
mkt = rng.normal(0, 0.01, T)
R = pd.DataFrame(mkt[:, None] * rng.uniform(0.8, 1.2, N) + rng.normal(0, 0.005, (T, N)),
                 columns=[f"S{i}" for i in range(N)], index=pd.bdate_range("2020-01-01", periods=T))


def test_pca_first_component_is_market_and_reconstruction():
    m = factors.fit_pca(R)
    assert m.explained.iloc[0] > 0.6
    assert (m.loadings["PC1"] > 0).all()
    assert np.allclose(m.reconstruct(N).values, R.values, atol=1e-10)
    assert m.explained.sum() == pytest.approx(1.0)


def test_absorption_ratio_bounds():
    ar = factors.absorption_ratio(R, window=200)
    assert ((ar > 0) & (ar <= 1)).all()


def test_lasso_recovers_sparse_index():
    true_w = pd.Series(0.0, index=R.columns); true_w[["S1", "S4", "S7"]] = [0.5, 0.3, 0.2]
    idx = R @ true_w
    w = tracking.lasso_weights(R, idx, alpha=1e-7)
    assert set(w[w > 0.01].index) == {"S1", "S4", "S7"}
    assert tracking.evaluate(w, R, idx)["tracking error"] < 1e-3


def test_weights_for_n_names_is_sparse():
    idx = R.mean(axis=1)
    w, _ = tracking.weights_for_n_names(R, idx, 4)
    assert 4 <= (w > 0).sum() <= N and w.sum() == pytest.approx(1.0)


def test_walk_forward_never_uses_future():
    X = pd.DataFrame({"x": rng.normal(size=200)}, index=pd.date_range("2000-01-31", periods=200, freq="ME"))
    y = pd.Series((rng.normal(size=200) > 0).astype(float), index=X.index)
    p = regime.expanding_walk_forward(LogisticRegression(), X, y, min_train=60, step=12)
    assert p.index[0] == X.index[60] and len(p) == 140
    assert set(regime.scores(p, y)) >= {"accuracy", "AUC"}


def test_two_stage_tracker_beats_plain_lasso():
    idx = R.mean(axis=1)
    w2, _ = tracking.sparse_tracking_portfolio(R, idx, 5)
    w1, _ = tracking.weights_for_n_names(R, idx, 5)
    assert (w2 > 0).sum() <= 5 and w2.sum() == pytest.approx(1.0)
    assert tracking.evaluate(w2, R, idx)["tracking error"] <= tracking.evaluate(w1, R, idx)["tracking error"] + 1e-9


def test_oos_r2_and_no_lookahead():
    from sklearn.linear_model import LinearRegression
    from finml import timing
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    x = pd.Series(rng.normal(size=240), idx)
    y = 0.5 * x + rng.normal(0, 0.1, 240)
    X = x.to_frame("x")
    f = timing.expanding_forecast(X, y, LinearRegression, start=60, refit=12)
    hm = timing.historical_mean_forecast(y, 60)
    assert f.iloc[:60].isna().all() and f.iloc[60:].notna().all()
    assert timing.oos_r2(y, f, hm) > 0.9
    assert timing.oos_r2(y, hm, hm) == pytest.approx(0.0)
    # changing a future value must not change earlier forecasts
    y2 = y.copy(); y2.iloc[-1] += 100
    f2 = timing.expanding_forecast(X, y2, LinearRegression, start=60, refit=12)
    assert np.allclose(f.iloc[:-1], f2.iloc[:-1], equal_nan=True)


def test_timing_weights_bounds():
    from finml import timing
    w = timing.timing_weights(pd.Series([-0.01, 0.0, 0.01, 1.0]), pd.Series([0.002] * 4), cap=2.0)
    assert list(w) == [0.0, 0.0, pytest.approx(0.01 / 0.006), 2.0]
