"""finml - classical machine learning for financial data.

factors   : PCA statistical factor models and the absorption ratio
tracking  : sparse (LASSO) index replication and tracking-error analysis
regime    : monthly macro/market features and walk-forward classifiers for equity regimes
timing    : expanding-window return forecasts, out-of-sample R², factor timing and volatility management
"""
from . import factors, regime, timing, tracking

__all__ = ["factors", "tracking", "regime", "timing"]
__version__ = "1.0.0"
