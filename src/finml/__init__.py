"""finml - classical machine learning for financial data.

factors   : PCA statistical factor models and the absorption ratio
tracking  : sparse (LASSO) index replication and tracking-error analysis
regime    : monthly macro/market features and walk-forward classifiers for equity regimes
"""
from . import factors, regime, tracking

__all__ = ["factors", "tracking", "regime"]
__version__ = "1.0.0"
