"""Principal-component factor models for asset returns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAModel:
    loadings: pd.DataFrame          # assets x components (eigenvectors)
    explained: pd.Series            # explained variance ratio per component
    factor_returns: pd.DataFrame    # dates x components
    mean: pd.Series
    std: pd.Series

    def reconstruct(self, k: int) -> pd.DataFrame:
        """Returns rebuilt from the first k factors (in original units)."""
        L = self.loadings.iloc[:, :k]
        z = self.factor_returns.iloc[:, :k] @ L.T
        return z * self.std + self.mean


def fit_pca(returns: pd.DataFrame, n_components: int | None = None, standardize=True) -> PCAModel:
    """PCA on (optionally standardised) returns; signs fixed so loadings sum positive."""
    r = returns.dropna()
    mu, sd = r.mean(), (r.std() if standardize else pd.Series(1.0, index=r.columns))
    z = (r - mu) / sd
    pca = PCA(n_components=n_components).fit(z.values)
    comps = pca.components_.copy()
    signs = np.sign(comps.sum(axis=1)); signs[signs == 0] = 1
    comps *= signs[:, None]
    names = [f"PC{i + 1}" for i in range(comps.shape[0])]
    loadings = pd.DataFrame(comps.T, index=r.columns, columns=names)
    f = pd.DataFrame(z.values @ comps.T, index=r.index, columns=names)
    return PCAModel(loadings, pd.Series(pca.explained_variance_ratio_, index=names), f, mu, sd)


def absorption_ratio(returns: pd.DataFrame, window=252, k=1) -> pd.Series:
    """Share of total variance explained by the first k PCs on a rolling window (Kritzman et al., 2011).

    Rising values mean markets are more tightly coupled, which historically comes with fragility.
    """
    r = returns.dropna()
    out = {}
    for end in range(window, len(r) + 1, 5):         # weekly step keeps it fast
        ev = np.linalg.eigvalsh(np.corrcoef(r.iloc[end - window:end].values.T))[::-1]
        out[r.index[end - 1]] = ev[:k].sum() / ev.sum()
    return pd.Series(out, name=f"absorption_ratio_k{k}")
