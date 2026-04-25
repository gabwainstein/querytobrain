"""
Utilities for evaluating text-to-brain predictions, including residual correlation for PET evaluation.

When comparing a predicted map to a PET receptor map, raw correlation can be inflated by shared
spatial autocorrelation (dominant gene-expression gradients). Regressing out the top gene-expression
PCs from both maps then correlating the residuals measures pharmacologically specific agreement.
Informed by Fulcher et al. Nat Commun 2021 (https://www.nature.com/articles/s41467-021-22862-1),
who show that spatial autocorrelation inflates associations; residual correlation is our extension.
"""

import numpy as np


def residual_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    gradient_components: np.ndarray,
) -> float:
    """
    Correlation between pred and target after regressing out dominant spatial gradients from both.

    Args:
        pred: (n_parcels,) predicted map.
        target: (n_parcels,) ground-truth map (e.g. PET receptor).
        gradient_components: (K, n_parcels) e.g. abagen_gradient_components.npy from merged cache.

    Returns:
        Pearson r between residual(pred) and residual(target). Use as specificity metric:
        high residual r = model captured target-specific pattern, not just generic gradient.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    G = np.asarray(gradient_components, dtype=np.float64)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    n_parcels = pred.shape[0]
    if target.shape[0] != n_parcels or G.shape[1] != n_parcels:
        return np.nan
    # Regress G out: residual = x - G.T @ (G @ G.T)^{-1} @ G @ x
    # With G (K, n_parcels): coef = (G @ G.T)^{-1} @ G @ x  (K,), residual = x - G.T @ coef
    GGT = G @ G.T
    if np.linalg.cond(GGT) > 1e10:
        return np.nan
    coef_p = np.linalg.solve(GGT, G @ pred)
    coef_t = np.linalg.solve(GGT, G @ target)
    res_p = pred - G.T @ coef_p
    res_t = target - G.T @ coef_t
    if np.std(res_p) < 1e-12 or np.std(res_t) < 1e-12:
        return np.nan
    return float(np.corrcoef(res_p, res_t)[0, 1])
