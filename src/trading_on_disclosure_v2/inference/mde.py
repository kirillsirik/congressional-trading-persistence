"""Dependence-aware minimum detectable effects (PWR-001).

MDEs derive only from an actual dependence-aware standard error or
covariance-with-contrast; there is deliberately no raw-N interface, because
row counts overstate the effective sample (05 spec section 12.1). Historical
outputs are labeled retrospective design diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

LABEL = "retrospective_design_diagnostic"


def mde_from_se(se: float, alpha: float = 0.05, power: float = 0.80) -> dict:
    if se <= 0:
        raise ValueError("se must be positive")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    z = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    return {
        "mde": float(z * se),
        "se": float(se),
        "alpha": alpha,
        "power": power,
        "label": LABEL,
    }


def mde_from_covariance(
    cov: np.ndarray,
    contrast: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    cov = np.asarray(cov, dtype=float)
    c = np.asarray(contrast, dtype=float)
    var = float(c @ cov @ c)
    if var <= 0:
        raise ValueError("contrast variance must be positive")
    return mde_from_se(float(np.sqrt(var)), alpha=alpha, power=power)
