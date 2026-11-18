"""A6 time-to-detect (plan §7.4): years of n purchases/year needed for 80% power at one-sided 5%.

T = [(z_{1-alpha} + z_{power}) * sigma * sqrt(d) / theta_20]^2 / n, theta_20 = theta_annual / 12.6,
sigma the event-level outcome SD (bp per 20 sessions), d a design effect (variance inflation).
Best-of-H replaces z_{1-alpha} by the Bonferroni z_{1-alpha/H}.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .tau import ANNUALIZED_SCALE

Z_ALPHA_05 = 1.645
Z_POWER_80 = 0.842
THETA_ANNUAL_GRID_PCT = (2.0, 5.0, 10.0, 20.0)
N_PER_YEAR_GRID = (10, 25, 50, 100, 250, 500)


def theta_20_from_annual(theta_annual_bp: float) -> float:
    """Per-20-session effect implied by an annualized-scale effect (divide by 252/20 = 12.6)."""
    return float(theta_annual_bp) / ANNUALIZED_SCALE


def years_to_detect(
    sigma_bp: float, theta_annual_bp: float, n_per_year: float, d: float = 1.0,
    z_alpha: float = Z_ALPHA_05, z_power: float = Z_POWER_80,
) -> float:
    if sigma_bp <= 0 or theta_annual_bp <= 0 or n_per_year <= 0 or d <= 0:
        raise ValueError("sigma, theta, n and d must be positive")
    theta_20 = theta_20_from_annual(theta_annual_bp)
    return float(((z_alpha + z_power) * sigma_bp * np.sqrt(d) / theta_20) ** 2 / n_per_year)


def bonferroni_z(alpha: float, n_households: int) -> float:
    """z_{1 - alpha/H} for the best-of-H search."""
    if n_households < 1:
        raise ValueError("n_households must be positive")
    return float(stats.norm.ppf(1.0 - alpha / n_households))


def years_to_detect_best_of_h(
    sigma_bp: float, theta_annual_bp: float, n_per_year: float, n_households: int, d: float = 1.0,
    alpha: float = 0.05, z_power: float = Z_POWER_80,
) -> float:
    return years_to_detect(sigma_bp, theta_annual_bp, n_per_year, d, bonferroni_z(alpha, n_households), z_power)


def ttd_grid(
    sigma_bp: float, d: float = 1.0, n_households: int | None = None,
    theta_pct: tuple[float, ...] = THETA_ANNUAL_GRID_PCT, n_grid: tuple[int, ...] = N_PER_YEAR_GRID,
) -> pd.DataFrame:
    """Figure-4 table: years to detect by (theta annualized %, purchases/year), with and without best-of-H."""
    rows = []
    for th, n in itertools.product(theta_pct, n_grid):
        theta_bp = th * 100.0
        row = {"theta_annual_pct": th, "theta_20_bp": theta_20_from_annual(theta_bp), "n_per_year": n, "sigma_bp": sigma_bp, "d": d,
               "years": years_to_detect(sigma_bp, theta_bp, n, d)}
        if n_households:
            row["years_best_of_h"] = years_to_detect_best_of_h(sigma_bp, theta_bp, n, n_households, d)
            row["n_households"] = n_households
        rows.append(row)
    return pd.DataFrame(rows)
