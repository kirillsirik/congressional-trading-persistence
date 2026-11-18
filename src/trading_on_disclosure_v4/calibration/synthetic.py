"""Fast synthetic two-period panel generator for testing E's estimators and harness without N's modules.

Heterogeneous household activity (Pareto), anchor-date clustering (common date shocks on a
business-day calendar), heavy-tailed idiosyncratic noise (Student-t), optional persistent
household ability theta_h. Outcomes are in basis points. Columns follow e_panel_interface.md.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

PANEL_COLUMNS = ("draw", "event_id", "household_id", "clock", "period", "y_h20", "anchor_date")


def seed_from_name(name: str, base: int = 20260921) -> int:
    """Deterministic integer seed from a test/analysis name (sha256, first 8 hex)."""
    return int(hashlib.sha256(f"{base}|{name}".encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class SyntheticSpec:
    n_households: int = 60
    mean_events_a: float = 15.0
    mean_events_b: float = 10.0
    pareto_shape: float = 2.5
    sigma_bp: float = 900.0
    date_shock_bp: float = 200.0
    tail_df: float = 4.0
    tau_bp: float = 0.0
    activity_slope_bp: float = 0.0
    clock: str = "T"
    period_a: tuple[str, str] = ("2015-01-01", "2020-12-01")
    period_b: tuple[str, str] = ("2021-01-01", "2025-12-01")


def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _t_noise(rng: np.random.Generator, size: int, df: float) -> np.ndarray:
    """Student-t scaled to unit variance (df > 2), Gaussian when df is inf."""
    if not np.isfinite(df):
        return rng.standard_normal(size)
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2.0))


def make_panel(spec: SyntheticSpec, seed: int, draw: int = 0) -> pd.DataFrame:
    """One synthetic panel: y = theta_h + slope*log(1+n_hA) + date shock + heavy-tailed noise."""
    rng = np.random.default_rng([int(seed), int(draw)])
    H = spec.n_households
    intensity = rng.pareto(spec.pareto_shape, size=H) + 1.0
    intensity = intensity / intensity.mean()
    theta = rng.normal(0.0, spec.tau_bp, size=H) if spec.tau_bp > 0 else np.zeros(H)
    cal_a = business_days(*spec.period_a)
    cal_b = business_days(*spec.period_b)
    shock_a = rng.normal(0.0, spec.date_shock_bp, size=len(cal_a))
    shock_b = rng.normal(0.0, spec.date_shock_bp, size=len(cal_b))
    n_a = 1 + rng.poisson(spec.mean_events_a * intensity)
    n_b = 1 + rng.poisson(spec.mean_events_b * intensity)
    parts = []
    for h in range(H):
        for period, n, cal, shock in (("A", n_a[h], cal_a, shock_a), ("B", n_b[h], cal_b, shock_b)):
            # date clustering: households trade in bursts around a few favourite sessions
            centers = rng.integers(0, len(cal), size=max(1, n // 3))
            pos = np.clip(rng.choice(centers, size=n) + rng.integers(-2, 3, size=n), 0, len(cal) - 1)
            y = theta[h] + spec.activity_slope_bp * np.log1p(n_a[h]) + shock[pos]
            y = y + spec.sigma_bp * _t_noise(rng, n, spec.tail_df)
            parts.append(
                pd.DataFrame(
                    {
                        "household_id": f"H{h:04d}",
                        "period": period,
                        "y_h20": y,
                        "anchor_date": cal[pos],
                    }
                )
            )
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "draw", draw)
    out.insert(1, "event_id", [f"{draw}-{i:07d}" for i in range(len(out))])
    out.insert(3, "clock", spec.clock)
    return out[list(PANEL_COLUMNS)]


def iter_panels(n: int, seed: int, spec: SyntheticSpec | None = None) -> Iterator[pd.DataFrame]:
    """Harness interface: yield n independent null (or alternative) panels, draw = 0..n-1."""
    spec = spec or SyntheticSpec()
    for d in range(n):
        yield make_panel(spec, seed, draw=d)


def inject_synthetic(panel: pd.DataFrame, spec: dict, rng: np.random.Generator, y: str = "y_h20") -> pd.DataFrame:
    """Reference implementation of the harness ``inject`` interface (N's inject/ must match it).

    spec keys: kind in {'none','dense','sparse_households','sparse_trades'}; dense: tau_bp;
    sparse_households: pi_h, mu_bp, informed in {'uniform','activity'}; sparse_trades: pi_t, mu_bp.
    Returns a copy with the effect added to y and a column 'theta_h' (per-event true household effect).
    """
    out = panel.copy()
    kind = spec.get("kind", "none")
    hh, codes = np.unique(out["household_id"].to_numpy(), return_inverse=True)
    theta = np.zeros(len(hh))
    if kind == "dense":
        theta = rng.normal(0.0, float(spec["tau_bp"]), size=len(hh))
    elif kind == "sparse_households":
        k = max(1, int(round(float(spec["pi_h"]) * len(hh))))
        if spec.get("informed", "uniform") == "activity":
            n_h = np.bincount(codes, minlength=len(hh)).astype(float)
            informed = rng.choice(len(hh), size=k, replace=False, p=n_h / n_h.sum())
        else:
            informed = rng.choice(len(hh), size=k, replace=False)
        theta[informed] = float(spec["mu_bp"])
    elif kind == "sparse_trades":
        hit = rng.random(len(out)) < float(spec["pi_t"])
        out[y] = out[y].to_numpy() + hit * float(spec["mu_bp"])
    elif kind != "none":
        raise ValueError(f"unknown injection kind {kind!r}")
    out[y] = out[y].to_numpy() + theta[codes]
    out["theta_h"] = theta[codes]
    return out


def truth_from_spec(spec: dict) -> dict:
    """Population targets implied by an injection spec: tau2 (bp^2) = Var_h(theta_h) and mean (bp).

    dense: tau^2. sparse households: p(1-p) mu^2 (uniform selection; under 'activity' selection the
    A1 estimand is the variance of theta residualized on log(1+n_A), which is smaller — the harness
    scores coverage against ``realized_tau2_target`` whenever the panel carries ``theta_h``). sparse
    trades: 0 for a common pi_T; (cv pi_T mu)^2 = (pi_T mu)^2 when pi_h ~ Gamma with CV 1
    (``household_specific``/``pi_mode == 'household'``), because theta_h = pi_h mu.
    """
    kind = spec.get("kind", "none")
    if kind == "dense":
        return {"tau2": float(spec["tau_bp"]) ** 2, "mean": 0.0}
    if kind == "sparse_households":
        p, m = float(spec["pi_h"]), float(spec["mu_bp"])
        return {"tau2": p * (1 - p) * m * m, "mean": p * m}
    if kind == "sparse_trades":
        p, m = float(spec["pi_t"]), float(spec["mu_bp"])
        hh_specific = bool(spec.get("household_specific", False)) or spec.get("pi_mode") == "household"
        cv = float(spec.get("pi_cv", 1.0)) if hh_specific else 0.0
        return {"tau2": (cv * p * m) ** 2, "mean": p * m}
    return {"tau2": 0.0, "mean": 0.0}


def realized_tau2_target(
    panel: pd.DataFrame, weights: str = "ES", y: str = "y_h20", household: str = "household_id",
    period: str = "period", theta: str = "theta_h",
) -> float:
    """What A1 estimates on THIS panel: the w-weighted variance of theta_h residualized on
    [1, log(1+n_hA)] over the two-period households (E[tau2_hat | theta, n] under independent noise)."""
    from ..estimators import tau as tau_mod

    table = tau_mod.household_table(panel, weights, y=y, household=household, period=period)
    th = panel.groupby(household)[theta].first().reindex(table.household_id).to_numpy(dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError(f"{theta} must be finite and constant within household")
    resid = tau_mod._wls_residual(th, tau_mod.activity_design(table.n_a), table.w)
    return tau_mod.var_w(resid, table.w)
