"""A1 headline estimator: tau^2 = Var(theta_h) from the two-period covariance design (plan §5).

x_h is the w-weighted residual of Ybar_hA on [1, log(1+n_hA)]; b_hat is the event-level WLS slope
of period-B outcomes on x_h (control log(1+n_hA), weights w_h/n_hB); tau2_hat = b_hat * Var_w(x).

Non-finite outcomes: events whose ``y`` is NaN/inf are dropped ONCE, before any household table or
design is built (``finite_outcome_rows``); the count travels in ``HouseholdTable.n_dropped_nonfinite``
and ``TauResult.n_dropped_nonfinite`` so no estimator or bootstrap shrinks silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

WEIGHTS = ("EH", "ES", "PW")
ANNUALIZED_SCALE = 252.0 / 20.0
PERIOD_A = "A"
PERIOD_B = "B"


@dataclass(frozen=True)
class HouseholdTable:
    """Household-level sufficient statistics for the A1 design."""

    household_id: np.ndarray
    n_a: np.ndarray
    n_b: np.ndarray
    ybar_a: np.ndarray
    ybar_b: np.ndarray
    w: np.ndarray
    x: np.ndarray
    log_n_a: np.ndarray
    n_dropped_nonfinite: int = 0


@dataclass(frozen=True)
class TauResult:
    weights: str
    b_hat: float
    var_wx: float
    tau2_hat: float
    tau_hat: float
    n_households: int
    n_events_a: int
    n_events_b: int
    activity_control: bool
    n_dropped_nonfinite: int = 0


def finite_outcome_rows(panel: pd.DataFrame, y: str = "y_h20") -> tuple[pd.DataFrame, int]:
    """Drop events with a non-finite outcome; returns (kept rows, number dropped). The single NaN policy."""
    yv = pd.to_numeric(panel[y], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(yv)
    n_dropped = int((~ok).sum())
    if n_dropped == 0:
        return panel, 0
    return panel.loc[ok], n_dropped


def household_weights(n_a: np.ndarray, n_b: np.ndarray, weights: str) -> np.ndarray:
    """EH: 1; ES: sqrt(nA*nB); PW: nA*nB."""
    n_a = np.asarray(n_a, dtype=float)
    n_b = np.asarray(n_b, dtype=float)
    if weights == "EH":
        return np.ones_like(n_a)
    if weights == "ES":
        return np.sqrt(n_a * n_b)
    if weights == "PW":
        return n_a * n_b
    raise ValueError(f"unknown weights {weights!r}; expected one of {WEIGHTS}")


def _wls_residual(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return y - X @ beta


def activity_design(n_a: np.ndarray) -> np.ndarray:
    """Regressors [1, log(1+n_hA)] used both to residualize x_h and as the WLS control."""
    return np.column_stack([np.ones(len(n_a)), np.log1p(np.asarray(n_a, dtype=float))])


def residual_design(n_a: np.ndarray, residualize_activity: bool = True) -> np.ndarray:
    """The design x_h is the residual from: [1, log(1+n_hA)] with the activity control, [1] without it
    (then x_h is Ybar_hA minus its w-weighted mean and the estimand is the plain w-weighted covariance
    of the two unresidualized period means)."""
    if residualize_activity:
        return activity_design(n_a)
    return np.ones((len(n_a), 1), dtype=float)


def household_table(
    panel: pd.DataFrame,
    weights: str,
    y: str = "y_h20",
    household: str = "household_id",
    period: str = "period",
    residualize_activity: bool = True,
) -> HouseholdTable:
    """Collapse an event panel to households present in both periods; compute w_h and x_h.

    With ``residualize_activity=False`` x_h is the w-demeaned Ybar_hA (no activity control at all).
    Events with a non-finite outcome are dropped first and counted (``n_dropped_nonfinite``)."""
    panel, n_dropped = finite_outcome_rows(panel, y)
    sub = panel.loc[panel[period].isin([PERIOD_A, PERIOD_B]), [household, period, y]]
    g = sub.groupby([household, period], sort=True)[y].agg(["count", "mean"]).unstack(period)
    if g.empty or set(g["count"].columns) != {PERIOD_A, PERIOD_B}:
        raise ValueError("no household has events in both periods")
    g = g.dropna()
    if g.empty:
        raise ValueError("no household has events in both periods")
    n_a = g[("count", PERIOD_A)].to_numpy(dtype=float)
    n_b = g[("count", PERIOD_B)].to_numpy(dtype=float)
    ybar_a = g[("mean", PERIOD_A)].to_numpy(dtype=float)
    ybar_b = g[("mean", PERIOD_B)].to_numpy(dtype=float)
    w = household_weights(n_a, n_b, weights)
    x = _wls_residual(ybar_a, residual_design(n_a, residualize_activity), w)
    return HouseholdTable(
        household_id=g.index.to_numpy(),
        n_a=n_a,
        n_b=n_b,
        ybar_a=ybar_a,
        ybar_b=ybar_b,
        w=w,
        x=x,
        log_n_a=activity_design(n_a)[:, 1],
        n_dropped_nonfinite=n_dropped,
    )


def var_w(x: np.ndarray, w: np.ndarray) -> float:
    """w-weighted second moment about zero (x is w-mean-zero by construction)."""
    return float(np.sum(w * x * x) / np.sum(w))


def corr_w(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """w-weighted correlation of two household vectors (NaN when either is w-degenerate)."""
    a, b, w = (np.asarray(v, dtype=float) for v in (a, b, w))
    sw = float(np.sum(w))
    da, db = a - float(np.sum(w * a)) / sw, b - float(np.sum(w * b)) / sw
    den = np.sqrt(float(np.sum(w * da * da)) * float(np.sum(w * db * db)))
    return float(np.sum(w * da * db) / den) if den > 0 else np.nan


def event_design(
    panel: pd.DataFrame,
    table: HouseholdTable,
    y: str = "y_h20",
    household: str = "household_id",
    period: str = "period",
    anchor: str = "anchor_date",
) -> pd.DataFrame:
    """Period-B event rows (finite outcomes only) with x_h, log(1+n_hA) and event weight w_h/n_hB attached."""
    panel, _ = finite_outcome_rows(panel, y)
    hh = pd.DataFrame(
        {
            household: table.household_id,
            "x": table.x,
            "log_n_a": table.log_n_a,
            "event_weight": table.w / table.n_b,
        }
    )
    b = panel.loc[panel[period] == PERIOD_B]
    keep = [household, y] + ([anchor] if anchor in b.columns else [])
    out = b[keep].merge(hh, on=household, how="inner")
    return out.rename(columns={y: "y"}).reset_index(drop=True)


def wls_slope(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Coefficients of the w-weighted least-squares fit of y on X."""
    sw = np.sqrt(np.asarray(w, dtype=float))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], np.asarray(y, dtype=float) * sw, rcond=None)
    return beta


def design_matrix(ev: pd.DataFrame, activity_control: bool = True) -> np.ndarray:
    """[1, x_h, log(1+n_hA)] (column 1 is the slope of interest); control optional (robustness)."""
    cols = [np.ones(len(ev)), ev["x"].to_numpy(dtype=float)]
    if activity_control:
        cols.append(ev["log_n_a"].to_numpy(dtype=float))
    return np.column_stack(cols)


def tau2_household_covariance(table: HouseholdTable) -> float:
    """Same estimand from household rows: Cov_w(x_h, Ybar_hB) = sum w x Ybar_B / sum w."""
    return float(np.sum(table.w * table.x * table.ybar_b) / np.sum(table.w))


def signed_root(tau2: float) -> float:
    return float(np.sign(tau2) * np.sqrt(abs(tau2)))


def estimate_tau2(
    panel: pd.DataFrame,
    weights: str = "ES",
    y: str = "y_h20",
    household: str = "household_id",
    period: str = "period",
    activity_control: bool = True,
    residualize_activity: bool = True,
) -> TauResult:
    """A1 point estimate via the event-level WLS slope; tau2 = b_hat * Var_w(x)."""
    table = household_table(panel, weights, y=y, household=household, period=period,
                            residualize_activity=residualize_activity)
    ev = event_design(panel, table, y=y, household=household, period=period)
    beta = wls_slope(ev["y"].to_numpy(), design_matrix(ev, activity_control), ev["event_weight"].to_numpy())
    b_hat = float(beta[1])
    v = var_w(table.x, table.w)
    tau2 = b_hat * v
    return TauResult(
        weights=weights,
        b_hat=b_hat,
        var_wx=v,
        tau2_hat=tau2,
        tau_hat=signed_root(tau2),
        n_households=int(len(table.household_id)),
        n_events_a=int(table.n_a.sum()),
        n_events_b=int(table.n_b.sum()),
        activity_control=activity_control,
        n_dropped_nonfinite=table.n_dropped_nonfinite,
    )


def tau_upper_bound(
    b_hat: float, se_b: float, var_wx: float, dof: int | None = None, level: float = 0.95
) -> dict[str, float]:
    """One-sided upper bound for tau from a t (or normal) bound on b_hat: tau2_up = Var_w(x)*(b_hat + c*se)."""
    if dof is None or not np.isfinite(dof) or dof <= 0:
        crit = float(stats.norm.ppf(level))
    else:
        crit = float(stats.t.ppf(level, df=dof))
    tau2_up = var_wx * (b_hat + crit * se_b)
    tau_up = float(np.sqrt(max(tau2_up, 0.0)))
    return {
        "level": level,
        "crit": crit,
        "tau2_upper": float(tau2_up),
        "tau_upper": tau_up,
        "tau_upper_annualized_scale": tau_up * ANNUALIZED_SCALE,
    }


def tau_ci_from_b_ci(b_low: float, b_high: float, var_wx: float) -> dict[str, float]:
    """CI for tau2 = Var_w(x) x CI(b_hat); tau interval as signed roots (plan §5, I1)."""
    lo, hi = var_wx * b_low, var_wx * b_high
    return {"tau2_low": float(lo), "tau2_high": float(hi), "tau_low": signed_root(lo), "tau_high": signed_root(hi)}


def annualized_scale(tau_bp_per_20: float) -> float:
    """Scale convention only (x 252/20 = 12.6): 'annualized scale', not a portfolio alpha."""
    return float(tau_bp_per_20 * ANNUALIZED_SCALE)


def leave_one_household_out(
    panel: pd.DataFrame, weights: str = "ES", y: str = "y_h20", household: str = "household_id",
    period: str = "period", activity_control: bool = True,
) -> pd.DataFrame:
    """A1 influence: tau2 re-estimated dropping each household in turn (ids stay identifiers, never names)."""
    table = household_table(panel, weights, y=y, household=household, period=period)
    rows = []
    for h in table.household_id:
        r = estimate_tau2(panel.loc[panel[household] != h], weights, y=y, household=household, period=period,
                          activity_control=activity_control)
        rows.append({"dropped": str(h), "tau2_hat": r.tau2_hat, "tau_hat": r.tau_hat, "n_households": r.n_households})
    return pd.DataFrame(rows)


def leave_one_year_out(
    panel: pd.DataFrame, weights: str = "ES", anchor: str = "anchor_date", y: str = "y_h20",
    household: str = "household_id", period: str = "period", activity_control: bool = True,
) -> pd.DataFrame:
    """A1 influence: tau2 re-estimated dropping every event anchored in one calendar year."""
    years = pd.DatetimeIndex(panel[anchor]).year
    rows = []
    for yr in sorted(set(years)):
        r = estimate_tau2(panel.loc[years != yr], weights, y=y, household=household, period=period,
                          activity_control=activity_control)
        rows.append({"dropped": int(yr), "tau2_hat": r.tau2_hat, "tau_hat": r.tau_hat, "n_households": r.n_households})
    return pd.DataFrame(rows)


def influence_range(loo: pd.DataFrame) -> dict:
    return {"tau2_min": float(loo["tau2_hat"].min()), "tau2_max": float(loo["tau2_hat"].max()),
            "argmin": loo.loc[loo["tau2_hat"].idxmin(), "dropped"], "argmax": loo.loc[loo["tau2_hat"].idxmax(), "dropped"], "n": int(len(loo))}
