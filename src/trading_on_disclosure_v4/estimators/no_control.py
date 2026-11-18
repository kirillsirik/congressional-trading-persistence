"""The genuine no-control estimand: the w-weighted covariance of the two UNRESIDUALIZED period means.

The plan's estimator (§4) residualizes the first-period mean on [1, log(1 + n_hA)] and carries the same
control into the second-stage regression. Because x_h is w-orthogonal to that design, adding the control
to the second stage changes nothing: the `no_activity_control` robustness variant reproduces the headline
tau2 to the last digit. Dropping the control from BOTH stages is a different estimand,

    tau2_nc = Cov_w(Ybar_hA, Ybar_hB),  w_h as in the headline (event-share, equal-household or pair),

and this module computes it with the headline's own inference: the same event-level WLS slope with
household x block clustered standard errors (`cgm.clustered_slope`) and the same household bootstrap
(households resampled with replacement, w_h fixed, percentile interval and one-sided upper bound).
Nothing here reads data; the caller passes the harness panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..inference import block_bootstrap as bb
from ..inference import cgm
from . import tau as tau_mod


def household_table(panel: pd.DataFrame, weights: str = "ES", y: str = "y_h20", household: str = "household_id",
                    period: str = "period") -> tau_mod.HouseholdTable:
    """The household table whose x_h is Ybar_hA minus its w-weighted mean (no activity control)."""
    return tau_mod.household_table(panel, weights, y=y, household=household, period=period, residualize_activity=False)


def activity_correlation(table: tau_mod.HouseholdTable) -> float:
    """w-weighted correlation between the first-period mean and log(1 + n_hA) — why the control matters or does not."""
    return tau_mod.corr_w(table.ybar_a, table.log_n_a, table.w)


def tau2_no_control(panel: pd.DataFrame, weights: str = "ES", y: str = "y_h20", household: str = "household_id",
                    period: str = "period") -> float:
    """Point estimate only: sum_h w_h (Ybar_hA - mean_w Ybar_A)(Ybar_hB) / sum_h w_h."""
    return tau_mod.tau2_household_covariance(household_table(panel, weights, y=y, household=household, period=period))


def tau2_no_control_i1(panel: pd.DataFrame, weights: str = "ES", y: str = "y_h20", household: str = "household_id",
                       period: str = "period", anchor: str = "anchor_date", level: float = 0.95) -> dict:
    """`cgm.tau2_i1` for the no-control estimand: the event-level WLS of period-B outcomes on [1, x_h]
    (weights w_h / n_hB), two-way clustered by household and `anchor`; tau2 = b_hat x Var_w(x)."""
    table = household_table(panel, weights, y=y, household=household, period=period)
    ev = tau_mod.event_design(panel, table, y=y, household=household, period=period, anchor=anchor)
    if anchor not in ev.columns:
        raise ValueError(f"the no-control estimand needs the {anchor!r} column for two-way clustering")
    X = np.column_stack([np.ones(len(ev)), ev["x"].to_numpy(dtype=float)])
    clusters = pd.DataFrame({cgm.CLUSTER_COLS[0]: ev[household].to_numpy(), cgm.CLUSTER_COLS[1]: ev[anchor].to_numpy()})
    sl = cgm.clustered_slope(ev["y"].to_numpy(), X, ev["event_weight"].to_numpy(), clusters, level=level)
    v = tau_mod.var_w(table.x, table.w)
    tau2 = sl.coef * v
    out = {
        "weights": weights, "activity_control": False, "residualize_activity": False, "inference": "I1_cgm_two_way",
        "b_hat": sl.coef, "se_b": sl.se, "t_stat": sl.t_stat, "p_two_sided": sl.p, "dof": sl.dof,
        "var_wx": v, "tau2_hat": tau2, "tau_hat": tau_mod.signed_root(tau2),
        "n_households": int(len(table.household_id)), "n_events_a": int(table.n_a.sum()),
        "n_events_b": int(table.n_b.sum()), "n_anchor_dates": sl.cluster_counts.get(cgm.CLUSTER_COLS[1], 0),
        "psd_corrected": sl.psd_corrected, "n_dropped_nonfinite": table.n_dropped_nonfinite,
        "activity_correlation": activity_correlation(table),
    }
    out.update(tau_mod.tau_ci_from_b_ci(sl.ci_low, sl.ci_high, v))
    return out


def _tau2_from_sums_no_control(n_a: np.ndarray, s_a: np.ndarray, n_b: np.ndarray, s_b: np.ndarray,
                               w: np.ndarray) -> tuple[float, float]:
    """(tau2*, b*) on one household resample with w fixed; x is the w-demeaned first-period mean."""
    keep = (n_a > 0) & (n_b > 0) & (w > 0)
    if keep.sum() < bb.MIN_HOUSEHOLDS_PER_DRAW:
        return np.nan, np.nan
    ybar_a, ybar_b, ww = s_a[keep] / n_a[keep], s_b[keep] / n_b[keep], w[keep]
    x = ybar_a - float(np.sum(ww * ybar_a)) / float(np.sum(ww))
    sxx = float(np.sum(ww * x * x))
    if sxx <= 0:
        return np.nan, np.nan
    b = float(np.sum(ww * x * ybar_b)) / sxx
    return b * sxx / float(np.sum(ww)), b


def tau2_no_control_household_bootstrap(panel: pd.DataFrame, weights: str = "ES", n_draws: int = bb.DEFAULT_DRAWS,
                                        seed: int = 0, y: str = "y_h20", household: str = "household_id",
                                        period: str = "period") -> dict:
    """The headline household bootstrap (I2b) for the no-control estimand: households resampled with
    replacement, each carrying all its purchases in both periods, w_h fixed, percentile interval and the
    one-sided 95% upper bound (95th percentile of tau2*, root clipped at zero)."""
    table = household_table(panel, weights, y=y, household=household, period=period)
    n_h = int(len(table.household_id))
    if n_h < bb.MIN_HOUSEHOLDS_PER_DRAW:
        raise ValueError(f"household bootstrap needs at least {bb.MIN_HOUSEHOLDS_PER_DRAW} two-period households")
    rng = np.random.default_rng(seed)
    tau2s, bs = np.full(n_draws, np.nan), np.full(n_draws, np.nan)
    s_a, s_b = table.ybar_a * table.n_a, table.ybar_b * table.n_b
    for d in range(n_draws):
        m = np.bincount(rng.integers(0, n_h, size=n_h), minlength=n_h).astype(float)
        tau2s[d], bs[d] = _tau2_from_sums_no_control(table.n_a, s_a, table.n_b, s_b, table.w * m)
    n_valid = int(np.isfinite(tau2s).sum())
    res = bb.BootstrapResult(tau2s, 0, n_draws, seed, n_valid, n_draws - n_valid, "households", table.n_dropped_nonfinite)
    lo, hi = res.percentile_interval()
    up = res.one_sided_upper()
    fb = bs[np.isfinite(bs)]
    return {
        "weights": weights, "inference": "I2b_household_bootstrap_no_control", "n_draws": n_draws, "n_valid": n_valid,
        "n_invalid_draws": n_draws - n_valid, "seed": seed, "n_households": n_h,
        "tau2_hat": tau_mod.tau2_household_covariance(table),
        "tau2_low": lo, "tau2_high": hi, "tau_low": tau_mod.signed_root(lo), "tau_high": tau_mod.signed_root(hi),
        "tau2_upper_one_sided": up, "tau_upper_one_sided": float(np.sqrt(max(up, 0.0))) if np.isfinite(up) else np.nan,
        "b_low": float(np.quantile(fb, 0.025)) if len(fb) else np.nan,
        "b_high": float(np.quantile(fb, 0.975)) if len(fb) else np.nan,
        "statistics": res.statistics,
    }
