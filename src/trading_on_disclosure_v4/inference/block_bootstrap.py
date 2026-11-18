"""I2 calendar-block bootstrap (plan §5), the I2b/I2c household bootstraps, and the joint
bootstrap of household means (A4/A5).

I2 (plan spec, kept exactly): within each period the trading-session grid between the period's first
and last anchor is covered by CIRCULAR blocks of L consecutive sessions (random starts on the whole
grid, wrapping at the end, concatenated, truncated to the grid length); a selected session carries
every event anchored on it (whole cross-sections); w_h fixed; x_h recomputed per draw. The
non-circular moving-block scheme (starts on [0, T-L]) remains available as ``scheme="moving"`` but
under-samples the first and last L sessions (referee MINOR-1).

I2b (``tau2_household_bootstrap``, two_way=False): households resampled with replacement, each
carrying all its events in both periods; w_h fixed; percentile interval. I2c (two_way=True): the
household resample, then circular calendar blocks within the resample.

NaN policy: events with a non-finite outcome are dropped once, up front, and counted
(``n_dropped_nonfinite``); draws that cannot be evaluated (fewer than three two-period households
or a degenerate x) are counted in ``n_invalid_draws`` and never silently shrink the interval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..estimators import tau as tau_mod

BLOCK_LENGTHS = (20, 40, 60)
PRIMARY_BLOCK_LENGTH = 20
DEFAULT_DRAWS = 4999
SCHEMES = ("circular", "moving")
PRIMARY_SCHEME = "circular"
MIN_HOUSEHOLDS_PER_DRAW = 3


@dataclass(frozen=True)
class PeriodGrid:
    period: str
    sessions: pd.DatetimeIndex
    rows_by_session: list[np.ndarray]


def _period_grid(panel: pd.DataFrame, calendar: pd.DatetimeIndex, period: str, anchor: str) -> PeriodGrid:
    mask = (panel["period"] == period).to_numpy()
    dates = pd.DatetimeIndex(panel.loc[mask, anchor])
    if len(dates) == 0:
        raise ValueError(f"no events in period {period}")
    pos = calendar.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors in period {period} are not calendar sessions")
    lo, hi = int(pos.min()), int(pos.max())
    sessions = calendar[lo : hi + 1]
    rows = np.flatnonzero(mask)
    order = np.argsort(pos - lo, kind="stable")
    bounds = np.searchsorted(np.sort(pos - lo), np.arange(len(sessions) + 1))
    by_session = [rows[order[bounds[s] : bounds[s + 1]]] for s in range(len(sessions))]
    return PeriodGrid(period=period, sessions=sessions, rows_by_session=by_session)


def block_sessions(rng: np.random.Generator, n_sessions: int, block_length: int, scheme: str = PRIMARY_SCHEME) -> np.ndarray:
    """Session positions of one block draw over a grid of n_sessions (length n_sessions).

    ``circular``: starts uniform on [0, T), blocks wrap modulo T so every session has the same expected
    multiplicity; ``moving``: starts uniform on [0, T-L] (edge sessions under-sampled)."""
    T = int(n_sessions)
    L = min(int(block_length), T)
    n_blocks = -(-T // L)
    if scheme == "circular":
        starts = rng.integers(0, T, size=n_blocks)
        seq = (starts[:, None] + np.arange(L)[None, :]).ravel() % T
    elif scheme == "moving":
        starts = rng.integers(0, T - L + 1, size=n_blocks)
        seq = (starts[:, None] + np.arange(L)[None, :]).ravel()
    else:
        raise ValueError(f"unknown block scheme {scheme!r}; expected one of {SCHEMES}")
    return seq[:T]


def moving_block_sessions(rng: np.random.Generator, n_sessions: int, block_length: int) -> np.ndarray:
    """Non-circular moving blocks (kept for comparison; see MINOR-1)."""
    return block_sessions(rng, n_sessions, block_length, scheme="moving")


def draw_rows(rng: np.random.Generator, grids: list[PeriodGrid], block_length: int, scheme: str = PRIMARY_SCHEME) -> np.ndarray:
    """Row indices of one two-period draw (a session selected k times contributes its rows k times)."""
    parts = []
    for g in grids:
        sel = block_sessions(rng, len(g.sessions), block_length, scheme)
        parts.extend(g.rows_by_session[s] for s in sel)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class BootstrapResult:
    statistics: np.ndarray
    block_length: int
    n_draws: int
    seed: int
    n_valid: int
    n_invalid: int = 0
    scheme: str = PRIMARY_SCHEME
    n_dropped_nonfinite: int = 0

    def finite(self) -> np.ndarray:
        s = np.asarray(self.statistics, dtype=float)
        return s[np.isfinite(s)]

    def percentile_interval(self, level: float = 0.95) -> tuple[float, float]:
        a = (1.0 - level) / 2.0
        f = self.finite()
        if len(f) == 0:
            return np.nan, np.nan
        lo, hi = np.quantile(f, [a, 1.0 - a])
        return float(lo), float(hi)

    def one_sided_upper(self, level: float = 0.95) -> float:
        f = self.finite()
        return float(np.quantile(f, level)) if len(f) else np.nan


def two_period_block_bootstrap(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    stat_fn: Callable[[pd.DataFrame], float],
    block_length: int = PRIMARY_BLOCK_LENGTH,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    anchor: str = "anchor_date",
    y: str = "y_h20",
    scheme: str = PRIMARY_SCHEME,
) -> BootstrapResult:
    """Generic engine: ``stat_fn`` on each resampled panel (rows duplicated per selection).

    Non-finite outcomes are dropped first; a draw on which ``stat_fn`` raises ValueError or returns a
    non-finite value is counted in ``n_invalid`` (never silently)."""
    panel, n_dropped = tau_mod.finite_outcome_rows(panel, y)
    panel = panel.reset_index(drop=True)
    grids = [_period_grid(panel, calendar, p, anchor) for p in (tau_mod.PERIOD_A, tau_mod.PERIOD_B)]
    rng = np.random.default_rng(seed)
    stats = np.full(n_draws, np.nan)
    for d in range(n_draws):
        rows = draw_rows(rng, grids, block_length, scheme)
        try:
            stats[d] = stat_fn(panel.iloc[rows])
        except ValueError:
            stats[d] = np.nan
    n_valid = int(np.isfinite(stats).sum())
    return BootstrapResult(stats, block_length, n_draws, seed, n_valid, n_draws - n_valid, scheme, n_dropped)


def _household_sums(codes: np.ndarray, y: np.ndarray, n_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-household event counts and outcome sums."""
    n = np.bincount(codes, minlength=n_h).astype(float)
    s = np.bincount(codes, weights=y, minlength=n_h)
    return n, s


def _tau2_from_sums(n_a: np.ndarray, s_a: np.ndarray, n_b: np.ndarray, s_b: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """(tau2*, b*) from resampled household sums with fixed w; households absent from a period drop."""
    keep = (n_a > 0) & (n_b > 0) & (w > 0)
    if keep.sum() < MIN_HOUSEHOLDS_PER_DRAW:
        return np.nan, np.nan
    ybar_a, ybar_b, ww = s_a[keep] / n_a[keep], s_b[keep] / n_b[keep], w[keep]
    Z = tau_mod.activity_design(n_a[keep])
    x = tau_mod._wls_residual(ybar_a, Z, ww)
    sxx = float(np.sum(ww * x * x))
    if sxx <= 0:
        return np.nan, np.nan
    b = float(np.sum(ww * x * ybar_b)) / sxx
    return b * sxx / float(np.sum(ww)), b


def _summary(res: BootstrapResult, bs: np.ndarray, table: tau_mod.HouseholdTable, weights: str, inference: str) -> dict:
    lo, hi = res.percentile_interval()
    up = res.one_sided_upper()
    fb = bs[np.isfinite(bs)]
    return {
        "weights": weights, "inference": inference, "block_length": res.block_length, "scheme": res.scheme,
        "n_draws": res.n_draws, "n_valid": res.n_valid, "n_invalid_draws": res.n_invalid, "seed": res.seed,
        "n_dropped_nonfinite": table.n_dropped_nonfinite, "n_households": int(len(table.household_id)),
        "tau2_hat": tau_mod.tau2_household_covariance(table),
        "tau2_low": lo, "tau2_high": hi, "tau_low": tau_mod.signed_root(lo), "tau_high": tau_mod.signed_root(hi),
        "tau2_upper_one_sided": up, "tau_upper_one_sided": float(np.sqrt(max(up, 0.0))) if np.isfinite(up) else np.nan,
        "b_low": float(np.quantile(fb, 0.025)) if len(fb) else np.nan, "b_high": float(np.quantile(fb, 0.975)) if len(fb) else np.nan,
        "statistics": res.statistics,
    }


def _prepare(panel: pd.DataFrame, weights: str, y: str, household: str) -> tuple[tau_mod.HouseholdTable, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Finite-outcome rows of the two-period households, with household codes aligned to the table."""
    table = tau_mod.household_table(panel, weights, y=y, household=household)  # drops and counts non-finite y
    panel, _ = tau_mod.finite_outcome_rows(panel, y)
    panel = panel.reset_index(drop=True)
    codes_all = pd.Index(table.household_id).get_indexer(panel[household])
    keep = codes_all >= 0
    sub = panel.loc[keep].reset_index(drop=True)
    codes = codes_all[keep]
    yv = sub[y].to_numpy(dtype=float)
    is_a = (sub["period"] == tau_mod.PERIOD_A).to_numpy()
    return table, sub, codes, yv, is_a


def tau2_block_bootstrap(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    weights: str = "ES",
    block_length: int = PRIMARY_BLOCK_LENGTH,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    y: str = "y_h20",
    household: str = "household_id",
    anchor: str = "anchor_date",
    scheme: str = PRIMARY_SCHEME,
) -> dict:
    """I2 for A1: tau2* per draw via the household-covariance identity, w_h fixed at original values."""
    table, sub, codes, yv, is_a = _prepare(panel, weights, y, household)
    n_dropped = table.n_dropped_nonfinite
    grids = [_period_grid(sub, calendar, p, anchor) for p in (tau_mod.PERIOD_A, tau_mod.PERIOD_B)]
    rng = np.random.default_rng(seed)
    n_h = len(table.household_id)
    tau2s, bs = np.full(n_draws, np.nan), np.full(n_draws, np.nan)
    for d in range(n_draws):
        rows = draw_rows(rng, grids, block_length, scheme)
        ra, rb = rows[is_a[rows]], rows[~is_a[rows]]
        n_a, s_a = _household_sums(codes[ra], yv[ra], n_h)
        n_b, s_b = _household_sums(codes[rb], yv[rb], n_h)
        tau2s[d], bs[d] = _tau2_from_sums(n_a, s_a, n_b, s_b, table.w)
    n_valid = int(np.isfinite(tau2s).sum())
    res = BootstrapResult(tau2s, block_length, n_draws, seed, n_valid, n_draws - n_valid, scheme, n_dropped)
    return _summary(res, bs, table, weights, "I2_two_period_block_bootstrap")


def _household_draw(rng: np.random.Generator, n_h: int) -> np.ndarray:
    """Multiplicity of each household in one with-replacement resample of the n_h households."""
    return np.bincount(rng.integers(0, n_h, size=n_h), minlength=n_h).astype(float)


def tau2_household_bootstrap(
    panel: pd.DataFrame,
    weights: str = "ES",
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    two_way: bool = False,
    calendar: pd.DatetimeIndex | None = None,
    block_length: int = PRIMARY_BLOCK_LENGTH,
    y: str = "y_h20",
    household: str = "household_id",
    anchor: str = "anchor_date",
    scheme: str = PRIMARY_SCHEME,
) -> dict:
    """I2b (two_way=False): household-cluster (pairs) bootstrap — households resampled with replacement,
    each carrying all its events in both periods; w_h fixed; x_h re-residualized on the resample;
    percentile interval. I2c (two_way=True): household resample, then circular calendar blocks within
    it (a session drawn k times contributes its rows k times to every copy of the household). A household
    drawn m_h times enters every sum with weight m_h * w_h (identical to stacking m_h copies)."""
    table, sub, codes, yv, is_a = _prepare(panel, weights, y, household)
    n_dropped = table.n_dropped_nonfinite
    n_h = len(table.household_id)
    if n_h < MIN_HOUSEHOLDS_PER_DRAW:
        raise ValueError(f"household bootstrap needs at least {MIN_HOUSEHOLDS_PER_DRAW} two-period households")
    if two_way:
        if calendar is None:
            raise ValueError("two_way=True needs the trading calendar")
        grids = [_period_grid(sub, calendar, p, anchor) for p in (tau_mod.PERIOD_A, tau_mod.PERIOD_B)]
    rng = np.random.default_rng(seed)
    tau2s, bs = np.full(n_draws, np.nan), np.full(n_draws, np.nan)
    for d in range(n_draws):
        m = _household_draw(rng, n_h)
        if two_way:
            rows = draw_rows(rng, grids, block_length, scheme)
            ra, rb = rows[is_a[rows]], rows[~is_a[rows]]
            n_a, s_a = _household_sums(codes[ra], yv[ra], n_h)
            n_b, s_b = _household_sums(codes[rb], yv[rb], n_h)
        else:
            n_a, s_a = table.n_a, table.ybar_a * table.n_a
            n_b, s_b = table.n_b, table.ybar_b * table.n_b
        tau2s[d], bs[d] = _tau2_from_sums(n_a, s_a, n_b, s_b, table.w * m)
    n_valid = int(np.isfinite(tau2s).sum())
    res = BootstrapResult(tau2s, block_length if two_way else 0, n_draws, seed, n_valid, n_draws - n_valid,
                          scheme if two_way else "households", n_dropped)
    name = "I2c_household_x_block_bootstrap" if two_way else "I2b_household_bootstrap"
    return _summary(res, bs, table, weights, name)


@dataclass(frozen=True)
class HouseholdMeanBootstrap:
    household_id: np.ndarray
    means: np.ndarray
    n_events: np.ndarray
    boot_means: np.ndarray
    block_length: int
    seed: int
    scheme: str = PRIMARY_SCHEME
    n_dropped_nonfinite: int = 0
    n_draws_with_absent_household: int = 0

    def se(self) -> np.ndarray:
        return np.nanstd(self.boot_means, axis=0, ddof=1)


def _grids_for(panel: pd.DataFrame, calendar: pd.DatetimeIndex, periods: tuple[str, ...] | None, anchor: str) -> list[PeriodGrid]:
    if periods is None:
        tmp = panel.assign(period="ALL")
        return [_period_grid(tmp, calendar, "ALL", anchor)]
    return [_period_grid(panel, calendar, p, anchor) for p in periods]


def household_mean_bootstrap(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    block_length: int = PRIMARY_BLOCK_LENGTH,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    periods: tuple[str, ...] | None = None,
    y: str = "y_h20",
    household: str = "household_id",
    anchor: str = "anchor_date",
    scheme: str = PRIMARY_SCHEME,
) -> HouseholdMeanBootstrap:
    """Joint block-bootstrap distribution of all household means (A4 rank sets, A5 sampling variances).

    A household absent from a draw has a NaN mean in that draw; the number of such draws is reported
    (``n_draws_with_absent_household``) and ``ranks.rank_confidence_sets`` drops those draws whole."""
    panel, n_dropped = tau_mod.finite_outcome_rows(panel, y)
    panel = panel.reset_index(drop=True)
    if periods is not None:
        panel = panel.loc[panel["period"].isin(periods)].reset_index(drop=True)
    codes, hh = pd.factorize(panel[household], sort=True)
    yv = panel[y].to_numpy(dtype=float)
    n_h = len(hh)
    n0, s0 = _household_sums(codes, yv, n_h)
    grids = _grids_for(panel, calendar, periods, anchor)
    rng = np.random.default_rng(seed)
    boot = np.full((n_draws, n_h), np.nan)
    for d in range(n_draws):
        rows = draw_rows(rng, grids, block_length, scheme)
        n, s = _household_sums(codes[rows], yv[rows], n_h)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[d] = np.where(n > 0, s / n, np.nan)
    return HouseholdMeanBootstrap(
        household_id=np.asarray(hh), means=s0 / n0, n_events=n0.astype(int), boot_means=boot,
        block_length=block_length, seed=seed, scheme=scheme, n_dropped_nonfinite=n_dropped,
        n_draws_with_absent_household=int(np.isnan(boot).any(axis=1).sum()),
    )


def household_block_bootstrap(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    block_length: int = PRIMARY_BLOCK_LENGTH,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    periods: tuple[str, ...] | None = None,
    y: str = "y_h20",
    household: str = "household_id",
    anchor: str = "anchor_date",
) -> HouseholdMeanBootstrap:
    """Within-household block bootstrap of every household mean: each household's events are grouped by
    20-trading-day calendar block and its blocks are resampled with replacement (as many as it has), so
    every household is present in every draw and within-household time clustering is kept. Households are
    resampled independently of one another, so common date shocks are not reproduced across households
    (conservative for pairwise differences)."""
    panel, n_dropped = tau_mod.finite_outcome_rows(panel, y)
    panel = panel.reset_index(drop=True)
    if periods is not None:
        panel = panel.loc[panel["period"].isin(periods)].reset_index(drop=True)
    pos = calendar.get_indexer(pd.DatetimeIndex(panel[anchor]))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors are not calendar sessions")
    blk = pos // block_length
    codes, hh = pd.factorize(panel[household], sort=True)
    yv = panel[y].to_numpy(dtype=float)
    n_h = len(hh)
    rng = np.random.default_rng(seed)
    boot = np.full((n_draws, n_h), np.nan)
    means, n_events = np.zeros(n_h), np.zeros(n_h, dtype=int)
    for h in range(n_h):
        m = codes == h
        b = pd.factorize(blk[m])[0]
        s = np.bincount(b, weights=yv[m])
        n = np.bincount(b).astype(float)
        G = len(s)
        means[h], n_events[h] = float(yv[m].mean()), int(m.sum())
        idx = rng.integers(0, G, size=(n_draws, G))
        boot[:, h] = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return HouseholdMeanBootstrap(
        household_id=np.asarray(hh), means=means, n_events=n_events, boot_means=boot, block_length=block_length,
        seed=seed, scheme="within_household_blocks", n_dropped_nonfinite=n_dropped, n_draws_with_absent_household=0,
    )
