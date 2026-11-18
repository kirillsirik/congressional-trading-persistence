"""Event-study simultaneous confidence bands (INF-013, INF-014).

Bands for the event-time mean CAR path of purchases, sales, and their
difference, from a signals x event-time matrix plus the three cluster keys.
Two path-wide constructions:

- ``block_bootstrap``: sup-t over entry-date block resamples that carry the
  full cross-section of each selected date (consistent with
  ``block_bootstrap.draw_block_indices``; an externally supplied index matrix
  shares draws across series, INF-008);
- ``mvt``: finite-cluster multivariate-t sup-t on a three-way
  inclusion-exclusion path covariance.

A pointwise band exists only as a labelled diagnostic. Requesting a
path-wide claim from a pointwise or naive-Gaussian construction raises
:class:`GaussianShortcutError` — the legacy Gaussian shortcut undercovered
and is prohibited (INF-014).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .block_bootstrap import (
    PRIMARY_BLOCK_LENGTH,
    draw_block_indices,
    seed_from_freeze_hash,
)

REQUIRED_KEYS = ("filer_uid", "permno", "entry_date")
SERIES = ("purchase", "sale", "difference")

CLAIM_PATH_WIDE = "path_wide"
CLAIM_POINTWISE_DIAGNOSTIC = "pointwise_diagnostic"
CLAIM_DIAGNOSTIC = "diagnostic_only"

METHOD_BLOCK_BOOTSTRAP = "block_bootstrap_supt"
METHOD_MVT = "finite_cluster_multivariate_t_supt"
METHOD_POINTWISE = "pointwise_t_diagnostic"
METHOD_DAY_MULTIPLIER = "calendar_day_multiplier_supt"
METHOD_BONFERRONI_DAY = "bonferroni_day_variance"


class GaussianShortcutError(RuntimeError):
    """Path-wide claim requested from a construction known to undercover."""


class EventBandNumericalError(RuntimeError):
    """A band computation produced non-finite or inconsistent values.

    Raised rather than returning a silently degraded band — the HS-08
    acceptance criteria require zero silent numerical failures; callers count
    and report these.
    """


@dataclass
class EventBandResult:
    ks: np.ndarray
    estimate: np.ndarray
    se: np.ndarray
    critical: float
    lower: np.ndarray
    upper: np.ndarray
    method: str
    band_scope: str
    level: float
    dof: int | None
    meta: dict = field(default_factory=dict)

    def covers(self, path: np.ndarray) -> bool:
        """True when ``path`` lies inside the band at every event time."""
        p = np.asarray(path, dtype=float)
        return bool(np.all((p >= self.lower) & (p <= self.upper)))


def _validate(car: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    missing = [c for c in REQUIRED_KEYS if c not in keys.columns]
    if missing:
        raise ValueError(f"keys frame missing cluster columns: {missing}")
    if len(car) != len(keys):
        raise ValueError("car matrix and keys frame must align row-wise")
    complete = ~car.isna().any(axis=1).to_numpy()
    n_dropped = int((~complete).sum())
    return car.loc[complete], keys.loc[complete], n_dropped


def _relative_day_labels(car: pd.DataFrame) -> np.ndarray:
    """Event-time axis from the CAR column labels (relative trading days).

    The columns of ``car`` carry the relative-day labels (e.g. -10..60 from
    ``build_event_time_car``); returning array positions here mislabeled the
    axis downstream. Non-numeric columns fall back to positions 1..K.
    """
    try:
        return np.asarray([int(c) for c in car.columns], dtype=int)
    except (TypeError, ValueError):
        return np.arange(1, car.shape[1] + 1)


def _series_weights(direction: pd.Series, series: str) -> np.ndarray:
    is_buy = direction.astype(str).str.lower().isin(["purchase", "buy", "p"]).to_numpy()
    n_p, n_s = int(is_buy.sum()), int((~is_buy).sum())
    if series == "purchase":
        if n_p == 0:
            raise ValueError("no purchase signals")
        return np.where(is_buy, 1.0 / n_p, 0.0)
    if series == "sale":
        if n_s == 0:
            raise ValueError("no sale signals")
        return np.where(is_buy, 0.0, 1.0 / n_s)
    if series == "difference":
        if n_p == 0 or n_s == 0:
            raise ValueError("difference path needs both purchase and sale signals")
        return np.where(is_buy, 1.0 / n_p, -1.0 / n_s)
    raise ValueError(f"unknown series {series!r}; expected one of {SERIES}")


def _own_series_resid(values: np.ndarray, is_buy: np.ndarray) -> np.ndarray:
    resid = values.copy()
    for mask in (is_buy, ~is_buy):
        if mask.any():
            resid[mask] -= values[mask].mean(axis=0, keepdims=True)
    return resid


def threeway_path_covariance(
    car: pd.DataFrame, keys: pd.DataFrame, direction: pd.Series, series: str
) -> tuple[np.ndarray, dict]:
    """K x K covariance of the weighted mean path via 7-term inclusion-exclusion.

    Same cluster algebra as ``cgm3`` (filer + permno + entry date, minus
    pairwise, plus triple), applied to per-signal path contributions
    ``w_i * resid_ik`` so single-series means and the purchase-minus-sale
    difference share one construction.
    """
    values = car.to_numpy(dtype=float)
    is_buy = direction.astype(str).str.lower().isin(["purchase", "buy", "p"]).to_numpy()
    w = _series_weights(direction, series)
    u = w[:, None] * _own_series_resid(values, is_buy)

    cov = np.zeros((values.shape[1], values.shape[1]))
    counts: dict[str, int] = {}
    for r in (1, 2, 3):
        for combo in combinations(REQUIRED_KEYS, r):
            codes = pd.factorize(
                pd.MultiIndex.from_frame(keys[list(combo)].astype(str))
            )[0]
            g = int(codes.max()) + 1
            sums = np.zeros((g, values.shape[1]))
            np.add.at(sums, codes, u)
            term = (g / max(g - 1, 1)) * (sums.T @ sums)
            cov += term if r % 2 == 1 else -term
            counts["G_" + "_x_".join(combo)] = g
    cov = 0.5 * (cov + cov.T)
    eig, vec = np.linalg.eigh(cov)
    n_neg = int((eig < 0).sum())
    cov_psd = vec @ np.diag(np.maximum(eig, 0.0)) @ vec.T
    meta = {
        "cluster_counts": counts,
        "psd_floor_applied": bool(n_neg > 0),
        "min_eigenvalue_raw": float(eig.min()) if len(eig) else float("nan"),
        "dof": min(
            counts["G_filer_uid"], counts["G_permno"], counts["G_entry_date"]
        )
        - 1,
    }
    return cov_psd, meta


def _path_estimate(car: pd.DataFrame, direction: pd.Series, series: str) -> np.ndarray:
    w = _series_weights(direction, series)
    return w @ car.to_numpy(dtype=float)


def _per_date_sums(
    car: pd.DataFrame, keys: pd.DataFrame, direction: pd.Series
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-entry-date column sums and counts, split by series."""
    values = car.to_numpy(dtype=float)
    is_buy = direction.astype(str).str.lower().isin(["purchase", "buy", "p"]).to_numpy()
    dates = pd.DatetimeIndex(pd.unique(keys["entry_date"])).sort_values()
    pos = dates.get_indexer(pd.DatetimeIndex(keys["entry_date"]))
    k = values.shape[1]
    sums_p = np.zeros((len(dates), k))
    sums_s = np.zeros((len(dates), k))
    n_p = np.zeros(len(dates))
    n_s = np.zeros(len(dates))
    np.add.at(sums_p, pos[is_buy], values[is_buy])
    np.add.at(sums_s, pos[~is_buy], values[~is_buy])
    np.add.at(n_p, pos[is_buy], 1.0)
    np.add.at(n_s, pos[~is_buy], 1.0)
    return dates, sums_p, sums_s, n_p, n_s


def _bootstrap_paths(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    indices: np.ndarray,
) -> np.ndarray:
    """(n_draws, K) resampled mean paths; whole dates carried per draw."""
    _, sums_p, sums_s, n_p, n_s = _per_date_sums(car, keys, direction)
    sp = sums_p[indices].sum(axis=1)
    ss = sums_s[indices].sum(axis=1)
    cp = n_p[indices].sum(axis=1)[:, None]
    cs = n_s[indices].sum(axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(cp > 0, sp / cp, np.nan)
        mean_s = np.where(cs > 0, ss / cs, np.nan)
    if series == "purchase":
        return mean_p
    if series == "sale":
        return mean_s
    return mean_p - mean_s


def auto_block_length(n_dates: int, horizon: int) -> int:
    """Deterministic block-length rule: max(2h, 1.3 n^(1/3)), capped at n/4."""
    rule = max(2 * horizon, int(np.ceil(1.3 * n_dates ** (1.0 / 3.0))))
    return max(1, min(rule, max(n_dates // 4, 1)))


def _finite_b_critical(sup_stats: np.ndarray, level: float) -> float:
    """(B+1)-convention order statistic — conservative at finite draw counts."""
    finite = np.sort(sup_stats[np.isfinite(sup_stats)])
    if len(finite) == 0:
        raise EventBandNumericalError("no finite sup statistics in bootstrap draws")
    idx = min(int(np.ceil(level * (len(finite) + 1))) - 1, len(finite) - 1)
    return float(finite[idx])


def supt_band_block_bootstrap(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    freeze_hash: str,
    purpose: str,
    n_draws: int = 999,
    block_length: int | str = PRIMARY_BLOCK_LENGTH,
    level: float = 0.95,
    indices: np.ndarray | None = None,
    bootstrap_method: str = "moving_block",
    recentre: str = "estimate",
    finite_b: bool = False,
) -> EventBandResult:
    """Entry-date block sup-t band.

    ``recentre='bootstrap_mean'`` studentizes draws around the bootstrap mean
    (removes resampling-induced miscentring); ``block_length='auto'`` applies
    :func:`auto_block_length`; ``finite_b`` uses the (B+1) order-statistic
    convention. All outcome-independent variants under HS-08 evaluation.
    """
    car, keys, n_dropped = _validate(car, keys)
    direction = direction.loc[car.index]
    est = _path_estimate(car, direction, series)
    dates = pd.DatetimeIndex(pd.unique(keys["entry_date"])).sort_values()
    if block_length == "auto":
        block_length = auto_block_length(len(dates), car.shape[1])
    if indices is None:
        indices = draw_block_indices(
            len(dates), n_draws, block_length, freeze_hash, purpose, bootstrap_method
        )
    paths = _bootstrap_paths(car, keys, direction, series, indices)
    se_boot = np.nanstd(paths, axis=0, ddof=1)
    centre = np.nanmean(paths, axis=0) if recentre == "bootstrap_mean" else est
    with np.errstate(invalid="ignore", divide="ignore"):
        t_star = np.abs(paths - centre[None, :]) / se_boot[None, :]
    sup_t = np.nanmax(t_star, axis=1)
    critical = (
        _finite_b_critical(sup_t, level)
        if finite_b
        else float(np.nanquantile(sup_t, level))
    )
    if not (np.all(np.isfinite(se_boot)) and np.isfinite(critical)):
        raise EventBandNumericalError("non-finite bootstrap SE or critical value")
    half = critical * se_boot
    return EventBandResult(
        ks=_relative_day_labels(car),
        estimate=est,
        se=se_boot,
        critical=critical,
        lower=est - half,
        upper=est + half,
        method=METHOD_BLOCK_BOOTSTRAP,
        band_scope=CLAIM_PATH_WIDE,
        level=level,
        dof=None,
        meta={
            "n_draws": int(indices.shape[0]),
            "block_length": int(block_length),
            "bootstrap_method": bootstrap_method,
            "recentre": recentre,
            "finite_b": finite_b,
            "n_dropped_incomplete": n_dropped,
            "indices_shared": True,
            "seed_purpose": purpose,
        },
    )


def dwb_bandwidth_rule(horizon: int) -> int:
    """Deterministic multiplier-smoothing bandwidth: the horizon itself.

    Overlap dependence cannot extend past the horizon by construction; under
    day-level independence any bandwidth is variance-unbiased (cross terms
    have zero mean), so the rule costs only estimator noise, never bias.
    """
    return max(3, horizon)


def day_contributions(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    trading_days: pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact per-calendar-day decomposition of the mean CAR path.

    Returns ``(c, load, meta)`` where ``c[tau, k]`` sums every signal's
    fixed-weight day-``tau`` increment contributing to path point ``k``, so
    ``c.sum(axis=0)`` reconstructs the estimate exactly and each day's full
    cross-section stays together (whole-date resampling requirement).
    ``load[tau, k]`` is the same aggregation of the bare weights, used for
    drift adjustment. With ``trading_days=None`` the sorted unique entry
    dates serve as the day grid — market days without an entry are then
    compressed out, so two entries separated by many trading days sit on
    adjacent kernel slots and the Bartlett dependence structure is misstated
    unless every trading day hosts an entry. Pass the true trading calendar
    so intervening no-entry market days enter the kernel spacing. Slot
    indices are a uniform translation of true calendar positions (constant
    ``pre + 1`` shift), which leaves all slot differences — the only thing
    the kernel sees — exact.
    """
    values = car.to_numpy(dtype=float)
    incr = np.diff(values, axis=1, prepend=0.0)
    w = _series_weights(direction, series)
    entry = pd.DatetimeIndex(keys["entry_date"])
    grid = (
        pd.DatetimeIndex(pd.unique(entry)).sort_values()
        if trading_days is None
        else pd.DatetimeIndex(trading_days)
    )
    pos = grid.get_indexer(entry)
    if (pos < 0).any():
        raise ValueError("entry dates missing from the trading-day grid")
    k = values.shape[1]
    n_slots = len(grid) + k
    m = np.zeros((n_slots, k))
    load = np.zeros((n_slots, k))
    for j in range(k):
        np.add.at(m[:, j], pos + j + 1, w * incr[:, j])
        np.add.at(load[:, j], pos + j + 1, w)
    c = np.cumsum(m, axis=1)
    load_c = np.cumsum(load, axis=1)
    meta = {
        "n_day_slots": n_slots,
        "n_grid_days": len(grid),
        "grid_from_entry_dates": trading_days is None,
    }
    return c, load_c, meta


def _smoothed_multipliers(
    rng: np.random.Generator, n_draws: int, n_days: int, bandwidth: int
) -> np.ndarray:
    """Unit-variance normal multipliers with Bartlett MA dependence over days."""
    if bandwidth <= 1:
        return rng.standard_normal((n_draws, n_days))
    kern = 1.0 - np.arange(bandwidth) / bandwidth
    kern = kern / np.sqrt((kern**2).sum())
    xi = rng.standard_normal((n_draws, n_days + bandwidth - 1))
    windows = np.lib.stride_tricks.sliding_window_view(xi, bandwidth, axis=1)
    return windows @ kern


def _dwb_variance(c: np.ndarray, bandwidth: int) -> np.ndarray:
    """Analytic per-k variance implied by the smoothed multipliers."""
    if bandwidth <= 1:
        return (c * c).sum(axis=0)
    kern = 1.0 - np.arange(bandwidth) / bandwidth
    kern = kern / np.sqrt((kern**2).sum())
    v = (c * c).sum(axis=0)
    for j in range(1, bandwidth):
        rho = float((kern[:-j] * kern[j:]).sum())
        v = v + 2.0 * rho * (c[:-j] * c[j:]).sum(axis=0)
    return v


def supt_band_day_multiplier(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    freeze_hash: str,
    purpose: str,
    n_draws: int = 999,
    level: float = 0.95,
    bandwidth: int | None = None,
    centering: str = "null_imposed",
    trading_days: pd.DatetimeIndex | None = None,
    multipliers: np.ndarray | None = None,
) -> EventBandResult:
    """Calendar-day multiplier sup-t band (dependent wild bootstrap).

    Decomposes the path into per-day contributions — overlap dependence
    becomes within-day cross-section, kept intact — then draws smoothed
    normal multipliers per day. ``null_imposed`` uses raw contributions
    (valid under H0, conservative under drift); ``drift_adjusted`` removes a
    first-order per-day drift via the load matrix. Deterministic under the
    freeze-hash seed; shared ``multipliers`` reuse draws across a family
    (INF-008). Non-finite variance or criticals raise
    :class:`EventBandNumericalError` — never silently degraded.
    """
    car, keys, n_dropped = _validate(car, keys)
    direction = direction.loc[car.index]
    est = _path_estimate(car, direction, series)
    c, load_c, cmeta = day_contributions(car, keys, direction, series, trading_days)
    if not np.allclose(c.sum(axis=0), est, atol=1e-10):
        raise EventBandNumericalError("day decomposition failed to reconstruct path")
    horizon = car.shape[1]
    bw = dwb_bandwidth_rule(horizon) if bandwidth is None else bandwidth
    if bw < 1:
        raise ValueError("bandwidth must be a positive integer")
    bw = min(bw, c.shape[0])
    if centering == "drift_adjusted":
        drift_per_day = est[-1] / horizon
        c_used = c - load_c * drift_per_day
    elif centering == "null_imposed":
        c_used = c
    else:
        raise ValueError(f"unknown centering {centering!r}")
    v = _dwb_variance(c_used, bw)
    se = np.sqrt(v)
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise EventBandNumericalError("day-multiplier variance non-finite or zero")
    if multipliers is None:
        rng = np.random.default_rng(seed_from_freeze_hash(freeze_hash, purpose))
        multipliers = _smoothed_multipliers(rng, n_draws, c.shape[0], bw)
    g = multipliers @ c_used
    sup_t = np.max(np.abs(g) / se[None, :], axis=1)
    critical = _finite_b_critical(sup_t, level)
    half = critical * se
    return EventBandResult(
        ks=_relative_day_labels(car),
        estimate=est,
        se=se,
        critical=critical,
        lower=est - half,
        upper=est + half,
        method=METHOD_DAY_MULTIPLIER,
        band_scope=CLAIM_PATH_WIDE,
        level=level,
        dof=None,
        meta={
            **cmeta,
            "n_draws": int(multipliers.shape[0]),
            "bandwidth": int(bw),
            "centering": centering,
            "multiplier": "smoothed_normal_bartlett",
            "n_dropped_incomplete": n_dropped,
            "seed_purpose": purpose,
        },
    )


def bonferroni_band_day_variance(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    level: float = 0.95,
    bandwidth: int | None = None,
    trading_days: pd.DatetimeIndex | None = None,
) -> EventBandResult:
    """Bonferroni simultaneous band on day-decomposition variances.

    Guaranteed-level fallback: valid marginal SEs from the
    overlap-aware day decomposition plus a Bonferroni split need no joint
    distribution, at the price of extra width.
    """
    car, keys, n_dropped = _validate(car, keys)
    direction = direction.loc[car.index]
    est = _path_estimate(car, direction, series)
    c, _, cmeta = day_contributions(car, keys, direction, series, trading_days)
    horizon = car.shape[1]
    bw = dwb_bandwidth_rule(horizon) if bandwidth is None else bandwidth
    se = np.sqrt(_dwb_variance(c, min(bw, c.shape[0])))
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise EventBandNumericalError("day-variance non-finite or zero")
    critical = float(stats.norm.ppf(1.0 - (1.0 - level) / (2.0 * horizon)))
    half = critical * se
    return EventBandResult(
        ks=_relative_day_labels(car),
        estimate=est,
        se=se,
        critical=critical,
        lower=est - half,
        upper=est + half,
        method=METHOD_BONFERRONI_DAY,
        band_scope=CLAIM_PATH_WIDE,
        level=level,
        dof=None,
        meta={**cmeta, "bandwidth": int(min(bw, c.shape[0])),
              "n_dropped_incomplete": n_dropped},
    )


def supt_band_mvt(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    freeze_hash: str,
    purpose: str,
    level: float = 0.95,
    n_draw: int = 20000,
    dof: int | None = -1,
) -> EventBandResult:
    """Finite-cluster multivariate-t sup-t band.

    ``dof=-1`` (default) resolves to ``min(G) - 1`` from the path covariance.
    ``dof=None`` is the pure-Gaussian reference and stays a labelled
    diagnostic — :func:`event_study_bands` refuses it for a path-wide claim.
    """
    car, keys, n_dropped = _validate(car, keys)
    direction = direction.loc[car.index]
    est = _path_estimate(car, direction, series)
    cov, meta = threeway_path_covariance(car, keys, direction, series)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    resolved_dof = meta["dof"] if dof == -1 else dof
    rng = np.random.default_rng(seed_from_freeze_hash(freeze_hash, purpose))
    ok = se > 0
    eig, vec = np.linalg.eigh(cov[np.ix_(ok, ok)])
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0.0)))
    z = root @ rng.standard_normal((root.shape[1], n_draw))
    t = np.abs(z) / se[ok][:, None]
    if resolved_dof is not None and resolved_dof > 0:
        t = t / np.sqrt(rng.chisquare(resolved_dof, size=n_draw) / resolved_dof)[None, :]
    critical = float(np.quantile(t.max(axis=0), level))
    half = critical * se
    return EventBandResult(
        ks=_relative_day_labels(car),
        estimate=est,
        se=se,
        critical=critical,
        lower=est - half,
        upper=est + half,
        method=METHOD_MVT,
        band_scope=CLAIM_DIAGNOSTIC,
        level=level,
        dof=resolved_dof,
        meta={**meta, "n_draw": n_draw, "n_dropped_incomplete": n_dropped,
              "reference": "multivariate_t" if resolved_dof else "gaussian"},
    )


def pointwise_band_diagnostic(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    series: str,
    level: float = 0.95,
) -> EventBandResult:
    """Pointwise t band — DIAGNOSTIC ONLY, never a path-wide claim (INF-014)."""
    car, keys, n_dropped = _validate(car, keys)
    direction = direction.loc[car.index]
    est = _path_estimate(car, direction, series)
    cov, meta = threeway_path_covariance(car, keys, direction, series)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    critical = float(stats.t(max(meta["dof"], 1)).ppf(0.5 + level / 2.0))
    half = critical * se
    return EventBandResult(
        ks=_relative_day_labels(car),
        estimate=est,
        se=se,
        critical=critical,
        lower=est - half,
        upper=est + half,
        method=METHOD_POINTWISE,
        band_scope=CLAIM_POINTWISE_DIAGNOSTIC,
        level=level,
        dof=meta["dof"],
        meta={**meta, "n_dropped_incomplete": n_dropped},
    )


def event_study_bands(
    car: pd.DataFrame,
    keys: pd.DataFrame,
    direction: pd.Series,
    freeze_hash: str,
    method: str = "block_bootstrap",
    claim: str = CLAIM_PATH_WIDE,
    series: tuple[str, ...] = SERIES,
    **kwargs,
) -> dict[str, EventBandResult]:
    """Bands for purchases, sales, and the difference under one method.

    A path-wide claim from a pointwise or Gaussian-reference construction
    raises :class:`GaussianShortcutError` (INF-014).
    """
    if claim == CLAIM_PATH_WIDE:
        if method == "pointwise":
            raise GaussianShortcutError(
                "pointwise bands cannot support a path-wide claim (INF-014)"
            )
        if method == "mvt":
            raise GaussianShortcutError(
                "the multivariate-t band is diagnostic only and may not "
                "support path-wide inference (2026-08-07 directive, HS-08); "
                "measured undercoverage under adjacent-date overlap"
            )
    out: dict[str, EventBandResult] = {}
    for s in series:
        purpose = f"event_band|{method}|{s}"
        if method == "block_bootstrap":
            out[s] = supt_band_block_bootstrap(
                car, keys, direction, s, freeze_hash, purpose, **kwargs
            )
        elif method == "day_multiplier":
            out[s] = supt_band_day_multiplier(
                car, keys, direction, s, freeze_hash, purpose, **kwargs
            )
        elif method == "bonferroni_day":
            out[s] = bonferroni_band_day_variance(car, keys, direction, s, **kwargs)
        elif method == "mvt":
            out[s] = supt_band_mvt(
                car, keys, direction, s, freeze_hash, purpose, **kwargs
            )
        elif method == "pointwise":
            out[s] = pointwise_band_diagnostic(car, keys, direction, s, **kwargs)
        else:
            raise ValueError(f"unknown band method {method!r}")
    return out
