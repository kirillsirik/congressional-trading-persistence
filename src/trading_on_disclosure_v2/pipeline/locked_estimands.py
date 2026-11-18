"""Executors for the locked two-clock registry (final directive §11-§18).

Primary inference is two-way Cameron-Gelbach-Miller clustering by canonical
``filer_uid`` and the EXACT public-entry trading date, with the frozen
finite-sample correction and min-cluster degrees of freedom, plus a
calendar-block bootstrap companion. Every registry row is executed or carries
an explicit blocked reason; no IID path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v2.inference.block_bootstrap import calendar_block_bootstrap
from trading_on_disclosure_v2.inference.cgm3 import two_way_cluster_cov
from trading_on_disclosure_v2.outcomes.returns import window_abnormal_return
from trading_on_disclosure_v2.timing import BENCHMARK_ENTRY_COL, FOLLOWER_ENTRY_COL

#: outcome anchor -> entry-date column used BOTH for timing and as the exact
#: entry-date cluster dimension of that anchor's estimand
ENTRY_COLS = {
    "E": FOLLOWER_ENTRY_COL,
    "ERAW": "entry_raw_ct_date",
    "EPLUS1": "entry_plus1_date",
    "T": BENCHMARK_ENTRY_COL,
    "TRAW": "transaction_raw_entry_date",
}

PRIMARY_OUTCOME = "ar_E_h20"


class LockedExecutionError(RuntimeError):
    pass


@dataclass
class EstimandContext:
    signals: pd.DataFrame
    events: pd.DataFrame
    portfolio_daily: pd.DataFrame | None
    factors: pd.DataFrame | None
    freeze_hash: str
    daily: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    calendar: pd.DatetimeIndex | None = None
    n_boot: int = 499
    artifacts: dict = field(default_factory=dict)


def primary_sample(signals: pd.DataFrame) -> pd.DataFrame:
    """Non-mixed unique signals (§8): the primary directional contrast."""
    if "signal_direction" in signals.columns:
        return signals[signals["signal_direction"].isin(["purchase", "sale"])]
    return signals[signals["direction"].isin(["purchase", "sale"])]


def _spread_stat(outcome_col: str, direction_col: str = "direction"):
    def stat(df: pd.DataFrame) -> float:
        buys = df.loc[df[direction_col] == "purchase", outcome_col]
        sells = df.loc[df[direction_col] == "sale", outcome_col]
        if buys.dropna().empty or sells.dropna().empty:
            return np.nan
        return float(buys.mean() - sells.mean())

    return stat


def two_way_spread(
    frame: pd.DataFrame,
    outcome_col: str,
    entry_col: str,
    freeze_hash: str,
    boot_purpose: str,
    household_col: str = "filer_uid",
    weight_col: str | None = None,
    n_boot: int = 499,
) -> dict:
    """Purchase-minus-sale spread with two-way CGM covariance and a
    calendar-block bootstrap companion interval."""
    if outcome_col not in frame.columns:
        return {"status": f"blocked_missing_input:{outcome_col}"}
    if entry_col not in frame.columns:
        return {"status": f"blocked_missing_input:{entry_col}"}
    data = frame.dropna(subset=[outcome_col, entry_col]).copy()
    data = data[data["direction"].isin(["purchase", "sale"])]
    if weight_col is not None:
        if weight_col not in data.columns:
            return {"status": f"blocked_missing_input:{weight_col}"}
        reps = data[weight_col].fillna(1).astype(int).clip(lower=1)
        data = data.loc[data.index.repeat(reps)].reset_index(drop=True)
    if data.empty or data["direction"].nunique() < 2:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    if household_col not in data.columns:
        return {"status": f"blocked_missing_input:{household_col}"}

    y = data[outcome_col].to_numpy(float)
    purchase = (data["direction"] == "purchase").to_numpy(float)
    X = np.column_stack([np.ones(len(y)), purchase])
    clusters = pd.DataFrame(
        {
            household_col: data[household_col].to_numpy(),
            entry_col: data[entry_col].to_numpy(),
        }
    )
    res = two_way_cluster_cov(y, X, clusters, required=(household_col, entry_col))
    se = float(np.sqrt(res.cov[1, 1]))
    est = float(res.beta[1])
    t_stat = est / se if se > 0 else np.nan
    dof = max(res.dof, 1)
    p = float(2 * stats.t.sf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan
    crit = float(stats.t.ppf(0.975, df=dof))
    out = {
        "status": "executed",
        "estimate": est,
        "se_multiway": se,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "raw_p": p,
        "ci_low": est - crit * se,
        "ci_high": est + crit * se,
        "n": int(len(data)),
        "n_households": int(res.cluster_counts.get(household_col, 0)),
        "n_entry_dates": int(res.cluster_counts.get(entry_col, 0)),
        "dof": int(dof),
        "purchase_mean": float(data.loc[data["direction"] == "purchase", outcome_col].mean()),
        "sale_mean": float(data.loc[data["direction"] == "sale", outcome_col].mean()),
    }
    boot = calendar_block_bootstrap(
        data,
        entry_col,
        _spread_stat(outcome_col),
        freeze_hash,
        boot_purpose,
        n_draws=n_boot,
    )
    out["block_ci_low"], out["block_ci_high"] = boot.percentile_interval()
    return out


def intercept_two_way(
    data: pd.DataFrame,
    value_col: str,
    entry_col: str,
    household_col: str = "filer_uid",
) -> dict:
    """Mean of ``value_col`` with two-way clustered SE (paired/signed rows)."""
    data = data.dropna(subset=[value_col, entry_col])
    if len(data) < 3:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    y = data[value_col].to_numpy(float)
    X = np.ones((len(y), 1))
    clusters = pd.DataFrame(
        {
            household_col: data[household_col].to_numpy(),
            entry_col: data[entry_col].to_numpy(),
        }
    )
    res = two_way_cluster_cov(y, X, clusters, required=(household_col, entry_col))
    se = float(np.sqrt(res.cov[0, 0]))
    est = float(res.beta[0])
    dof = max(res.dof, 1)
    t_stat = est / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan
    crit = float(stats.t.ppf(0.975, df=dof))
    return {
        "status": "executed",
        "estimate": est,
        "se_multiway": se,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "raw_p": p,
        "ci_low": est - crit * se,
        "ci_high": est + crit * se,
        "n": int(len(data)),
        "n_households": int(res.cluster_counts.get(household_col, 0)),
        "n_entry_dates": int(res.cluster_counts.get(entry_col, 0)),
        "dof": int(dof),
    }


def _paired_clock(ctx: EstimandContext) -> dict:
    data = primary_sample(ctx.signals).copy()
    for col in ("ar_T_h20", PRIMARY_OUTCOME):
        if col not in data.columns:
            return {"status": f"blocked_missing_input:{col}"}
    data["paired_diff"] = data["ar_T_h20"] - data[PRIMARY_OUTCOME]
    return intercept_two_way(data, "paired_diff", FOLLOWER_ENTRY_COL)


def _signed_payoff(ctx: EstimandContext) -> dict:
    data = primary_sample(ctx.signals).copy()
    if PRIMARY_OUTCOME not in data.columns:
        return {"status": f"blocked_missing_input:{PRIMARY_OUTCOME}"}
    sign = np.where(data["direction"] == "purchase", 1.0, -1.0)
    data["sar_E_h20"] = data[PRIMARY_OUTCOME] * sign
    return intercept_two_way(data, "sar_E_h20", FOLLOWER_ENTRY_COL)


def _equal_household(ctx: EstimandContext) -> dict:
    data = primary_sample(ctx.signals).dropna(subset=[PRIMARY_OUTCOME])
    per_hh = (
        data.groupby(["filer_uid", "direction"])[PRIMARY_OUTCOME]
        .mean()
        .unstack("direction")
    )
    diffs = (per_hh.get("purchase") - per_hh.get("sale")).dropna()
    if len(diffs) < 3:
        return {"status": "blocked_insufficient_data", "n": int(len(diffs))}
    t_stat, p = stats.ttest_1samp(diffs, 0.0)
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return {
        "status": "executed",
        "estimate": float(diffs.mean()),
        "se_multiway": se,
        "t_stat": float(t_stat),
        "raw_p": float(p),
        "ci_low": float(diffs.mean() - stats.t.ppf(0.975, len(diffs) - 1) * se),
        "ci_high": float(diffs.mean() + stats.t.ppf(0.975, len(diffs) - 1) * se),
        "n": int(len(diffs)),
    }


def _welch_satterthwaite_dof(v1: float, n1: int, v2: float, n2: int) -> float:
    if n1 < 2 or n2 < 2:
        return 1.0
    denom = v1**2 / (n1 - 1) + v2**2 / (n2 - 1)
    return float((v1 + v2) ** 2 / denom) if denom > 0 else 1.0


def _event_level(ctx: EstimandContext) -> dict:
    events = ctx.events
    if "ar_E_h20" not in events.columns:
        return {"status": "blocked_missing_input:event_outcomes"}
    data = events.dropna(subset=["ar_E_h20"])
    pure_buy = data[
        (data["purchase_household_count"] > 0) & (data["sale_household_count"] == 0)
    ]["ar_E_h20"]
    pure_sell = data[
        (data["sale_household_count"] > 0) & (data["purchase_household_count"] == 0)
    ]["ar_E_h20"]
    if pure_buy.empty or pure_sell.empty:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    diff = float(pure_buy.mean() - pure_sell.mean())
    vb = pure_buy.var(ddof=1) / len(pure_buy)
    vs = pure_sell.var(ddof=1) / len(pure_sell)
    se = float(np.sqrt(vb + vs))
    t_stat = diff / se if se > 0 else np.nan
    dof = _welch_satterthwaite_dof(vb, len(pure_buy), vs, len(pure_sell))
    crit = float(stats.t.ppf(0.975, df=dof))
    return {
        "status": "executed",
        "estimate": diff,
        "se_multiway": se,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "raw_p": float(2 * stats.t.sf(abs(t_stat), df=dof)),
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "dof": float(dof),
        "n": int(len(data)),
    }


def _portfolio_regression(
    daily: pd.DataFrame | None,
    factors: pd.DataFrame | None,
    factor_cols: tuple[str, ...],
    lag: int = 20,
) -> dict:
    """Newey-West (Bartlett, frozen lag rule = 20 trading days) portfolio
    mean-excess or alpha regression."""
    if daily is None or factors is None:
        return {"status": "blocked_missing_input:portfolio_or_factors"}
    frame = daily.join(factors, how="inner").dropna(
        subset=["strategy_return", "rf", *factor_cols]
    )
    if len(frame) < 60:
        return {"status": "blocked_insufficient_data", "n": int(len(frame))}
    y = (frame["strategy_return"] - frame["rf"]).to_numpy(float)
    X = np.column_stack(
        [np.ones(len(y))] + [frame[c].to_numpy(float) for c in factor_cols]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    Xu = X * resid[:, None]
    meat = Xu.T @ Xu
    for ell in range(1, lag + 1):
        w = 1 - ell / (lag + 1)
        gamma = Xu[ell:].T @ Xu[:-ell]
        meat += w * (gamma + gamma.T)
    cov = bread @ meat @ bread * n / max(n - k, 1)
    se = float(np.sqrt(cov[0, 0]))
    est_annual = float(beta[0] * 252)
    se_annual = se * 252
    t_stat = float(beta[0] / se) if se > 0 else np.nan
    dof = n - k
    crit = float(stats.t.ppf(0.975, df=dof))
    return {
        "status": "executed",
        "estimate": est_annual,
        "se_multiway": se_annual,
        "t_stat": t_stat,
        "raw_p": float(2 * stats.t.sf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan,
        "ci_low": est_annual - crit * se_annual,
        "ci_high": est_annual + crit * se_annual,
        "n": int(n),
    }


def placebo_direction_permutation(
    ctx: EstimandContext, n_perm: int = 999
) -> dict:
    """§17 placebo 1: permute purchase/sale within disclosure month."""
    from trading_on_disclosure_v2.dates import ANALYSIS_PUBLIC_COL
    from trading_on_disclosure_v2.release.seeds import derive_seed

    data = primary_sample(ctx.signals).dropna(subset=[PRIMARY_OUTCOME]).copy()
    if data.empty or data["direction"].nunique() < 2:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    observed = _spread_stat(PRIMARY_OUTCOME)(data)
    rng = np.random.default_rng(
        derive_seed(ctx.freeze_hash, "placebo_battery", "direction_permutation")
    )
    month = pd.to_datetime(data[ANALYSIS_PUBLIC_COL]).dt.to_period("M")
    draws = np.empty(n_perm)
    directions = data["direction"].to_numpy(object)
    month_codes, _ = pd.factorize(month)
    order = np.argsort(month_codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(month_codes[order])) + 1
    for i in range(n_perm):
        permuted = directions.copy()
        for grp in np.split(order, boundaries):
            permuted[grp] = directions[grp][rng.permutation(len(grp))]
        shuffled = data.copy()
        shuffled["direction"] = permuted
        draws[i] = _spread_stat(PRIMARY_OUTCOME)(shuffled)
    finite = draws[np.isfinite(draws)]
    p = float((1 + np.sum(np.abs(finite) >= abs(observed))) / (len(finite) + 1))
    return {
        "status": "executed",
        "estimate": float(observed),
        "raw_p": p,
        "n": int(len(data)),
        "placebo_mean": float(finite.mean()),
        "placebo_sd": float(finite.std(ddof=1)),
        "n_draws": int(len(finite)),
    }


def placebo_pseudo_entry(
    ctx: EstimandContext,
    exclusion_window: int = 20,
    horizon: int = 20,
    n_draws: int = 19,
) -> dict:
    """§17 placebo 2: pseudo entry dates within the same security and year,
    outside the frozen exclusion window, run through the SAME outcome
    constructor as the primary analysis."""
    from trading_on_disclosure_v2.outcomes.returns import combine_with_delisting
    from trading_on_disclosure_v2.release.seeds import derive_seed

    if ctx.daily is None or ctx.benchmark is None or ctx.calendar is None:
        return {"status": "blocked_missing_input:market_frames"}
    data = primary_sample(ctx.signals).dropna(
        subset=[PRIMARY_OUTCOME, FOLLOWER_ENTRY_COL]
    ).copy()
    if data.empty or data["direction"].nunique() < 2:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    observed = _spread_stat(PRIMARY_OUTCOME)(data)
    combined = combine_with_delisting(ctx.daily)
    by_permno = {
        int(p): g.set_index("date")["total_ret"]
        for p, g in combined.groupby("permno")
    }
    calendar = ctx.calendar
    rng = np.random.default_rng(
        derive_seed(ctx.freeze_hash, "placebo_battery", "pseudo_entry_dates")
    )
    entry_pos = calendar.searchsorted(pd.DatetimeIndex(data[FOLLOWER_ENTRY_COL]))
    years = pd.DatetimeIndex(data[FOLLOWER_ENTRY_COL]).year
    year_positions: dict[int, np.ndarray] = {
        y: np.flatnonzero(calendar.year == y) for y in np.unique(years)
    }
    draws = np.empty(n_draws)
    permnos = data["permno"].to_numpy(int)
    for d in range(n_draws):
        pseudo = np.full(len(data), np.nan)
        for i in range(len(data)):
            candidates = year_positions[years[i]]
            valid = candidates[
                (np.abs(candidates - entry_pos[i]) > exclusion_window)
                & (candidates + horizon < len(calendar))
            ]
            if len(valid) == 0:
                continue
            pos = int(valid[rng.integers(len(valid))])
            ar, reason = window_abnormal_return(
                permnos[i],
                calendar[pos],
                horizon,
                by_permno.get(permnos[i], pd.Series(dtype=float)),
                ctx.benchmark,
                calendar,
            )
            pseudo[i] = ar
        shuffled = data.copy()
        shuffled["pseudo_ar"] = pseudo
        draws[d] = _spread_stat("pseudo_ar")(shuffled.dropna(subset=["pseudo_ar"]))
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return {"status": "blocked_insufficient_data", "n": int(len(data))}
    p = float((1 + np.sum(np.abs(finite) >= abs(observed))) / (len(finite) + 1))
    return {
        "status": "executed",
        "estimate": float(observed),
        "raw_p": p,
        "n": int(len(data)),
        "placebo_mean": float(finite.mean()),
        "placebo_sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
        "n_draws": int(len(finite)),
    }


def _segment_masks(signals: pd.DataFrame) -> dict[str, pd.Series]:
    from trading_on_disclosure_v2.dates import (
        ANALYSIS_PUBLIC_PROXY_COL,
        ANALYSIS_PUBLIC_SOURCE_COL,
        SOURCE_OFFICIAL_OBSERVED,
    )

    src = signals.get(
        ANALYSIS_PUBLIC_SOURCE_COL, pd.Series(index=signals.index, dtype=object)
    )
    proxy = signals.get(
        ANALYSIS_PUBLIC_PROXY_COL, pd.Series(False, index=signals.index)
    ).fillna(False)
    record_source = signals.get(
        "record_source", pd.Series("official", index=signals.index)
    )
    return {
        "official_date_only": src == SOURCE_OFFICIAL_OBSERVED,
        "official_segment": record_source == "official",
        "vendor_segment": proxy.astype(bool),
    }


def execute_locked_registry(
    registry: pd.DataFrame, ctx: EstimandContext
) -> pd.DataFrame:
    """Execute every registry row; each ends executed or explicitly blocked."""
    prim = primary_sample(ctx.signals)
    masks = _segment_masks(prim)

    def spread(outcome: str, anchor: str, mask=None, **kw) -> dict:
        frame = prim if mask is None else prim[mask]
        return two_way_spread(
            frame,
            outcome,
            ENTRY_COLS[anchor],
            ctx.freeze_hash,
            boot_purpose=f"locked|{outcome}|{anchor}|{kw.get('household_col','filer_uid')}",
            n_boot=ctx.n_boot,
            **kw,
        )

    executors = {
        "TW_P1_E20": lambda: spread("ar_E_h20", "E"),
        "TW_P2_E5": lambda: spread("ar_E_h5", "E"),
        "TW_P2_E10": lambda: spread("ar_E_h10", "E"),
        "TW_P2_E40": lambda: spread("ar_E_h40", "E"),
        "TW_P2_E60": lambda: spread("ar_E_h60", "E"),
        "TW_P2_E127": lambda: spread("ar_E_h127", "E"),
        "TW_P3_RAW_CT": lambda: spread("ar_ERAW_h20", "ERAW"),
        "TW_P3_OFFICIAL_ONLY": lambda: spread(
            "ar_E_h20", "E", mask=masks["official_date_only"]
        ),
        "TW_P3_OFFICIAL_SEGMENT": lambda: spread(
            "ar_E_h20", "E", mask=masks["official_segment"]
        ),
        "TW_P3_VENDOR_SEGMENT": lambda: spread(
            "ar_E_h20", "E", mask=masks["vendor_segment"]
        ),
        "TW_P3_ENTRY_PLUS1": lambda: spread("ar_EPLUS1_h20", "EPLUS1"),
        "TW_C1_T20": lambda: spread("ar_T_h20", "T"),
        "TW_C1_TRAW20": lambda: spread("ar_TRAW_h20", "TRAW"),
        "TW_C1_PAIRED": lambda: _paired_clock(ctx),
        "TW_D_SAR20": lambda: _signed_payoff(ctx),
        "TW_S1_RAWROWS": lambda: spread("ar_E_h20", "E", weight_col="lot_count"),
        "TW_S1_EQHH": lambda: _equal_household(ctx),
        "TW_S1_EVENT": lambda: _event_level(ctx),
        "TW_S1_SRCID": lambda: spread(
            "ar_E_h20", "E", household_col="source_native_household_id"
        ),
        "TW_S2_CAPM": lambda: spread("car_capm_h20", "E"),
        "TW_S2_FF5MOM": lambda: spread("car_ff5mom_h20", "E"),
        "TW_IMPL_PORT_NW": lambda: _portfolio_regression(
            ctx.portfolio_daily, ctx.factors, ()
        ),
        "TW_IMPL_PORT_CAPM": lambda: _portfolio_regression(
            ctx.portfolio_daily, ctx.factors, ("mkt_rf",)
        ),
        "TW_IMPL_PORT_FF5MOM": lambda: _portfolio_regression(
            ctx.portfolio_daily, ctx.factors,
            ("mkt_rf", "smb", "hml", "rmw", "cma", "umd"),
        ),
        "TW_PLC_DIRECTION": lambda: placebo_direction_permutation(ctx),
        "TW_PLC_DATES": lambda: placebo_pseudo_entry(ctx),
    }

    rows = []
    for reg in registry.itertuples(index=False):
        test_id = reg.test_id
        eligible = str(reg.inference_eligible).strip().lower() == "yes"
        row = {
            "test_id": test_id,
            "family": reg.family,
            "evidence_tier": reg.evidence_tier,
            "inference_eligible": "yes" if eligible else "no",
            "estimate": np.nan,
            "raw_p": np.nan,
            "t_stat": np.nan,
            "n": 0,
            "status": "blocked_no_executor",
        }
        fn = executors.get(test_id)
        if fn is None:
            rows.append(row)
            continue
        row.update(fn())
        rows.append(row)
    frame = pd.DataFrame(rows)
    terminal = frame["status"].str.startswith(("executed", "blocked"))
    if not terminal.all():
        bad = frame.loc[~terminal, "test_id"].tolist()
        raise LockedExecutionError(f"rows with no terminal status: {bad}")
    return frame
