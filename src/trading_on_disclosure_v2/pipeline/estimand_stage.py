"""Registry-driven estimand execution producing the canonical result frame.

Every inference-eligible row of the frozen hypothesis registry is executed
or carries an explicit blocked reason; deleting or substituting rows is
impossible because the registry frame drives the loop (MUL-001/002,
registry §12.1). Headline covariance is the three-way CGM (INF-002) with a
calendar-block companion (INF-006); shared draws serve Romano-Wolf within
the P1 family (INF-008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.inference.block_bootstrap import (
    calendar_block_bootstrap,
)
from trading_on_disclosure_v2.inference.cgm3 import three_way_cluster_cov
from trading_on_disclosure_v2.timing import (
    BENCHMARK_ENTRY_COL,
    FOLLOWER_ENTRY_COL,
)

HEADLINE_CLUSTERS = ("filer_uid", "permno", "follower_entry_date")


class RegistryExecutionError(RuntimeError):
    pass


def _purchase_regression(
    frame: pd.DataFrame, outcome_col: str, entry_col: str
) -> dict:
    """AR ~ 1 + purchase with three-way CGM covariance; returns row stats."""
    data = frame.dropna(subset=[outcome_col]).copy()
    if data.empty or data["direction"].nunique() < 2:
        return {"status": "blocked_insufficient_data", "n": len(data)}
    y = data[outcome_col].to_numpy(float)
    purchase = (data["direction"] == "purchase").to_numpy(float)
    X = np.column_stack([np.ones(len(y)), purchase])
    clusters = pd.DataFrame(
        {
            "filer_uid": data["filer_uid"].to_numpy(),
            "permno": data["permno"].to_numpy(),
            "follower_entry_date": data[entry_col].to_numpy(),
        }
    )
    res = three_way_cluster_cov(y, X, clusters)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    se = float(np.sqrt(res.cov[1, 1]))
    t_stat = float(beta[1] / se) if se > 0 else np.nan
    from scipy import stats

    p = float(2 * stats.t.sf(abs(t_stat), df=res.dof)) if np.isfinite(t_stat) else np.nan
    return {
        "status": "executed",
        "estimate": float(beta[1]),
        "se_multiway": se,
        "t_stat": t_stat,
        "raw_p": p,
        "n": int(len(data)),
        "dof": int(res.dof),
    }


def _spread_stat(outcome_col: str):
    def stat(df: pd.DataFrame) -> float:
        buys = df.loc[df["direction"] == "purchase", outcome_col]
        sells = df.loc[df["direction"] == "sale", outcome_col]
        if buys.dropna().empty or sells.dropna().empty:
            return np.nan
        return float(buys.mean() - sells.mean())

    return stat


def execute_registry(
    registry: pd.DataFrame,
    signals: pd.DataFrame,
    events: pd.DataFrame,
    portfolio_daily: pd.DataFrame | None,
    factors: pd.DataFrame | None,
    freeze_hash: str,
    n_boot: int = 499,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (canonical result frame, shared P1 bootstrap t-draws)."""
    rows = []
    p1_draws: dict[str, np.ndarray] = {}

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
            "status": "not_inference_eligible",
        }
        if str(getattr(reg, "status_before_freeze", "")).startswith("blocked"):
            row["status"] = f"blocked_by_registry:{reg.status_before_freeze}"
            rows.append(row)
            continue
        if not eligible:
            rows.append(row)
            continue

        stats_out: dict | None = None
        if test_id in ("H_P1_01", "L_P1_01"):
            stats_out = _purchase_regression(signals, "ar_E_h20", FOLLOWER_ENTRY_COL)
            if stats_out.get("status") == "executed":
                boot = calendar_block_bootstrap(
                    signals.dropna(subset=["ar_E_h20"]),
                    FOLLOWER_ENTRY_COL,
                    _spread_stat("ar_E_h20"),
                    freeze_hash,
                    "p1_family_shared",
                    n_draws=n_boot,
                )
                stats_out["block_ci_low"], stats_out["block_ci_high"] = boot.percentile_interval()
                sd = float(np.nanstd(boot.statistics))
                p1_draws[test_id] = (
                    (boot.statistics - np.nanmean(boot.statistics)) / sd if sd > 0 else boot.statistics * 0
                )
        elif test_id == "H_C1_01":
            stats_out = _purchase_regression(signals, "ar_T_h20", BENCHMARK_ENTRY_COL)
        elif test_id == "H_C1_02":
            stats_out = _purchase_regression(signals, "ar_D_h1", FOLLOWER_ENTRY_COL)
        elif test_id.startswith("H_S2_") and test_id[5:].isdigit():
            h = int(test_id[5:])
            stats_out = _purchase_regression(signals, f"ar_E_h{h}", FOLLOWER_ENTRY_COL)
        elif test_id == "H_S2_EW":
            if "ar_EW_h20" in signals.columns:
                stats_out = _purchase_regression(signals, "ar_EW_h20", FOLLOWER_ENTRY_COL)
            else:
                stats_out = {"status": "blocked_missing_input:equal_weighted_benchmark"}
        elif test_id in ("H_S3_01", "H_S3_02"):
            moderator = (
                "reporting_lag_days" if test_id == "H_S3_01" else "is_scanned_format"
            )
            stats_out = _interaction_regression(signals, moderator)
        elif test_id == "H_S1_01":
            stats_out = _event_level(events)
        elif test_id == "H_S1_02":
            stats_out = _equal_household(signals)
        elif test_id in ("H_REC_01", "H_REC_02"):
            stats_out = _event_level(events)
        elif test_id in ("H_P1_02", "L_P1_02"):
            stats_out = _portfolio_alpha(portfolio_daily, factors)
            if stats_out.get("status") == "executed" and portfolio_daily is not None:
                excess = (
                    portfolio_daily["strategy_return"]
                    - factors["rf"].reindex(portfolio_daily.index)
                ).dropna()
                boot_frame = excess.rename("x").reset_index().rename(columns={"index": "date"})
                boot = calendar_block_bootstrap(
                    boot_frame,
                    "date",
                    lambda df: float(df["x"].mean()),
                    freeze_hash,
                    "p1_portfolio",
                    n_draws=n_boot,
                )
                stats_out["block_ci_low"], stats_out["block_ci_high"] = boot.percentile_interval()
                sd = float(np.nanstd(boot.statistics))
                p1_draws[test_id] = (
                    (boot.statistics - np.nanmean(boot.statistics)) / sd if sd > 0 else boot.statistics * 0
                )
        else:
            stats_out = {"status": "blocked_no_executor"}

        if stats_out is None:
            stats_out = {"status": "blocked_no_executor"}
        row.update(stats_out)
        rows.append(row)

    frame = pd.DataFrame(rows)
    executed_or_blocked = frame["status"].str.startswith(
        ("executed", "blocked", "not_inference_eligible")
    )
    if not executed_or_blocked.all():
        bad = frame.loc[~executed_or_blocked, "test_id"].tolist()
        raise RegistryExecutionError(f"rows with no terminal status: {bad}")
    draws = pd.DataFrame(p1_draws) if p1_draws else pd.DataFrame()
    return frame, draws


def _interaction_regression(signals: pd.DataFrame, moderator: str) -> dict:
    if moderator == "is_scanned_format":
        signals = signals.copy()
        signals[moderator] = (
            signals.get("document_format", pd.Series(index=signals.index, dtype=object))
            .astype(str)
            .str.startswith("scanned")
            .astype(float)
        )
    if moderator not in signals.columns:
        return {"status": f"blocked_missing_input:{moderator}"}
    data = signals.dropna(subset=["ar_E_h20", moderator]).copy()
    if data.empty or data["direction"].nunique() < 2:
        return {"status": "blocked_insufficient_data", "n": len(data)}
    y = data["ar_E_h20"].to_numpy(float)
    purchase = (data["direction"] == "purchase").to_numpy(float)
    m = data[moderator].to_numpy(float)
    m = (m - m.mean()) / (m.std() or 1.0)
    X = np.column_stack([np.ones(len(y)), purchase, m, purchase * m])
    clusters = pd.DataFrame(
        {
            "filer_uid": data["filer_uid"].to_numpy(),
            "permno": data["permno"].to_numpy(),
            "follower_entry_date": data[FOLLOWER_ENTRY_COL].to_numpy(),
        }
    )
    res = three_way_cluster_cov(y, X, clusters)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    se = float(np.sqrt(res.cov[3, 3]))
    t_stat = float(beta[3] / se) if se > 0 else np.nan
    from scipy import stats

    p = float(2 * stats.t.sf(abs(t_stat), df=res.dof)) if np.isfinite(t_stat) else np.nan
    return {
        "status": "executed",
        "estimate": float(beta[3]),
        "se_multiway": se,
        "t_stat": t_stat,
        "raw_p": p,
        "n": int(len(data)),
    }


def _event_level(events: pd.DataFrame) -> dict:
    data = events.dropna(subset=["ar_E_h20"]).copy()
    pure_buy = data[
        (data["purchase_household_count"] > 0) & (data["sale_household_count"] == 0)
    ]["ar_E_h20"]
    pure_sell = data[
        (data["sale_household_count"] > 0) & (data["purchase_household_count"] == 0)
    ]["ar_E_h20"]
    if pure_buy.empty or pure_sell.empty:
        return {"status": "blocked_insufficient_data", "n": len(data)}
    diff = float(pure_buy.mean() - pure_sell.mean())
    se = float(
        np.sqrt(pure_buy.var(ddof=1) / len(pure_buy) + pure_sell.var(ddof=1) / len(pure_sell))
    )
    from scipy import stats

    t_stat = diff / se if se > 0 else np.nan
    dof = max(len(pure_buy) + len(pure_sell) - 2, 1)
    return {
        "status": "executed",
        "estimate": diff,
        "se_multiway": se,
        "t_stat": t_stat,
        "raw_p": float(2 * stats.t.sf(abs(t_stat), df=dof)),
        "n": int(len(data)),
    }


def _equal_household(signals: pd.DataFrame) -> dict:
    data = signals.dropna(subset=["ar_E_h20"])
    per_hh = (
        data.groupby(["filer_uid", "direction"])["ar_E_h20"].mean().unstack("direction")
    )
    diffs = (per_hh.get("purchase") - per_hh.get("sale")).dropna()
    if len(diffs) < 3:
        return {"status": "blocked_insufficient_data", "n": len(diffs)}
    from scipy import stats

    t_stat, p = stats.ttest_1samp(diffs, 0.0)
    return {
        "status": "executed",
        "estimate": float(diffs.mean()),
        "se_multiway": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "t_stat": float(t_stat),
        "raw_p": float(p),
        "n": int(len(diffs)),
    }


def _portfolio_alpha(daily: pd.DataFrame | None, factors: pd.DataFrame | None) -> dict:
    if daily is None or factors is None:
        return {"status": "blocked_missing_input:portfolio_or_factors"}
    frame = daily.join(factors, how="inner").dropna(
        subset=["strategy_return", "mkt_rf", "rf"]
    )
    if len(frame) < 60:
        return {"status": "blocked_insufficient_data", "n": len(frame)}
    y = (frame["strategy_return"] - frame["rf"]).to_numpy(float)
    X = np.column_stack(
        [np.ones(len(y))]
        + [frame[c].to_numpy(float) for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")]
    )
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    # Newey-West lag 20 with Bartlett weights (contract HAC minimum)
    lag = 20
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    for ell in range(1, lag + 1):
        w = 1 - ell / (lag + 1)
        gamma = (X[ell:] * resid[ell:, None]).T @ (X[:-ell] * resid[:-ell, None])
        meat += w * (gamma + gamma.T)
    cov = bread @ meat @ bread * n / max(n - k, 1)
    se = float(np.sqrt(cov[0, 0]))
    from scipy import stats

    t_stat = float(beta[0] / se) if se > 0 else np.nan
    return {
        "status": "executed",
        "estimate": float(beta[0] * 252),  # annualized daily alpha
        "se_multiway": se,
        "t_stat": t_stat,
        "raw_p": float(2 * stats.t.sf(abs(t_stat), df=n - k)),
        "n": int(n),
    }
