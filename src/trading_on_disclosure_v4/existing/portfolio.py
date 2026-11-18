"""The funded long-after-purchase portfolio on the de-duplicated signal set.

The v2 constructor `portfolio_stage.simulate_long_after_purchase` is called BY IMPORT on a substituted
purchase-signal frame with the released frozen rule (`configs/top_journal_two_clock_locked.yaml`
`portfolio:` block as `canonical_locked._s11_portfolio` reads it): entry at the public-entry close,
20-trading-day hold, equal target notional of 5% of account value, 5% security cap, no household cap,
no leverage, 50 open lots, 10 bp round trip charged half per side, cash at the risk-free rate, full
liquidation at the sample end; the portfolio calendar ends where the risk-free or benchmark series
ends. Market inputs are loaded as the release loads them (`outcomes.panel`). The released signal set
reproduces `outputs/frames/portfolio_daily.csv` and the three registry rows (asserted); the
Newey-West lag-20 mean excess return and the CAPM / FF5+MOM alphas (annualized) are
`locked_estimands._portfolio_regression` unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.pipeline import locked_estimands as le
from trading_on_disclosure_v2.pipeline.portfolio_stage import (
    TERMINAL_LIQUIDATE_ALL,
    PortfolioParams,
    PortfolioResult,
    simulate_long_after_purchase,
)
from trading_on_disclosure_v2.reporting.locked_outputs import annualized_total_return
from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.outcomes import panel

RELEASED_DAILY_PATH = panel.FRAMES / "portfolio_daily.csv"
DAILY_PATH = rerun.OUT_DIR / "portfolio_dedup_daily.csv"
SUMMARY_PATH = rerun.OUT_DIR / "portfolio_dedup_summary.csv"
INITIAL_EQUITY = 1_000_000.0
REGRESSIONS = {"TW_IMPL_PORT_NW": (), "TW_IMPL_PORT_CAPM": ("mkt_rf",),
               "TW_IMPL_PORT_FF5MOM": ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")}
REG_FIELDS = ("estimate", "se_multiway", "t_stat", "raw_p", "ci_low", "ci_high", "n")
LEDGER_KEYS = {"annualized_return": "t6.annualized_return", "benchmark_annualized_return": "t6.benchmark_annualized_return",
               "avg_invested_fraction": "t6.avg_invested_fraction", "max_security_exposure": "t6.max_security_exposure",
               "signals_accepted": "t6.signals_accepted", "signals_rejected": "t6.signals_rejected",
               "n_lots": "t6.n_lots", "annual_turnover": "t6.annual_turnover", "sharpe": "t6.sharpe"}


def released_params(config_path: Path | None = None) -> PortfolioParams:
    """`PortfolioParams` exactly as `_s11_portfolio` builds them from the locked config."""
    from trading_on_disclosure_v2.locked_config import load_locked_config

    port = load_locked_config(config_path.parents[1] if config_path else None).data["portfolio"]
    return PortfolioParams(holding_days=int(port["holding_days"]), signal_target_fraction=float(port["security_cap_nav"]),
                           security_cap=float(port["security_cap_nav"]), household_cap=1.0, gross_cap=1.0,
                           one_way_cost_bp=float(port["round_trip_cost_bps"]) / 2.0,
                           max_open_lots=int(port["max_open_lots"]), terminal_rule=TERMINAL_LIQUIDATE_ALL, data_is_real=True)


def load_market() -> dict:
    """Combined daily panel as a per-permno return dict, benchmark, risk-free, factors, calendars."""
    daily = panel.load_local_daily()
    stock_returns = {int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")}
    benchmark, factors = panel.load_benchmark().dropna(), panel.load_factors()
    risk_free = factors["rf"].dropna()
    calendar = panel.v2_trading_calendar()
    end = min(risk_free.index.max(), benchmark.index.max())
    return {"stock_returns": stock_returns, "benchmark": benchmark, "risk_free": risk_free, "factors": factors,
            "calendar": calendar, "portfolio_calendar": calendar[calendar <= end]}


def purchases_of(signals: pd.DataFrame) -> pd.DataFrame:
    """The constructor's input: non-mixed purchase signals with a public-entry date."""
    prim = le.primary_sample(signals)
    return prim[prim["direction"] == "purchase"].dropna(subset=[FOLLOWER_ENTRY_COL])


def simulate(signals: pd.DataFrame, market: dict, params: PortfolioParams) -> PortfolioResult:
    return simulate_long_after_purchase(purchases_of(signals), market["stock_returns"], market["benchmark"],
                                        market["risk_free"], market["portfolio_calendar"], params, INITIAL_EQUITY)


def summarize(result: PortfolioResult, factors: pd.DataFrame, risk_free: pd.Series, n_eligible: int) -> dict:
    """`locked_outputs.build_numerical_ledger`'s t6 block plus the three regressions (decimal units)."""
    d = result.daily
    n = len(d)
    out = {"n_days": int(n), "n_eligible_purchases": int(n_eligible),
           "annualized_return": annualized_total_return(d["equity"].iloc[-1], result.initial_equity, n),
           "benchmark_annualized_return": annualized_total_return(d["benchmark_equity"].iloc[-1], result.initial_equity, n),
           "avg_invested_fraction": float(d["risky_fraction"].mean()),
           "max_security_exposure": float(d["max_security_exposure_frac"].max())}
    counts = result.routing["state"].value_counts()
    out["signals_accepted"] = int(counts.get("accepted", 0))
    out["signals_rejected"] = int(len(result.routing) - out["signals_accepted"])
    out["n_lots"] = int(len(result.lots))
    traded = float(result.lots["entry_notional"].sum() + result.lots["final_value"].sum())
    out["annual_turnover"] = float(traded / 2 / d["equity"].mean() / (n / 252))
    excess = (d["strategy_return"] - risk_free.reindex(d.index)).dropna()
    sd = float(excess.std(ddof=1))
    out["sharpe"] = float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    for test_id, cols in REGRESSIONS.items():
        r = le._portfolio_regression(d, factors, cols)
        if r.get("status") != "executed":
            raise RuntimeError(f"{test_id}: {r.get('status')}")
        out.update({f"{test_id}.{k}": r[k] for k in REG_FIELDS})
    return out


def account_path(result: PortfolioResult, risk_free: pd.Series) -> pd.DataFrame:
    """date, nav, strategy return, excess return, benchmark nav, invested fraction, open lots (no ids)."""
    d = result.daily
    return pd.DataFrame({"date": d.index, "nav": d["equity"].to_numpy(), "strategy_return": d["strategy_return"].to_numpy(),
                         "excess_return": (d["strategy_return"] - risk_free.reindex(d.index)).to_numpy(),
                         "benchmark_nav": d["benchmark_equity"].to_numpy(), "risky_fraction": d["risky_fraction"].to_numpy(),
                         "n_open_lots": d["n_open_lots"].to_numpy()})


def check_released(result: PortfolioResult, summary: dict, released_daily_path: Path = RELEASED_DAILY_PATH,
                   registry_path: Path = rerun.REGISTRY_PATH, ledger_path: Path | None = None,
                   tol: float = 1e-8) -> dict:
    """The released signal set must reproduce the released account path (returns and fractions to
    ``tol``; dollar columns to ``tol`` relative to the initial equity) and the three registry rows."""
    rel = pd.read_csv(released_daily_path)
    if len(rel) != len(result.daily):
        raise AssertionError(f"released path has {len(rel)} days, reproduction {len(result.daily)}")
    diffs = {}
    for c in rel.columns:
        scale = INITIAL_EQUITY if c in ("cash", "positions_value", "equity", "benchmark_equity", "interest", "costs_paid") else 1.0
        diffs[f"daily.{c}"] = float(np.nanmax(np.abs(result.daily[c].to_numpy(float) - rel[c].to_numpy(float))) / scale)
    reg = pd.read_csv(registry_path).set_index("test_id")
    for test_id in REGRESSIONS:
        for k in ("estimate", "se_multiway", "raw_p"):
            diffs[f"{test_id}.{k}"] = abs(summary[f"{test_id}.{k}"] - float(reg.loc[test_id, k]))
        if int(summary[f"{test_id}.n"]) != int(reg.loc[test_id, "n"]):
            raise AssertionError(f"{test_id}: n differs from the registry")
    if ledger_path is not None:
        ledger = json.loads(ledger_path.read_text())
        for field, key in LEDGER_KEYS.items():
            diffs[key] = abs(float(summary[field]) - float(ledger[key]))
    bad = {k: v for k, v in diffs.items() if not v < tol}
    if bad:
        raise AssertionError(f"released portfolio not reproduced: {bad}")
    return diffs


def main(argv: list[str] | None = None) -> pd.DataFrame:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="funded long-after-purchase portfolio, released and de-duplicated signal sets")
    ap.add_argument("--daily-out", type=Path, default=DAILY_PATH)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    args = ap.parse_args(argv)
    t0 = time.time()
    signals = rerun.load_signals()
    dedup, report = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    market, params = load_market(), released_params()
    ledger_path = panel.REPO_ROOT / "outputs" / "canonical" / "numerical_ledger.json"
    rows = []
    for sample, frame in (("original", signals), ("dedup", dedup)):
        res = simulate(frame, market, params)
        summary = summarize(res, market["factors"], market["risk_free"], len(purchases_of(frame)))
        if sample == "original":
            diffs = check_released(res, summary, ledger_path=ledger_path)
            report["max_abs_released_reproduction_diff"] = max(diffs.values())
        else:
            account_path(res, market["risk_free"]).to_csv(args.daily_out, index=False, date_format="%Y-%m-%d")
        rows.append({"sample": sample, **summary})
        print(f"{sample}: accepted {summary['signals_accepted']} of {summary['n_eligible_purchases']}, {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.summary_out, index=False)
    report.update({"portfolio_sample_end": str(market["portfolio_calendar"].max().date()),
                   "params": {"holding_days": params.holding_days, "signal_target_fraction": params.signal_target_fraction,
                              "security_cap": params.security_cap, "max_open_lots": params.max_open_lots,
                              "round_trip_cost_bps": 2 * params.one_way_cost_bp, "terminal_rule": params.terminal_rule,
                              "initial_equity": INITIAL_EQUITY}})
    args.summary_out.with_name("portfolio_dedup_report.json").write_text(json.dumps(report, indent=2))
    print(out.T.to_string())
    return out


if __name__ == "__main__":
    main()
