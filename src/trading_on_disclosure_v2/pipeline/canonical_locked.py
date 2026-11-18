"""The locked streamlined two-clock canonical pipeline (final directive).

Single linear execution path — no stage-selection parameter exists. Human
workflow gates are replaced by the author parse-review attestation (§2);
amendments resolve deterministically (§7); the public/transaction date
hierarchies are frozen (§3-§4); the primary estimand is the 20-day
purchase-minus-sale spread on unique public signals with two-way CGM
inference by household and exact public-entry trading date (§8, §11-§12).

Superseded (preserved, no longer executed by the canonical path):
``pipeline/canonical.py`` and the multi-gate architecture in ``gates.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v2.amendments import (
    DUPLICATE_FLAG,
    LINEAGE_STATUS_COL,
    apply_amendment_hierarchy,
)
from trading_on_disclosure_v2.dates import (
    ANALYSIS_PUBLIC_COL,
    ANALYSIS_PUBLIC_PROXY_COL,
    ANALYSIS_PUBLIC_SOURCE_COL,
    CT_PUBLIC_RAW_COL,
    OFFICIAL_PUBLIC_COL,
    SOURCE_UNUSABLE,
    TRADE_ANALYSIS_COL,
    TRADE_CT_RAW_COL,
    TRADE_OFFICIAL_COL,
    apply_public_date_hierarchy,
    apply_trade_date_hierarchy,
    public_date_overlap_diagnostics,
)
from trading_on_disclosure_v2.firewall import assert_design_only
from trading_on_disclosure_v2.inference.influence import (
    equal_household_estimate,
    household_concentration,
    leave_one_household_out,
)
from trading_on_disclosure_v2.inference.multiplicity import (
    adjust_family,
    validate_result_frame,
)
from trading_on_disclosure_v2.locked_config import LockedConfig, load_locked_config
from trading_on_disclosure_v2.outcomes.factor_models import compute_factor_model_cars
from trading_on_disclosure_v2.outcomes.returns import (
    HORIZONS,
    combine_with_delisting,
    compute_signal_outcomes,
)
from trading_on_disclosure_v2.pipeline.locked_estimands import (
    ENTRY_COLS,
    EstimandContext,
    execute_locked_registry,
    primary_sample,
)
from trading_on_disclosure_v2.pipeline.portfolio_stage import (
    TERMINAL_LIQUIDATE_ALL,
    PortfolioParams,
    simulate_long_after_purchase,
)
from trading_on_disclosure_v2.securities.matching import (
    SecurityMatchError,
    apply_common_stock_screen,
    assign_tiers,
    generate_candidates,
    selection_diagnostics,
)
from trading_on_disclosure_v2.signals.build import (
    SIGNAL_KEY,
    AttritionLedger,
    build_public_signals,
    build_security_entry_events,
)
from trading_on_disclosure_v2.timing import (
    BENCHMARK_ENTRY_COL,
    FOLLOWER_ENTRY_COL,
    close_on_or_next,
    derive_two_clock_entries,
    entry_strictly_after,
)

LOCKED_REGISTRY_RELPATH = Path("configs") / "two_clock_locked_hypothesis_registry.csv"

MODES = ("canonical", "SYNTHETIC_DRY_RUN", "HISTORICAL_SMOKE_TEST")


class LockedStageError(RuntimeError):
    pass


@dataclass
class LockedPipelineContext:
    repo_root: Path
    out_root: Path
    freeze_hash: str
    mode: str
    calendar: pd.DatetimeIndex | None = None
    config: LockedConfig | None = None
    frames: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    ledger: AttritionLedger = field(default_factory=AttritionLedger)
    n_boot: int = 499
    band_draws: int = 199

    def require(self, key: str):
        if key not in self.frames:
            raise LockedStageError(f"stage input {key!r} not produced yet")
        return self.frames[key]


def _s01_validate_inputs(ctx: LockedPipelineContext) -> None:
    if ctx.mode not in MODES:
        raise LockedStageError(f"unknown mode {ctx.mode!r}")
    if ctx.calendar is None or len(ctx.calendar) == 0:
        raise LockedStageError("trading calendar required")
    if ctx.config is None:
        ctx.config = load_locked_config(ctx.repo_root)
    txns = ctx.frames.get("canonical_transactions")
    if txns is None:
        raise LockedStageError("canonical_transactions frame absent")
    for col in (
        "transaction_uid",
        "filer_uid",
        "document_uid",
        "direction",
        OFFICIAL_PUBLIC_COL,
        CT_PUBLIC_RAW_COL,
        TRADE_OFFICIAL_COL,
        TRADE_CT_RAW_COL,
    ):
        if col not in txns.columns:
            raise LockedStageError(f"canonical_transactions missing column {col!r}")
    bad_ids = txns["filer_uid"].isna() | (txns["filer_uid"].astype(str) == "")
    if bad_ids.any():
        raise LockedStageError(
            f"{int(bad_ids.sum())} rows lack a deterministic filer_uid (§6)"
        )
    if ctx.mode == "canonical":
        from trading_on_disclosure_v2.release.authorization import load_attestation

        att = load_attestation(ctx.repo_root)
        if att.status != "APPROVED":
            raise LockedStageError(
                "canonical mode requires the APPROVED author parse-review "
                "attestation (§2); status is " + att.status
            )
        ctx.artifacts["attestation"] = {
            "status": att.status,
            "reviewer_id": att.reviewer_id,
            "date": att.date,
        }


def _s02_date_hierarchies(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("canonical_transactions")
    txns = apply_public_date_hierarchy(txns)
    txns = apply_trade_date_hierarchy(txns)
    # downstream consumers (timing, amendments, signals) read the analysis
    # transaction date under the legacy field name
    txns["transaction_date_reported"] = txns[TRADE_ANALYSIS_COL]
    by_cols = tuple(
        c for c in ("chamber", "document_format", "report_year") if c in txns.columns
    )
    ctx.artifacts["date_proxy_diagnostics"] = public_date_overlap_diagnostics(
        txns, by=by_cols
    )
    ctx.frames["dated_transactions"] = txns


def _s03_sample_window(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("dated_transactions")
    n_in = len(txns)
    unusable = txns[ANALYSIS_PUBLIC_SOURCE_COL] == SOURCE_UNUSABLE
    txns = txns[~unusable]
    sample_start = pd.Timestamp(str(ctx.config["sample_start"]))
    early = txns[ANALYSIS_PUBLIC_COL] < sample_start
    txns = txns[~early]
    ctx.ledger.record(
        "public_date_and_sample_window",
        n_in=n_in,
        n_out=len(txns),
        reasons={
            "no_usable_public_date": int(unusable.sum()),
            "before_sample_start": int(early.sum()),
        },
    )
    ctx.frames["windowed_transactions"] = txns


def _s04_amendments(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("windowed_transactions")
    flagged, summary = apply_amendment_hierarchy(txns)
    kept = flagged[~flagged[DUPLICATE_FLAG]]
    ctx.ledger.record(
        "amendment_exact_duplicates",
        n_in=len(flagged),
        n_out=len(kept),
        reasons={"amendment_exact_duplicate": int(flagged[DUPLICATE_FLAG].sum())},
    )
    ctx.artifacts["amendment_summary"] = summary
    ctx.frames["deduplicated_transactions"] = kept
    ctx.frames["amendment_provenance"] = flagged[
        ["transaction_uid", DUPLICATE_FLAG, LINEAGE_STATUS_COL]
    ]


def _s05_direction_filter(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("deduplicated_transactions")
    n_in = len(txns)
    good = txns["direction"].isin(["purchase", "sale"])
    ctx.ledger.record(
        "purchase_sale_filter",
        n_in=n_in,
        n_out=int(good.sum()),
        reasons={"no_purchase_or_sale_interpretation": int((~good).sum())},
    )
    ctx.frames["directional_transactions"] = txns[good]


def _s06_security_matching(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("directional_transactions")
    names = ctx.require("crsp_names")
    candidates = generate_candidates(txns, names)
    resolved = assign_tiers(txns, candidates, ctx.frames.get("security_reviews"))
    screened = apply_common_stock_screen(resolved, txns, names)
    ctx.frames["security_resolution"] = screened
    ctx.artifacts["security_diagnostics"] = selection_diagnostics(resolved)
    tier_d = int((resolved["match_tier"] == "D").sum())
    ctx.ledger.record(
        "security_matching",
        n_in=len(txns),
        n_out=len(txns) - tier_d,
        reasons={"unresolved_security_match": tier_d},
    )
    accepted = screened[
        screened["match_tier"].isin(["A", "B", "C"])
        & screened["common_stock_screen_pass"]
    ][["transaction_uid", "permno"]]
    dropped_screen = (len(txns) - tier_d) - len(accepted)
    ctx.ledger.record(
        "common_stock_screen",
        n_in=len(txns) - tier_d,
        n_out=len(accepted),
        reasons={"failed_common_stock_screen": dropped_screen},
    )
    dup_accepted = accepted["transaction_uid"].duplicated(keep=False)
    if dup_accepted.any():
        raise SecurityMatchError(
            "accepted security matches carry duplicated transaction_uid: "
            f"{int(dup_accepted.sum())} rows across "
            f"{accepted.loc[dup_accepted, 'transaction_uid'].nunique()} uids"
        )
    matched = txns.drop(columns=["permno"], errors="ignore").merge(
        accepted, on="transaction_uid", how="inner", validate="one_to_one"
    )
    matched["permno"] = matched["permno"].astype(int)
    ctx.frames["matched_transactions"] = matched


def _s07_timing(ctx: LockedPipelineContext) -> None:
    txns = ctx.require("matched_transactions")
    timed = derive_two_clock_entries(
        txns,
        ctx.calendar,
        transaction_col="transaction_date_reported",
        public_col=ANALYSIS_PUBLIC_COL,
    )
    proxy = timed[ANALYSIS_PUBLIC_PROXY_COL].astype(bool)
    raw_public = timed[OFFICIAL_PUBLIC_COL].where(
        ~proxy, timed[CT_PUBLIC_RAW_COL]
    )
    timed[ENTRY_COLS["ERAW"]] = entry_strictly_after(raw_public, ctx.calendar)
    timed[ENTRY_COLS["EPLUS1"]] = entry_strictly_after(
        timed[FOLLOWER_ENTRY_COL], ctx.calendar
    )
    raw_trade = timed[TRADE_CT_RAW_COL].where(
        timed[TRADE_CT_RAW_COL].notna(), timed[TRADE_OFFICIAL_COL]
    )
    timed[ENTRY_COLS["TRAW"]] = close_on_or_next(raw_trade, ctx.calendar)
    ctx.frames["timed_transactions"] = timed


def _s08_signals(ctx: LockedPipelineContext) -> None:
    timed = ctx.require("timed_transactions")
    signals = build_public_signals(timed, ctx.ledger)
    key = list(SIGNAL_KEY)
    carry = [
        c
        for c in (
            FOLLOWER_ENTRY_COL,
            BENCHMARK_ENTRY_COL,
            ENTRY_COLS["ERAW"],
            ENTRY_COLS["EPLUS1"],
            ENTRY_COLS["TRAW"],
            "record_source",
            "source_native_household_id",
            "reporting_lag_days",
            "document_format",
            "chamber",
        )
        if c in timed.columns
    ]
    merged = signals.merge(
        timed[key + carry].drop_duplicates(subset=key), on=key, how="left"
    )
    assert_design_only(merged.columns, "public_signals")
    ctx.artifacts["mixed_direction_signals"] = int(merged["mixed_direction"].sum())
    ctx.frames["signals"] = merged


def _s09_events(ctx: LockedPipelineContext) -> None:
    ctx.frames["events"] = build_security_entry_events(ctx.require("signals"))


def _s10_returns(ctx: LockedPipelineContext) -> None:
    signals = ctx.require("signals")
    daily = ctx.require("daily")
    benchmark = ctx.require("benchmark")
    anchors = (
        ("E", ENTRY_COLS["E"], HORIZONS),
        ("T", ENTRY_COLS["T"], (20,)),
        ("ERAW", ENTRY_COLS["ERAW"], (20,)),
        ("EPLUS1", ENTRY_COLS["EPLUS1"], (20,)),
        ("TRAW", ENTRY_COLS["TRAW"], (20,)),
    )
    with_outcomes = compute_signal_outcomes(
        signals, daily, benchmark, ctx.calendar, anchors=anchors
    )
    factors = ctx.frames.get("factors")
    if factors is not None:
        with_outcomes = compute_factor_model_cars(
            with_outcomes, daily, factors, ctx.calendar, entry_col=ENTRY_COLS["E"]
        )
    ctx.frames["signals"] = with_outcomes
    events = ctx.require("events").merge(
        with_outcomes.groupby(["permno", FOLLOWER_ENTRY_COL])["ar_E_h20"]
        .mean()
        .reset_index(),
        on=["permno", FOLLOWER_ENTRY_COL],
        how="left",
    )
    ctx.frames["events"] = events


def _s11_portfolio(ctx: LockedPipelineContext) -> None:
    signals = primary_sample(ctx.require("signals"))
    purchases = signals[signals["direction"] == "purchase"].dropna(
        subset=[FOLLOWER_ENTRY_COL]
    )
    daily = combine_with_delisting(ctx.require("daily"))
    stock_returns = {
        int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")
    }
    port = ctx.config["portfolio"]
    params = PortfolioParams(
        holding_days=int(port["holding_days"]),
        signal_target_fraction=float(port["security_cap_nav"]),
        security_cap=float(port["security_cap_nav"]),
        household_cap=1.0,  # §18 defines no household cap
        gross_cap=1.0,  # no leverage
        one_way_cost_bp=float(port["round_trip_cost_bps"]) / 2.0,  # half per side
        max_open_lots=int(port["max_open_lots"]),
        terminal_rule=TERMINAL_LIQUIDATE_ALL,
        data_is_real=ctx.mode != "SYNTHETIC_DRY_RUN",
    )
    # the portfolio sample ends where its required inputs end (risk-free and
    # benchmark coverage); the truncation is recorded, never zero-filled
    risk_free = ctx.require("risk_free").dropna()
    benchmark = ctx.require("benchmark").dropna()
    input_end = min(risk_free.index.max(), benchmark.index.max())
    portfolio_calendar = ctx.calendar[ctx.calendar <= input_end]
    if len(portfolio_calendar) < len(ctx.calendar):
        ctx.artifacts["portfolio_sample_truncation"] = {
            "portfolio_sample_end": str(pd.Timestamp(input_end).date()),
            "market_calendar_end": str(pd.Timestamp(ctx.calendar.max()).date()),
            "reason": "risk_free_or_benchmark_coverage_ends",
        }
    result = simulate_long_after_purchase(
        purchases,
        stock_returns,
        benchmark,
        risk_free,
        portfolio_calendar,
        params,
    )
    ctx.frames["portfolio_daily"] = result.daily
    ctx.artifacts["portfolio_lots"] = result.lots
    ctx.artifacts["portfolio_routing"] = result.routing
    ctx.artifacts["portfolio_params"] = {
        "holding_days": params.holding_days,
        "security_cap": params.security_cap,
        "max_open_lots": params.max_open_lots,
        "round_trip_cost_bps": float(port["round_trip_cost_bps"]),
        "terminal_rule": params.terminal_rule,
        "initial_equity": float(result.initial_equity),
    }


def load_locked_registry(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(Path(repo_root) / LOCKED_REGISTRY_RELPATH)


def _s12_estimands(ctx: LockedPipelineContext) -> None:
    registry = load_locked_registry(ctx.repo_root)
    ectx = EstimandContext(
        signals=ctx.require("signals"),
        events=ctx.require("events"),
        portfolio_daily=ctx.frames.get("portfolio_daily"),
        factors=ctx.frames.get("factors"),
        freeze_hash=ctx.freeze_hash,
        daily=ctx.frames.get("daily"),
        benchmark=ctx.frames.get("benchmark"),
        calendar=ctx.calendar,
        n_boot=ctx.n_boot,
    )
    frame = execute_locked_registry(registry, ectx)
    ctx.frames["result_frame"] = frame
    ctx.frames["registry"] = registry


def _s13_multiplicity(ctx: LockedPipelineContext) -> None:
    frame = ctx.require("result_frame")
    executed = frame[frame["status"] == "executed"].copy()
    validate_result_frame(executed, registry=None)
    ctx.frames["adjusted_results"] = adjust_family(executed)


def _s14_influence(ctx: LockedPipelineContext) -> None:
    signals = primary_sample(ctx.require("signals")).dropna(subset=["ar_E_h20"])
    if signals.empty:
        raise LockedStageError("no outcomes for influence diagnostics")
    ctx.artifacts["equal_household"] = equal_household_estimate(
        signals, household_col="filer_uid", value_col="ar_E_h20"
    )
    conc = household_concentration(signals, household_col="filer_uid")
    ctx.artifacts["concentration"] = {k: v for k, v in conc.items() if k != "shares"}
    top = conc["shares"].index[: min(10, len(conc["shares"]))].tolist()
    ctx.artifacts["leave_one_out"] = leave_one_household_out(
        signals,
        household_col="filer_uid",
        value_col="ar_E_h20",
        households=top,
        change_threshold=0.50,
    )


def _s15_event_bands(ctx: LockedPipelineContext) -> None:
    from trading_on_disclosure_v2.inference.event_bands import (
        supt_band_day_multiplier,
    )
    from trading_on_disclosure_v2.pipeline.dryrun_outputs import build_event_time_car

    band_cfg = ctx.config["event_band"]
    window = band_cfg["window"]
    horizon = int(ctx.config["primary_outcome"]["horizon_trading_days"])
    signals = primary_sample(ctx.require("signals")).dropna(
        subset=[FOLLOWER_ENTRY_COL]
    )
    car, keys, direction = build_event_time_car(
        signals,
        ctx.require("daily"),
        ctx.require("benchmark"),
        ctx.calendar,
        pre=abs(int(window[0])),
        post=int(window[1]),
    )
    draws = ctx.band_draws
    results = {}
    for series in ("purchase", "sale", "difference"):
        results[series] = supt_band_day_multiplier(
            car,
            keys,
            direction,
            series,
            ctx.freeze_hash,
            purpose=f"locked_event_band|{series}",
            n_draws=draws,
            bandwidth=horizon,
            centering="null_imposed",
            trading_days=ctx.calendar,
        )
    ctx.artifacts["event_bands"] = results
    diff = results["difference"]
    ctx.artifacts["event_band_runtime_manifest"] = {
        "method_id": "day_mult_nullimp_bwH",
        "estimator": "calendar_day_multiplier_supt",
        "multiplier_distribution": "gaussian_bartlett_smoothed",
        "dependence_kernel": "bartlett",
        "bandwidth": int(diff.meta["bandwidth"]),
        "bandwidth_rule": "fixed_equal_to_registered_horizon",
        "studentization": diff.meta["centering"],
        "bootstrap_draws": int(diff.meta["n_draws"]),
        "draws_match_locked_config": int(diff.meta["n_draws"])
        == int(band_cfg["draws"]),
        "window": [int(window[0]), int(window[1])],
        "axis": "relative_trading_days",
        "relative_days": [int(diff.ks[0]), int(diff.ks[-1])],
        "dependence_grid": "pipeline_trading_calendar",
        "n_grid_days": int(diff.meta["n_grid_days"]),
        "series": list(results.keys()),
        "freeze_hash": ctx.freeze_hash,
        "mode": ctx.mode,
    }


def _s16_ledgers(ctx: LockedPipelineContext) -> None:
    ctx.artifacts["attrition_ledger"] = ctx.ledger.to_frame()


def _s17_outputs(ctx: LockedPipelineContext) -> None:
    from trading_on_disclosure_v2.reporting.locked_outputs import (
        write_canonical_outputs,
    )

    ctx.artifacts["output_paths"] = write_canonical_outputs(ctx)


def _s18_release_manifest(ctx: LockedPipelineContext) -> None:
    from trading_on_disclosure_v2.release.manifests import write_manifest

    frames_dir = ctx.out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in ctx.frames.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(frames_dir / f"{name}.csv", index=False)
    (ctx.out_root / "MODE").write_text(ctx.mode + "\n")
    ctx.artifacts["manifest_path"] = write_manifest(ctx.out_root)


LOCKED_STAGES = (
    ("validate_inputs", _s01_validate_inputs),
    ("date_hierarchies", _s02_date_hierarchies),
    ("sample_window", _s03_sample_window),
    ("amendments", _s04_amendments),
    ("direction_filter", _s05_direction_filter),
    ("security_matching", _s06_security_matching),
    ("timing", _s07_timing),
    ("signals", _s08_signals),
    ("events", _s09_events),
    ("returns", _s10_returns),
    ("portfolio", _s11_portfolio),
    ("estimands", _s12_estimands),
    ("multiplicity", _s13_multiplicity),
    ("influence", _s14_influence),
    ("event_bands", _s15_event_bands),
    ("ledgers", _s16_ledgers),
    ("outputs", _s17_outputs),
    ("release_manifest", _s18_release_manifest),
)


def run_locked_pipeline(ctx: LockedPipelineContext) -> LockedPipelineContext:
    """Run ALL locked stages in order; fail rather than silently skip (§24)."""
    for name, fn in LOCKED_STAGES:
        try:
            fn(ctx)
        except Exception as exc:
            raise LockedStageError(f"stage {name!r} failed: {exc}") from exc
    return ctx
