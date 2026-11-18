"""The canonical two-clock pipeline: 30 declared stages, one orchestrator.

``run_pipeline`` executes every stage in canonical order; there is no stage
selection parameter anywhere in this module, and the entry-point CLI exposes
none (selective generation prohibited). Stages that require human-gated
inputs consume only sealed/adjudicated artifacts and hard-fail with a clear
message when those are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.firewall import assert_design_only
from trading_on_disclosure_v2.inference.equivalence import SESOIBound, tost_equivalence
from trading_on_disclosure_v2.inference.influence import (
    equal_household_estimate,
    household_concentration,
    leave_one_household_out,
)
from trading_on_disclosure_v2.inference.mde import mde_from_se
from trading_on_disclosure_v2.inference.multiplicity import (
    adjust_family,
    validate_result_frame,
)
from trading_on_disclosure_v2.outcomes.returns import (
    combine_with_delisting,
    compute_signal_outcomes,
)
from trading_on_disclosure_v2.pipeline.estimand_stage import execute_registry
from trading_on_disclosure_v2.pipeline.portfolio_stage import (
    PortfolioParams,
    simulate_long_after_purchase,
)
from trading_on_disclosure_v2.reporting.shells import build_all_shells
from trading_on_disclosure_v2.securities.matching import (
    apply_common_stock_screen,
    assign_tiers,
    generate_candidates,
    selection_diagnostics,
)
from trading_on_disclosure_v2.signals.build import (
    AttritionLedger,
    build_public_signals,
    build_security_entry_events,
)
from trading_on_disclosure_v2.timing import (
    FOLLOWER_ENTRY_COL,
    PUBLIC_DATE_COL,
    derive_two_clock_entries,
)


class StageError(RuntimeError):
    pass


class HumanArtifactMissing(StageError):
    """A stage needs a sealed/adjudicated human artifact that is absent."""


@dataclass
class PipelineContext:
    repo_root: Path
    out_root: Path
    freeze_hash: str
    mode: str  # "canonical" | "SYNTHETIC_DRY_RUN" | "HISTORICAL_SMOKE_TEST"
    calendar: pd.DatetimeIndex | None = None
    frames: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    ledger: AttritionLedger = field(default_factory=AttritionLedger)
    portfolio_params: PortfolioParams | None = None

    def require(self, key: str) -> object:
        if key not in self.frames:
            raise StageError(f"stage input {key!r} not produced yet")
        return self.frames[key]

    def stage_dir(self, name: str) -> Path:
        path = self.out_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class Stage:
    number: int
    key: str
    title: str
    fn: Callable[[PipelineContext], None]
    needs_human_artifacts: str | None = None


def _s01_census(ctx: PipelineContext) -> None:
    census = ctx.frames.get("census")
    if census is None:
        raise StageError("official census frame absent from context")
    ctx.artifacts["census_rows"] = len(census)


def _s02_document_states(ctx: PipelineContext) -> None:
    census = ctx.require("census")
    nonterminal = census[~census["terminal"]] if "terminal" in census.columns else census.iloc[0:0]
    if len(nonterminal):
        raise StageError(f"{len(nonterminal)} nonterminal documents (SRC-004)")


def _s03_amendment_lineage(ctx: PipelineContext) -> None:
    if ctx.mode == "canonical" and "amendment_decisions" not in ctx.frames:
        raise HumanArtifactMissing(
            "amendment lineage requires the completed two-judgment review "
            "(VER-001); sealed decisions are absent"
        )
    ctx.artifacts["amendment_decisions"] = len(ctx.frames.get("amendment_decisions", ()))


def _s04_transcription_import(ctx: PipelineContext) -> None:
    if ctx.mode == "canonical" and "sealed_transcriptions" not in ctx.frames:
        raise HumanArtifactMissing(
            "transcription import requires sealed, adjudicated Mode A output; "
            "none present in the locked namespace"
        )


def _s05_identity(ctx: PipelineContext) -> None:
    txns = ctx.require("canonical_transactions")
    if txns["filer_uid"].isna().any() or (txns["filer_uid"] == "").any():
        raise StageError("unresolved filer_uid rows in canonical transactions (ID-001)")


def _s06_security_matching(ctx: PipelineContext) -> None:
    txns = ctx.require("canonical_transactions")
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
        reasons={"tier_D_excluded": tier_d},
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
    # the RESOLVED permno is authoritative; input frames may lack one entirely
    matched = txns.drop(columns=["permno"], errors="ignore").merge(
        accepted, on="transaction_uid", how="inner"
    )
    matched["permno"] = matched["permno"].astype(int)
    ctx.frames["matched_transactions"] = matched


def _s07_first_sufficient(ctx: PipelineContext) -> None:
    # first sufficient disclosure per economic transaction: earliest public
    # date within each (filer, permno, direction, transaction_date) group
    txns = ctx.require("matched_transactions").copy()
    txns = txns.sort_values(PUBLIC_DATE_COL, kind="mergesort")
    key = ["filer_uid", "permno", "direction", "transaction_date_reported"]
    first = txns.drop_duplicates(subset=key, keep="first")
    ctx.ledger.record(
        "first_sufficient_disclosure",
        n_in=len(txns),
        n_out=len(first),
        reasons={"superseded_versions_collapsed": len(txns) - len(first)},
    )
    ctx.frames["first_sufficient"] = first


def _s08_s09_timing(ctx: PipelineContext) -> None:
    txns = ctx.require("first_sufficient")
    ctx.frames["timed_transactions"] = derive_two_clock_entries(txns, ctx.calendar)


def _s10_signals(ctx: PipelineContext) -> None:
    timed = ctx.require("timed_transactions")
    signals = build_public_signals(timed, ctx.ledger, public_date_col=PUBLIC_DATE_COL)
    key = ["filer_uid", "document_uid", "permno", "direction", PUBLIC_DATE_COL]
    carry = [
        FOLLOWER_ENTRY_COL,
        "transaction_benchmark_entry_date",
        "transaction_date_reported",
    ] + [
        c
        for c in ("reporting_lag_days", "document_format")
        if c in timed.columns
    ]
    merged = signals.merge(
        timed[key + carry].drop_duplicates(subset=key), on=key, how="left"
    )
    assert_design_only(merged.columns, "public_signals")
    ctx.frames["signals"] = merged


def _s11_events(ctx: PipelineContext) -> None:
    ctx.frames["events"] = build_security_entry_events(ctx.require("signals"))


def _s12_returns(ctx: PipelineContext) -> None:
    signals = ctx.require("signals")
    daily = ctx.require("daily")
    benchmark = ctx.require("benchmark")
    with_outcomes = compute_signal_outcomes(signals, daily, benchmark, ctx.calendar)
    # historical disclosure-day market response (noninvestable, C1 context):
    # single-day AR on the first trading day on or after the public date
    combined = combine_with_delisting(daily)
    by_permno = {
        int(p): g.set_index("date")["total_ret"] for p, g in combined.groupby("permno")
    }
    d_day, d_reason = [], []
    for row in with_outcomes.itertuples(index=False):
        pub = pd.Timestamp(getattr(row, PUBLIC_DATE_COL))
        pos = ctx.calendar.searchsorted(pub, side="left")
        if pos >= len(ctx.calendar):
            d_day.append(np.nan)
            d_reason.append("end_of_sample_truncation")
            continue
        day = ctx.calendar[pos]
        r = by_permno.get(int(row.permno), pd.Series(dtype=float)).get(day, np.nan)
        b = benchmark.get(day, np.nan)
        if np.isnan(r) or np.isnan(b):
            d_day.append(np.nan)
            d_reason.append("missing_return_days")
        else:
            d_day.append(float(r - b))
            d_reason.append("ok")
    with_outcomes["ar_D_h1"] = d_day
    with_outcomes["ar_D_h1_reason"] = d_reason
    ctx.frames["signals"] = with_outcomes
    events = ctx.require("events").merge(
        with_outcomes.groupby(["permno", FOLLOWER_ENTRY_COL])["ar_E_h20"]
        .mean()
        .reset_index(),
        on=["permno", FOLLOWER_ENTRY_COL],
        how="left",
    )
    ctx.frames["events"] = events


def _s22_portfolio(ctx: PipelineContext) -> None:
    signals = ctx.require("signals")
    purchases = signals[signals["direction"] == "purchase"].dropna(
        subset=[FOLLOWER_ENTRY_COL]
    )
    daily = combine_with_delisting(ctx.require("daily"))
    stock_returns = {
        int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")
    }
    params = ctx.portfolio_params or PortfolioParams(
        terminal_rule=None if ctx.mode == "canonical" else
        "SYNTHETIC_RULE_exclude_incomplete_exits",
        data_is_real=ctx.mode == "canonical",
    )
    result = simulate_long_after_purchase(
        purchases,
        stock_returns,
        ctx.require("benchmark"),
        ctx.require("risk_free"),
        ctx.calendar,
        params,
    )
    ctx.frames["portfolio_daily"] = result.daily
    ctx.artifacts["portfolio_lots"] = result.lots
    ctx.artifacts["portfolio_routing"] = result.routing


def _s13_estimands(ctx: PipelineContext) -> None:
    from trading_on_disclosure_v2.reporting.shells import load_hypothesis_registry

    registry = load_hypothesis_registry(ctx.repo_root)
    frame, draws = execute_registry(
        registry,
        ctx.require("signals"),
        ctx.require("events"),
        ctx.frames.get("portfolio_daily"),
        ctx.frames.get("factors"),
        ctx.freeze_hash,
    )
    ctx.frames["result_frame"] = frame
    ctx.frames["p1_draws"] = draws
    ctx.frames["registry"] = registry


def _s17_multiplicity(ctx: PipelineContext) -> None:
    frame = ctx.require("result_frame")
    executed = frame[frame["status"] == "executed"].copy()
    validate_result_frame(executed, registry=None)
    draws = ctx.require("p1_draws")
    adjusted = adjust_family(
        executed, t_boot=draws if len(draws) else None
    )
    ctx.frames["adjusted_results"] = adjusted


def _s18_equivalence(ctx: PipelineContext) -> None:
    adjusted = ctx.require("adjusted_results")
    rows = []
    bound = SESOIBound(
        value=0.0025,
        unit="decimal_return",
        rationale="candidate delta_E20 bound (25 bp); approval PENDING HS-03",
        approval_status="PENDING_HUMAN_SIGNOFF",
        signatory=None,
        freeze_hash=ctx.freeze_hash,
    )
    for row in adjusted.itertuples(index=False):
        if row.test_id not in ("H_P1_01", "L_P1_01"):
            continue
        se = getattr(row, "se_multiway", np.nan)
        if not np.isfinite(se) or se <= 0:
            continue
        n = int(getattr(row, "n", 0) or 0)
        result = tost_equivalence(
            float(row.estimate), float(se), bound, df=max(n - 2, 1)
        )
        rows.append(
            {
                "test_id": row.test_id,
                "estimate": result.estimate,
                "p_lower": result.p_lower,
                "p_upper": result.p_upper,
                "tost_reject": result.tost_reject,
                "decision": result.decision,
                "advisory": result.advisory,
                "bound_status": bound.approval_status,
            }
        )
    ctx.frames["equivalence"] = pd.DataFrame(rows)


def _s19_power(ctx: PipelineContext) -> None:
    adjusted = ctx.require("adjusted_results")
    rows = []
    for row in adjusted.itertuples(index=False):
        se = getattr(row, "se_multiway", np.nan)
        if np.isfinite(se) and se > 0:
            rows.append(
                {
                    "test_id": row.test_id,
                    **mde_from_se(float(se)),
                }
            )
    ctx.frames["mde"] = pd.DataFrame(rows)


def _s20_influence(ctx: PipelineContext) -> None:
    signals = ctx.require("signals").dropna(subset=["ar_E_h20"])
    if signals.empty:
        raise StageError("no outcomes for influence diagnostics")
    ctx.artifacts["equal_household"] = equal_household_estimate(
        signals, household_col="filer_uid", value_col="ar_E_h20"
    )
    conc = household_concentration(signals, household_col="filer_uid")
    ctx.artifacts["concentration"] = {
        k: v for k, v in conc.items() if k != "shares"
    }
    top = conc["shares"].index[: min(10, len(conc["shares"]))].tolist()
    ctx.artifacts["leave_one_out"] = leave_one_household_out(
        signals,
        household_col="filer_uid",
        value_col="ar_E_h20",
        households=top,
        change_threshold=0.50,
    )


def _s26_ledgers(ctx: PipelineContext) -> None:
    ctx.artifacts["attrition_ledger"] = ctx.ledger.to_frame()


def _s27_tables_figures(ctx: PipelineContext) -> None:
    ctx.artifacts["shells"] = build_all_shells(ctx.repo_root)


def _s30_release_manifest(ctx: PipelineContext) -> None:
    from trading_on_disclosure_v2.release.manifests import write_manifest

    for name, frame in ctx.frames.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(ctx.stage_dir("frames") / f"{name}.csv", index=False)
    for name, artifact in ctx.artifacts.items():
        if isinstance(artifact, pd.DataFrame):
            artifact.to_csv(ctx.stage_dir("artifacts") / f"{name}.csv", index=False)
    (ctx.out_root / "MODE").write_text(ctx.mode + "\n")
    ctx.artifacts["manifest_path"] = write_manifest(ctx.out_root)


def _noop_factory(what: str) -> Callable[[PipelineContext], None]:
    def _fn(ctx: PipelineContext) -> None:
        ctx.artifacts.setdefault("stage_notes", []).append(what)

    return _fn


STAGES: tuple[Stage, ...] = (
    Stage(1, "census", "official-document census", _s01_census),
    Stage(2, "document_states", "document-state resolution", _s02_document_states),
    Stage(3, "amendment_lineage", "amendment lineage", _s03_amendment_lineage,
          needs_human_artifacts="sealed two-judgment amendment decisions"),
    Stage(4, "transcription_import", "transcription import", _s04_transcription_import,
          needs_human_artifacts="sealed adjudicated Mode A transcriptions"),
    Stage(5, "identity", "identity resolution", _s05_identity),
    Stage(6, "security_matching", "security matching", _s06_security_matching),
    Stage(7, "first_sufficient", "first-sufficient-disclosure construction", _s07_first_sufficient),
    Stage(8, "transaction_clock", "transaction-clock date construction", _s08_s09_timing),
    Stage(9, "public_entry", "public-action entry construction",
          _noop_factory("public entry derived jointly with stage 8 (timing.derive_two_clock_entries)")),
    Stage(10, "signal_aggregation", "signal aggregation", _s10_signals),
    Stage(11, "event_aggregation", "security-entry-date event aggregation", _s11_events),
    Stage(12, "returns", "return construction", _s12_returns),
    Stage(13, "estimands", "primary and secondary estimands", _s13_estimands),
    Stage(14, "three_way_covariance", "three-way covariance",
          _noop_factory("three-way CGM computed inside estimand execution (INF-001/002)")),
    Stage(15, "calendar_block", "calendar-block inference",
          _noop_factory("calendar-block companion computed inside estimand execution (INF-006)")),
    Stage(16, "event_bands", "simultaneous event-study bands",
          _noop_factory("frozen band method day_mult_nullimp_bwH; executed on event paths at canonical run")),
    Stage(17, "multiplicity", "multiplicity correction", _s17_multiplicity),
    Stage(18, "equivalence", "equivalence testing", _s18_equivalence),
    Stage(19, "power_mde", "power and MDE reporting", _s19_power),
    Stage(20, "influence", "household influence diagnostics", _s20_influence),
    Stage(21, "weighting_views", "event and household weighting",
          _noop_factory("event/equal-household views executed as registry rows H_S1_01/02")),
    Stage(22, "portfolio", "literal long-after-purchase portfolio", _s22_portfolio),
    Stage(23, "portfolio_accounting", "portfolio accounting and costs",
          _noop_factory("lot accounting identities asserted in engine (POR-010)")),
    Stage(24, "factor_alpha", "factor alpha",
          _noop_factory("FF5+momentum alpha executed as registry row H_P1_02")),
    Stage(25, "portfolio_inference", "portfolio bootstrap and HAC inference",
          _noop_factory("HAC lag-20 + block bootstrap inside H_P1_02 executor")),
    Stage(26, "ledgers", "attrition and provenance ledgers", _s26_ledgers),
    Stage(27, "tables_figures", "tables and figures", _s27_tables_figures),
    Stage(28, "manuscript_exports", "manuscript-number exports",
          _noop_factory("exports resolve exclusively via the numbers ledger (K)")),
    Stage(29, "claim_ledger", "claim ledger",
          _noop_factory("claim ledger populated from adjusted results at release build")),
    Stage(30, "release_manifest", "release manifest", _s30_release_manifest),
)


def run_pipeline(ctx: PipelineContext) -> PipelineContext:
    """Run ALL stages in canonical order. No subset parameter exists."""
    if ctx.calendar is None:
        raise StageError("pipeline context requires a trading calendar")
    # portfolio must precede estimands so H_P1_02 sees daily returns; the
    # declared stage numbering is the reporting order, execution order is
    # dependency order:
    execution_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]
    by_number = {s.number: s for s in STAGES}
    for number in execution_order:
        stage = by_number[number]
        stage.fn(ctx)
    return ctx
