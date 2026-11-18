"""Full output surface for pipeline dry runs (standing directive J.1).

Given a completed :class:`PipelineContext`, produce EVERY catalog output —
all tables, figures, diagnostics, the result registry, the claim ledger, the
release manifest, and a mock manuscript-number export — through the numbers
ledger only (JC-023). Catalog outputs the synthetic universe cannot support
receive explicitly-reasoned ``SYNTHETIC_NOT_APPLICABLE`` rows, never a
silent skip (RPT-001). ``verify_output_completeness`` fails on any missing
output file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.inference.event_bands import supt_band_day_multiplier
from trading_on_disclosure_v2.outcomes.returns import combine_with_delisting
from trading_on_disclosure_v2.reporting.claims import Claim, ClaimLedger
from trading_on_disclosure_v2.reporting.exports import export_all
from trading_on_disclosure_v2.reporting.figures import (
    event_study_paths,
    portfolio_wealth_exposure,
    two_clock_schematic,
)
from trading_on_disclosure_v2.reporting.numbers_ledger import (
    LedgerEntry,
    NumbersLedger,
)
from trading_on_disclosure_v2.reporting.shells import build_shell, load_catalog
from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL


class CompletenessError(RuntimeError):
    """A catalog output is missing from the dry-run output surface."""


def _code_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def _frame_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _na_row(reason: str) -> dict:
    return {
        "row_id": "not_applicable",
        "display_label": f"SYNTHETIC_NOT_APPLICABLE: {reason}",
        "estimand_label": "",
        "evidence_tier": "D",
        "ledger_key": "",
    }


#: Catalog outputs the synthetic universe cannot meaningfully populate, with
#: the explicit reason recorded in the emitted table (never silently skipped).
NOT_APPLICABLE_REASONS = {
    "A01": "source census terminal states require the real official store",
    "A02": "public-date provenance decision tables are human work (TC-020/021)",
    "A03": "vendor reconciliation is gated until official rules freeze (TIM-006)",
    "A04": "scan transcription agreement requires human Mode A coding",
    "A05": "Mode B bounds are inapplicable; Mode A signed (HS-04)",
    "A06": "amendment lineage requires the two-judgment review (VER-001)",
    "A07": "identity audit evidence packs live in the design namespace (HD-05)",
    "A08": "security tier-C review has not occurred (HD-06)",
    "A17": "placebo battery executes at the canonical run under frozen seeds",
    "A20": "cost/amount scenarios require the signed HS-05 parameters",
    "A23": "sale/short diagnostics require signed borrow assumptions (HS-05)",
    "A24": "legacy registry is preserved in the legacy namespace (H0)",
    "A25": "legacy prediction appendix is preserved Tier H (JC-004)",
    "A26": "source families R1-R12 are legacy Tier H artifacts",
    "A27": "literature reconciliation needs the real House event subset",
    "A28": "cross-software validation is a real D-tier artifact (INF-015)",
    "A29": "coverage simulations are real D-tier artifacts; see the validation namespace",
    "A30": "provenance claims assemble at the canonical release build",
}


def build_event_time_car(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.Series,
    calendar: pd.DatetimeIndex,
    pre: int = 10,
    post: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-signal cumulative abnormal-return paths around public entry."""
    combined = combine_with_delisting(daily)
    by_permno = {
        int(p): g.set_index("date")["total_ret"]
        for p, g in combined.groupby("permno")
    }
    ks = list(range(-pre, post + 1))
    rows, keys, direction = [], [], []
    for row in signals.dropna(subset=[FOLLOWER_ENTRY_COL]).itertuples(index=False):
        entry = pd.Timestamp(getattr(row, FOLLOWER_ENTRY_COL))
        pos = calendar.searchsorted(entry)
        if pos - pre < 0 or pos + post >= len(calendar):
            continue
        window = calendar[pos - pre : pos + post + 1]
        stock = by_permno.get(int(row.permno))
        if stock is None:
            continue
        ar = (stock.reindex(window) - benchmark.reindex(window)).to_numpy()
        if np.isnan(ar).any():
            continue
        rows.append(np.cumsum(ar))
        keys.append({
            "filer_uid": row.filer_uid,
            "permno": int(row.permno),
            "entry_date": entry,
        })
        direction.append(row.direction)
    car = pd.DataFrame(rows, columns=ks)
    return car, pd.DataFrame(keys), pd.Series(direction, name="direction")


def _path(car: pd.DataFrame, direction: pd.Series, which: str) -> np.ndarray:
    mask = (direction == which).to_numpy()
    return car.loc[mask].mean(axis=0).to_numpy()


class _Surface:
    """Stateful builder so helpers share the ledger/commit/sample context."""

    def __init__(self, ctx, repo_root: Path) -> None:
        self.ctx = ctx
        self.repo_root = Path(repo_root)
        self.commit = _code_commit(self.repo_root)
        self.ledger = NumbersLedger()
        self.signals = ctx.frames["signals"]
        self.adjusted = ctx.frames["adjusted_results"]
        self.res = {str(r["test_id"]): r for _, r in self.adjusted.iterrows()}
        self.sha_results = _frame_sha(self.adjusted.astype(str))
        self.sha_signals = _frame_sha(self.signals.astype(str))
        self.sample = f"{ctx.mode} n={len(self.signals)}"

    def add(self, key: str, value: float, unit: str, test_id: str,
            sha: str | None = None) -> str:
        self.ledger.add(LedgerEntry(
            ledger_key=key, output_id=key.split(".")[0],
            raw_value=float(value), rounded_value=round(float(value), 6),
            unit=unit, rounding_rule="6dp", sample=self.sample,
            evidence_tier="D", estimator_or_test_id=test_id,
            source_artifact_sha256=sha or self.sha_results,
            code_commit=self.commit, release_id="SYNTHETIC_DRY_RUN",
        ))
        return key

    def key_or_empty(self, test_id: str, field: str, unit: str) -> str:
        row = self.res.get(test_id)
        if row is None:
            return ""
        value = row.get(field, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(value):
            return ""
        key = f"T.{test_id}.{field}"
        if key not in self.ledger.entries:
            self.add(key, value, unit, test_id)
        return key

    def adjusted_p_key(self, test_id: str) -> str:
        for field in ("p_romano_wolf", "p_holm", "p_bonferroni_global"):
            key = self.key_or_empty(test_id, field, "p")
            if key:
                return key
        return ""


def generate_full_output_surface(ctx, repo_root: Path) -> dict:
    """Emit every catalog output from a completed dry-run context."""
    out = ctx.out_root
    s = _Surface(ctx, repo_root)
    signals = s.signals
    equivalence = ctx.frames["equivalence"]
    mde = ctx.frames["mde"]
    daily_port = ctx.frames["portfolio_daily"].reset_index()
    sha_portfolio = _frame_sha(daily_port.astype(str))
    tables: dict[str, object] = {}

    # ---- T01 / T02 -------------------------------------------------------
    counts = {
        "signals": len(signals),
        "purchases": int((signals["direction"] == "purchase").sum()),
        "sales": int((signals["direction"] == "sale").sum()),
        "households": signals["filer_uid"].nunique(),
        "securities": signals["permno"].nunique(),
        "entry_dates": signals[FOLLOWER_ENTRY_COL].nunique(),
    }
    t01_rows = []
    for name, value in counts.items():
        k = s.add(f"T01.count.{name}", value, "count", "census_composition",
                  s.sha_signals)
        t01_rows.append({
            "row_id": name, "display_label": f"[SYNTHETIC] {name}",
            "estimand_label": "sample composition", "evidence_tier": "D",
            "ledger_key": "", "chamber": "SYNTHETIC", "year": "2024",
            "count_ledger_key": k,
        })
    tables["T01"] = build_shell(repo_root, "T01", rows=t01_rows)

    lag = signals["reporting_lag_days"].astype(float)
    t02_rows = []
    for stat, value in (
        ("lag_mean", lag.mean()), ("lag_p50", lag.median()),
        ("lag_p90", lag.quantile(0.9)),
        ("effective_households", counts["households"]),
        ("entry_dates", counts["entry_dates"]),
    ):
        k = s.add(f"T02.{stat}", value, "days_or_count", "timing_composition",
                  s.sha_signals)
        t02_rows.append({
            "row_id": stat, "display_label": f"[SYNTHETIC] {stat}",
            "estimand_label": "timing/composition", "evidence_tier": "D",
            "ledger_key": k, "statistic_group": "dependence_structure",
        })
    tables["T02"] = build_shell(repo_root, "T02", rows=t02_rows)

    # ---- T03 ---------------------------------------------------------------
    have_ar = signals["ar_E_h20"].notna()
    mu_signed = float(
        (np.where(signals.loc[have_ar, "direction"] == "purchase", 1.0, -1.0)
         * signals.loc[have_ar, "ar_E_h20"]).mean())
    mu_key = s.add("T03.mu_signed_E20.estimate", mu_signed, "decimal_return",
                   "mu_signed_E20_descriptive", s.sha_signals)

    t03_rows = []
    for row_id, test_id, clock in (
        ("transaction_benchmark_delta_T20", "H_C1_01", "transaction_clock"),
        ("disclosure_day_response_delta_D0", "H_C1_02", "disclosure_day"),
        ("public_action_delta_E20", "H_P1_01", "public_action_clock"),
    ):
        t03_rows.append({
            "row_id": row_id, "display_label": f"[SYNTHETIC] {row_id}",
            "estimand_label": row_id, "evidence_tier": "D", "ledger_key": "",
            "clock": clock, "unit": "decimal_return",
            "estimate_key": s.key_or_empty(test_id, "estimate", "decimal_return"),
            "multiway_interval_key": s.key_or_empty(test_id, "se_multiway", "se"),
            "block_interval_key": s.key_or_empty(test_id, "block_ci_low",
                                                 "decimal_return"),
            "raw_p_key": s.key_or_empty(test_id, "raw_p", "p"),
            "adjusted_p_key": s.adjusted_p_key(test_id),
            "n_signals_key": s.key_or_empty(test_id, "n", "count"),
        })
    t03_rows.append({
        "row_id": "legacy_signed_mu_E20",
        "display_label": "[SYNTHETIC] legacy signed payoff (continuity)",
        "estimand_label": "mu_signed_E20", "evidence_tier": "D",
        "ledger_key": "", "clock": "public_action_clock",
        "unit": "decimal_return", "estimate_key": mu_key,
        "multiway_interval_key": "", "block_interval_key": "",
        "raw_p_key": "", "adjusted_p_key": "", "n_signals_key": "",
    })
    tables["T03"] = build_shell(repo_root, "T03", rows=t03_rows)

    # ---- T04 ---------------------------------------------------------------
    eq_key = sesoi_key = ""
    if len(equivalence):
        sha_eq = _frame_sha(equivalence.astype(str))
        eq_key = s.add("T04.equivalence.p_upper",
                       float(equivalence.iloc[0]["p_upper"]), "p",
                       "TOST_delta_E20_advisory", sha_eq)
        sesoi_key = s.add("T04.sesoi.bound", 0.0025, "decimal_return",
                          "SESOI_candidate_PENDING_HS03", sha_eq)
    eh = ctx.artifacts["equal_household"]
    eh_key = s.add("T04.equal_household.estimate", eh["estimate"],
                   "decimal_return", "H_S1_02", s.sha_signals)
    loo = ctx.artifacts["leave_one_out"]
    if isinstance(loo, tuple):  # (full_estimate, [LeaveOneOutRow, ...])
        loo = loo[1]
    loo_frame = (loo if isinstance(loo, pd.DataFrame)
                 else pd.DataFrame([dataclasses.asdict(r) for r in loo]))
    max_change = (float(loo_frame["abs_change_fraction"].max())
                  if len(loo_frame) else 0.0)
    loo_key = s.add("T04.leave_one_out.max_abs_change", max_change,
                    "relative_change", "H_S1_03",
                    _frame_sha(loo_frame.astype(str)))
    tables["T04"] = build_shell(repo_root, "T04", rows=[
        {"row_id": "signal_view", "display_label": "[SYNTHETIC] signal view",
         "estimand_label": "delta_E20", "evidence_tier": "D", "ledger_key": "",
         "view": "equal_signal",
         "estimate_key": s.key_or_empty("H_P1_01", "estimate", "decimal_return"),
         "interval_key": s.key_or_empty("H_P1_01", "se_multiway", "se"),
         "flag": "", "sesoi_key": sesoi_key, "equivalence_decision_key": ""},
        {"row_id": "security_entry_event_view",
         "display_label": "[SYNTHETIC] event view",
         "estimand_label": "delta_E20", "evidence_tier": "D", "ledger_key": "",
         "view": "security_entry_event",
         "estimate_key": s.key_or_empty("H_S1_01", "estimate", "decimal_return"),
         "interval_key": s.key_or_empty("H_S1_01", "se_multiway", "se"),
         "flag": "", "sesoi_key": "", "equivalence_decision_key": ""},
        {"row_id": "equal_household_view",
         "display_label": "[SYNTHETIC] equal-household view",
         "estimand_label": "delta_E20", "evidence_tier": "D", "ledger_key": "",
         "view": "equal_household", "estimate_key": eh_key, "interval_key": "",
         "flag": "", "sesoi_key": "", "equivalence_decision_key": ""},
        {"row_id": "leave_one_top_household_out",
         "display_label": "[SYNTHETIC] leave-one-out max |change|",
         "estimand_label": "influence", "evidence_tier": "D", "ledger_key": "",
         "view": "leave_one_out", "estimate_key": loo_key, "interval_key": "",
         "flag": "threshold_0.50", "sesoi_key": "",
         "equivalence_decision_key": ""},
        {"row_id": "equivalence_delta_E20",
         "display_label": "[SYNTHETIC] TOST (advisory; HS-03 pending)",
         "estimand_label": "delta_E20 equivalence", "evidence_tier": "D",
         "ledger_key": "", "view": "equivalence", "estimate_key": "",
         "interval_key": "", "flag": "advisory", "sesoi_key": sesoi_key,
         "equivalence_decision_key": eq_key},
    ])

    # ---- T05 ---------------------------------------------------------------
    equity0 = float(daily_port["equity"].iloc[0])
    net_cum = float(daily_port["equity"].iloc[-1] / equity0 - 1)
    costs_total = float(daily_port["costs_paid"].sum())
    gross_cum = float(
        (daily_port["equity"].iloc[-1] + costs_total) / equity0 - 1)
    net_key = s.add("T05.net.cum_return", net_cum, "decimal_return",
                    "portfolio_engine", sha_portfolio)
    gross_key = s.add("T05.gross.cum_return_costs_rebated", gross_cum,
                      "decimal_return", "portfolio_engine", sha_portfolio)
    exposure_key = s.add("T05.exposure.mean",
                         float(daily_port["risky_fraction"].mean()),
                         "fraction", "portfolio_engine", sha_portfolio)
    cost_key = s.add("T05.costs.total", costs_total, "decimal_wealth",
                     "portfolio_engine", sha_portfolio)
    tables["T05"] = build_shell(repo_root, "T05", rows=[
        {"row_id": "gross_performance",
         "display_label": "[SYNTHETIC] gross cumulative return (costs rebated)",
         "estimand_label": "portfolio", "evidence_tier": "D", "ledger_key": "",
         "gross_net": "gross", "estimate_key": gross_key,
         "hac_interval_key": "", "bootstrap_interval_key": "",
         "exposure_key": exposure_key, "cost_key": ""},
        {"row_id": "net_performance",
         "display_label": "[SYNTHETIC] net cumulative return",
         "estimand_label": "portfolio", "evidence_tier": "D", "ledger_key": "",
         "gross_net": "net", "estimate_key": net_key, "hac_interval_key": "",
         "bootstrap_interval_key": "", "exposure_key": exposure_key,
         "cost_key": cost_key},
        {"row_id": "net_alpha_ff5_mom",
         "display_label": "[SYNTHETIC] net FF5+momentum alpha",
         "estimand_label": "long_purchase_net_alpha", "evidence_tier": "D",
         "ledger_key": "", "gross_net": "net",
         "estimate_key": s.key_or_empty("H_P1_02", "estimate",
                                        "annualized_decimal"),
         "hac_interval_key": s.key_or_empty("H_P1_02", "se_multiway", "se"),
         "bootstrap_interval_key": s.key_or_empty("H_P1_02", "block_ci_low",
                                                  "annualized_decimal"),
         "exposure_key": exposure_key, "cost_key": cost_key},
    ])

    # ---- T06 ---------------------------------------------------------------
    t06_rows = []
    for test_id, label in (("H_S3_01", "direction x reporting lag"),
                           ("H_S3_02", "direction x scanned format")):
        est = s.key_or_empty(test_id, "estimate", "decimal_return")
        if not est:
            t06_rows.append({**_na_row(f"{test_id} not executable on synthetic data"),
                             "interaction": label, "estimate_key": "",
                             "interval_key": "", "adjusted_p_key": ""})
            continue
        t06_rows.append({
            "row_id": test_id, "display_label": f"[SYNTHETIC] {label}",
            "estimand_label": label, "evidence_tier": "D", "ledger_key": "",
            "interaction": label, "estimate_key": est,
            "interval_key": s.key_or_empty(test_id, "se_multiway", "se"),
            "adjusted_p_key": s.adjusted_p_key(test_id),
        })
    tables["T06"] = build_shell(repo_root, "T06", rows=t06_rows)

    # ---- appendix tables ---------------------------------------------------
    attrition = ctx.artifacts["attrition_ledger"]
    sha_attr = _frame_sha(attrition.astype(str))
    tables["A09"] = build_shell(repo_root, "A09", rows=[
        {"row_id": str(r["stage"]), "display_label": f"[SYNTHETIC] {r['stage']}",
         "estimand_label": "attrition", "evidence_tier": "D",
         "ledger_key": s.add(f"A09.{r['stage']}.n_out", float(r["n_out"]),
                             "count", "attrition_ledger", sha_attr)}
        for _, r in attrition.iterrows()
    ])
    tables["A10"] = build_shell(repo_root, "A10", rows=[
        {"row_id": n, "display_label": f"[SYNTHETIC] {n}",
         "estimand_label": "descriptives", "evidence_tier": "D",
         "ledger_key": f"T01.count.{n}"}
        for n in ("signals", "households", "securities")
    ])
    tables["A11"] = build_shell(repo_root, "A11", rows=[
        ({"row_id": t, "display_label": f"[SYNTHETIC] {t}",
          "estimand_label": "horizon sensitivity", "evidence_tier": "D",
          "ledger_key": s.key_or_empty(t, "estimate", "decimal_return")}
         if s.key_or_empty(t, "estimate", "decimal_return")
         else _na_row(f"{t} blocked or ineligible"))
        for t in ("H_S2_05", "H_S2_10", "H_S2_40", "H_S2_60", "H_S2_127")
    ])
    ew_key = s.key_or_empty("H_S2_EW", "estimate", "decimal_return")
    tables["A12"] = build_shell(repo_root, "A12", rows=[
        {"row_id": "H_S2_EW", "display_label": "[SYNTHETIC] H_S2_EW",
         "estimand_label": "benchmark sensitivity", "evidence_tier": "D",
         "ledger_key": ew_key} if ew_key
        else _na_row("EW/DGTW benchmark inputs blocked in this dry run")
    ])
    a13_rows = []
    for test_id in ("H_P1_01", "H_C1_01"):
        for field in ("se_multiway", "block_ci_low", "block_ci_high"):
            key = s.key_or_empty(test_id, field, "decimal_return")
            if key:
                a13_rows.append({
                    "row_id": f"{test_id}.{field}",
                    "display_label": f"[SYNTHETIC] {test_id} {field}",
                    "estimand_label": "inference comparison",
                    "evidence_tier": "D", "ledger_key": key,
                })
    tables["A13"] = build_shell(repo_root, "A13", rows=a13_rows)
    sha_loo = _frame_sha(loo_frame.astype(str))
    tables["A14"] = build_shell(repo_root, "A14", rows=[
        {"row_id": f"loo_{i}",
         "display_label": f"[SYNTHETIC] drop {r['household']}",
         "estimand_label": "leave-one-out", "evidence_tier": "D",
         "ledger_key": s.add(f"A14.loo.{i}", float(r["estimate"]),
                             "decimal_return", "H_S1_03", sha_loo)}
        for i, r in loo_frame.iterrows()
    ] or [_na_row("no leave-one-out rows produced")])
    a15_map = (
        ("signal_view", s.key_or_empty("H_P1_01", "estimate", "decimal_return")),
        ("event_view", s.key_or_empty("H_S1_01", "estimate", "decimal_return")),
        ("equal_household_view", eh_key),
    )
    tables["A15"] = build_shell(repo_root, "A15", rows=[
        {"row_id": rid, "display_label": f"[SYNTHETIC] {rid}",
         "estimand_label": "aggregation bridge", "evidence_tier": "D",
         "ledger_key": k}
        for rid, k in a15_map if k
    ])

    # event-time paths + frozen-method band (also feeds F03)
    car, car_keys, car_dir = build_event_time_car(
        signals, ctx.frames["daily"], ctx.frames["benchmark"], ctx.calendar)
    band = supt_band_day_multiplier(
        car, car_keys, car_dir, series="difference",
        freeze_hash=ctx.freeze_hash, purpose="dryrun-f03-band",
        n_draws=299, trading_days=ctx.calendar)
    sha_band = hashlib.sha256(band.estimate.tobytes()).hexdigest()
    a16_rows = []
    for k_idx, k_time in enumerate(band.ks):
        if int(k_time) % 5:
            continue
        lk = s.add(f"A16.diff_car.k{int(k_time)}", float(band.estimate[k_idx]),
                   "decimal_return", "event_band_day_mult_nullimp_bwH",
                   sha_band)
        a16_rows.append({
            "row_id": f"k_{int(k_time)}",
            "display_label": f"[SYNTHETIC] diff CAR at k={int(k_time)}",
            "estimand_label": "event-study path", "evidence_tier": "D",
            "ledger_key": lk,
        })
    tables["A16"] = build_shell(repo_root, "A16", rows=a16_rows)

    a18_rows = []
    if eq_key:
        a18_rows.append({
            "row_id": "tost_delta_E20",
            "display_label": "[SYNTHETIC] TOST p_upper (advisory)",
            "estimand_label": "equivalence", "evidence_tier": "D",
            "ledger_key": eq_key,
        })
    sha_mde = _frame_sha(mde.astype(str))
    mde_value_col = next(
        (c for c in mde.columns if c not in ("test_id",) and
         pd.api.types.is_numeric_dtype(mde[c])), None)
    for i, r in mde.iterrows():
        a18_rows.append({
            "row_id": f"mde_{r['test_id']}_{i}",
            "display_label": f"[SYNTHETIC] MDE {r['test_id']}",
            "estimand_label": "MDE", "evidence_tier": "D",
            "ledger_key": s.add(f"A18.mde.{i}", float(r[mde_value_col]),
                                "decimal_return",
                                "mde_retrospective_design_diagnostic",
                                sha_mde),
        })
    tables["A18"] = build_shell(repo_root, "A18", rows=a18_rows)

    routing = ctx.artifacts["portfolio_routing"]
    sha_rout = _frame_sha(routing.astype(str))
    tables["A19"] = build_shell(repo_root, "A19", rows=[
        {"row_id": str(state), "display_label": f"[SYNTHETIC] {state}",
         "estimand_label": "routing", "evidence_tier": "D",
         "ledger_key": s.add(f"A19.routing.{state}", float(n), "count",
                             "portfolio_routing", sha_rout)}
        for state, n in routing["state"].value_counts().items()
    ])
    lots = ctx.artifacts["portfolio_lots"]
    sha_lots = _frame_sha(lots.astype(str))
    a21_rows = [
        {"row_id": str(state), "display_label": f"[SYNTHETIC] {state}",
         "estimand_label": "dispositions", "evidence_tier": "D",
         "ledger_key": s.add(f"A21.disposition.{state}", float(n), "count",
                             "portfolio_lots", sha_lots)}
        for state, n in lots["disposition"].value_counts().items()
    ] if len(lots) else []
    tables["A21"] = build_shell(repo_root, "A21", rows=a21_rows or [
        _na_row("no lots created on this synthetic universe")])
    conc = ctx.artifacts["concentration"]
    tables["A22"] = build_shell(repo_root, "A22", rows=[
        {"row_id": name, "display_label": f"[SYNTHETIC] {name}",
         "estimand_label": "concentration", "evidence_tier": "D",
         "ledger_key": s.add(f"A22.concentration.{name}", float(value),
                             "share", "household_concentration",
                             s.sha_signals)}
        for name, value in conc.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    ])
    for output_id, reason in NOT_APPLICABLE_REASONS.items():
        tables[output_id] = build_shell(repo_root, output_id,
                                        rows=[_na_row(reason)])

    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for output_id, shell in tables.items():
        shell.frame.to_csv(tables_dir / f"{output_id}.csv", index=False)

    # ---- figures -----------------------------------------------------------
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    two_clock_schematic(fig_dir / "F01.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.hist(lag.dropna(), bins=24)
    ax1.set_title("[SYNTHETIC] reporting lag (days)", fontsize=9)
    per_month = signals.set_index(pd.DatetimeIndex(
        signals[FOLLOWER_ENTRY_COL]))["direction"].resample("ME").count()
    ax2.plot(per_month.index, per_month.to_numpy())
    ax2.set_title("[SYNTHETIC] signals per month", fontsize=9)
    ax2.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "F02.png", dpi=150)
    plt.close(fig)

    event_study_paths(
        fig_dir / "F03.png",
        event_time=np.asarray(band.ks, dtype=float),
        buy_path=_path(car, car_dir, "purchase"),
        sell_path=_path(car, car_dir, "sale"),
        band_lower=band.lower, band_upper=band.upper,
        band_type="sup_t", entry_marker=0,
        ledger_keys=tuple(k for k in s.ledger.entries if k.startswith("A16.")),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    forest = s.adjusted[
        pd.to_numeric(s.adjusted["estimate"], errors="coerce").notna()
    ]
    y = np.arange(len(forest))
    est = forest["estimate"].astype(float).to_numpy()
    se = pd.to_numeric(forest["se_multiway"], errors="coerce").to_numpy()
    ax.errorbar(est, y, xerr=1.96 * np.where(np.isfinite(se), se, 0.0),
                fmt="o", ms=3)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["test_id"].tolist(), fontsize=6)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_title("[SYNTHETIC] estimates with multiway 95% intervals",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "F04.png", dpi=150)
    plt.close(fig)

    portfolio_wealth_exposure(
        fig_dir / "F05.png",
        dates=daily_port["date"].to_numpy(),
        strategy_wealth=(daily_port["equity"] / equity0).to_numpy(),
        benchmark_wealth=(daily_port["benchmark_equity"] / equity0).to_numpy(),
        risky_exposure=daily_port["risky_fraction"].to_numpy(),
        ledger_keys=(net_key, exposure_key),
    )

    # ---- release ledgers/reports --------------------------------------------
    rel = out / "release"
    rel.mkdir(parents=True, exist_ok=True)
    ctx.frames["census"].to_csv(rel / "R01_source_census_ledger.csv",
                                index=False)
    pd.DataFrame([{"note": "SYNTHETIC_NOT_APPLICABLE: scan ledger requires "
                           "the real scan inventory and Mode A workflow"}]
                 ).to_csv(rel / "R02_scan_ledger.csv", index=False)
    attrition.to_csv(rel / "R03_attrition_ledger.csv", index=False)
    ctx.frames["result_frame"].to_csv(rel / "R04_hypothesis_results.csv",
                                      index=False)
    routing.to_csv(rel / "R05_portfolio_routing_ledger.csv", index=False)
    lots.to_csv(rel / "R06_portfolio_lot_ledger.csv", index=False)
    ledger_digest = s.ledger.write(rel / "R07_numbers_ledger.json")

    registry = ctx.frames["registry"]
    claims = ClaimLedger()
    claims.add(Claim(
        claim_id="SYN-C1",
        text_fragment="[SYNTHETIC_DRY_RUN] the pipeline computes a "
                      "purchase-minus-sale public-action spread on null data",
        supporting_test_ids=("H_P1_01",), evidence_tier="H",
        claim_class="diagnostic",
    ), registry)
    claims.add(Claim(
        claim_id="SYN-C2",
        text_fragment="[SYNTHETIC_DRY_RUN] the event path display carries "
                      "simultaneous sup-t bands",
        supporting_test_ids=("H_C1_03",), evidence_tier="H",
        claim_class="diagnostic", path_wide=True, band_type="sup_t",
    ), registry)
    claims.add(Claim(
        claim_id="SYN-C3",
        text_fragment="[SYNTHETIC_DRY_RUN] engineering wiring exercised end "
                      "to end on synthetic null data",
        supporting_test_ids=(), evidence_tier="D",
        claim_class="descriptive",
    ), registry)
    claims.to_frame().to_csv(rel / "R08_claim_ledger.csv", index=False)

    (rel / "R09_reproduction_report.md").write_text(
        "# SYNTHETIC_DRY_RUN reproduction report\n\n"
        f"mode: {ctx.mode}\nseed: two-clock-synthetic-dry-run-v1\n"
        f"code_commit: {s.commit}\nnumbers_ledger_sha256: {ledger_digest}\n"
        "Rerun: scripts/v2/21_canonical_top_journal_release.py --synthetic-dry-run\n")
    (rel / "R10_optional_extension_status.md").write_text(
        "# Optional extension status\n\noptional_disabled — the prospective/"
        "processability extension is not activated (contract scope_lock).\n")

    export_all(tables, s.ledger, out / "manuscript_export")

    (out / "WATERMARK").write_text(
        "SYNTHETIC_DRY_RUN — every number on this surface derives from "
        "synthetic null data; nothing here is empirical evidence.\n")

    summary = {
        "tables": len(tables),
        "figures": 5,
        "ledger_entries": len(s.ledger.entries),
        "claims": len(claims.claims),
        "not_applicable_tables": len(NOT_APPLICABLE_REASONS),
        "numbers_ledger_sha256": ledger_digest,
    }
    (out / "output_surface_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def verify_output_completeness(out_root: Path, repo_root: Path) -> None:
    """Every catalog output must exist on disk (RPT-001). Raises listing all."""
    catalog = load_catalog(repo_root)
    missing = []
    for output_id, spec in catalog.items():
        if spec.kind == "figure":
            if not (out_root / "figures" / f"{output_id}.png").exists():
                missing.append(f"{output_id} (figure)")
        elif spec.kind == "table":
            if not (out_root / "tables" / f"{output_id}.csv").exists():
                missing.append(f"{output_id} (table)")
            if not (out_root / "manuscript_export" / f"{output_id}.md").exists():
                missing.append(f"{output_id} (manuscript export)")
        else:  # ledger | report
            if not list((out_root / "release").glob(f"{output_id}_*")):
                missing.append(f"{output_id} ({spec.kind})")
    if missing:
        raise CompletenessError(
            f"{len(missing)} catalog outputs missing: {missing}")
