from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_on_disclosure_v4.reporting_labels import display_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GEN = ROOT / "generated"
FIG = ROOT / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

reg = pd.read_csv(DATA / "result_registry.csv").set_index("test_id")
ledger = json.loads((DATA / "numerical_ledger.json").read_text())
release = json.loads((DATA / "RELEASE_RECORD.json").read_text())
attrition = pd.read_csv(DATA / "attrition_ledger.csv")
source_diag = json.loads((DATA / "source_construction_diagnostics.json").read_text())
date_universes = pd.read_csv(DATA / "date_proxy_universes.csv")
inference_map = pd.read_csv(DATA / "inference_map.csv").set_index("test_id")

# The earlier version's appendix analyses re-run on the de-duplicated sample
# (trading_on_disclosure_v4.existing.appendix; files named in paper/generated/existing_macros.md). The
# appendix tables print these; the released ledger under paper/data stays the comparison column.
DEDUP_DIR = ROOT.parent / "data/v4/existing/appendix_dedup"
DEDUP_FILES = {name: f"data/v4/existing/appendix_dedup/{name}.csv" for name in
               ["full_results", "attrition", "source_assembly", "date_gap", "comparison_portfolio", "event_study_path", "horizons"]}
full_results = pd.read_csv(DEDUP_DIR / "full_results.csv")
dedup_attrition = pd.read_csv(DEDUP_DIR / "attrition.csv")
source_assembly = pd.read_csv(DEDUP_DIR / "source_assembly.csv").set_index("field")
date_gap = pd.read_csv(DEDUP_DIR / "date_gap.csv")
comparison_portfolio = pd.read_csv(DEDUP_DIR / "comparison_portfolio.csv").set_index("field")
event_dedup = pd.read_csv(DEDUP_DIR / "event_study_path.csv")
horizons_dedup = pd.read_csv(DEDUP_DIR / "horizons.csv")


def num(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):,.{decimals}f}"


def pval(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "--"
    if float(x) < 0.001:
        return r"$<$0.001"  # text-mode form, so "p-value \Macro" reads "p-value <0.001" (never "p =<0.001")
    return f"{float(x):.{decimals}f}"


def bps(x: float, decimals: int = 2) -> str:
    return num(float(x) * 10_000, decimals)


def pct(x: float, decimals: int = 2) -> str:
    return num(float(x) * 100, decimals)


def integer(x: float) -> str:
    return f"{int(round(float(x))):,}"


def signed_integer(x: float) -> str:
    """An integer with a typeset minus sign, for prose that gives a negative value."""
    text = integer(x)
    return text.replace("-", r"\ensuremath{-}", 1) if text.startswith("-") else text


# Released event-level test (TW_S1_EVENT): the registry's n counts every outcome-bearing
# security-entry-date event, but the Welch contrast uses purchase-only and sale-only events
# only (data/v4/existing/event_level_counts.csv). The
# sensitivity table prints the events that enter the estimate; the complete-results table
# reproduces the released number verbatim and its note says what it counts.
EVENT_COUNTS_FILE = "data/v4/existing/event_level_counts.csv"
event_counts = pd.read_csv(ROOT.parent / EVENT_COUNTS_FILE).iloc[0]
if int(event_counts.purchase_only_events + event_counts.sale_only_events) != int(event_counts.events_in_estimate):
    raise ValueError(f"{EVENT_COUNTS_FILE}: purchase-only + sale-only events differ from events_in_estimate")
if int(event_counts.events_in_estimate + event_counts.mixed_events_excluded) != int(event_counts.all_outcome_bearing_events):
    raise ValueError(f"{EVENT_COUNTS_FILE}: events in the estimate plus mixed events differ from all outcome-bearing events")


def estimation_n(test_id: str, value: float, released: bool = False) -> str:
    """Display the estimation sample size. For the event-level test the estimate uses the
    purchase-only and sale-only events (the ExEventLevelN macro); `released=True` keeps the registry's
    count of all outcome-bearing events for the table that reproduces the released numbers."""
    if test_id == "TW_S1_EVENT" and not released:
        return r"\ExEventLevelN"
    return integer(value)


def tex_escape(value: object) -> str:
    out = str(value)
    for src, dst in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        out = out.replace(src, dst)
    return out


def tex_row(*cells: object) -> str:
    return " & ".join(str(c) for c in cells) + r" \\"


def rerun(test_id: str, sample: str = "dedup") -> pd.Series:
    """One row of the re-run results file (estimates already in bp; portfolio rows in annual percent)."""
    rows = full_results[(full_results.test_id == test_id) & (full_results["sample"] == sample)]
    if len(rows) != 1:
        raise ValueError(f"{DEDUP_FILES['full_results']}: expected one {sample} row for {test_id}, found {len(rows)}")
    return rows.iloc[0]


def interval(low: float, high: float, decimals: int = 2) -> str:
    return "--" if pd.isna(low) else f"[{num(low, decimals)}, {num(high, decimals)}]"


# The original-sample rows of the re-run file must reproduce the released registry, so the
# de-duplicated rows printed in the appendix are tied to the same code path as the released numbers.
for test_id in reg.index:
    row, rel = rerun(test_id, "original"), reg.loc[test_id]
    scale = 100.0 if row.units == "pct_annual" else 10_000.0
    # the registry's event-level n counts every outcome-bearing event; the re-run reports the estimate's events
    released_n = event_counts.events_in_estimate if test_id == "TW_S1_EVENT" else rel.n
    checks = [(row.estimate, rel.estimate * scale), (row.p_day, rel.raw_p), (row.n, released_n)]
    if pd.notna(rel.p_holm):
        checks.append((row.p_holm_day, rel.p_holm))
    for got, want in checks:
        if pd.isna(got) != pd.isna(want) or (pd.notna(got) and abs(float(got) - float(want)) > 1e-6):
            raise ValueError(f"{DEDUP_FILES['full_results']}: original-sample {test_id} differs from the registry ({got} vs {want})")
if len(full_results) != 2 * len(reg):
    raise ValueError(f"{DEDUP_FILES['full_results']}: expected {2 * len(reg)} rows, found {len(full_results)}")


# ---------------------------------------------------------------------------
# Numerical macros
# ---------------------------------------------------------------------------
macros: list[str] = []


def macro(name: str, value: object) -> None:
    macros.append(rf"\newcommand{{\{name}}}{{{value}}}")


macro("ReleaseCommit", release["authorization"]["head_sha"])
macro("ReleaseFreezeHash", release["freeze_hash"])
macro("ReleaseCommitShort", release["authorization"]["head_sha"][:8])
macro("ReleaseFreezeShort", release["freeze_hash"][:12])
macro("ReleaseDate", "August 2026")

count_mapping = {
    "NOfficial": "t1.official_records",
    "NVendor": "t1.vendor_records",
    "NAssembled": "t1.parsed_records_total",
    "NObservedOfficial": "t1.observed_official_dates",
    "NProxy": "t1.vendor_one_day_proxies",
    "NMatched": "t1.matched_transactions",
    "NSignals": "t1.unique_signals",
    "NMixed": "t1.mixed_direction_excluded",
    "NPrimary": "t1.complete_h20_outcomes",
    "NHouseholds": "t1.households",
    "NSecurities": "t1.securities",
    "NPrimaryHouseholds": "t1.concentration.n_households",
}
for name, key in count_mapping.items():
    macro(name, integer(ledger[key]))

macro("TopOneShare", pct(ledger["t1.concentration.top_1_share"], 1) + r"\%")
macro("TopFiveShare", pct(ledger["t1.concentration.top_5_share"], 1) + r"\%")
macro("TopTenShare", pct(ledger["t1.concentration.top_10_share"], 1) + r"\%")
macro("EffectiveHouseholds", num(ledger["t1.concentration.effective_household_count"], 1))
macro("ObservedOfficialShare", pct(ledger["t1.observed_official_dates"] / ledger["t1.parsed_records_total"], 1) + r"\%")
macro("ProxyShare", pct(ledger["t1.vendor_one_day_proxies"] / ledger["t1.parsed_records_total"], 1) + r"\%")
macro("NCapitolRaw", integer(source_diag["capitol_trades_snapshot_rows"]))
macro("NVendorOfficialDup", integer(source_diag["vendor_duplicates_of_parsed_official"]))
macro("NRepeatScrape", integer(source_diag["repeat_scrape_duplicates"]))
macro("NScanLinked", integer(source_diag["scan_linked_vendor_rows_retained"]))
macro("NVendorOnly", integer(source_diag["vendor_only_rows_retained"]))
macro("NOfficialFilings", integer(source_diag["official_ptr_census_filings"]))
macro("NImageOnlyFilings", integer(source_diag["image_only_unparsed_filings"]))
macro("NAmendOfficialLineage", integer(source_diag["amendment_rows_official_lineage"]))
macro("NAmendUnresolved", integer(source_diag["amendment_rows_unresolved_lineage"]))
macro("NExactCrossDocDup", integer(source_diag["later_exact_cross_document_duplicates"]))

macro("DateOverlapN", integer(ledger["dates.n_matched"]))
macro("DateGapMean", num(ledger["dates.mean_diff_days"], 2))
macro("DateGapMedian", num(ledger["dates.median_diff_days"], 0))
macro("DateSameDay", pct(ledger["dates.share_ct_same_day"], 1) + r"\%")
macro("DatePlusOne", pct(ledger["dates.share_ct_exactly_one_day_later"], 1) + r"\%")
macro("DateMoreThanOne", pct(ledger["dates.share_ct_more_than_one_day_later"], 1) + r"\%")
# Outcome-free post-repair source-linkage diagnostics, computed from the
# bundled date-universe table rather than entered manually.
plus_one = date_universes.loc[date_universes["gap_days"] == 1].iloc[0]
macro("DateBroadN", integer(date_universes["row_weighted_n"].sum()))
macro("DateBroadPlusOne", pct(plus_one["row_weighted_share"], 1) + r"\%")
macro("DateFilingN", integer(date_universes["filing_weighted_n"].sum()))
macro("DateFilingPlusOne", pct(plus_one["filing_weighted_share"], 1) + r"\%")
# Counts formerly hard-coded in the Internet Appendix prose (column sums of the
# date-universe table; estimation sample sizes from the registry).
macro("DateRowUniverseN", integer(date_universes["row_weighted_n"].sum()))
macro("DateFilingUniverseN", integer(date_universes["filing_weighted_n"].sum()))
macro("DateOverlapUniverseN", integer(date_universes["retained_overlap_n"].sum()))
macro("FactorSampleN", integer(reg.loc["TW_S2_FF5MOM"].n))
macro("EventReleasedN", integer(reg.loc["TW_S1_EVENT"].n))  # all outcome-bearing events, as released
if int(reg.loc["TW_S1_EVENT"].n) != int(event_counts.all_outcome_bearing_events):
    raise ValueError("registry n for TW_S1_EVENT differs from the all-outcome-bearing event count")
macro("NActOneAnalyses", str(len(reg)))

primary = reg.loc["TW_P1_E20"]
for name, value in [
    ("PrimaryBps", bps(primary.estimate)),
    ("PrimarySEBps", bps(primary.se_multiway)),
    ("PrimaryCILowBps", bps(primary.ci_low)),
    ("PrimaryCIHighBps", bps(primary.ci_high)),
    ("PrimaryP", pval(primary.raw_p)),
    ("PrimaryN", integer(primary.n)),
    ("PrimaryHH", integer(primary.n_households)),
    ("PrimaryDates", integer(primary.n_entry_dates)),
    ("PurchaseMeanBps", bps(primary.purchase_mean)),
    ("SaleMeanBps", bps(primary.sale_mean)),
    ("PrimaryBlockLowBps", bps(primary.block_ci_low)),
    ("PrimaryBlockHighBps", bps(primary.block_ci_high)),
]:
    macro(name, value)

signed = reg.loc["TW_D_SAR20"]
for name, value in [
    ("SignedBps", bps(signed.estimate)),
    ("SignedSEBps", bps(signed.se_multiway)),
    ("SignedCILowBps", bps(signed.ci_low)),
    ("SignedCIHighBps", bps(signed.ci_high)),
    ("SignedP", pval(signed.raw_p)),
]:
    macro(name, value)

for test_id, prefix in [
    ("TW_C1_T20", "Transaction"),
    ("TW_C1_TRAW20", "RawTransaction"),
    ("TW_C1_PAIRED", "Paired"),
]:
    row = reg.loc[test_id]
    for suffix, value in [
        ("Bps", bps(row.estimate)),
        ("SEBps", bps(row.se_multiway)),
        ("CILowBps", bps(row.ci_low)),
        ("CIHighBps", bps(row.ci_high)),
        ("P", pval(row.raw_p)),
        ("N", estimation_n(test_id, row.n)),
    ]:
        macro(prefix + suffix, value)

for name, key, formatter in [
    ("PortfolioReturn", "t6.annualized_return", pct),
    ("BenchmarkReturn", "t6.benchmark_annualized_return", pct),
    ("AvgInvested", "t6.avg_invested_fraction", pct),
    ("AnnualTurnover", "t6.annual_turnover", pct),
    ("SignalsAccepted", "t6.signals_accepted", integer),
    ("SignalsRejected", "t6.signals_rejected", integer),
    ("MaxExposure", "t6.max_security_exposure", pct),
    ("Sharpe", "t6.sharpe", lambda x: num(x, 2)),
]:
    value = formatter(ledger[key])
    if name in {"PortfolioReturn", "BenchmarkReturn", "AvgInvested", "AnnualTurnover", "MaxExposure"}:
        value += r"\%"
    macro(name, value)

for test_id, prefix in [
    ("TW_IMPL_PORT_NW", "PortMean"),
    ("TW_IMPL_PORT_CAPM", "PortCAPM"),
    ("TW_IMPL_PORT_FF5MOM", "PortFFFiveMom"),
]:
    row = reg.loc[test_id]
    for suffix, value in [
        ("Pct", pct(row.estimate)),
        ("SEPct", pct(row.se_multiway)),
        ("CILowPct", pct(row.ci_low)),
        ("CIHighPct", pct(row.ci_high)),
        ("P", pval(row.raw_p)),
        ("HolmP", pval(row.p_holm)),
        ("N", estimation_n(test_id, row.n)),
    ]:
        macro(prefix + suffix, value)

for test_id, prefix in [
    ("TW_PLC_DIRECTION", "PlaceboDirection"),
    ("TW_PLC_DATES", "PlaceboDates"),
]:
    row = reg.loc[test_id]
    macro(prefix + "P", pval(row.raw_p))
    macro(prefix + "Draws", integer(row.n_draws))
    macro(prefix + "MeanBps", bps(row.placebo_mean))
    macro(prefix + "SdBps", bps(row.placebo_sd))

# Friendly aliases used in prose.
alias_rows = {
    "RtwPTwoEFive": "TW_P2_E5",
    "RtwPTwoEOneZero": "TW_P2_E10",
    "RtwPTwoEFourZero": "TW_P2_E40",
    "RtwPTwoESixZero": "TW_P2_E60",
    "RtwPTwoEOneTwoSeven": "TW_P2_E127",
    "RtwPThreeRawCt": "TW_P3_RAW_CT",
    "RtwPThreeOfficialOnly": "TW_P3_OFFICIAL_ONLY",
    "RtwPThreeOfficialSegment": "TW_P3_OFFICIAL_SEGMENT",
    "RtwPThreeVendorSegment": "TW_P3_VENDOR_SEGMENT",
    "RtwPThreeEntryPlusOne": "TW_P3_ENTRY_PLUS1",
    "RtwSOneRawrows": "TW_S1_RAWROWS",
    "RtwSOneEqhh": "TW_S1_EQHH",
    "RtwSOneEvent": "TW_S1_EVENT",
    "RtwSOneSrcid": "TW_S1_SRCID",
    "RtwSTwoCapm": "TW_S2_CAPM",
    "RtwSTwoFfFiveMom": "TW_S2_FF5MOM",
}
for prefix, test_id in alias_rows.items():
    row = reg.loc[test_id]
    for suffix, value in [
        ("Bps", bps(row.estimate)),
        ("SEBps", bps(row.se_multiway)),
        ("CILowBps", bps(row.ci_low)),
        ("CIHighBps", bps(row.ci_high)),
        ("P", pval(row.raw_p)),
        ("HolmP", pval(row.p_holm)),
        ("N", estimation_n(test_id, row.n)),
    ]:
        macro(prefix + suffix, value)

(GEN / "original_macros.tex").write_text("\n".join(macros) + "\n")
# Until version 4 this file was generated/results_macros.tex; that name now belongs to the
# household-results emitter (scripts/v4/run_results.py, \renewcommand only). A stale copy of
# the legacy file would redefine every original macro, so it is removed.
legacy = GEN / "results_macros.tex"
if legacy.exists() and r"\newcommand{\ReleaseCommit}" in legacy.read_text():
    legacy.unlink()

# ---------------------------------------------------------------------------
# Main-paper table bodies
# ---------------------------------------------------------------------------
main: list[str] = []
# Table 1 prints the de-duplicated sample: every field is one of
# the \Ex<Orig|Dedup>TableOne<Field> macros (paper/generated/existing_macros.md), so the table and the
# prose of Section 2 read the same file; the released-sample twins serve the appendix comparison.
TABLE_ONE_PANEL_A = [
    ("Official-source transaction records", "OfficialRecords"),
    ("Retained Capitol Trades records", "VendorRecords"),
    ("Assembled transaction records", "AssembledRecords"),
    ("Records with observed official public dates", "ObservedOfficialDates"),
    ("Records using the vendor-date proxy", "VendorProxyDates"),
    ("CRSP-matched common-stock transaction records", "MatchedRecords"),
    ("Distinct public directional signals", "Signals"),
    ("Mixed-direction signals excluded from the primary contrast", "MixedSignals"),
    ("Vendor signals that duplicate an official transaction, removed", "DuplicateSignalsRemoved"),
    ("Signals with complete 20-day outcomes", "SignalsCompleteTwenty"),
]
TABLE_ONE_PANEL_B = [
    ("Households in full signal sample", "HouseholdsAllSignals"),
    ("Households in primary 20-day sample", "Households"),
    ("Securities", "Securities"),
    ("20-trading-day blocks", "Blocks"),
    ("Largest household share", "TopOneShare"),
    ("Top five household share", "TopFiveShare"),
    ("Top ten household share", "TopTenShare"),
    ("Herfindahl-implied effective household count", "EffectiveHouseholds"),
]
main.append(r"\newcommand{\TableOnePanelA}{%")
for label, field in TABLE_ONE_PANEL_A:
    main.append(tex_row(label, rf"\ExDedupTableOne{field}"))
main.append("}")

main.append(r"\newcommand{\TableOnePanelB}{%")
for label, field in TABLE_ONE_PANEL_B:
    main.append(tex_row(label, rf"\ExDedupTableOne{field}"))
main.append("}")

# The appendix comparison of the two samples, field by field (released sample beside this one).
main.append(r"\newcommand{\TableOneComparisonRows}{%")
for panel, rows in [("Panel A: Source construction and analytical units", TABLE_ONE_PANEL_A), ("Panel B: Coverage and household concentration", TABLE_ONE_PANEL_B)]:
    if rows is TABLE_ONE_PANEL_B:
        main.append(r"\midrule")
    main.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{panel}}}}}\\")
    main.append(r"\midrule")
    for label, field in rows:
        main.append(tex_row(label, rf"\ExOrigTableOne{field}", rf"\ExDedupTableOne{field}"))
main.append("}")

main.append(r"\newcommand{\TableThreeRows}{%")
horizon_ids = {5: "TW_P2_E5", 10: "TW_P2_E10", 20: "TW_P1_E20", 40: "TW_P2_E40", 60: "TW_P2_E60", 127: "TW_P2_E127"}
for horizon in [5, 10, 20, 40, 60, 127]:
    row = reg.loc[horizon_ids[horizon]]
    main.append(tex_row(
        "20 (primary)" if horizon == 20 else str(horizon),
        bps(row.purchase_mean),
        bps(row.sale_mean),
        bps(row.estimate),
        bps(row.se_multiway),
        f"[{bps(row.ci_low)}, {bps(row.ci_high)}]",
        pval(row.raw_p),
        "--" if horizon == 20 else pval(row.p_holm),
        integer(row.n),
    ))
main.append("}")

# Version-4 main-text exhibits: Table 2 merges the primary estimate with the
# horizon family; Table 3 has a compact (sample-mode) and a full (journal-mode) body.
main.append(r"\newcommand{\TableTwoPanelARows}{%")
main.append(tex_row(
    "Purchase-minus-sale spread", bps(primary.purchase_mean), bps(primary.sale_mean),
    bps(primary.estimate), bps(primary.se_multiway),
    f"[{bps(primary.ci_low)}, {bps(primary.ci_high)}]", pval(primary.raw_p), integer(primary.n),
))
main.append(tex_row(
    "\\quad 20-date block-bootstrap interval", "", "", "", "",
    f"[{bps(primary.block_ci_low)}, {bps(primary.block_ci_high)}]", "", "",
))
main.append(tex_row(
    "Signed mimicking payoff", "", "", bps(signed.estimate), bps(signed.se_multiway),
    f"[{bps(signed.ci_low)}, {bps(signed.ci_high)}]", pval(signed.raw_p), integer(signed.n),
))
main.append("}")

main.append(r"\newcommand{\TableTwoPanelBRows}{%")
for horizon in [5, 10, 40, 60, 127]:
    row = reg.loc[horizon_ids[horizon]]
    main.append(tex_row(
        f"{horizon} trading days", bps(row.purchase_mean), bps(row.sale_mean),
        bps(row.estimate), bps(row.se_multiway),
        f"[{bps(row.ci_low)}, {bps(row.ci_high)}]",
        f"{pval(row.raw_p)} ({pval(row.p_holm)})", integer(row.n),
    ))
main.append("}")

table_three_specs = [
    ("Primary official-first date rule", "TW_P1_E20", True),
    ("Raw Capitol Trades publication date for vendor-only records", "TW_P3_RAW_CT", True),
    ("Observed-official-date sample", "TW_P3_OFFICIAL_ONLY", True),
    ("Official-source segment", "TW_P3_OFFICIAL_SEGMENT", False),
    ("Vendor-proxy segment", "TW_P3_VENDOR_SEGMENT", False),
    ("Entry delayed by one additional trading day", "TW_P3_ENTRY_PLUS1", True),
]
for macro_name, compact in [("TableThreeRowsCompact", True), ("TableThreeRowsFull", False)]:
    main.append(rf"\newcommand{{\{macro_name}}}{{%")
    for label, test_id, in_compact in table_three_specs:
        if compact and not in_compact:
            continue
        row = reg.loc[test_id]
        holm = "--" if test_id == "TW_P1_E20" or pd.isna(row.p_holm) else pval(row.p_holm)
        main.append(tex_row(label, bps(row.estimate), bps(row.se_multiway), f"[{bps(row.ci_low)}, {bps(row.ci_high)}]", pval(row.raw_p), holm, integer(row.n)))
    main.append("}")

main.append(r"\newcommand{\TableFourRows}{%")
for label, test_id in [
    ("Primary official-first date rule", "TW_P1_E20"),
    ("Raw Capitol Trades publication date for vendor-only records", "TW_P3_RAW_CT"),
    ("Observed-official-date sample", "TW_P3_OFFICIAL_ONLY"),
    ("Official-source segment", "TW_P3_OFFICIAL_SEGMENT"),
    ("Vendor-proxy segment", "TW_P3_VENDOR_SEGMENT"),
    ("Entry delayed by one additional trading day", "TW_P3_ENTRY_PLUS1"),
]:
    row = reg.loc[test_id]
    holm = "--" if test_id == "TW_P1_E20" or pd.isna(row.p_holm) else pval(row.p_holm)
    main.append(tex_row(label, bps(row.estimate), bps(row.se_multiway), f"[{bps(row.ci_low)}, {bps(row.ci_high)}]", pval(row.raw_p), holm, integer(row.n)))
main.append("}")

main.append(r"\newcommand{\TableFiveRows}{%")
for label, test_id in [
    ("Public-action clock", "TW_P1_E20"),
    ("Transaction clock using standardized reported date", "TW_C1_T20"),
    ("Transaction clock using raw vendor trade date", "TW_C1_TRAW20"),
    ("Paired trade-date minus disclosure-date difference", "TW_C1_PAIRED"),
]:
    row = reg.loc[test_id]
    main.append(tex_row(label, bps(row.estimate), bps(row.se_multiway), f"[{bps(row.ci_low)}, {bps(row.ci_high)}]", pval(row.raw_p), integer(row.n)))
main.append("}")

# The funded portfolio (de-duplicated sample; the released constructor re-run on the de-duplicated
# purchase set, A5's full_results.csv): annual percent, Newey-West lag 20, Holm across the three tests.
main.append(r"\newcommand{\TableSixInferenceRows}{%")
for label, test_id in [
    ("Mean account excess return over risk-free, annualized", "TW_IMPL_PORT_NW"),
    ("CAPM alpha, annualized", "TW_IMPL_PORT_CAPM"),
    ("FF5 plus momentum alpha, annualized", "TW_IMPL_PORT_FF5MOM"),
]:
    row = rerun(test_id)
    main.append(tex_row(label, num(row.estimate), num(row.se_day), interval(row.ci_low_day, row.ci_high_day), pval(row.p_day), pval(row.p_holm_day), integer(row.n)))
main.append("}")

(GEN / "main_table_rows.tex").write_text(display_text("\n".join(main) + "\n"))

# ---------------------------------------------------------------------------
# Internet Appendix table bodies
# ---------------------------------------------------------------------------
appendix: list[str] = []


def panel_row(ncols: int, title: str) -> str:
    return rf"\addlinespace[3pt]\multicolumn{{{ncols}}}{{l}}{{\textit{{{title}}}}} \\"


appendix.append(r"\newcommand{\IAAttritionRows}{%")
stage_labels = {
    "public_date_and_sample_window": "Public date and sample window",
    "amendment_exact_duplicates": "Exact amendment duplicates",
    "purchase_sale_filter": "Purchase or sale classification",
    "security_matching": "Security matching",
    "common_stock_screen": "Common-stock screen",
    "collapse_to_signals": "Collapse to directional signals",
    "mixed_direction_exclusion": "Mixed-direction signals excluded",
    "vendor_duplicate_rule": "Vendor signals that duplicate an official transaction",
    "complete_h20_outcome": "Complete 20-day outcome window",
}
reason_labels = {
    "no_usable_public_date": "No usable public date",
    "before_sample_start": "Before sample start",
    "amendment_exact_duplicate": "Exact amendment duplicate",
    "no_purchase_or_sale_interpretation": "No purchase or sale classification",
    "unresolved_security_match": "Unresolved security match",
    "failed_common_stock_screen": "Failed common-stock screen",
    "collapsed_repeated_lots": "Repeated records collapsed into signals",
    "mixed_direction_signal": "Both directions in one signal group",
    "duplicate_of_official_transaction": "Vendor duplicate of an official transaction",
    "incomplete_h20_outcome": "Incomplete 20-day return window",
}
# The six released record-level stages (paper/data/attrition_ledger.csv) must agree with the re-run's
# frozen-frame counts; the three signal-level stages of this paper's sample come from the re-run.
released_stages = dedup_attrition[dedup_attrition["sample"] == "both"].set_index("stage")
for _, row in attrition.iterrows():
    if int(released_stages.loc[row["stage"], "n_in"]) != int(row["n_in"]) or int(released_stages.loc[row["stage"], "n_out"]) != int(row["n_out"]):
        raise ValueError(f"{DEDUP_FILES['attrition']}: stage {row['stage']} differs from the released attrition ledger")
reason_cols = [c for c in dedup_attrition.columns if c not in {"stage", "n_in", "n_out", "sample"}]
appendix.append(panel_row(4, "Record-level stages, the same on both samples"))
for _, row in dedup_attrition.iterrows():
    if row["stage"] == "mixed_direction_exclusion":
        appendix.append(panel_row(4, "Signal-level stages, this paper's sample"))
    reasons = []
    for column in reason_cols:
        value = row[column]
        if pd.notna(value) and float(value) != 0:
            reasons.append(f"{reason_labels[column]}: {integer(value)}")
    appendix.append(tex_row(stage_labels[row["stage"]], integer(row["n_in"]), integer(row["n_out"]), "; ".join(reasons) if reasons else "--"))
if int(dedup_attrition.iloc[-1]["n_out"]) != int(rerun("TW_P1_E20").n):
    raise ValueError("attrition.csv: the last stage does not end at the de-duplicated headline sample")
appendix.append("}")

# Vendor-minus-official date gap: the three released universes (pre-assembly linkage, rule-independent)
# and, from the re-run, the retained-overlap records that are parents of a signal and of a de-duplicated
# non-mixed signal. The rule removes none of the retained-overlap records, so that column is shared.
appendix.append(r"\newcommand{\IADateUniverseRows}{%")
gap_rows = date_gap[date_gap.gap_days != "total"].assign(gap_days=lambda d: d.gap_days.astype(int)).set_index("gap_days")
gap_total = date_gap[date_gap.gap_days == "total"].iloc[0]
gap_totals = {
    key: int(date_universes[f"{key}_n"].sum())
    for key in ("row_weighted", "filing_weighted", "retained_overlap")
}
for key in ("in_signals", "dedup_nonmixed_signals"):
    if int(gap_rows[f"{key}_n"].sum()) != int(gap_total[f"{key}_n"]):
        raise ValueError(f"date_gap.csv: inconsistent total for {key}")

for _, drow in date_universes.iterrows():
    gap = int(drow["gap_days"])
    if int(gap_rows.loc[gap, "retained_overlap_n"]) != int(drow["retained_overlap_n"]):
        raise ValueError(f"{DEDUP_FILES['date_gap']}: retained-overlap count at gap {gap} differs from the released table")
    appendix.append(tex_row(
        f"+{gap}",
        integer(drow["row_weighted_n"]), pct(drow["row_weighted_n"] / gap_totals["row_weighted"], 1) + r"\%",
        integer(drow["filing_weighted_n"]), pct(drow["filing_weighted_n"] / gap_totals["filing_weighted"], 1) + r"\%",
        integer(drow["retained_overlap_n"]), pct(drow["retained_overlap_n"] / gap_totals["retained_overlap"], 1) + r"\%",
        integer(gap_rows.loc[gap, "in_signals_n"]), pct(gap_rows.loc[gap, "in_signals_n"] / gap_total["in_signals_n"], 1) + r"\%",
        integer(gap_rows.loc[gap, "dedup_nonmixed_signals_n"]), pct(gap_rows.loc[gap, "dedup_nonmixed_signals_n"] / gap_total["dedup_nonmixed_signals_n"], 1) + r"\%",
    ))
appendix.append(r"\midrule")
appendix.append(tex_row("Total", integer(date_universes["row_weighted_n"].sum()), r"100.0\%", integer(date_universes["filing_weighted_n"].sum()), r"100.0\%",
                        integer(date_universes["retained_overlap_n"].sum()), r"100.0\%",
                        integer(gap_total["in_signals_n"]), r"100.0\%", integer(gap_total["dedup_nonmixed_signals_n"]), r"100.0\%"))
appendix.append("}")

# Alternative units and return models on the de-duplicated sample. Clustered rows use household and
# 20-trading-day-block clusters (the main text's convention); the equal-household and event-level rows
# are cluster-free and carry one standard error. p at three decimals, as the released table.
SENSITIVITY_ROWS = [
    ("Transaction-record weighting", "TW_S1_RAWROWS"),
    ("Equal-household weighting", "TW_S1_EQHH"),
    ("Security-entry-date event aggregation", "TW_S1_EVENT"),
    ("Source-native household identifiers", "TW_S1_SRCID"),
    ("CAPM-adjusted returns", "TW_S2_CAPM"),
    ("Five-factor-plus-momentum returns", "TW_S2_FF5MOM"),
]
appendix.append(r"\newcommand{\IASensitivityRows}{%")
for label, test_id in SENSITIVITY_ROWS:
    row = rerun(test_id)
    appendix.append(tex_row(label, num(row.estimate), num(row.se_block), interval(row.ci_low_block, row.ci_high_block), pval(row.p_block), integer(row.n)))
appendix.append("}")

# Analytical against calendar-block intervals, de-duplicated sample, with the earlier version's
# moving-block interval beside it so the released bootstrap values remain in print.
appendix.append(r"\newcommand{\IABlockRows}{%")
block_panels = [
    ("Panel A: Public-entry horizons", [
        ("20 trading days (headline)", "TW_P1_E20"),
        ("5 trading days", "TW_P2_E5"),
        ("10 trading days", "TW_P2_E10"),
        ("40 trading days", "TW_P2_E40"),
        ("60 trading days", "TW_P2_E60"),
        ("127 trading days", "TW_P2_E127"),
    ]),
    ("Panel B: Public-date and source checks", [
        ("Raw Capitol Trades publication date", "TW_P3_RAW_CT"),
        ("Observed official dates only", "TW_P3_OFFICIAL_ONLY"),
        ("Official-source segment", "TW_P3_OFFICIAL_SEGMENT"),
        ("Vendor-proxy segment", "TW_P3_VENDOR_SEGMENT"),
        ("Entry delayed one additional trading day", "TW_P3_ENTRY_PLUS1"),
    ]),
    ("Panel C: Return-date, unit, and return-model checks", [
        ("Transaction clock: standardized reported date", "TW_C1_T20"),
        ("Transaction clock: raw vendor-reported date", "TW_C1_TRAW20"),
        ("Transaction-record weighting", "TW_S1_RAWROWS"),
        ("Source-native household clusters", "TW_S1_SRCID"),
        ("CAPM-adjusted returns", "TW_S2_CAPM"),
        ("Five-factor-plus-momentum returns", "TW_S2_FF5MOM"),
    ]),
]
for panel, rows in block_panels:
    appendix.append(panel_row(5, panel))
    for label, test_id in rows:
        row, orig = rerun(test_id), rerun(test_id, "original")
        appendix.append(tex_row(label, num(row.estimate), interval(row.ci_low_block, row.ci_high_block), interval(row.boot_ci_low, row.boot_ci_high), interval(orig.boot_ci_low, orig.boot_ci_high)))
appendix.append("}")

# Design-preserving placebos on both samples (the same draws: freeze-hash seeds).
appendix.append(r"\newcommand{\IAPlaceboRows}{%")
for label, test_id in [
    ("Within-month direction permutation", "TW_PLC_DIRECTION"),
    ("Within-security-year pseudo-entry dates", "TW_PLC_DATES"),
]:
    for sample, sample_label in [("dedup", "This paper"), ("original", "Earlier version")]:
        row = rerun(test_id, sample)
        appendix.append(tex_row(label if sample == "dedup" else "", sample_label, num(row.estimate), num(row.placebo_mean), num(row.placebo_sd), integer(row.n_draws), pval(row.p_day)))
appendix.append("}")

appendix.append(r"\newcommand{\IAInferenceMapRows}{%")
inference_panels = [
    ("Panel A: Main estimate and horizons", [
        (
            "20-day public-entry spread",
            "Distinct public directional signal",
            "Purchase mean minus sale mean",
            r"Two-way CGM: household $\times$ exact public-entry date; 20-date block interval also reported",
            "None; single primary test",
            r"\path{TW_P1_E20}",
        ),
        (
            "Alternative horizons: 5, 10, 40, 60, and 127 days",
            "Distinct public directional signal",
            "Purchase mean minus sale mean",
            "Same two-way CGM and block procedures",
            "Holm across five horizons",
            r"\path{TW_P2_E5}, \path{TW_P2_E10}, \path{TW_P2_E40}, \path{TW_P2_E60}, \path{TW_P2_E127}",
        ),
    ]),
    ("Panel B: Source and entry-date checks", [
        (
            "Raw vendor date; official dates only; source segments; entry delayed one day",
            "Distinct public directional signal",
            "Purchase mean minus sale mean",
            "Two-way CGM using the exact entry date implied by each specification; block interval",
            "Holm across five source-and-timing checks",
            r"\path{TW_P3_RAW_CT}, \path{TW_P3_OFFICIAL_ONLY}, \path{TW_P3_OFFICIAL_SEGMENT}, \path{TW_P3_VENDOR_SEGMENT}, \path{TW_P3_ENTRY_PLUS1}",
        ),
        (
            "Standardized and raw trade dates",
            "Distinct public directional signal",
            "Purchase mean minus sale mean",
            "Two-way CGM using the exact trade-date entry session; block interval",
            "Holm across three return-date comparisons; descriptive",
            r"\path{TW_C1_T20}, \path{TW_C1_TRAW20}",
        ),
        (
            "Paired trade-date minus disclosure-date return",
            "Signal observed under both entry dates",
            "Mean within-signal return difference",
            r"Two-way CGM: household $\times$ exact public-entry date",
            "Same return-date comparison family; descriptive",
            r"\path{TW_C1_PAIRED}",
        ),
        (
            "Signed public-entry payoff",
            "Distinct public directional signal",
            r"Mean of $(2B_i-1)\AR^E_{i,20}$",
            r"Two-way CGM: household $\times$ exact public-entry date",
            "Diagnostic; no adjustment",
            r"\path{TW_D_SAR20}",
        ),
    ]),
    ("Panel C: Alternative units and return models", [
        (
            "Transaction-record weighting",
            "Underlying source transaction record",
            "Transaction-record-weighted purchase-minus-sale difference",
            "Two-way CGM; block interval",
            "Holm across four alternative-unit and identifier checks; sensitivity",
            r"\path{TW_S1_RAWROWS}",
        ),
        (
            "Equal-household weighting",
            "Household with purchase and sale signals",
            "Mean household-specific spread",
            r"One-sample Student $t$, $G-1$ df",
            "Same alternative-unit family; sensitivity",
            r"\path{TW_S1_EQHH}",
        ),
        (
            "Security-entry-date aggregation",
            "Purchase-only and sale-only security-date events",
            "Difference in event means",
            r"Welch two-sample $t$",
            "Same alternative-unit family; sensitivity",
            r"\path{TW_S1_EVENT}",
        ),
        (
            "Source-native household identifiers",
            "Distinct public directional signal",
            "Purchase mean minus sale mean",
            "Two-way CGM using source-native household clusters; block interval",
            "Same alternative-unit family; sensitivity",
            r"\path{TW_S1_SRCID}",
        ),
        (
            "CAPM and five-factor-plus-momentum returns",
            "Distinct public directional signal",
            "Purchase mean minus sale mean in factor-adjusted returns",
            "Two-way CGM; block interval",
            "Holm across two return models",
            r"\path{TW_S2_CAPM}, \path{TW_S2_FF5MOM}",
        ),
    ]),
    ("Panel D: Funded portfolio", [
        (
            "Mean account excess return",
            "Portfolio trading day",
            "Intercept-only mean of account return minus risk-free rate",
            "Newey--West HAC, lag 20; estimate and SE annualized",
            "Holm across three portfolio tests",
            r"\path{TW_IMPL_PORT_NW}",
        ),
        (
            "CAPM and five-factor-plus-momentum alpha",
            "Portfolio trading day",
            "Time-series alpha",
            "Newey--West HAC, lag 20; alpha and SE annualized",
            "Same portfolio family",
            r"\path{TW_IMPL_PORT_CAPM}, \path{TW_IMPL_PORT_FF5MOM}",
        ),
    ]),
    ("Panel E: Placebos", [
        (
            "Direction permutation",
            "Distinct public directional signal",
            "Observed spread relative to 999 within-month label permutations",
            r"Two-sided finite-randomization $p$-value",
            "Diagnostic; no adjustment",
            r"\path{TW_PLC_DIRECTION}",
        ),
        (
            "Pseudo-entry dates",
            "Distinct public directional signal",
            "Observed spread relative to 19 within-security-year pseudo-date draws",
            r"Two-sided finite-randomization $p$-value",
            "Diagnostic; no adjustment",
            r"\path{TW_PLC_DATES}",
        ),
    ]),
]
# The registry IDs (sixth element of each tuple) are rendered in a note below the
# table rather than as a column, so the landscape table fits with every column visible.
id_notes: list[str] = []
for panel, rows in inference_panels:
    appendix.append(panel_row(5, panel))
    for row in rows:
        appendix.append(tex_row(*row[:5]))
        id_notes.append(f"{row[0]}: {row[5]}")
appendix.append("}")
appendix.append(r"\newcommand{\IAInferenceMapIDNotes}{" + "; ".join(id_notes) + "}")

appendix.append(r"\newcommand{\IARegistryRows}{%")
registry_panels = [
    ("Panel A: Primary estimate", [
        ("20-day public-entry spread", "TW_P1_E20", False),
    ]),
    ("Panel B: Alternative horizons -- Holm across five", [
        ("5 trading days", "TW_P2_E5", True),
        ("10 trading days", "TW_P2_E10", True),
        ("40 trading days", "TW_P2_E40", True),
        ("60 trading days", "TW_P2_E60", True),
        ("127 trading days", "TW_P2_E127", True),
    ]),
    ("Panel C: Public-date and source checks -- Holm across five", [
        ("Raw Capitol Trades publication date", "TW_P3_RAW_CT", True),
        ("Observed official dates only", "TW_P3_OFFICIAL_ONLY", True),
        ("Official-source segment", "TW_P3_OFFICIAL_SEGMENT", True),
        ("Vendor-proxy segment", "TW_P3_VENDOR_SEGMENT", True),
        ("Entry delayed one additional trading day", "TW_P3_ENTRY_PLUS1", True),
    ]),
    ("Panel D: Transaction-clock comparisons -- Holm across three; descriptive", [
        ("Standardized reported trade date", "TW_C1_T20", True),
        ("Raw vendor-reported transaction date", "TW_C1_TRAW20", True),
        ("Paired trade-date minus disclosure-date difference", "TW_C1_PAIRED", True),
    ]),
    ("Panel E: Directional diagnostic", [
        ("Signed public-entry payoff", "TW_D_SAR20", False),
    ]),
    ("Panel F: Alternative units and identifiers -- Holm across four; sensitivity", [
        ("Transaction-record weighting", "TW_S1_RAWROWS", True),
        ("Equal-household weighting", "TW_S1_EQHH", True),
        ("Security-entry-date event aggregation", "TW_S1_EVENT", True),
        ("Source-native household identifiers", "TW_S1_SRCID", True),
    ]),
    ("Panel G: Factor-adjusted returns -- Holm across two", [
        ("CAPM-adjusted spread", "TW_S2_CAPM", True),
        ("Five-factor-plus-momentum spread", "TW_S2_FF5MOM", True),
    ]),
    ("Panel H: Funded portfolio -- annualized basis points; Holm across three", [
        ("Mean excess return", "TW_IMPL_PORT_NW", True),
        ("CAPM alpha", "TW_IMPL_PORT_CAPM", True),
        ("Five-factor-plus-momentum alpha", "TW_IMPL_PORT_FF5MOM", True),
    ]),
    ("Panel I: Placebo tests -- finite-randomization probabilities", [
        ("Within-month direction permutation", "TW_PLC_DIRECTION", False),
        ("Within-security-year pseudo-entry dates", "TW_PLC_DATES", False),
    ]),
]
# Every analysis of the earlier version's plan on the de-duplicated sample: block clusters and Holm
# within family with block clusters; portfolio rows in annualized basis points (annual percent x 100),
# as the released table printed them; placebo rows carry the observed spread and the randomization p.
for panel, rows in registry_panels:
    appendix.append(panel_row(6, panel))
    for label, test_id, show_adjusted in rows:
        row = rerun(test_id)
        scale = 100.0 if row.units == "pct_annual" else 1.0
        if test_id.startswith("TW_PLC"):
            ci = "--"
        elif row.units == "pct_annual":
            ci = interval(row.ci_low_day * scale, row.ci_high_day * scale)
        else:
            ci = interval(row.ci_low_block, row.ci_high_block)
        raw_p = row.p_day if row.units == "pct_annual" or test_id.startswith("TW_PLC") else row.p_block
        holm = row.p_holm_day if row.units == "pct_annual" else row.p_holm_block
        adjusted = pval(holm) if show_adjusted and pd.notna(holm) else "--"
        appendix.append(tex_row(label, num(row.estimate * scale), ci, pval(raw_p), adjusted, integer(row.n)))
appendix.append("}")

(GEN / "appendix_table_rows.tex").write_text(display_text("\n".join(appendix) + "\n"))

# ---------------------------------------------------------------------------
# Vector figures from canonical machine-readable results
# ---------------------------------------------------------------------------
horizon_df = pd.read_csv(DATA / "table3_horizons.csv")
horizons = np.array([5, 10, 20, 40, 60, 127])
estimate = horizon_df["estimate"].to_numpy() * 10_000
lower = horizon_df["ci_low"].to_numpy() * 10_000
upper = horizon_df["ci_high"].to_numpy() * 10_000
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.errorbar(horizons, estimate, yerr=[estimate - lower, upper - estimate], fmt="o", capsize=4, elinewidth=1.4, markersize=5.4)
ax.axhline(0, linewidth=0.8, color="0.35")
ax.set_xlabel("Trading days after public entry")
ax.set_ylabel("Purchase-minus-sale abnormal return (basis points)")
ax.set_xticks(horizons)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18)
fig.tight_layout()
fig.savefig(FIG / "figure2_horizons.pdf", bbox_inches="tight", metadata={"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None})
fig.savefig(FIG / "figure2_horizons.png", dpi=300, bbox_inches="tight")
plt.close(fig)

event_df = pd.read_csv(DATA / "figure3_event_study_path.csv")
difference = event_df[event_df.series == "difference"].sort_values("relative_trading_day")
x = difference.relative_trading_day.to_numpy()
y = difference.estimate.to_numpy() * 10_000
lo = difference.lower.to_numpy() * 10_000
hi = difference.upper.to_numpy() * 10_000
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.fill_between(x, lo, hi, alpha=0.20, label="Simultaneous 95% band")
ax.plot(x, y, linewidth=1.6, label="Purchase-minus-sale cumulative path")
ax.axhline(0, linewidth=0.8, color="0.35")
ax.axvline(0, linewidth=1.0, linestyle="--", color="0.45")
ax.set_xlabel("Trading days relative to public entry (day 0 = entry close)")
ax.set_ylabel("Cumulative abnormal-return difference (basis points)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.16)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "figure3_event_study.pdf", bbox_inches="tight", metadata={"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None})
fig.savefig(FIG / "figure3_event_study.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.5))
for series, label in [("purchase", "Purchases"), ("sale", "Sales")]:
    subset = event_df[event_df.series == series].sort_values("relative_trading_day")
    ax.plot(subset.relative_trading_day, subset.estimate * 10_000, linewidth=1.4, label=label)
ax.axhline(0, linewidth=0.8, color="0.35")
ax.axvline(0, linewidth=1.0, linestyle="--", color="0.45")
ax.set_xlabel("Trading days relative to public entry")
ax.set_ylabel("Cumulative market-adjusted return (basis points)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.16)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "ia_purchase_sale_paths.pdf", bbox_inches="tight", metadata={"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None})
fig.savefig(FIG / "ia_purchase_sale_paths.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# The same three figures on the de-duplicated sample (A5's horizons.csv with block-clustered intervals
# and event_study_path.csv with the 4,999-draw simultaneous band); the released files above stay as the
# comparison. The appendix includes the de-duplicated versions.
FIG_META = {"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None}
hd = horizons_dedup[horizons_dedup["sample"] == "dedup"].sort_values("horizon")
if list(hd.horizon) != [5, 10, 20, 40, 60, 127]:
    raise ValueError(f"{DEDUP_FILES['horizons']}: unexpected de-duplicated horizon set {list(hd.horizon)}")
for _, hrow in hd.iterrows():
    if abs(float(hrow.estimate) - float(rerun(hrow.test_id).estimate)) > 1e-9:
        raise ValueError(f"{DEDUP_FILES['horizons']}: {hrow.test_id} differs from full_results.csv")
est_d, lo_d, hi_d = hd.estimate.to_numpy(), hd.ci_low_block.to_numpy(), hd.ci_high_block.to_numpy()
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.errorbar(hd.horizon.to_numpy(), est_d, yerr=[est_d - lo_d, hi_d - est_d], fmt="o", capsize=4, elinewidth=1.4, markersize=5.4)
ax.axhline(0, linewidth=0.8, color="0.35")
ax.set_xlabel("Trading days after public entry")
ax.set_ylabel("Purchase-minus-sale abnormal return (basis points)")
ax.set_xticks(hd.horizon.to_numpy())
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18)
fig.tight_layout()
fig.savefig(FIG / "figure2_horizons_dedup.pdf", bbox_inches="tight", metadata=FIG_META)
plt.close(fig)

ed = event_dedup[event_dedup["sample"] == "dedup"]
diff_d = ed[ed.series == "difference"].sort_values("relative_trading_day")
if len(diff_d) != 71 or (diff_d.lower > diff_d.estimate).any() or (diff_d.upper < diff_d.estimate).any():
    raise ValueError(f"{DEDUP_FILES['event_study_path']}: the de-duplicated difference path is incomplete or its band does not contain it")
if ((diff_d.lower > 0) | (diff_d.upper < 0)).any():
    raise ValueError(f"{DEDUP_FILES['event_study_path']}: the de-duplicated band excludes zero on some day; the appendix prose says it never does")
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.fill_between(diff_d.relative_trading_day, diff_d.lower * 10_000, diff_d.upper * 10_000, alpha=0.20, label="Simultaneous 95% band")
ax.plot(diff_d.relative_trading_day, diff_d.estimate * 10_000, linewidth=1.6, label="Purchase-minus-sale cumulative path")
ax.axhline(0, linewidth=0.8, color="0.35")
ax.axvline(0, linewidth=1.0, linestyle="--", color="0.45")
ax.set_xlabel("Trading days relative to public entry (day 0 = entry close)")
ax.set_ylabel("Cumulative abnormal-return difference (basis points)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.16)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "figure3_event_study_dedup.pdf", bbox_inches="tight", metadata=FIG_META)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.5))
for series, label in [("purchase", "Purchases"), ("sale", "Sales")]:
    subset = ed[ed.series == series].sort_values("relative_trading_day")
    ax.plot(subset.relative_trading_day, subset.estimate * 10_000, linewidth=1.4, label=label)
ax.axhline(0, linewidth=0.8, color="0.35")
ax.axvline(0, linewidth=1.0, linestyle="--", color="0.45")
ax.set_xlabel("Trading days relative to public entry")
ax.set_ylabel("Cumulative market-adjusted return (basis points)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.16)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "ia_purchase_sale_paths_dedup.pdf", bbox_inches="tight", metadata=FIG_META)
plt.close(fig)


# ---------------------------------------------------------------------------
# Version-4 household-analysis placeholder macros (\H<Analysis><Field>). Every macro is
# a \providecommand with a visible default. Two override files are input after this one
# and use \renewcommand: generated/results_macros.tex (household results, written once by
# scripts/v4/run_results.py from data/v4/results.csv) and generated/existing_rerun_macros.tex
# (the existing paper's tables re-run on the de-duplicated sample with block clusters).
# ---------------------------------------------------------------------------
PENDING = r"\text{[PENDING]}"
v4_macros: list[tuple[str, str, str, str]] = []  # (name, default, description, source)


def v4(name: str, description: str, default: str = PENDING, source: str = "") -> None:
    """Register a placeholder. A non-pending default must name its source: a plan
    constant ("plan") or the outcome-free diagnostics file it was read from."""
    if default != PENDING and not default.startswith("%") and not source:
        raise ValueError(f"placeholder {name} has a non-pending default without a source")
    v4_macros.append((name, default, description, source))


# Household results on actual outcomes are NOT placeholders here: scripts/v4/run_results.py
# writes generated/results_macros.tex with the \Res... names, and the existing-results re-run
# writes existing_rerun_macros.tex with the \Ex... names of paper/generated/existing_macros.md. Every \Res/\Ex name the manuscript
# uses is \providecommand'd as [PENDING] in generated/v4_results_pending.tex (written below),
# input after both files, so the build never breaks on a missing value. The \H... macros
# below are outcome-free: design constants, counts, placebo calibration, frontiers.

# What the record could detect: 80%-power frontiers computed on
# placebo data (data/v4/calibration/frontiers_80.csv), read at every build. Only
# size-correct, non-superseded rows enter. "Best" is the smallest effect at which any correctly
# sized test reaches 80% power; a frontier not reached inside the effect grid prints "--".
FRONTIER_FILE = "data/v4/calibration/frontiers_80.csv"
frontiers = pd.read_csv(ROOT.parent / FRONTIER_FILE)
frontiers = frontiers[(~frontiers.superseded.astype(bool)) & (frontiers.size_correct.astype(bool))]
PLAN_VARIANT = {"T1_blk20": "raw", "T1_blk20_2s": "raw", "T4_z2": "raw", "T4_z3": "raw"}  # tau2 and max-t tests: winsorized


def frontier_rows(clock: str, alternative: str, **where) -> pd.DataFrame:
    rows = frontiers[(frontiers.clock == clock) & (frontiers.alternative == alternative)]
    for col, val in where.items():
        rows = rows[rows[col] == val]
    keep = [i for i, r in rows.iterrows() if r.variant == PLAN_VARIANT.get(r.test, "wins")]
    return rows.loc[keep]


def frontier_value(rows: pd.DataFrame, test: str | None = None) -> str:
    if test is not None:
        rows = rows[rows.test == test]
    reached = rows[rows.status != "not_reached"].dropna(subset=["effect_bp_at_target"])
    if reached.empty:
        return "--"
    return num(reached.effect_bp_at_target.min(), 0)


for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    dense = frontier_rows(clock, "dense")
    v4(f"DetectDenseBest{tag}", f"dense ability: smallest tau (bp per 20 trading days) at which any correctly sized test reaches 80% power, clock {clock}", frontier_value(dense), FRONTIER_FILE)
    v4(f"DetectDenseTauTest{tag}", f"dense ability: tau at 80% power for the placebo test of tau-squared (event-share weights), clock {clock}", frontier_value(dense, "T2_I1cal_ES"), FRONTIER_FILE)
    v4(f"DetectDenseBestHousehold{tag}", f"dense ability: tau at 80% power for the best-household test, clock {clock}", frontier_value(dense, "T3"), FRONTIER_FILE)

# Figure 2: one frontier per panel, the best correctly
# sized test at each share; (a) share of households informed, (b) share of purchases informed.
# The prose quotes the frontier at reference shares, read from the figure's own CSVs when they
# exist (else the macros stay pending and the caption still describes the figure).
FIG2_DIR = ROOT.parent / "data" / "v4" / "calibration"


def at_share(frame: pd.DataFrame, x: float) -> str:
    row = frame.iloc[(frame.x - x).abs().argmin()]
    return "--" if pd.isna(row.frontier_bp) else num(row.frontier_bp, 0)


# the main-text figure now judges every test on one common
# sample (the households the estimand is defined on), so the unprefixed files carry the common-sample
# frontiers and the "testspecific" files carry the version in which each test keeps its own analysis
# set, which is now an appendix figure. Both sets of macros are generated; the prose picks the one
# that matches the figure it describes.
for stem, name in [("", "Detect"), ("testspecific_", "DetectTestSpecific")]:
    for clock, tag, prefix in [("T", "Txn", ""), ("P", "Pub", "public_")]:
        files = {panel: FIG2_DIR / f"figure2_detection_region_v3_{stem}{prefix}{panel}_frontier.csv" for panel in ["a", "b"]}
        if not all(f.exists() for f in files.values()):
            continue
        fa = pd.read_csv(files["a"]).sort_values("x")
        fb = pd.read_csv(files["b"]).sort_values("x")
        src_a, src_b = str(files["a"].relative_to(ROOT.parent)), str(files["b"].relative_to(ROOT.parent))
        v4(f"{name}HHCount{tag}", f"households in the sample from which informed households are drawn on Figure 2(a), the smallest share being one of them (count), clock {clock}", integer(round(1 / fa.x.min())), src_a)
        v4(f"{name}ShareHHOne{tag}", f"Figure 2(a): edge per purchase (bp) at which one informed household is detectable with 80% power, clock {clock}", at_share(fa, fa.x.min()), src_a)
        for x, word in [(0.05, "Five"), (0.10, "Ten"), (0.5, "Half"), (1.0, "All")]:
            v4(f"{name}ShareHH{word}{tag}", f"Figure 2(a): edge per purchase (bp) at which informed households at share {x:g} are detectable with 80% power, clock {clock}", at_share(fa, x), src_a)
        for x, word in [(0.01, "One"), (0.02, "Two"), (0.05, "Five"), (0.10, "Ten"), (0.5, "Half"), (1.0, "All")]:
            v4(f"{name}ShareTrade{word}{tag}", f"Figure 2(b): edge per informed purchase (bp) at which informed purchases at share {x:g} are detectable with 80% power, clock {clock}", at_share(fb, x), src_b)
        # The dense case drawn as the short bar inside the right edge of panel (a). The dense runs
        # exist only on the full panel, so on the common-sample figure the bar spans the two
        # variance-component tests, which are the ones that read the persistence households there.
        dense_file = FIG2_DIR / f"figure2_detection_region_v3_{stem}{prefix}dense.csv"
        if dense_file.exists():
            dense_csv = pd.read_csv(dense_file).dropna(subset=["effect_bp_at_target"])
            src_d = str(dense_file.relative_to(ROOT.parent))
            v4(f"{name}DenseBarLow{tag}", f"Figure 2(a) dense bar: smallest tau (bp per 20 trading days) at which a test on this figure's sample detects dense ability with 80% power, clock {clock}", num(dense_csv.effect_bp_at_target.min(), 0), src_d)
            v4(f"{name}DenseBarHigh{tag}", f"Figure 2(a) dense bar: largest such tau across the tests that reach 80% power on this figure, clock {clock}", num(dense_csv.effect_bp_at_target.max(), 0), src_d)
v4("DetectPanels", "effect-injected placebo datasets per grid point in the power surfaces (count)", integer(frontiers.n_panels.min()), FRONTIER_FILE)

# The common sample of the main-text figure: the households that buy in both periods and the
# purchases they carry, averaged over the placebo datasets.
COMMON_SIZE_FILE = "data/v4/calibration/common_sample_size.csv"
common_size = pd.read_csv(ROOT.parent / COMMON_SIZE_FILE)
for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    row = common_size[common_size.clock == clock]
    if len(row) != 1:
        raise ValueError(f"{COMMON_SIZE_FILE}: expected one row per clock, found {len(row)} for {clock}")
    row = row.iloc[0]
    v4(f"DetectCommonHouseholds{tag}", f"Figure 2 common sample: households that buy in both periods (count), clock {clock}", integer(row.n_households_r1), COMMON_SIZE_FILE)
    v4(f"DetectCommonEvents{tag}", f"Figure 2 common sample: purchases on average across the placebo datasets (count), clock {clock}", integer(round(row.n_events_mean)), COMMON_SIZE_FILE)
    v4(f"DetectCommonEventsLow{tag}", f"Figure 2 common sample: fewest purchases across the placebo datasets (count), clock {clock}", integer(row.n_events_min), COMMON_SIZE_FILE)
    v4(f"DetectCommonEventsHigh{tag}", f"Figure 2 common sample: most purchases across the placebo datasets (count), clock {clock}", integer(row.n_events_max), COMMON_SIZE_FILE)

# Design constants used in the prose (grid values), not results.
v4("TTDReferenceN", "A6 reference purchases per year used in prose (plan grid value)", "50", "plan")
v4("TTDReferenceTheta", "A6 reference annual edge used in prose (plan grid value)", r"5\%", "plan")
# The inputs of the years-to-detect formula at the precision that reproduces the printed years:
# sigma to one decimal and d to three (932 and 1.08 give 73.7, not the 74.0 of the results file).
TTD_FILE = "data/v4/results_time_to_detect.csv"
ttd = pd.read_csv(ROOT.parent / TTD_FILE)
for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    cell = ttd[(ttd.clock == clock) & np.isclose(ttd.theta_annual_pct, 5.0) & (ttd.n_per_year == 50)]
    if len(cell) != 1 or ttd[ttd.clock == clock].sigma_bp.nunique() != 1 or ttd[ttd.clock == clock].design_effect.nunique() != 1:
        raise ValueError(f"{TTD_FILE}: expected one reference cell and one sigma and d per clock ({clock})")
    cell = cell.iloc[0]
    sigma_one, d_three = round(float(cell.sigma_bp), 1), round(float(cell.design_effect), 3)
    years_from_printed = ((1.645 + 0.842) * sigma_one * d_three**0.5 / float(cell.theta_20_bp)) ** 2 / float(cell.n_per_year)
    if round(years_from_printed, 1) != round(float(cell.years), 1):
        raise ValueError(f"{TTD_FILE}: sigma {sigma_one} and d {d_three} give {years_from_printed:.2f} years, the file {cell.years:.2f} ({clock})")
    v4(f"TTDSigma{tag}", f"outcome SD sigma of the years-to-detect formula, bp per 20 trading days, one decimal, clock {clock}", num(sigma_one, 1), TTD_FILE)
    v4(f"TTDDesignEffect{tag}", f"design effect d of the years-to-detect formula, three decimals, clock {clock}", num(d_three, 3), TTD_FILE)

# The date on which the baseline design was fixed, typeset as a macro so it is written once.
v4("PlanCommitDate", "date on which the pre-analysis plan was committed to the repository (before any household estimate on actual outcomes)", "21 September 2026", "plan")

# Filing delay as a signal (exploratory).
# Pooled slopes with household and year-month fixed effects absorbed; no household-level
# quantity. Read from the committed outputs; the trigger row is S1 on purchases, h20.
DELAY_FILE = "data/v4/delay/stage1_results.csv"
DELAY_MANIFEST = "data/v4/delay/stage1_manifest.json"
delay = pd.read_csv(ROOT.parent / DELAY_FILE)
delay = delay[delay.is_trigger_spec.astype(str).str.lower() == "true"]
if len(delay) != 1:
    raise ValueError(f"expected one trigger row in {DELAY_FILE}, found {len(delay)}")
delay = delay.iloc[0]
delay_manifest = json.loads((ROOT.parent / DELAY_MANIFEST).read_text())
delay_triggered = bool(delay_manifest["trigger"]["triggered"]) if isinstance(delay_manifest.get("trigger"), dict) else abs(float(delay.t_stat)) >= 2.5
for name, value, desc in [
    ("DelayLogLagCoef", num(delay.coef_bp, 1), "A8 stage-1 coefficient on log(lag), household + year-month FE, purchases, h20 raw Y, bp per log unit"),
    ("DelayLogLagSE", num(delay.se_bp, 1), "A8 stage-1 two-way-clustered standard error, bp"),
    ("DelayLogLagT", num(delay.t_stat, 2), "A8 stage-1 t-statistic on log(lag) (trigger |t| >= 2.5)"),
    ("DelayLogLagCILow", num(delay.ci_low_bp, 1), "A8 stage-1 95% interval lower bound, bp per log unit"),
    ("DelayLogLagCIHigh", num(delay.ci_high_bp, 1), "A8 stage-1 95% interval upper bound, bp per log unit"),
    ("DelayN", integer(delay.n_events), "A8 stage-1 purchases with an observed official date, lag 1-365 days and a complete outcome (count)"),
    ("DelayHouseholds", integer(delay.n_households), "A8 stage-1 households (count)"),
]:
    v4(name, desc, value, DELAY_FILE)
if delay_triggered:
    raise ValueError("the filing-delay first stage is recorded as triggered; the manuscript footnote states that the second stage was not run")

# Outcome-free diagnostics written (counts and shares only; no outcome
# enters). Read from the named files; never typed. These counts describe the
# stream BEFORE the cross-source duplicate filter, as D built it.
DIAG = ROOT.parent / "data" / "v4" / "diagnostics"
DUP_FILE = "data/v4/diagnostics/duplicate_audit_summary.csv"
DUP_T_FILE = "data/v4/diagnostics/duplicate_audit_t_episodes.csv"
CONC_FILE = "data/v4/diagnostics/concentration_by_clock_period.csv"
CONT_FILE = "data/v4/diagnostics/continuation_rates.csv"
dup = pd.read_csv(ROOT.parent / DUP_FILE).set_index("variant").loc["primary"]
dup_t = pd.read_csv(ROOT.parent / DUP_T_FILE).set_index("variant").loc["primary"]
# D renamed this column during its fix pass; accept either name.
dup_t_share = dup_t["share_t_events_with_flagged_line" if "share_t_events_with_flagged_line" in dup_t.index else "share_t_events_affected"]
for name, value, desc in [
    ("DupLines", integer(dup.vendor_rows_matched), "CP-3 vendor lines reproducing an official transaction (count)"),
    ("DupVendorLines", integer(dup.vendor_rows), "CP-3 vendor-only CRSP-matched transaction lines audited (count)"),
    ("DupVendorShare", pct(dup.share_vendor_rows_matched, 1) + r"\%", "CP-3 share of vendor lines that reproduce an official transaction"),
    ("DupHouseholds", integer(dup.households_affected), "CP-3 households with at least one such line (count)"),
    ("DupEventsP", integer(dup.p_events_affected), "CP-3 public-clock events containing such a line (count)"),
    ("DupEventsPTotal", integer(dup.p_events_total), "CP-3 public-clock events audited (count)"),
    ("DupShareP", pct(dup.share_p_events_affected, 1) + r"\%", "CP-3 share of public-clock events containing such a line"),
    ("DupPurchaseShareP", pct(dup.share_p_purchase_events_affected, 1) + r"\%", "CP-3 share of public-clock purchase events affected"),
    ("DupShareT", pct(dup_t_share, 1) + r"\%", "CP-3 share of transaction-clock episodes containing such a line"),
]:
    v4(name, desc, value, DUP_FILE if not name.endswith("T") else DUP_T_FILE)

ACT_TWO_STREAM = "drop_dup_primary"  # default; D writes one block per dup_filter


def act_two_stream(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rows of a D diagnostics table for the Act II default stream (or all rows if the
    file carries no dup_filter column). Returns the rows and a label for the note."""
    if "dup_filter" not in frame.columns:
        return frame, "before the cross-source duplicate filter"
    return frame[frame.dup_filter == ACT_TWO_STREAM].drop(columns="dup_filter"), "duplicate vendor lines dropped"


conc, conc_stream = act_two_stream(pd.read_csv(ROOT.parent / CONC_FILE))
conc = conc.set_index(["clock", "period", "households"]).sort_index()
v4("ConcStreamNote", "which stream the concentration macros describe (from the dup_filter column of D's file)", conc_stream, CONC_FILE)
for clock in ["T", "P"]:
    for period in ["A", "B"]:
        for hh, tag in [("all", ""), ("R1", "Rone")]:
            row = conc.loc[(clock, period, hh)]
            base = f"Conc{clock}{period}{tag}"
            who = "all households" if hh == "all" else "R1-eligible households"
            v4(base + "Eff", f"concentration of purchases, clock {clock}, period {period}, {who}: effective household count (inverse Herfindahl)", num(row.effective_household_count, 1), CONC_FILE)
            v4(base + "Top", f"concentration of purchases, clock {clock}, period {period}, {who}: largest household share", pct(row.top_1_share, 1) + r"\%", CONC_FILE)
            v4(base + "NHH", f"purchases, clock {clock}, period {period}, {who}: households (count)", integer(row.n_households), CONC_FILE)
            v4(base + "NEvents", f"purchases, clock {clock}, period {period}, {who}: purchase events (count)", integer(row.n_events), CONC_FILE)

cont, cont_stream = act_two_stream(pd.read_csv(ROOT.parent / CONT_FILE))
v4("ContinuationStreamNote", "which stream the continuation macros describe (from the dup_filter column of D's file)", cont_stream, CONT_FILE)
cont_rows = []
for clock, clock_tag in [("T", "Txn"), ("P", "Pub")]:
    sub = cont[cont.clock == clock].set_index("bin_A")
    active = sub.drop(index="0")
    v4(f"ContinuationBase{clock_tag}", f"households with at least one period-A purchase, clock {clock} (count)", integer(active.n_households.sum()), CONT_FILE)
    v4(f"ContinuationRate{clock_tag}", f"share of households with at least one period-A purchase that record at least one period-B purchase, clock {clock}", pct(active.n_with_nB_ge1.sum() / active.n_households.sum(), 1) + r"\%", CONT_FILE)
for bin_a in ["0", "1-4", "5-9", "10-19", "20-49", "50+"]:
    cells = [bin_a.replace("-", "--")]
    for clock in ["T", "P"]:
        row = cont[(cont.clock == clock) & (cont.bin_A == bin_a)].iloc[0]
        nested_counts = [row.n_households, row.n_with_nB_ge1, row.n_with_nB_ge5, row.n_with_nB_ge10]
        if any(left < right for left, right in zip(nested_counts, nested_counts[1:])):
            raise ValueError(f"continuation_rates.csv: non-nested counts for {clock}, {bin_a}")

        cells += [integer(row.n_households), pct(row.share_nB_ge1, 1) + r"\%", pct(row.share_nB_ge5, 1) + r"\%", pct(row.share_nB_ge10, 1) + r"\%"]
    cont_rows.append(tex_row(*cells))
v4("IAContinuationRows", "continuation table body: rows are period-A purchase-count bins, columns households and shares with n_B >= 1 / 5 / 10 on each clock", "%\n" + "\n".join(cont_rows) + "\n", CONT_FILE)

# The post-2023 vendor stream is dominated by households with no parsed official filing. Counts and
# shares only, from the coverage and stream-size files.
COV_FILE = "data/v4/diagnostics/vendor_household_coverage.csv"
CLASS_FILE = "data/v4/diagnostics/vendor_household_classes.csv"
STREAM_FILE = "data/v4/diagnostics/stream_sizes.csv"
cov = pd.read_csv(ROOT.parent / COV_FILE)
cov = cov[cov.direction == "purchase"].set_index(["household_class", "clock", "period"]).sort_index()
classes = pd.read_csv(ROOT.parent / CLASS_FILE).set_index("household_class")
blind = classes.loc["vendor_without_official_2023plus"]
all_vendor = classes.loc["all_vendor_households"]
v4("VendorHouseholds", "households with any retained vendor line, 2023 onward (count)", integer(all_vendor.n_households), CLASS_FILE)
v4("VendorBlindHouseholds", "vendor households with no parsed official transaction line at all (count)", integer(blind.n_households_no_official_line_at_all), CLASS_FILE)
v4("VendorLinesTotal", "retained vendor-only CRSP-matched lines (count)", integer(all_vendor.vendor_lines), CLASS_FILE)
v4("VendorBlindLineShare", "share of all vendor lines supplied by households with no parsed official line", pct(blind.vendor_lines / all_vendor.vendor_lines, 1) + r"\%", CLASS_FILE)
v4("VendorBlindTopTwoLineShare", "share of all vendor lines supplied by the two largest such households", pct(blind.top2_households_vendor_lines / all_vendor.vendor_lines, 1) + r"\%", CLASS_FILE)
for clock in ["T", "P"]:
    b = cov.loc[("vendor_without_official_2023plus", clock, "B")]
    total_b = float(b.period_total_events)
    vendor_src_b = float(cov.loc[(slice(None), clock, "B"), "n_events_vendor_source"].sum())
    v4(f"VendorBlindHH{clock}B", f"households with no parsed official line that have period-B purchases, clock {clock} (count)", integer(b.n_households), COV_FILE)
    v4(f"VendorBlindShare{clock}B", f"share of period-B purchases from households with no parsed official line, clock {clock}", pct(b.share_of_period_events, 1) + r"\%", COV_FILE)
    v4(f"VendorBlindTopTwoShare{clock}B", f"share of period-B purchases from the two largest such households, clock {clock}", pct(b.top2_share_of_period_events, 1) + r"\%", COV_FILE)
    v4(f"OfficialOnlyRemovedShare{clock}B", f"share of period-B purchases that are vendor-sourced and leave under the official-only run, clock {clock}", pct(vendor_src_b / total_b, 1) + r"\%", COV_FILE)
    v4(f"PeriodBPurchases{clock}", f"period-B purchases on the full Act II stream (all-lines duplicate filter), clock {clock} (count)", integer(total_b), COV_FILE)
stream = pd.read_csv(ROOT.parent / STREAM_FILE)
stream = stream[stream.direction == "purchase"].set_index(["dup_filter", "clock"]).n_events
for clock in ["T", "P"]:
    v4(f"StreamPurchasesAll{clock}", f"purchase events on the full stream, clock {clock} (count; all periods)", integer(stream[("none", clock)]), STREAM_FILE)
    v4(f"StreamPurchasesDup{clock}", f"purchase events after the duplicate filter, clock {clock} (count)", integer(stream[("drop_dup_primary", clock)]), STREAM_FILE)
    v4(f"StreamPurchasesOfficial{clock}", f"purchase events on the official-only stream, clock {clock} (count)", integer(stream[("official_only", clock)]), STREAM_FILE)
    v4(f"OfficialOnlyRemovedShareAll{clock}", f"share of filtered-stream purchases removed by official-only, clock {clock}", pct(1 - stream[("official_only", clock)] / stream[("drop_dup_primary", clock)], 1) + r"\%", STREAM_FILE)

# Size of each inference procedure at zero effect on the N2 block-shift null panels. Statistics of
# placebo panels only. The outcome variant matches the test's outcome in the paper: T1 on raw Y,
# the tau-squared family on winsorized Y. Regenerated from the file at every build.
# Two size files: `size_null_v2.csv` (the procedures on the final event
# tables) is preferred cell by cell; `size_null.csv` supplies the cells v2 does not carry
# (the calendar-block bootstrap I2, reported in the appendix with its measured behaviour). Each
# macro's `source` names the file its value came from.
SIZE_FILES = [f for f in ["data/v4/calibration/size_null_v2.csv", "data/v4/calibration/size_null.csv"] if (ROOT.parent / f).exists()]
if not SIZE_FILES:
    raise FileNotFoundError("no size calibration file under data/v4/calibration/")
size_tables = {f: pd.read_csv(ROOT.parent / f) for f in SIZE_FILES}
SIZE_FILE = SIZE_FILES[0]


def size_row(clock: str, variant: str, test: str, inference: str, weights: str = "none", sided: str = "one-sided") -> pd.Series:
    """The one row for a (clock, variant, test, inference, weights, sidedness) cell; v2 first. The
    returned row carries the file it came from in `source_file`."""
    for f, size in size_tables.items():
        rows = size[(size.clock == clock) & (size.variant == variant) & (size.test == test) & (size.inference == inference) & (size.weights == weights) & (size.sidedness == sided)]
        if len(rows) > 1:
            raise ValueError(f"{f}: {len(rows)} rows for {(clock, variant, test, inference, weights, sided)}")
        if len(rows) == 1:
            row = rows.iloc[0].copy()
            row["source_file"] = f
            return row
    raise ValueError(f"no size row for {(clock, variant, test, inference, weights, sided)} in {SIZE_FILES}")


def rate(x: float) -> str:
    return pct(x, 1) + r"\%"


v4("SizeNPanels", "null panels per clock and outcome variant in the size calibration (count)", integer(min(t.n_panels.min() for t in size_tables.values())), SIZE_FILE)
for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    day = size_row(clock, "raw", "T1", "I1")
    blk = size_row(clock, "raw", "T1", "I1blk20")
    v4(f"SizeTOneDay{tag}", f"T1 with household x anchor-day clusters: one-sided rejection at nominal 5% on the N2 null, clock {clock}", rate(day.reject_rate), day.source_file)
    v4(f"SizeTOneDayCov{tag}", f"T1 with household x anchor-day clusters: 95% interval coverage of zero on the N2 null, clock {clock}", rate(day.coverage), day.source_file)
    v4(f"SizeTOneBlock{tag}", f"T1 with household x 20-session-block clusters: one-sided rejection at nominal 5%, clock {clock}", rate(blk.reject_rate), blk.source_file)
    v4(f"SizeTOneBlockCov{tag}", f"T1 with household x 20-session-block clusters: 95% interval coverage of zero, clock {clock}", rate(blk.coverage), blk.source_file)
    for w in ["EH", "ES", "PW"]:
        i1 = size_row(clock, "wins", "T2", "I1", w)
        v4(f"SizeTTwoIOne{w}{tag}", f"tau-squared test with the nominal I1 t-test, weights {w}, winsorized Y: one-sided rejection at nominal 5%, clock {clock}", rate(i1.reject_rate), i1.source_file)
        v4(f"SizeTTwoIOneCov{w}{tag}", f"I1 t-interval for tau-squared, weights {w}, winsorized Y: two-sided coverage of zero on the N2 null, clock {clock}", rate(i1.coverage), i1.source_file)
        v4(f"SizeTTwoICal{w}{tag}", f"I1cal (placebo-calibrated critical value), weights {w}, winsorized Y: one-sided rejection, clock {clock}", rate(size_row(clock, "wins", "T2", "I1cal", w).reject_rate), size_row(clock, "wins", "T2", "I1cal", w).source_file)
        i2b = size_row(clock, "wins", "T2", "I2b", w)
        v4(f"SizeTTwoITwob{w}{tag}", f"I2b household-cluster bootstrap, weights {w}, winsorized Y: one-sided rejection, clock {clock}", rate(i2b.reject_rate), i2b.source_file)
        v4(f"SizeTTwoITwobCov{w}{tag}", f"I2b household-cluster bootstrap, weights {w}, winsorized Y: coverage of zero, clock {clock}", rate(i2b.coverage), i2b.source_file)
        i2 = size_row(clock, "wins", "T2", "I2L20", w)
        v4(f"SizeTTwoITwo{w}{tag}", f"I2 calendar-block bootstrap (L = 20), weights {w}, winsorized Y: one-sided rejection, clock {clock}", rate(i2.reject_rate), i2.source_file)
        v4(f"SizeTTwoITwoCov{w}{tag}", f"I2 calendar-block bootstrap (L = 20), weights {w}, winsorized Y: coverage of zero, clock {clock}", rate(i2.coverage), i2.source_file)
        v4(f"SizeTTwoITwoDisp{w}{tag}", f"I2 bootstrap SD of tau-hat squared over its Monte Carlo SD, weights {w}, winsorized Y, clock {clock}", num(i2.boot_sd_over_mc_sd, 2), i2.source_file)
    v4(f"SizeTThree{tag}", f"T3 max-t against the N2 critical value: rejection at nominal 5%, winsorized Y, clock {clock}", rate(size_row(clock, "wins", "T3", "N2_critical_value").reject_rate), size_row(clock, "wins", "T3", "N2_critical_value").source_file)
    v4(f"SizeTFourTwo{tag}", f"T4 z > 2 against the N2 share distribution: rejection at nominal 5%, raw Y, clock {clock}", rate(size_row(clock, "raw", "T4_z2", "N2_placebo_p").reject_rate), size_row(clock, "raw", "T4_z2", "N2_placebo_p").source_file)
    v4(f"SizeTFourThree{tag}", f"T4 z > 3 against the N2 share distribution: rejection at nominal 5%, raw Y, clock {clock}", rate(size_row(clock, "raw", "T4_z3", "N2_placebo_p").reject_rate), size_row(clock, "raw", "T4_z3", "N2_placebo_p").source_file)
def size_cells(inference: str, column: str) -> list[pd.Series]:
    return [size_row(c, "wins", "T2", inference, w) for c in ["T", "P"] for w in ["EH", "ES", "PW"]]


def size_extreme(inference: str, column: str, fn) -> tuple[str, str]:
    cells = size_cells(inference, column)
    best = fn(cells, key=lambda r: r[column])
    return rate(best[column]), best.source_file


v4("SizeTTwoIOneMax", "largest one-sided rejection of the nominal I1 t-test across weights and clocks, winsorized Y", *size_extreme("I1", "reject_rate", max))
v4("SizeTTwoIOneMin", "smallest one-sided rejection of the day-clustered t-test of tau-squared across weights and clocks, winsorized Y", *size_extreme("I1", "reject_rate", min))
v4("SizeTTwoICalMin", "smallest I1cal one-sided rejection across weights and clocks, winsorized Y", *size_extreme("I1cal", "reject_rate", min))
v4("SizeTTwoICalMax", "largest I1cal one-sided rejection across weights and clocks, winsorized Y", *size_extreme("I1cal", "reject_rate", max))
v4("SizeTTwoITwoMax", "largest I2 calendar-block bootstrap rejection across weights and clocks, winsorized Y", *size_extreme("I2L20", "reject_rate", max))
v4("SizeTTwoITwoCovMin", "smallest I2 calendar-block bootstrap coverage of zero across weights and clocks, winsorized Y", *size_extreme("I2L20", "coverage", min))

# DRAFT exhibits from the null-side work (no actual household-level value in either):
# Figure 2 = the calibrated detection region (size-correct tests only; the newest
# figure2*calibrated*.pdf or figure2*v2*.pdf wins), Figure 4 = the time-to-detect
# chart. Copied into figures/ as v4_draft_*; the include chain is final > draft > shell.
DRAFT_FIGURES = {
    "v4_draft_fig2_detection_region.pdf": ("data/v4/calibration", ["figure2_detection_region_v3.pdf", "figure2_detection_region_v2.pdf", "figure2_detection_region_calibrated.pdf"]),
    "v4_draft_fig2_detection_region_public.pdf": ("data/v4/calibration", ["figure2_detection_region_v3_public.pdf"]),
    # The appendix version in which each test keeps its own analysis set (the "testspecific"
    # files). These are final includes, not drafts, so they carry the names the appendix asks for;
    # copying them here keeps them in step with the main-text figure when C's script is re-run.
    "v4_fig2_detection_region_testspecific.pdf": ("data/v4/calibration", ["figure2_detection_region_v3_testspecific.pdf"]),
    "v4_fig2_detection_region_testspecific_public.pdf": ("data/v4/calibration", ["figure2_detection_region_v3_testspecific_public.pdf"]),
    "v4_draft_fig4_time_to_detect.pdf": ("data/v4/nullside", ["figure4_T_raw.pdf"]),
}
for dest, (folder, patterns) in DRAFT_FIGURES.items():
    candidates = []
    for pattern in patterns:
        candidates = sorted((ROOT.parent / folder).glob(pattern))
        if candidates:
            break
    if not candidates:
        continue
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copyfile(newest, FIG / dest)
    print(f"draft figure {dest} <- {newest.relative_to(ROOT.parent)}")

# Table bodies. Cells reference the \Res... macros (written by
# trading_on_disclosure_v4.ledger.macros), so filling the macros fills every table; a missing value prints [PENDING].
WEIGHT_LABELS = {"ES": "Event-share", "EH": "Equal-household", "PW": "Pair"}
RES_WEIGHT = {"ES": "", "EH": "EqualHousehold", "PW": "Pair"}
RES_CLOCK = {"Txn": "Transaction", "Pub": "Public"}
# the main text reports the household estimates on the transaction clock only,
# and the public-clock counterparts move to an appendix table with the same columns. The clock column
# is therefore gone from both: each table states its clock once, in the caption.
def tau_panel_a(clock: str, headline: bool) -> str:
    rows = []
    for weight in ["ES", "EH", "PW"]:
        w = RES_WEIGHT[weight]
        label = WEIGHT_LABELS[weight] + (" (headline)" if weight == "ES" and headline else "")
        # the headline upper bound is the inverted placebo-injection
        # bound (\Res...UpperInjection); the nominal 95% household-bootstrap bound is printed beside it.
        rows.append(tex_row(
            label, rf"\ResTauSq{w}{clock}", rf"\ResTau{w}{clock}", rf"\ResTau{w}{clock}PlaceboP", rf"\ResTau{w}{clock}PlaceboPTiming",
            rf"[\ResTau{w}{clock}BootLo, \ResTau{w}{clock}BootHi]", rf"\ResTau{w}{clock}UpperInjection", rf"\ResTau{w}{clock}Upper",
        ))
    # the estimand without the activity control is the weighted covariance of the
    # unresidualized period means, a different number from the headline; the robustness row that changed
    # only the control is identical to the headline by construction and says nothing.
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{l}{\textit{Without the activity control: the covariance of the unresidualized period means}}\\")
    for weight in ["ES", "EH", "PW"]:
        w = RES_WEIGHT[weight]
        rows.append(tex_row(
            WEIGHT_LABELS[weight], rf"\ResTauNoControlSq{w}{clock}", rf"\ResTauNoControl{w}{clock}",
            rf"\ResTauNoControl{w}{clock}PlaceboP", rf"\ResTauNoControl{w}{clock}PlaceboPTiming",
            rf"[\ResTauNoControl{w}{clock}BootLo, \ResTauNoControl{w}{clock}BootHi]", "--", rf"\ResTauNoControl{w}{clock}Upper",
        ))
    # The two outcome variants that change the sign on the
    # transaction clock, from the robustness rows (event-share weights), with the same columns.
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{l}{\textit{Event-share weights, other outcomes}}\\")
    for key, label in [("NoAlpha", "Factor model without intercept"), ("MarketAdjusted", "Market-adjusted outcome")]:
        rows.append(tex_row(
            label, rf"\ResRob{key}{clock}TauSq", rf"\ResRob{key}{clock}Tau", rf"\ResRob{key}{clock}PlaceboP",
            rf"\ResRob{key}{clock}PlaceboPTiming", rf"[\ResRob{key}{clock}BootLo, \ResRob{key}{clock}BootHi]",
            rf"\ResRob{key}{clock}UpperInjection", rf"\ResRob{key}{clock}Upper",
        ))
    return "%\n" + "\n".join(rows) + "\n"


PANEL_A_DESC = ("body (8 columns: weights or outcome, tau2, tau, centered placebo p, timing-only placebo p, "
                "household-bootstrap 95% interval, inverted placebo-injection upper bound bp, nominal 95% "
                "household-bootstrap upper bound bp); three weightings, then the same three without the activity "
                "control, then the no-intercept and market-adjusted outcomes")
v4("TableTauRowsA", "Table 4 Panel A " + PANEL_A_DESC + ", transaction clock", tau_panel_a("Transaction", True))
v4("TableTauRowsAPublic", "appendix public-clock table, Panel A " + PANEL_A_DESC + ", public clock", tau_panel_a("Public", False))
tau_rows_b = [tex_row(RES_CLOCK[c], rf"\ResTau{RES_CLOCK[c]}Households", rf"\ResTau{RES_CLOCK[c]}EventsA", rf"\ResTau{RES_CLOCK[c]}EventsB") for c in ["Txn", "Pub"]]
tau_rows_b += [tex_row(r"\shortstack[l]{" + display_text(RES_CLOCK[c]) + r"\\official sources only}", rf"\ResOfficialHouseholdsPersistence{RES_CLOCK[c]}", rf"\ResOfficialEventsPersistenceA{RES_CLOCK[c]}", rf"\ResOfficialEventsPersistenceB{RES_CLOCK[c]}") for c in ["Txn", "Pub"]]
v4("TableTauRowsB", "persistence-sample body, appendix eligibility table (4 columns: clock, households, purchases 2015-2020, purchases 2021-2025; the persistence sample on the de-duplicated stream and on official sources only)",
   "%\n" + "\n".join(tau_rows_b) + "\n")
# Panel B shares Panel A's eight-column tabular (one tabular, so the table note spans the full width).
def tau_panel_b(clock: str) -> str:
    rows = [tex_row(label, rf"\multicolumn{{4}}{{l}}{{range of $\widehat\tau$ (bp): [\ResTau{stat}{clock}Min, \ResTau{stat}{clock}Max]}}", rf"\multicolumn{{3}}{{r}}{{re-estimates: \ResTau{stat}{clock}Refits}}")
            for stat, label in [("LeaveHouseholdOut", "Leave one household out"), ("LeaveYearOut", "Leave one calendar year out")]]
    return "%\n" + "\n".join(rows) + "\n"


PANEL_B_DESC = ("body (eight-column rows: drop label, the range of tau-hat across the re-estimates spanning four "
                "columns, the number of re-estimates spanning three)")
v4("TableTauRowsC", "Table 4 Panel B " + PANEL_B_DESC + ", transaction clock", tau_panel_b("Transaction"))
v4("TableTauRowsCPublic", "appendix public-clock table, Panel B " + PANEL_B_DESC + ", public clock", tau_panel_b("Public"))

ROBUST_VARIANTS = [
    # this row changes only the activity control, which drops out of the
    # covariance form, so it reproduces the headline digit for digit; the label says so, and the
    # estimand actually computed without the control is the separate block of Table 4.
    ("NoActivityControl", "No activity control (identical to the headline by construction)"),
    ("LooseRule", "Looser household rule (5 purchases in each period)"),
    ("StrictRule", "Stricter household rule (20 and 10 purchases, 5 securities)"),
    ("OddEven", "Odd versus even calendar years"),
    ("OfficialOnly", "Official sources only"),
    ("KeepDuplicates", "Duplicate vendor lines kept"),
    ("MarketAdjusted", "Market-adjusted outcome"),
    ("NoAlpha", "Factor model without the estimation-window intercept"),
    ("HorizonTen", "10-day horizon"),
    ("HorizonSixty", "60-day horizon"),
    ("RawOutcome", "Raw (unwinsorized) outcome"),
    ("OnePerSecurityMonth", "One purchase per household, security and month"),
    ("WithinSecondPeriod", "Second period split: 2021--2022 versus 2023--2025"),
    ("Sales", "Sales instead of purchases"),
    ("SignedAll", "Signed purchases and sales"),
]
def robust_row(key: str, label: str) -> str:
    no_placebo = key in ("Sales", "SignedAll")  # purchase-only placebo sets: no placebo p, no Holm value over them
    return tex_row(label, rf"\ResRob{key}TransactionTauSq", rf"\ResRob{key}TransactionTau",
                   "--" if no_placebo else rf"\ResRob{key}TransactionPlaceboP",
                   "--" if no_placebo else rf"\ResRob{key}TransactionPlaceboPTiming",
                   "--" if no_placebo else rf"\ResRob{key}TransactionPlaceboPHolm",
                   rf"\ResRob{key}TransactionPBlock", rf"\ResRob{key}TransactionPBlockHolm",
                   rf"[\ResRob{key}TransactionBootLo, \ResRob{key}TransactionBootHi]", rf"\ResRob{key}TransactionUpper", rf"\ResRob{key}TransactionHouseholds")


v4("RobustBlockFamily", "robustness rows on one clock over which the block-clustered p is Holm-adjusted (count; plan Section 7 variants)", str(len(ROBUST_VARIANTS)), "plan")
v4("RobustPlaceboFamily", "robustness rows on one clock with a centered placebo p, over which that p is Holm-adjusted (count; the sales and signed rows have none)", str(len([k for k, _ in ROBUST_VARIANTS if k not in ("Sales", "SignedAll")])), "plan")
v4("IATauRobustRows", "appendix robustness table body (11 columns: row, tau2, tau, centered placebo p, timing-only placebo p, Holm-adjusted centered placebo p, block-clustered p, Holm-adjusted block p, household-bootstrap interval, upper bound, households); event-share weights, transaction clock; one row per plan Section 7 item",
   "%\n" + "\n".join(robust_row(key, label) for key, label in ROBUST_VARIANTS) + "\n")

t1_rows, actual_tests = [], []
for c in ["Transaction", "Public"]:
    t1_rows.append(tex_row(c, rf"\ResMean{c}", rf"\ResMean{c}SE", rf"\ResMean{c}PBlock", rf"\ResMean{c}PlaceboMean", rf"\ResMean{c}PlaceboP", rf"\ResMean{c}Events"))
    actual_tests += [
        ("Best household: largest mean (bp)", c, rf"\ResBestMean{c}", rf"\ResBestMean{c}PlaceboNinetyFive", rf"\ResBestMean{c}PlaceboP", rf"\ResBestHouseholds{c}"),
        ("Best household: largest $t$", c, rf"\ResBestT{c}", rf"\ResBestT{c}PlaceboNinetyFive", rf"\ResBestT{c}PlaceboP", rf"\ResBestHouseholds{c}"),
        ("Tail: share above $z=2$", c, rf"\ResTailShareTwo{c}", rf"\ResTailShareTwo{c}PlaceboMean", rf"\ResTailShareTwo{c}PlaceboP", rf"\ResTailEvents{c}"),
    ]
# the share below minus three beside the share above three (the
# \ResTailShareBelowThree... names), by period, and both on the samples without anchors dated
# February-May 2020 (ExclCovid...).
TAIL_SAMPLES = [("", "All purchases, 2015--2025"), ("PeriodA", "2015--2020"), ("PeriodB", "2021--2025"),
                ("ExclCovid", "Without February--May 2020 anchors"), ("ExclCovidPeriodA", "2015--2020, without February--May 2020"),
                ("ExclCovidPeriodB", "2021--2025, without February--May 2020"),
                ("OfficialOnly", "Official sources only"), ("ExclTwo", "Without the two vendor-only households")]
tail_rows = []
for suffix, label in TAIL_SAMPLES:
    for c in ["Transaction", "Public"]:
        cells = [label if c == "Transaction" else "", c]
        for tail in ["Above", "Below"]:
            cells += [rf"\ResTailShare{tail}Three{c}{suffix}", rf"\ResTailShare{tail}Three{c}{suffix}PlaceboMean",
                      rf"\ResTailShare{tail}Three{c}{suffix}PlaceboP", rf"\ResTailShare{tail}Three{c}{suffix}TimingPlaceboP"]
        cells.append(rf"\ResTailEvents{c}{suffix}")
        tail_rows.append(tex_row(*cells))
v4("IATailRows", "supporting tests table, Panel C body (11 columns): shares of standardized purchase outcomes above three and below minus three by sample and period and clock: for each tail the share, the centered placebo mean, the centered placebo p and the timing-only placebo p; then N",
   "%\n" + "\n".join(tail_rows) + "\n")
v4("IAActualTestsRowsTOne", "supporting tests table, Panel A body (7 columns): purchase mean per clock: mean, block-clustered SE, p against zero, placebo mean, placebo p, N",
   "%\n" + "\n".join(t1_rows) + "\n")
v4("IAActualTestsRows", "supporting tests table, Panel B body (6 columns): best-household and tail tests per clock: actual, placebo reference (95th percentile or placebo mean), placebo p, N",
   "%\n" + "\n".join(tex_row(*cells) for cells in actual_tests) + "\n")
SAMPLE_SUFFIX = [("", "Primary", "De-duplicated (headline)"), ("OfficialOnly", "OfficialOnlyPrimary", "Official sources only"), ("ExclTwo", "ExclTwoPrimary", "Without the two households")]
two_sample_body = [panel_row(5, "Panel A: Headline 20-day purchase-minus-sale spread, public clock (bp)")]
for label, fmt in [("Spread", r"\ExDedup{k}Bps"), ("Block-clustered SE", r"\ExDedup{k}SeBlock"),
                   ("95\\% interval", r"[\ExDedup{k}CiLowBlock, \ExDedup{k}CiHighBlock]"), ("Signals", r"\ExDedup{k}N"), ("Households", r"\ExDedup{k}Households")]:
    two_sample_body.append(tex_row(label, "Public", *[fmt.format(k=k) for _, k, _ in SAMPLE_SUFFIX]))
two_sample_body.append(panel_row(5, "Panel B: Purchase mean, raw outcome (bp)"))
for c in ["Transaction", "Public"]:
    two_sample_body.append(tex_row("Mean (block-clustered SE)", c, *[rf"\ResMean{c}{sfx}\ (\ResMean{c}{sfx}SE)" for sfx, _, _ in SAMPLE_SUFFIX]))
    two_sample_body.append(tex_row("Timing-only placebo $p$", c, *[rf"\ResMean{c}{sfx}PlaceboP" for sfx, _, _ in SAMPLE_SUFFIX]))
    two_sample_body.append(tex_row("Purchases", c, *[rf"\ResMean{c}{sfx}Events" for sfx, _, _ in SAMPLE_SUFFIX]))
two_sample_body.append(panel_row(5, "Panel C: Tail test, share of standardized outcomes above three"))
for c in ["Transaction", "Public"]:
    two_sample_body.append(tex_row("Share (centered placebo $p$; timing-only $p$)", c, *[rf"\ResTailShareThree{c}{sfx}\ (\ResTailShareThree{c}{sfx}PlaceboP; \ResTailShareThree{c}{sfx}TimingPlaceboP)" for sfx, _, _ in SAMPLE_SUFFIX]))
two_sample_body.append(panel_row(5, "Panel D: Best household, households with at least 20 purchases"))
for c in ["Transaction", "Public"]:
    two_sample_body.append(tex_row("Households", c, *[rf"\ResBestHouseholds{c}{sfx}" for sfx, _, _ in SAMPLE_SUFFIX]))
    two_sample_body.append(tex_row("Largest mean, bp (centered placebo $p$)", c, *[rf"\ResBestMean{c}{sfx}\ (\ResBestMean{c}{sfx}PlaceboP)" for sfx, _, _ in SAMPLE_SUFFIX]))
    two_sample_body.append(tex_row("Largest $t$ (centered placebo $p$)", c, *[rf"\ResBestT{c}{sfx}\ (\ResBestT{c}{sfx}PlaceboP)" for sfx, _, _ in SAMPLE_SUFFIX]))
    two_sample_body.append(tex_row("Households with nominal $p<0.05$ (centered $p$; timing-only $p$)", c, *[rf"\ResCountSignificant{c}{sfx}\ (\ResCountSignificant{c}{sfx}PlaceboP; \ResCountSignificant{c}{sfx}TimingPlaceboP)" for sfx, _, _ in SAMPLE_SUFFIX]))
v4("IATwoSamplesRows", "appendix table body (5 columns: statistic, clock, de-duplicated sample, official sources only, without the two vendor-only households): headline spread, purchase mean, tail share above three, best-household statistics",
   "%\n" + "\n".join(two_sample_body) + "\n")

companion_rows = [
    ("Panel A: Best household, households with at least 20 purchases", None),
    ("Households", "BestHouseholds"), ("Largest household mean (bp), actual", "BestMean"),
    ("Largest household mean (bp), placebo 95th percentile", "BestMean{c}PlaceboNinetyFive"),
    ("Top-decile mean (bp)", "TopDecileMean"),
    ("Households with nominal $p<0.05$, actual", "CountSignificant"),
    ("Households with nominal $p<0.05$, placebo mean", "CountSignificant{c}PlaceboMean"),
    ("Panel B: Rank sets", None),
    ("Households", "RankHouseholds"),
    ("Median rank-set width (share of households)", "RankWidth"), ("Pairwise orderings resolved", "RankPairsResolved"),
    ("Households distinguishable from the median", "RankDistinctFromMedian"),
    ("Panel C: Shrinkage", None),
    ("Households", "ShrinkHouseholds"), (r"$\widehat\tau_{\mathrm{EB}}$ (bp)", "ShrinkTau"), ("Share of bootstrap draws at the zero boundary", "ShrinkBoundaryShare"),
    ("Forecast slope (SE)", ("ShrinkForecastSlope", "ShrinkForecastSlopeSE")), ("Reliability slope (SE)", ("ShrinkReliabilitySlope", "ShrinkReliabilitySlopeSE")),
    ("Panel D: Years to detect", None),
    (r"Outcome SD $\sigma$ (bp per 20 trading days)", "\\HTTDSigma{c}"), ("Design effect $d$", "\\HTTDDesignEffect{c}"),
    (r"Years at $n=\HTTDReferenceN$, $\theta=\HTTDReferenceTheta$", "DetectYearsFiveAtFifty"), ("Same, best-of-$H$ adjustment", "DetectYearsFiveAtFiftyBestOfH"),
]
companion_body = []
for label, key in companion_rows:
    if key is None:
        companion_body.append(panel_row(3, label))
    elif isinstance(key, tuple):
        companion_body.append(tex_row(label, *[rf"\Res{key[0]}{c}\ (\Res{key[1]}{c})" for c in ["Transaction", "Public"]]))
    elif key.startswith("\\H"):
        companion_body.append(tex_row(label, *[key.format(c=tag) for tag in ["Txn", "Pub"]]))
    elif "{c}" in key:
        companion_body.append(tex_row(label, *["\\Res" + key.format(c=c) for c in ["Transaction", "Public"]]))
    else:
        companion_body.append(tex_row(label, *[rf"\Res{key}{c}" for c in ["Transaction", "Public"]]))
v4("IACompanionRows", "supporting analyses by clock (appendix; 3 columns: statistic, transaction clock, public clock)",
   "%\n" + "\n".join(companion_body) + "\n")
v4("IAEligibilityRows", "eligibility counts table body (appendix; 4 columns: clock, households passing the persistence rule, households with at least 20 purchases, purchases with a finite outcome)",
   "%\n" + "\n".join(tex_row(c, rf"\ResHouseholdsPersistence{c}", rf"\ResHouseholdsFullTwenty{c}", rf"\ResEvents{c}") for c in ["Transaction", "Public"]) + "\n")

# the covariance matrix of household means across three
# consecutive subperiods of equal calendar length, with the centered placebo p of each entry.
SUBPERIOD_LABELS = [("One", "First: January 2015 to August 2018"), ("Two", "Second: September 2018 to April 2022"), ("Three", "Third: May 2022 to December 2025")]
v4("IASubperiodCovRows", "three-subperiod covariance table, Panel A body (appendix; 7 columns: the pair of subperiods, then for each clock the weighted covariance of the two subperiod means, their weighted correlation, and the centered placebo p of the covariance)",
   "%\n" + "\n".join(
       tex_row(label, *[cell for c in ["Transaction", "Public"] for cell in
                        (rf"\ResSubperiodCov{pair}{c}", rf"\ResSubperiodCorr{pair}{c}", rf"\ResSubperiodCov{pair}{c}PlaceboP")])
       for pair, label in [("OneTwo", "First and second"), ("OneThree", "First and third"), ("TwoThree", "Second and third")]) + "\n")
subperiod_var = [tex_row(label, *[cell for c in ["Transaction", "Public"] for cell in
                                  (rf"\ResSubperiodVar{tag}{c}", rf"\ResSubperiodEvents{tag}{c}", rf"\ResSubperiodVar{tag}{c}PlaceboP")])
                 for tag, label in SUBPERIOD_LABELS]
subperiod_var.append(r"\midrule")
subperiod_var.append(tex_row("Households in all three (pool); purchases dropped",
                             *[cell for c in ["Transaction", "Public"] for cell in
                               (rf"\ResSubperiodHouseholds{c}", rf"(\ResSubperiodHouseholdsPool{c})", rf"\ResSubperiodEventsDropped{c}")]))
v4("IASubperiodVarRows", "three-subperiod covariance table, Panel B body (appendix; 7 columns: the subperiod, then for each clock the weighted variance of its household means, the purchases it holds, and the centered placebo p of the variance), closing with the households, the pool and the purchases dropped by the completion rule",
   "%\n" + "\n".join(subperiod_var) + "\n")

# continuation into the second period regressed on the
# first-period mean given log activity, under two definitions of continuing.
CONTINUATION_RULES = [("", "At least one second-period purchase"), ("Rule", "The paper's second-period eligibility rule")]
continuation_body = []
for rule, rule_label in CONTINUATION_RULES:
    continuation_body.append(panel_row(5, f"Panel {'A' if rule == '' else 'B'}: {rule_label}"))
    for stem, label in [("Slope", "Linear probability: first-period mean (per 100 bp)"),
                        ("ActivitySlope", "Linear probability: log first-period purchases"),
                        ("LogitSlope", "Logit: first-period mean (per 100 bp)"),
                        ("LogitActivitySlope", "Logit: log first-period purchases")]:
        continuation_body.append(tex_row(label, *[cell for c in ["Transaction", "Public"]
                                                  for cell in (rf"\ResContinuation{rule}{stem}{c}", rf"\ResContinuation{rule}{stem}SE{c}")]))
    continuation_body.append(tex_row("Households (share continuing)",
                                     *[rf"\multicolumn{{2}}{{c}}{{\ResContinuation{rule}Households{c}\ (\ResContinuation{rule}Share{c})}}" for c in ["Transaction", "Public"]]))
v4("IAContinuationRegressionRows", "continuation-regression table body (appendix; 5 columns: row label, then the coefficient and its standard error on each clock); one panel per definition of continuing, four coefficients and the sample line in each",
   "%\n" + "\n".join(continuation_body) + "\n")

# The one day-clustered appendix table: every mean or spread with day
# clusters beside block clusters, each with its measured placebo rejection rate; tau-squared
# with the day-clustered t-test beside the placebo test and the household bootstrap. Rates are
# sourced from the size calibration file (household statistics) and from the placebo set
# (the spread); standard errors on actual outcomes come from the two result files.
day_rows = [panel_row(7, "Panel A: Means and spreads")]
for clock in ["Txn", "Pub"]:
    c = RES_CLOCK[clock]
    day_rows.append(tex_row("Purchase mean", c, rf"\ResMean{c}SEDay", rf"\HSizeTOneDay{clock}", rf"\ResMean{c}SE", rf"\HSizeTOneBlock{clock}", rf"\HSizeTOneBlockCov{clock}"))
day_rows.append(tex_row("Purchase-minus-sale spread", "Public", r"\ExDedupPrimarySeDay", r"\ExPlaceboSpreadRejectOfMeanDayTiming", r"\ExDedupPrimarySeBlock", r"\ExPlaceboSpreadRejectOfMeanBlockTiming", r"\ExPlaceboSpreadCoverageBlockTiming"))
day_rows.append(panel_row(7, r"Panel B: Persistent dispersion $\widehat\tau^2$ (day-clustered $t$-test against the placebo test and the household bootstrap)"))
for weight in ["ES", "EH", "PW"]:
    for clock in ["Txn", "Pub"]:
        c = RES_CLOCK[clock]
        day_rows.append(tex_row(WEIGHT_LABELS[weight] + " weights", c, rf"\ResTauSq{RES_WEIGHT[weight]}{c}SEDay", rf"\HSizeTTwoIOne{weight}{clock}", rf"\HSizeTTwoICal{weight}{clock}", rf"\HSizeTTwoITwob{weight}{clock}", rf"\HSizeTTwoITwobCov{weight}{clock}"))
v4("IADayClusterRows", "day-clustered appendix table body (7 columns; Panel A: statistic, clock, day-clustered SE, day placebo rejection, block-clustered SE, block placebo rejection, block coverage; Panel B: weights, clock, day-clustered SE of tau2, day t-test rejection, placebo-test rejection, bootstrap rejection, bootstrap coverage)",
   "%\n" + "\n".join(day_rows) + "\n", SIZE_FILE)

# ---------------------------------------------------------------------------
# Coverage of the one-sided upper bound on tau on placebo datasets with dense dispersion tau
# injected (read from
# data/v4/calibration/upper_bound_calibration.csv, injection before winsorization, 1,000 datasets per
# cell). Panel A: share of datasets on which the household-bootstrap percentile bound covers the
# injected tau, by nominal level; Panel B: the same against the dataset's own realized dispersion.
# The headline (inverted placebo-injection) bound covers the injected tau 95% at every grid point by
# construction (Section 13.4 of the report) and needs no table.
# ---------------------------------------------------------------------------
BOUND_CAL_FILE = "data/v4/calibration/upper_bound_calibration.csv"
bound_cal = pd.read_csv(ROOT.parent / BOUND_CAL_FILE)
bound_cal = bound_cal[bound_cal.injection == "before_winsor"]
BOUND_LEVELS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
BOUND_TAUS = [0, 50, 75, 100]
coverage_rows = []
for target_col, panel in [("coverage", "Panel A: Coverage of the injected dispersion $\\sigma_\\theta$"),
                          ("coverage_realized_target", "Panel B: Coverage of the dataset's realized dispersion")]:
    coverage_rows.append(panel_row(2 + len(BOUND_LEVELS), panel))
    for clock in ["T", "P"]:
        for tau in BOUND_TAUS:
            cells = [RES_CLOCK["Txn" if clock == "T" else "Pub"] if tau == BOUND_TAUS[0] else "", str(tau)]
            for level in BOUND_LEVELS:
                row = bound_cal[(bound_cal.clock == clock) & (bound_cal.tau_true == tau) & (np.isclose(bound_cal.level, level))]
                if len(row) != 1:
                    raise ValueError(f"upper_bound_calibration.csv: expected one row for {clock} tau={tau} level={level}, found {len(row)}")
                cells.append(pct(float(row[target_col].iloc[0]), 1))
            if any(float(left) > float(right) for left, right in zip(cells[2:], cells[3:])):
                raise ValueError(f"upper_bound_calibration.csv: non-monotone coverage for {target_col}, {clock}, {tau}")
            coverage_rows.append(tex_row(*cells))
v4("IABoundCoverageRows", "appendix upper-bound coverage table body (8 columns: clock, injected tau in bp, coverage in percent at nominal levels 0.90, 0.95, 0.975, 0.99, 0.995, 0.999); Panel A against the injected tau, Panel B against the dataset's realized dispersion",
   "%\n" + "\n".join(coverage_rows) + "\n", BOUND_CAL_FILE)
v4("BoundCalPanels", "placebo datasets per clock and injected tau in the upper-bound coverage table (count)", integer(bound_cal.n_panels.iloc[0]), BOUND_CAL_FILE)
if bound_cal.n_panels.nunique() != 1 or bound_cal.n_boot.nunique() != 1:
    raise ValueError("upper_bound_calibration.csv: panel or bootstrap-draw counts differ across cells")
v4("BoundCalBootDraws", "household-bootstrap draws per placebo dataset in the upper-bound coverage table (count)", integer(bound_cal.n_boot.iloc[0]), BOUND_CAL_FILE)
nominal95 = bound_cal[np.isclose(bound_cal.level, 0.95)]
v4("BoundCalCoverageNominalMinTxn", "smallest coverage of the injected tau at the nominal 95% level across injected taus, transaction clock (percent)", pct(nominal95[nominal95.clock == "T"].coverage.min(), 1) + r"\%", BOUND_CAL_FILE)
v4("BoundCalCoverageNominalMinPub", "smallest coverage of the injected tau at the nominal 95% level across injected taus, public clock (percent)", pct(nominal95[nominal95.clock == "P"].coverage.min(), 1) + r"\%", BOUND_CAL_FILE)
top = bound_cal[np.isclose(bound_cal.level, bound_cal.level.max())]
v4("BoundCalCoverageMaxLevelMinTxn", "smallest coverage of the injected tau at the largest level in the grid (0.999) across injected taus, transaction clock (percent)", pct(top[top.clock == "T"].coverage.min(), 1) + r"\%", BOUND_CAL_FILE)
v4("BoundCalCoverageMaxLevelMinPub", "smallest coverage of the injected tau at the largest level in the grid (0.999) across injected taus, public clock (percent)", pct(top[top.clock == "P"].coverage.min(), 1) + r"\%", BOUND_CAL_FILE)
# The attenuation behind Panel A: mean tau2-hat relative to the injected tau squared at tau = 100.
for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    row = bound_cal[(bound_cal.clock == clock) & (bound_cal.tau_true == 100) & np.isclose(bound_cal.level, 0.95)].iloc[0]
    v4(f"BoundCalAttenuation{tag}", f"mean tau2-hat divided by the injected tau squared at tau = 100 bp, clock {clock}", num(float(row.mean_tau2_hat) / 100.0**2, 2), BOUND_CAL_FILE)

# the inversion is built on placebo
# datasets 1-1000 and its coverage measured on the held-out 1001-2000, under four shapes of the
# household effect, each against a curve built under the same shape and against the normal curve.
BOUND_VAL_FILE = "data/v4/calibration/bound_validation.csv"
bound_val = pd.read_csv(ROOT.parent / BOUND_VAL_FILE)
BOUND_VAL_SHAPES = [("gaussian", "Normal", "Normal"), ("two_point_half", "TwoPointHalf", "Two-point, half and half"),
                    ("two_point_skew", "TwoPointSkew", "Two-point, a fifth high"), ("student_t3", "StudentT", "Student $t$ with 3 degrees of freedom")]
BOUND_VAL_TAUS = [50.0, 75.0, 100.0, 150.0, 200.0]
BOUND_VAL_CURVES = [("out_of_sample_matched_curve", "the same shape"), ("out_of_sample_normal_curve", "normal")]
bound_val_rows = []
for clock, clock_label in [("T", "Panel A: Transaction clock"), ("P", "Panel B: Public clock")]:
    bound_val_rows.append(panel_row(9, clock_label))
    for shape, _shape_tag, shape_label in BOUND_VAL_SHAPES:
        at_observed = bound_val[(bound_val.clock == clock) & (bound_val.effect_shape == shape) & (bound_val.level_or_method == "bound_at_observed_tau2")]
        if len(at_observed) != 1:
            raise ValueError(f"{BOUND_VAL_FILE}: expected one bound row for {clock} {shape}, found {len(at_observed)}")
        bound_bp = num(at_observed.median_bound.iloc[0], 1)
        for method, curve_label in BOUND_VAL_CURVES:
            cells = bound_val[(bound_val.clock == clock) & (bound_val.effect_shape == shape) & (bound_val.level_or_method == method)]
            if cells.empty:  # the normal truth has no separate normal-curve row: it is the matched one
                continue
            window = cells[(cells.tau_true >= 25) & (cells.tau_true <= 200)]
            bound_cell = bound_bp if method == BOUND_VAL_CURVES[0][0] else "--"
            bound_val_rows.append(tex_row(
                shape_label if method == BOUND_VAL_CURVES[0][0] else "", curve_label,
                *[pct(cells[np.isclose(cells.tau_true, tau)].coverage.iloc[0], 1) for tau in BOUND_VAL_TAUS],
                f"{pct(window.coverage.min(), 1)}--{pct(window.coverage.max(), 1)}", bound_cell))
v4("IABoundValidationRows", "out-of-sample bound-validation table body (appendix; 9 columns: the shape of the household effect, the shape the inversion curve was built under, the coverage of the injected tau at five injected values, the range of coverage over the injected values from 25 to 200 bp, and the bound at the observed estimate under that shape); one panel per clock",
   "%\n" + "\n".join(bound_val_rows) + "\n")
matched = bound_val[(bound_val.level_or_method == "out_of_sample_matched_curve") & (bound_val.tau_true >= 25) & (bound_val.tau_true <= 200)]
v4("BoundValCoverageMatchedMin", "smallest out-of-sample coverage with the effect shape known, over both clocks, all four shapes and injected taus 25-200 bp (percent)", pct(matched.coverage.min(), 1) + r"\%", BOUND_VAL_FILE)
v4("BoundValCoverageMatchedMax", "largest such coverage (percent)", pct(matched.coverage.max(), 1) + r"\%", BOUND_VAL_FILE)
normal_curve = bound_val[(bound_val.level_or_method == "out_of_sample_normal_curve") & (bound_val.tau_true >= 25) & (bound_val.tau_true <= 200)]
v4("BoundValCoverageNormalCurveMin", "smallest out-of-sample coverage when the normal curve is used and the truth is not normal, over both clocks and injected taus 25-200 bp (percent)", pct(normal_curve.coverage.min(), 1) + r"\%", BOUND_VAL_FILE)
v4("BoundValPanels", "held-out placebo datasets per cell in the bound-validation table (count)", integer(matched.n_panels.iloc[0]), BOUND_VAL_FILE)
for clock, tag in [("T", "Txn"), ("P", "Pub")]:
    for shape, shape_tag, _ in BOUND_VAL_SHAPES:
        row = bound_val[(bound_val.clock == clock) & (bound_val.effect_shape == shape) & (bound_val.level_or_method == "bound_at_observed_tau2")].iloc[0]
        v4(f"BoundVal{shape_tag}{tag}", f"upper bound on tau at the observed estimate when the household effect has the {shape} shape, clock {clock} (bp)", num(row.median_bound, 1), BOUND_VAL_FILE)
        v4(f"BoundVal{shape_tag}Pct{tag}", f"the same bound on the annualized scale, clock {clock} (percent)", pct(row.bound_annualized_pct / 100.0, 1) + r"\%", BOUND_VAL_FILE)

# Pyun (2026) numbers quoted in the main text: page-cited values typed into
# paper/data/pyun_quoted.csv from the paper (Pyun 2026, SSRN 5295880), emitted as \HPyunQuoted<Name>.
PYUN_QUOTED_FILE = "paper/data/pyun_quoted.csv"
pyun_quoted = pd.read_csv(DATA / "pyun_quoted.csv", dtype=str)
if pyun_quoted.macro.duplicated().any():
    raise ValueError("pyun_quoted.csv: duplicate macro name")
for _, qrow in pyun_quoted.iterrows():
    if not re.fullmatch(r"PyunQuoted[A-Za-z]+", qrow.macro):
        raise ValueError(f"pyun_quoted.csv: bad macro name {qrow.macro}")
    v4(qrow.macro, f"Pyun (2026), {qrow.location}: {qrow.description}", qrow.value.strip(), PYUN_QUOTED_FILE)

# Blonien, Crane & Crotty (2025) numbers quoted in the Section 5 paragraph: the same
# pattern, page-cited values typed into paper/data/bcc_quoted.csv from the paper (SSRN 5524863),
# emitted as \HBccQuoted<Name>. Everything computed
# on our own record for that paragraph is in generated/bcc_macros.tex instead.
BCC_QUOTED_FILE = "paper/data/bcc_quoted.csv"
bcc_quoted = pd.read_csv(DATA / "bcc_quoted.csv", dtype=str)
if bcc_quoted.macro.duplicated().any():
    raise ValueError("bcc_quoted.csv: duplicate macro name")
for _, qrow in bcc_quoted.iterrows():
    if not re.fullmatch(r"BccQuoted[A-Za-z]+", qrow.macro):
        raise ValueError(f"bcc_quoted.csv: bad macro name {qrow.macro}")
    v4(qrow.macro, f"Blonien, Crane and Crotty (2025), {qrow.location}: {qrow.description}", qrow.value.strip(), BCC_QUOTED_FILE)

# The compatibility cells the Blonien, Crane and Crotty table of the Internet Appendix prints, scored
# against both placebo streams: the transaction clock at the compatibility horizon, for the
# purchase mean, the tail share above three and the largest household t, one cell per injected version.
# The Section 5 clause and the table note need their count and how many change their 5% verdict.
BCC_COMPAT_FILE = "data/v4/calibration/bcc_compatibility.csv"
BCC_TABLE_STATISTICS = ["mean_bp", "share_gt_3", "max_t"]
bcc_compat = pd.read_csv(ROOT.parent / BCC_COMPAT_FILE)
bcc_cells = bcc_compat[(bcc_compat.clock == "T") & (bcc_compat.horizon == 20)
                       & bcc_compat.statistic.isin(BCC_TABLE_STATISTICS)]
if bcc_cells.observed.isna().any():
    raise ValueError(f"{BCC_COMPAT_FILE}: a printed compatibility cell has no observed statistic to score")
bcc_changed = bcc_cells[bcc_cells.verdict_same_at_5pct.ne(True)]
if not (bcc_changed.compatible_at_5pct_timing.eq(True) & bcc_changed.compatible_at_5pct_centered.eq(False)).all():
    raise ValueError(f"{BCC_COMPAT_FILE}: a printed verdict changes towards incompatibility, which the note denies")
v4("BccTableCompatCells", "compatibility cells printed in the Blonien, Crane and Crotty appendix table (count)",
   integer(len(bcc_cells)), BCC_COMPAT_FILE)
v4("BccTableCompatCellsChanged", "printed compatibility cells whose 5% verdict differs between the centered and the timing-only placebo (count)",
   integer(len(bcc_changed)), BCC_COMPAT_FILE)

# The household scatter drawn on the public clock (Internet Appendix): the plotted
# table behind the drawn figure carries the off-frame flag, the coordinates and the weight, so the
# caption reads them from it rather than from the in-image footer the figure no longer carries.
SCATTER_PLOTTED_FILE = "data/v4/figures/figure3_scatter_P.csv"
scatter_plotted = pd.read_csv(ROOT.parent / SCATTER_PLOTTED_FILE)
off_frame = scatter_plotted[scatter_plotted.off_frame.astype(bool)]
if len(off_frame) != 1:
    raise ValueError(f"{SCATTER_PLOTTED_FILE}: the caption names one off-frame household, found {len(off_frame)}")
off_household = off_frame.iloc[0]
v4("ScatterOffFrameXPub", "off-frame household of the public-clock scatter: first-period mean, residual of log activity (bp)",
   signed_integer(off_household.first_period_residual_bp), SCATTER_PLOTTED_FILE)
v4("ScatterOffFrameYPub", "the same household's second-period mean (bp)",
   signed_integer(off_household.second_period_mean_bp), SCATTER_PLOTTED_FILE)
v4("ScatterOffFrameWPub", "the same household's weight, the area of its marker",
   integer(off_household.weight), SCATTER_PLOTTED_FILE)

# ---------------------------------------------------------------------------
# The existing paper's main tables, re-run (generated/existing_rerun_macros.tex, names in
# paper/generated/existing_macros.md). The main text
# prints the de-duplicated sample with block clusters (\ExDedup<Test><Q>Block); the appendix
# re-run table prints the released numbers (original sample, day clusters), the original
# sample with block clusters and the de-duplicated sample with block clusters side by side.
# ---------------------------------------------------------------------------
EX_ROWS = [
    ("Primary", "TW_P1_E20", "20 trading days (headline)"),
    ("HorizonFive", "TW_P2_E5", "5 trading days"),
    ("HorizonTen", "TW_P2_E10", "10 trading days"),
    ("HorizonForty", "TW_P2_E40", "40 trading days"),
    ("HorizonSixty", "TW_P2_E60", "60 trading days"),
    ("HorizonOneTwentySeven", "TW_P2_E127", "127 trading days"),
    ("RawVendorDates", "TW_P3_RAW_CT", "Raw Capitol Trades date for vendor-only records"),
    ("OfficialDates", "TW_P3_OFFICIAL_ONLY", "Observed-official-date sample"),
    ("OfficialSource", "TW_P3_OFFICIAL_SEGMENT", "Official-source segment"),
    ("VendorSource", "TW_P3_VENDOR_SEGMENT", "Vendor-proxy segment"),
    ("EntryPlusOne", "TW_P3_ENTRY_PLUS1", "Entry delayed by one additional trading day"),
    ("TransactionClock", "TW_C1_T20", "Transaction clock, standardized reported date"),
    ("TransactionClockRawDates", "TW_C1_TRAW20", "Transaction clock, raw vendor trade date"),
    ("PairedClocks", "TW_C1_PAIRED", "Paired trade-date minus disclosure-date difference"),
    ("Capm", "TW_S2_CAPM", "CAPM-adjusted returns"),
    ("FiveFactorMomentum", "TW_S2_FF5MOM", "Five-factor-plus-momentum-adjusted returns"),
]
EX_LABEL = {k: label for k, _, label in EX_ROWS}


def ex_row(key: str, label: str | None = None, direction_means: bool = False) -> str:
    """One main-text row: de-duplicated sample, block clusters."""
    cells = [label or EX_LABEL[key]]
    if direction_means:
        cells += ["", ""]  # direction means are not part of the re-run tables (see the table note)
    cells += [rf"\ExDedup{key}Bps", rf"\ExDedup{key}SeBlock", rf"[\ExDedup{key}CiLowBlock, \ExDedup{key}CiHighBlock]", rf"\ExDedup{key}PBlock", rf"\ExDedup{key}N"]
    return tex_row(*cells)


v4("TableTwoPanelARows", "Table 2 Panel A body (6 columns: row, spread, SE, interval, p, N), de-duplicated sample, block clusters",
   "%\n" + ex_row("Primary", "Purchase-minus-sale spread") + "\n")
v4("TableTwoPanelBRows", "Table 2 Panel B body (6 columns; the five secondary horizons), de-duplicated sample, block clusters",
   "%\n" + "\n".join(ex_row(k) for k in ["HorizonFive", "HorizonTen", "HorizonForty", "HorizonSixty", "HorizonOneTwentySeven"]) + "\n")
v4("TableThreeRowsFull", "Table 3 body, full (6 columns: specification, spread, SE, interval, p, N)",
   "%\n" + "\n".join(ex_row(k, "Official-first date rule (headline)" if k == "Primary" else None)
                     for k in ["Primary", "RawVendorDates", "OfficialDates", "OfficialSource", "VendorSource", "EntryPlusOne"]) + "\n")
v4("TableThreeRowsCompact", "Table 3 body, compact (sample mode; the two source segments omitted)",
   "%\n" + "\n".join(ex_row(k, "Official-first date rule (headline)" if k == "Primary" else None)
                     for k in ["Primary", "RawVendorDates", "OfficialDates", "EntryPlusOne"]) + "\n")
v4("TableClocksRows", "clock-comparison table body (6 columns), de-duplicated sample, block clusters",
   "%\n" + "\n".join(ex_row(k, "Public clock (headline)" if k == "Primary" else None)
                     for k in ["Primary", "TransactionClock", "TransactionClockRawDates", "PairedClocks"]) + "\n")
rerun_body = []
for panel, keys in [("Panel A: Horizons after public entry", ["Primary", "HorizonFive", "HorizonTen", "HorizonForty", "HorizonSixty", "HorizonOneTwentySeven"]),
                    ("Panel B: Public-date and source variants", ["RawVendorDates", "OfficialDates", "OfficialSource", "VendorSource", "EntryPlusOne"]),
                    ("Panel C: Return-date comparison", ["TransactionClock", "TransactionClockRawDates", "PairedClocks"]),
                    ("Panel D: Factor-adjusted returns", ["Capm", "FiveFactorMomentum"])]:
    rerun_body.append(panel_row(8, panel))
    for key in keys:
        rerun_body.append(tex_row(EX_LABEL[key], rf"\ExOrig{key}Bps", rf"\ExOrig{key}SeDay", rf"\ExOrig{key}SeBlock", rf"\ExDedup{key}Bps", rf"\ExDedup{key}SeBlock", rf"\ExOrig{key}N", rf"\ExDedup{key}N"))
# Panel E: the signed payoff and the alternative units of the released plan; the
# equal-household and event-level tests are cluster-free and carry one standard error, printed once per sample.
rerun_body.append(panel_row(8, "Panel E: Signed payoff, alternative units, and identifiers"))
for key, label in [("SignedPayoff", "Signed mimicking payoff"), ("RawRows", "Transaction-record weighting"), ("SourceNativeIds", "Source-native household identifiers")]:
    rerun_body.append(tex_row(label, rf"\ExOrig{key}Bps", rf"\ExOrig{key}SeDay", rf"\ExOrig{key}SeBlock", rf"\ExDedup{key}Bps", rf"\ExDedup{key}SeBlock", rf"\ExOrig{key}N", rf"\ExDedup{key}N"))
for key, label in [("EqualHousehold", "Equal-household weighting (one SE)"), ("EventLevel", "Security-entry-date events (one SE)")]:
    rerun_body.append(tex_row(label, rf"\ExOrig{key}Bps", rf"\ExOrig{key}Se", "--", rf"\ExDedup{key}Bps", rf"\ExDedup{key}Se", rf"\ExOrig{key}N", rf"\ExDedup{key}N"))
# Panel F: the funded portfolio on both samples (A5's \ExCompPort<Field>{Released,Dedup} macros): account
# mechanics in the estimate columns, then the three Newey-West regressions with the SE under the released
# day column and the de-duplicated block column (there is no cluster dimension) and N = trading days.
rerun_body.append(panel_row(8, "Panel F: Funded long-after-purchase portfolio"))
for field, label in [("EligiblePurchases", "Eligible purchase signals offered"), ("Positions", "Positions opened"), ("Rejected", "Candidate signals not admitted"),
                     ("RejectedLotCap", r"\quad of which at the 50-position limit"), ("RejectedCashOrCap", r"\quad of which for want of cash or by the 5\% cap"),
                     ("RejectedTiming", r"\quad of which without an entry price"),
                     ("InvestedFraction", "Average invested fraction"), ("DaysAtLotCap", "Share of days at the 50-position limit"), ("SameInvestedDays", "Share of days with the same invested fraction on both samples"),
                     ("Turnover", "Annualized one-way turnover"), ("AnnualizedReturn", "Annualized total-account return"), ("Benchmark", "Exposure-matched benchmark return"),
                     ("MaxExposure", "Maximum realized single-security exposure"), ("Sharpe", "Sharpe ratio")]:
    rerun_body.append(tex_row(label, rf"\ExCompPort{field}Released", "", "", rf"\ExCompPort{field}Dedup", "", "", ""))
for field, label in [("NwMean", r"Mean excess return, annual \% (Newey--West SE)"), ("CapmAlpha", r"CAPM alpha, annual \% (Newey--West SE)"), ("FfFiveMomAlpha", r"Five-factor-plus-momentum alpha, annual \% (Newey--West SE)")]:
    rerun_body.append(tex_row(label, rf"\ExCompPort{field}Released", rf"\ExCompPort{field}SeReleased", "--", rf"\ExCompPort{field}Dedup", rf"\ExCompPort{field}SeDedup", r"\ExCompPortDaysReleased", r"\ExCompPortDaysDedup"))
v4("IARerunRows", "appendix re-run table body (8 columns: row, released estimate, released day-clustered SE, original-sample block-clustered SE, de-duplicated estimate, de-duplicated block-clustered SE, original N, de-duplicated N); Panels E and F add the alternative units and the funded portfolio on both samples",
   "%\n" + "\n".join(rerun_body) + "\n")

# The reconciliation bridge (generated/pyun_macros.tex): from Pyun's Table 3 design to this paper's registered estimates, one
# element per row. No-op rows (asset filter, direction, weighting) and near-no-op rows are left out.
BRIDGE_ROWS = [
    ("Zero", r"\citet{pyun2025disclosure}, Table~3 as printed (population record)", "30", False),
    ("One", "This paper's record, 2015--2025, his design", "30", True),
    ("Two", "Official lines only (his electronic segment)", "30", True),
    ("Three", "Official plus non-duplicate vendor lines", "30", True),
    ("Seven", "Public-clock entry one session after the public date", "30", True),
    ("EightA", "Lines collapsed to episodes and signals", "30", True),
    ("Nine", "Horizon 30 to 20 sessions", "20", True),
    ("ElevenA", r"Benchmark S\&P~500 in place of the value-weighted index", "20", True),
    ("TwelveA", "Household and 20-trading-day-block clusters", "20", True),
    ("ThirteenA", "All anchors, 2012--2026 (Section~3 sample)", "20", True),
]


def bridge_cell(step: str, stat: str, with_se: bool) -> str:
    return rf"\PyunBridgeStep{step}{stat}Estimate" + (rf" (\PyunBridgeStep{step}{stat}Se)" if with_se else "")


COMPACT_SKIP = {"Three", "ElevenA", "TwelveA"}  # the writing-sample cut omits three small steps


def bridge_body(compact: bool) -> str:
    rows = []
    for step, label, horizon, with_se in BRIDGE_ROWS:
        if compact and step in COMPACT_SKIP:
            continue
        spreads = [bridge_cell(step, "SpreadPublic", True), bridge_cell(step, "SpreadTxn", True)] if step != "Zero" else ["--", "--"]
        rows.append(tex_row(label, horizon, bridge_cell(step, "TxnMean", with_se), bridge_cell(step, "FilingMean", with_se), bridge_cell(step, "Diff", with_se), *spreads))
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{\textit{From the 20-session row with the factor benchmark of Section~4 (purchase means only)}}\\")
    for step, label in [("ElevenB", "Five-factor-plus-momentum benchmark"), ("ThirteenB", "Block clusters; the two periods of Section~4")]:
        rows.append(tex_row(label, "20", bridge_cell(step, "TxnMean", True), bridge_cell(step, "FilingMean", True), "--", "--", "--"))
    return "%\n" + "\n".join(rows) + "\n"


v4("TableBridgeRows", "Section 5 bridge table body (7 columns: element changed, horizon in sessions, purchase mean from the transaction date, purchase mean from the public date, paired difference for purchases, purchase-minus-sale spread on the public clock, spread on the transaction clock; estimate (SE) in bp over the horizon)", bridge_body(False))
v4("TableBridgeRowsCompact", "Section 5 bridge table body, writing-sample cut (three small steps omitted)", bridge_body(True))

placeholder_lines = [
    "% Generated by generate_assets.py: version-4 Act II placeholder macros.",
    "% Each macro is a \\providecommand with a visible default. Real values are",
    "% supplied by generated/results_macros.tex and generated/existing_rerun_macros.tex",
    "% (input after this file) using \\renewcommand. Never edit this file by hand.",
]
# every appendix file is scanned: the released-plan appendices D and F cite the event-level count macros of
manuscript_sources = list((ROOT / "sections").glob("v4_*.tex")) + list((ROOT / "appendix").glob("*.tex")) + [ROOT / "frontmatter" / "main.tex"]
result_names = set()
for path in manuscript_sources + [GEN / "v4_placeholders.tex"]:
    if path.exists():
        result_names |= set(re.findall(r"\\((?:Res|Ex|Pyun)[A-Z][A-Za-z]*)", path.read_text()))
for _name, default, _, _ in v4_macros:
    result_names |= set(re.findall(r"\\((?:Res|Ex|Pyun)[A-Z][A-Za-z]*)", default))
pending_lines = [
    "% Generated by generate_assets.py: every household-results (\\Res...) and existing-re-run (\\Ex...)",
    "% macro the manuscript uses, as a \\providecommand with a visible pending default. Input AFTER",
    "% generated/results_macros.tex and generated/existing_rerun_macros.tex, so a name those files",
    "% supply keeps its value and a name they do not supply prints the marker. Never edit by hand.",
]
pending_lines += [rf"\providecommand{{\{name}}}{{\text{{[PENDING]}}}}" for name in sorted(result_names)]
# The fallback list is read off the manuscript text; without the manuscript the stored file is kept.
if (ROOT / "frontmatter" / "main.tex").exists():
    (GEN / "v4_results_pending.tex").write_text("\n".join(pending_lines) + "\n")

names_seen = [name for name, *_ in v4_macros]
if len(names_seen) != len(set(names_seen)):
    raise ValueError("duplicate version-4 placeholder name")
for name, default, _, _ in v4_macros:
    placeholder_lines.append(rf"\providecommand{{\H{name}}}{{{default}}}")
(GEN / "v4_placeholders.tex").write_text(display_text("\n".join(placeholder_lines) + "\n"))
pd.DataFrame(v4_macros, columns=["macro", "default", "description", "source"]).assign(
    macro=lambda d: "\\H" + d["macro"],
    default=lambda d: d["default"].where(~d["default"].str.startswith("%"), "[table rows]"),
).to_csv(GEN / "v4_placeholders.csv", index=False)

# ---------------------------------------------------------------------------
# Version-4 figure shells: axes, labels, and a visible pending marker only. The
# manuscript prefers figures/v4_fig*.pdf when they exist (written by the results
# writer) and falls back to these shells otherwise.
# ---------------------------------------------------------------------------
SHELL_META = {"Creator": "Trading on Disclosure", "CreationDate": None, "ModDate": None}


def shell_stamp(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, PENDING, transform=ax.transAxes, ha="center", va="center", fontsize=13, color="0.55")


fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
for ax, title, xlabel, ylabel in [
    (axes[0], "Sparse households", "Share of informed households", "Edge per purchase (bp)"),
    (axes[1], "Sparse trades", "Share of informed purchases", "Edge per trade (bp)"),
]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.003, 0.3)
    ax.set_ylim(30, 5000)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16, which="both")
    shell_stamp(ax)
axes[0].text(0.02, 0.96, "80%-power frontier of the best correctly sized test (pending)", transform=axes[0].transAxes, fontsize=7.5, va="top", color="0.35")
fig.tight_layout()
fig.savefig(FIG / "v4_shell_fig2_detection_region.pdf", bbox_inches="tight", metadata=SHELL_META)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
axes[0].set_title("(a) Largest household mean under the null", fontsize=10)
axes[0].set_xlabel("Largest household mean abnormal return (bp)")
axes[0].set_ylabel("Density across null panels")
axes[1].set_title("(b) Simultaneous 95% rank sets", fontsize=10)
axes[1].set_xlabel("Household (anonymized index, ordered by point estimate)")
axes[1].set_ylabel("Rank")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    shell_stamp(ax)
fig.tight_layout()
fig.savefig(FIG / "v4_shell_fig3_winners_ranks.pdf", bbox_inches="tight", metadata=SHELL_META)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(8, 600)
ax.set_ylim(0.3, 3000)
ax.set_xlabel("Purchases per year")
ax.set_ylabel("Years of disclosures to 80% power")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.16, which="both")
ax.text(0.02, 0.96, "Curves: annual edge 2%, 5%, 10%, 20%; dashed = best-of-H adjustment (pending)", transform=ax.transAxes, fontsize=7.5, va="top", color="0.35")
shell_stamp(ax)
fig.tight_layout()
fig.savefig(FIG / "v4_shell_fig4_time_to_detect.pdf", bbox_inches="tight", metadata=SHELL_META)
plt.close(fig)

print("Generated LaTeX fragments and vector figures from corrected canonical release and version-4 placeholders.")
