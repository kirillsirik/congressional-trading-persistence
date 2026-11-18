"""Emitter: `paper/generated/existing_rerun_macros.tex` from the re-run tables and the placebo size
check, plus the macro list `paper/generated/existing_macros.md` (plain names; numbers only).

Name scheme: \\Ex<Sample><Test><Quantity>, Sample in {Orig, Dedup, DedupOfficialOnly, DedupExclTwo};
Test as in TEST_NAMES; Quantity in {Bps, SeDay, SeBlock, TDay, TBlock, PDay, PBlock, CiLowDay,
CiHighDay, CiLowBlock, CiHighBlock, N, Households, Days, Blocks}. Placebo macros:
\\ExPlaceboSpread<Quantity><Clusters><Flavour> for the market-adjusted outcome (the headline's) and
\\ExPlaceboSpreadFactor<Quantity><Clusters><Flavour> for the FF5+MOM-adjusted outcome; observed
placements \\ExDedup<Test>Placebo[Factor]P<Flavour>. Event-level counts: \\ExEventLevel<Quantity>
(released sample) and \\ExDedupEventLevel<Quantity>. Final wave: Table 1 fields
\\Ex<Orig|Dedup>TableOne<Field>; the Section 3.4 unit rows \\Ex<Sample><Test><Quantity> with Test in
{SignedPayoff, RawRows, SourceNativeIds} (clustered quantities as above), {EqualHousehold, EventLevel}
(Bps, Se, T, P, CiLow, CiHigh, N) and {PlaceboDirection, PlaceboDates} (Bps, P, Draws, MeanBps, SdBps);
the funded portfolio \\Ex<Sample>Port<Field>. Appendix wave (`existing/appendix.py`, files in
`data/v4/existing/appendix_dedup/`): \\Ex<Sample><Test>BootCi{Low,High} (20-date moving-block bootstrap
interval, 4,999 draws), \\Ex<Sample><Test>HolmP{Day,Block} and \\Ex<Sample>Port<Test>HolmP (Holm within the
released families), \\ExCompPort<Field>{Released,Dedup} (portfolio rows of the two-sample comparison),
\\ExDedupAppAttrition<Stage><Cell>, \\ExDedupAppAssembly<Field>, \\ExDedupAppDateGap<Gap><Set><Cell> and
\\Ex<Sample>AppEventStudy<Cell>.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.existing import appendix as ap
from trading_on_disclosure_v4.existing import placebo_spread as ps
from trading_on_disclosure_v4.existing import portfolio as pf
from trading_on_disclosure_v4.existing import rerun, table1, units
from trading_on_disclosure_v4.ledger import macros as ledger_macros
from trading_on_disclosure_v4.outcomes import panel

TEX_PATH = panel.REPO_ROOT / "paper" / "generated" / "existing_rerun_macros.tex"
MD_PATH = panel.REPO_ROOT / "paper" / "generated" / "existing_macros.md"

TEST_NAMES = {
    "TW_P1_E20": ("Primary", "20-day spread, public clock (Table 2 primary)"),
    "TW_P2_E5": ("HorizonFive", "5-day spread"),
    "TW_P2_E10": ("HorizonTen", "10-day spread"),
    "TW_P2_E40": ("HorizonForty", "40-day spread"),
    "TW_P2_E60": ("HorizonSixty", "60-day spread"),
    "TW_P2_E127": ("HorizonOneTwentySeven", "127-day spread"),
    "TW_P3_RAW_CT": ("RawVendorDates", "20-day spread on unadjusted vendor publication dates"),
    "TW_P3_OFFICIAL_ONLY": ("OfficialDates", "20-day spread, observed official public dates only"),
    "TW_P3_OFFICIAL_SEGMENT": ("OfficialSource", "20-day spread, official-source records"),
    "TW_P3_VENDOR_SEGMENT": ("VendorSource", "20-day spread, vendor-proxy records"),
    "TW_P3_ENTRY_PLUS1": ("EntryPlusOne", "20-day spread, one extra processing day"),
    "TW_C1_T20": ("TransactionClock", "20-day spread, transaction clock"),
    "TW_C1_TRAW20": ("TransactionClockRawDates", "20-day spread, transaction clock, uncorrected vendor dates"),
    "TW_C1_PAIRED": ("PairedClocks", "paired transaction-minus-public 20-day difference"),
    "TW_S2_CAPM": ("Capm", "20-day spread, CAPM abnormal returns"),
    "TW_S2_FF5MOM": ("FiveFactorMomentum", "20-day spread, FF5+momentum abnormal returns"),
    "TW_D_SAR20": ("SignedPayoff", "signed mimicking payoff (long after purchases, short after sales), 20 days"),
    "TW_S1_RAWROWS": ("RawRows", "20-day spread with each source record as an observation (raw-row weighting)"),
    "TW_S1_SRCID": ("SourceNativeIds", "20-day spread clustered by source-native household identifiers"),
}
UNIT_NAMES = {
    "TW_S1_EQHH": ("EqualHousehold", "equal-household 20-day spread (household t)"),
    "TW_S1_EVENT": ("EventLevel", "event-level 20-day spread (purchase-only minus sale-only events, Welch)"),
    "TW_PLC_DIRECTION": ("PlaceboDirection", "within-month direction-permutation placebo"),
    "TW_PLC_DATES": ("PlaceboDates", "within-security-year pseudo-entry placebo"),
}
TABLE_ONE_NAMES = {
    "official_records": "OfficialRecords", "vendor_records": "VendorRecords", "assembled_records": "AssembledRecords",
    "observed_official_dates": "ObservedOfficialDates", "vendor_proxy_dates": "VendorProxyDates",
    "matched_records": "MatchedRecords", "matched_vendor_duplicates": "MatchedVendorDuplicates",
    "signals": "Signals", "mixed_signals": "MixedSignals", "nonmixed_signals": "NonMixedSignals",
    "duplicate_signals_removed": "DuplicateSignalsRemoved", "signals_after_rule": "SignalsAfterRule",
    "signals_complete_h20": "SignalsCompleteTwenty", "households_all_signals": "HouseholdsAllSignals",
    "households_h20": "Households", "securities": "Securities", "purchases_h20": "Purchases", "sales_h20": "Sales",
    "official_source_h20": "OfficialSource", "vendor_source_h20": "VendorSource",
    "observed_date_h20": "ObservedDate", "proxy_date_h20": "ProxyDate", "top_1_share": "TopOneShare",
    "top_5_share": "TopFiveShare", "top_10_share": "TopTenShare", "herfindahl": "Herfindahl",
    "effective_households": "EffectiveHouseholds", "blocks_h20": "Blocks",
}
PORT_FIELDS = {  # field -> (macro suffix, formatter name, description)
    "n_eligible_purchases": ("EligiblePurchases", "int", "eligible purchase signals offered to the account"),
    "signals_accepted": ("SignalsAccepted", "int", "positions opened"),
    "signals_rejected": ("SignalsRejected", "int", "purchase signals rejected at entry"),
    "n_days": ("Days", "int", "trading days in the account path"),
    "annualized_return": ("AnnualizedReturn", "pct", "annualized total return"),
    "benchmark_annualized_return": ("BenchmarkReturn", "pct", "annualized return of the exposure-matched benchmark"),
    "avg_invested_fraction": ("AvgInvested", "pct", "average invested fraction"),
    "max_security_exposure": ("MaxExposure", "pct", "maximum single-security exposure"),
    "annual_turnover": ("Turnover", "pct", "annual turnover"),
    "sharpe": ("Sharpe", "num", "annualized Sharpe ratio of daily excess returns"),
}
PORT_TESTS = {"TW_IMPL_PORT_NW": ("Mean", "NW-lag-20 mean excess return"), "TW_IMPL_PORT_CAPM": ("Capm", "CAPM alpha"),
              "TW_IMPL_PORT_FF5MOM": ("FiveFactorMomentum", "FF5+MOM alpha")}
SAMPLE_NAMES = {"original": "Orig", "dedup": "Dedup", "dedup_official_only": "DedupOfficialOnly",
                "dedup_excl_two_vendor_households": "DedupExclTwo"}
SAMPLE_LABELS = {"original": "original sample", "dedup": "de-duplicated sample",
                 "dedup_official_only": "de-duplicated sample, official-source records only",
                 "dedup_excl_two_vendor_households": "de-duplicated sample without the two vendor-only households"}
CLUSTER_NAMES = {"day": "Day", "block20": "Block"}
FLAVOUR_NAMES = {"N1": "Timing", "N2": "Centered"}
OUTCOME_INFIX = {"market_adjusted": "", "ff5mom_adjusted": "Factor"}
OUTCOME_LABELS = {"market_adjusted": "market-adjusted outcome", "ff5mom_adjusted": "FF5+MOM-adjusted outcome"}
EVENT_LEVEL = {"events_in_estimate": ("N", "events entering the released event-level estimate (purchase-only + sale-only)"),
               "purchase_only_events": ("PurchaseOnly", "purchase-only events"),
               "sale_only_events": ("SaleOnly", "sale-only events"),
               "mixed_events_excluded": ("MixedExcluded", "mixed events excluded"),
               "all_outcome_bearing_events": ("All", "all outcome-bearing events")}
ATTRITION_STAGES = {"mixed_direction_exclusion": ("Mixed", "mixed-direction signals excluded"),
                    "vendor_duplicate_rule": ("Duplicates", "vendor signals that duplicate an official transaction removed"),
                    "complete_h20_outcome": ("CompleteTwenty", "signals with a complete 20-day public-clock outcome")}
ASSEMBLY_FIELDS = {"official_records": "OfficialRecords", "vendor_records": "VendorRecords", "scan_linked_records": "ScanLinked",
                   "vendor_only_records": "VendorOnly", "assembled_records": "Assembled", "removed_signals": "RemovedSignals",
                   "removed_signal_records": "RemovedSignalRecords", "removed_signal_records_scan_linked": "RemovedSignalRecordsScanLinked",
                   "removed_signal_records_vendor_only": "RemovedSignalRecordsVendorOnly", "vendor_records_after_rule": "VendorRecordsAfterRule"}
GAP_NAMES = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "total": "Total"}
GAP_SETS = {"retained_overlap": ("Overlap", "all retained-overlap vendor records (the released column)"),
            "in_signals": ("InSignals", "retained-overlap records that are parents of a signal"),
            "dedup_nonmixed_signals": ("Dedup", "retained-overlap records that are parents of a de-duplicated non-mixed signal")}
EVENT_DAYS = {-10: "MinusTen", 0: "Zero", 20: "Twenty", 40: "Forty", 60: "Sixty"}
# wave-2 renames: the factor-adjusted placebo (the only one before) now carries the 'Factor' infix;
# the un-infixed names hold the market-adjusted placebo, which is the headline outcome's.
RENAMES = [
    (r"\ExPlaceboSpread<Q><Cl><Fl>", r"\ExPlaceboSpreadFactor<Q><Cl><Fl>",
     "size rows (Q in Reject, RejectOfMean, Coverage, MeanSe, SdOverSe)"),
    (r"\ExPlaceboSpreadMean<Fl>", r"\ExPlaceboSpreadFactorMean<Fl>", "placebo mean spread"),
    (r"\ExPlaceboSpreadSd<Fl>", r"\ExPlaceboSpreadFactorSd<Fl>", "placebo SD of the spread"),
    (r"\ExDedupPrimaryPlaceboP<Fl>", r"\ExDedupPrimaryPlaceboP<Fl> (name kept, value changed: the headline spread "
     r"is now placed against the market-adjusted placebo)", "observed placement of the headline spread"),
    (r"(new)", r"\ExDedupCapmPlaceboFactorP<Fl>, \ExDedupFiveFactorMomentumPlaceboFactorP<Fl>",
     "the CAPM and FF5+MOM spreads placed against the factor-adjusted placebo"),
    (r"(new)", r"\ExPlaceboSpread[Factor]{QLow,QHigh,PurchaseMean,SaleMean}<Fl>",
     "placebo 2.5% / 97.5% quantiles and the placebo means of purchases and sales"),
]


def fmt_num(x: float, nd: int = 2) -> str:
    r"""Negatives carry the ledger emitter's minus (`\ensuremath{-}`): a proper minus in text and in math."""
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{x:,.{nd}f}"
    return ledger_macros.MINUS + s[1:] if s.startswith("-") else s


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}" if x < 0.01 else f"{x:.2f}"


def fmt_pct(x: float, nd: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else fmt_num(100 * x, nd).replace(",", "") + "\\%"


def fmt_p3(x: float) -> str:
    """Three decimals, the released tables' convention (`generate_assets.pval`)."""
    return "n/a" if not np.isfinite(x) else "$<$0.001" if x < 0.001 else f"{x:.3f}"


def rerun_macros(tables: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(name, value, meaning) for every test x sample; day and block quantities side by side."""
    out = []
    for test_id, (tname, desc) in TEST_NAMES.items():
        for sample, sname in SAMPLE_NAMES.items():
            sub = tables[(tables["test_id"] == test_id) & (tables["sample"] == sample)].set_index("clusters")
            if sub.empty:
                continue
            day, blk = sub.loc["day"], sub.loc["block20"]
            base = f"Ex{sname}{tname}"
            label = f"{desc}; {SAMPLE_LABELS[sample]}"
            out += [
                (f"{base}Bps", fmt_num(day["estimate"]), f"{label}: estimate, bp"),
                (f"{base}SeDay", fmt_num(day["se"]), f"{label}: day-clustered SE, bp"),
                (f"{base}SeBlock", fmt_num(blk["se"]), f"{label}: block-clustered SE, bp"),
                (f"{base}TDay", fmt_num(day["t"]), f"{label}: t, day clusters"),
                (f"{base}TBlock", fmt_num(blk["t"]), f"{label}: t, block clusters"),
                (f"{base}PDay", fmt_p(day["p"]), f"{label}: two-sided p, day clusters"),
                (f"{base}PBlock", fmt_p(blk["p"]), f"{label}: two-sided p, block clusters"),
                (f"{base}CiLowDay", fmt_num(day["ci_low"]), f"{label}: 95% CI low, day clusters, bp"),
                (f"{base}CiHighDay", fmt_num(day["ci_high"]), f"{label}: 95% CI high, day clusters, bp"),
                (f"{base}CiLowBlock", fmt_num(blk["ci_low"]), f"{label}: 95% CI low, block clusters, bp"),
                (f"{base}CiHighBlock", fmt_num(blk["ci_high"]), f"{label}: 95% CI high, block clusters, bp"),
                (f"{base}N", fmt_int(day["n"]), f"{label}: signals"),
                (f"{base}Households", fmt_int(day["n_households"]), f"{label}: households"),
                (f"{base}Days", fmt_int(day["n_dates_or_blocks"]), f"{label}: anchor-day clusters"),
                (f"{base}Blocks", fmt_int(blk["n_dates_or_blocks"]), f"{label}: 20-day block clusters"),
            ]
    return out


def sample_macros(report: dict) -> list[tuple[str, str, str]]:
    return [("ExDedupDroppedSignals", fmt_int(report["n_dropped"]), "signals dropped by the duplicate rule"),
            ("ExDedupSignals", fmt_int(report["n_out"]), "non-mixed signals after the duplicate rule"),
            ("ExOrigSignals", fmt_int(report["n_in"]), "non-mixed signals in the original sample")]


def placebo_macros(summary: pd.DataFrame) -> list[tuple[str, str, str]]:
    out = []
    if "outcome" not in summary.columns:
        summary = summary.assign(outcome="market_adjusted")
    for _, r in summary.iterrows():
        fl, ox, olab = FLAVOUR_NAMES[r["flavour"]], OUTCOME_INFIX[r["outcome"]], OUTCOME_LABELS[r["outcome"]]
        if r["clusters"] == "placebo_test":
            tname = TEST_NAMES[r["observed_test_id"]][0]
            out.append((f"ExDedup{tname}Placebo{ox}P{fl}", fmt_p(r["observed_placebo_p"]),
                        f"de-duplicated {TEST_NAMES[r['observed_test_id']][1]}: two-sided placebo p-value, "
                        f"{fl.lower()} placebo, {olab}"))
            continue
        cl = CLUSTER_NAMES[r["clusters"]]
        lab = f"spread test, {cl.lower()} clusters, {fl.lower()} placebo, {olab}"
        out += [(f"ExPlaceboSpread{ox}Reject{cl}{fl}", fmt_pct(r["reject_two_sided_5pct"]),
                 f"{lab}: two-sided rejection rate of zero at nominal 5%"),
                (f"ExPlaceboSpread{ox}RejectOfMean{cl}{fl}", fmt_pct(r["reject_two_sided_5pct_of_placebo_mean"]),
                 f"{lab}: two-sided rejection rate of the placebo mean at nominal 5%"),
                (f"ExPlaceboSpread{ox}Coverage{cl}{fl}", fmt_pct(r["coverage_of_placebo_mean"]),
                 f"{lab}: 95% interval coverage of the placebo mean"),
                (f"ExPlaceboSpread{ox}MeanSe{cl}{fl}", fmt_num(r["mean_se_bp"]), f"{lab}: mean SE, bp"),
                (f"ExPlaceboSpread{ox}SdOverSe{cl}{fl}", fmt_num(r["sd_over_mean_se"]),
                 f"{lab}: SD of estimates over mean SE")]
        if r["clusters"] == "day":  # cluster-free quantities once per outcome x flavour
            lab = f"{fl.lower()} placebo, {olab}"
            out += [(f"ExPlaceboSpread{ox}Mean{fl}", fmt_num(r["placebo_mean_spread_bp"]), f"mean placebo spread, {lab}, bp"),
                    (f"ExPlaceboSpread{ox}Sd{fl}", fmt_num(r["placebo_sd_spread_bp"]),
                     f"SD of the placebo spread across draws, {lab}, bp"),
                    (f"ExPlaceboSpread{ox}QLow{fl}", fmt_num(r["placebo_q025_spread_bp"]),
                     f"2.5% quantile of the placebo spread, {lab}, bp"),
                    (f"ExPlaceboSpread{ox}QHigh{fl}", fmt_num(r["placebo_q975_spread_bp"]),
                     f"97.5% quantile of the placebo spread, {lab}, bp"),
                    (f"ExPlaceboSpread{ox}PurchaseMean{fl}", fmt_num(r["placebo_mean_purchase_bp"]),
                     f"placebo mean of purchases, {lab}, bp"),
                    (f"ExPlaceboSpread{ox}SaleMean{fl}", fmt_num(r["placebo_mean_sale_bp"]),
                     f"placebo mean of sales, {lab}, bp")]
    r0 = summary.iloc[0]
    out += [("ExPlaceboSpreadDraws", fmt_int(r0["n_draws"]), "placebo draws"),
            ("ExPlaceboSpreadMcSe", fmt_pct(r0["mc_se_of_rejection_rate"]), "Monte Carlo SE of a 5% rejection rate")]
    return out


def event_level_macros(counts: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Released-sample counts keep their wave-2 names; the de-duplicated counts carry `Dedup` (its `N`
    is emitted by `unit_macros` from the event-level row, the same purchase-only + sale-only count)."""
    if "sample" not in counts.columns:
        counts = counts.assign(sample="original")
    out = []
    for r in counts.to_dict("records"):
        infix = "" if r["sample"] == "original" else SAMPLE_NAMES[r["sample"]]
        for col, (name, desc) in EVENT_LEVEL.items():
            if infix and name == "N":
                continue
            out.append((f"Ex{infix}EventLevel{name}", fmt_int(r[col]), f"event-level test, {SAMPLE_LABELS[r['sample']]}: {desc}"))
    return out


def unit_macros(tables: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Equal-household, event-level and the two released placebos, per sample."""
    out = []
    for test_id, (tname, desc) in UNIT_NAMES.items():
        for sample, sname in SAMPLE_NAMES.items():
            sub = tables[(tables["test_id"] == test_id) & (tables["sample"] == sample)]
            if sub.empty:
                continue
            r, base, label = sub.iloc[0], f"Ex{sname}{tname}", f"{desc}; {SAMPLE_LABELS[sample]}"
            if test_id in units.PLACEBO_TESTS:
                out += [(f"{base}Bps", fmt_num(r["estimate"]), f"{label}: observed 20-day spread, bp"),
                        (f"{base}P", fmt_p(r["p"]), f"{label}: placebo p-value"),
                        (f"{base}Draws", fmt_int(r["n_draws"]), f"{label}: draws"),
                        (f"{base}MeanBps", fmt_num(r["placebo_mean"]), f"{label}: mean placebo spread, bp"),
                        (f"{base}SdBps", fmt_num(r["placebo_sd"]), f"{label}: SD of the placebo spread, bp")]
            else:
                unit = "households with both directions" if test_id == "TW_S1_EQHH" else "events entering the estimate"
                out += [(f"{base}Bps", fmt_num(r["estimate"]), f"{label}: estimate, bp"),
                        (f"{base}Se", fmt_num(r["se"]), f"{label}: SE, bp"), (f"{base}T", fmt_num(r["t"]), f"{label}: t"),
                        (f"{base}P", fmt_p(r["p"]), f"{label}: two-sided p"),
                        (f"{base}CiLow", fmt_num(r["ci_low"]), f"{label}: 95% CI low, bp"),
                        (f"{base}CiHigh", fmt_num(r["ci_high"]), f"{label}: 95% CI high, bp"),
                        (f"{base}N", fmt_int(r["n"]), f"{label}: {unit}")]
    return out


def table_one_macros(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Every Table 1 field on both samples."""
    out = []
    for r in table.itertuples(index=False):
        name = TABLE_ONE_NAMES[r.field]
        for col, sname in (("original", "Orig"), ("dedup", "Dedup")):
            v = getattr(r, col)
            if r.field.endswith("_share"):
                value = fmt_pct(v)
            elif r.field == "herfindahl":
                value = fmt_num(v, 3)
            elif r.field == "effective_households":
                value = fmt_num(v, 1)
            else:
                value = fmt_int(v)
            out.append((f"Ex{sname}TableOne{name}", value, f"Table 1, {SAMPLE_LABELS[col]}: {r.description}"))
    return out


def portfolio_macros(summary: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Funded long-after-purchase portfolio, per signal set; percent units as the released macros."""
    out = []
    for r in summary.to_dict("records"):
        sname, label = SAMPLE_NAMES[r["sample"]], f"funded portfolio, {SAMPLE_LABELS[r['sample']]}"
        for field, (suffix, kind, desc) in PORT_FIELDS.items():
            v = r[field]
            value = fmt_int(v) if kind == "int" else fmt_pct(v, 2) if kind == "pct" else fmt_num(v)
            out.append((f"Ex{sname}Port{suffix}", value, f"{label}: {desc}"))
        for test_id, (tname, desc) in PORT_TESTS.items():
            base, g = f"Ex{sname}Port{tname}", {k: r[f"{test_id}.{k}"] for k in pf.REG_FIELDS}
            out += [(f"{base}Pct", fmt_num(100 * g["estimate"]), f"{label}: {desc}, annualized, percent"),
                    (f"{base}SEPct", fmt_num(100 * g["se_multiway"]), f"{label}: {desc}, SE, percent"),
                    (f"{base}T", fmt_num(g["t_stat"]), f"{label}: {desc}, t"),
                    (f"{base}P", fmt_p(g["raw_p"]), f"{label}: {desc}, two-sided p"),
                    (f"{base}CiLowPct", fmt_num(100 * g["ci_low"]), f"{label}: {desc}, 95% CI low, percent"),
                    (f"{base}CiHighPct", fmt_num(100 * g["ci_high"]), f"{label}: {desc}, 95% CI high, percent"),
                    (f"{base}N", fmt_int(g["n"]), f"{label}: {desc}, trading days")]
    return out


def _fmt_kind(v: float, kind: str) -> str:
    return fmt_int(v) if kind == "int" else fmt_pct(v, 2) if kind == "pct" else fmt_p(v) if kind == "p" else fmt_num(v)


def appendix_macros(out_dir: Path = ap.OUT_DIR) -> list[tuple[str, str, str]]:
    """Every macro of the appendix wave; each meaning names its file under ``out_dir``."""
    import json

    src = out_dir.relative_to(panel.REPO_ROOT) if out_dir.is_relative_to(panel.REPO_ROOT) else out_dir
    out = []
    boot = pd.read_csv(out_dir / "block_intervals.csv")
    for r in boot.itertuples(index=False):
        base, label = f"Ex{SAMPLE_NAMES[r.sample]}{TEST_NAMES[r.test_id][0]}", f"{TEST_NAMES[r.test_id][1]}; {SAMPLE_LABELS[r.sample]}"
        out += [(f"{base}BootCiLow", fmt_num(r.boot_ci_low), f"{label}: 20-date moving-block bootstrap 95% interval low, bp ({src}/block_intervals.csv)"),
                (f"{base}BootCiHigh", fmt_num(r.boot_ci_high), f"{label}: moving-block bootstrap 95% interval high, bp ({src}/block_intervals.csv)")]
    out.append(("ExBootDraws", fmt_int(boot["n_draws"].iloc[0]), f"moving-block bootstrap draws ({src}/block_intervals.csv)"))
    full = pd.read_csv(out_dir / "full_results.csv")
    sizes = full.groupby(["sample", "family"])["test_id"].transform("size")
    for r, size in zip(full.itertuples(index=False), sizes, strict=True):
        if not (isinstance(r.family, str) and size > 1 and str(r.inference_eligible).lower() == "yes"):
            continue
        sname = SAMPLE_NAMES[r.sample]
        if r.test_id in ap.PORT_TESTS:
            tname, desc = PORT_TESTS[r.test_id]
            out += [(f"Ex{sname}Port{tname}HolmP", fmt_p3(r.p_holm_day), f"funded portfolio, {SAMPLE_LABELS[r.sample]}: {desc}, Holm p within the three portfolio tests, three decimals ({src}/full_results.csv)"),
                    (f"Ex{sname}Port{tname}PThree", fmt_p3(r.p_day), f"funded portfolio, {SAMPLE_LABELS[r.sample]}: {desc}, two-sided p, three decimals ({src}/full_results.csv)")]
            continue
        tname, desc = {**TEST_NAMES, **UNIT_NAMES}[r.test_id]
        label = f"{desc}; {SAMPLE_LABELS[r.sample]}"
        out += [(f"Ex{sname}{tname}HolmPDay", fmt_p3(r.p_holm_day), f"{label}: Holm p within the released family, day clusters, three decimals ({src}/full_results.csv)"),
                (f"Ex{sname}{tname}HolmPBlock", fmt_p3(r.p_holm_block), f"{label}: Holm p within the released family, block clusters, three decimals ({src}/full_results.csv)"),
                (f"Ex{sname}{tname}PDayThree", fmt_p3(r.p_day), f"{label}: two-sided p, day clusters, three decimals ({src}/full_results.csv)"),
                (f"Ex{sname}{tname}PBlockThree", fmt_p3(r.p_block), f"{label}: two-sided p, block clusters, three decimals ({src}/full_results.csv)")]
    comp = pd.read_csv(out_dir / "comparison_portfolio.csv")
    for r in comp.itertuples(index=False):
        for col, suffix in (("released", "Released"), ("dedup", "Dedup")):
            out.append((f"ExCompPort{r.field}{suffix}", _fmt_kind(getattr(r, col), r.kind) if r.field != "MeanOpenLots" else fmt_num(getattr(r, col), 1),
                        f"two-sample comparison, funded portfolio, {'released' if col == 'released' else 'de-duplicated'} sample: {r.description} ({src}/comparison_portfolio.csv)"))
    att = pd.read_csv(out_dir / "attrition.csv")
    for r in att[att["sample"] == "dedup"].itertuples(index=False):
        name, desc = ATTRITION_STAGES[r.stage]
        out += [(f"ExDedupAppAttrition{name}In", fmt_int(r.n_in), f"attrition, de-duplicated stream, stage '{desc}': starting N ({src}/attrition.csv)"),
                (f"ExDedupAppAttrition{name}Out", fmt_int(r.n_out), f"attrition, de-duplicated stream, stage '{desc}': remaining N ({src}/attrition.csv)"),
                (f"ExDedupAppAttrition{name}Removed", fmt_int(r.n_in - r.n_out), f"attrition, de-duplicated stream, stage '{desc}': excluded ({src}/attrition.csv)")]
    asm = pd.read_csv(out_dir / "source_assembly.csv")
    for r in asm.itertuples(index=False):
        out.append((f"ExDedupAppAssembly{ASSEMBLY_FIELDS[r.field]}", fmt_int(r.value), f"source assembly: {r.description} ({src}/source_assembly.csv)"))
    gap = pd.read_csv(out_dir / "date_gap.csv", dtype={"gap_days": str})
    for r in gap.to_dict("records"):
        g = GAP_NAMES[r["gap_days"]]
        glabel = "all gaps" if g == "Total" else f"gap of {r['gap_days']} calendar day(s)"
        for key, (sname, sdesc) in GAP_SETS.items():
            out += [(f"ExDedupAppDateGap{g}{sname}N", fmt_int(r[f"{key}_n"]), f"vendor minus official date, {sdesc}, {glabel}: records ({src}/date_gap.csv)"),
                    (f"ExDedupAppDateGap{g}{sname}Share", fmt_pct(r[f"{key}_share"], 1), f"vendor minus official date, {sdesc}, {glabel}: share ({src}/date_gap.csv)")]
    report = json.loads((out_dir / "report.json").read_text())
    out.append(("ExDedupAppDateGapRemovedByRule", fmt_int(report["date_gap"]["removed_by_rule"]),
                f"retained-overlap vendor records that belong to a signal the duplicate rule removes ({src}/report.json)"))
    for key, (sname, sdesc) in GAP_SETS.items():
        g = report["date_gap"][key]
        out += [(f"ExDedupAppDateGap{sname}Mean", fmt_num(g["mean_gap_days"]), f"vendor minus official date, {sdesc}: mean gap, calendar days ({src}/report.json)"),
                (f"ExDedupAppDateGap{sname}Median", fmt_num(g["median_gap_days"], 0), f"vendor minus official date, {sdesc}: median gap, calendar days ({src}/report.json)")]
    path = out_dir / "event_study_path.csv"
    if path.exists() and "event_study_dedup" in report:
        es = pd.read_csv(path)
        for sample, sname in (("original", "Orig"), ("dedup", "Dedup")):
            meta, label = report[f"event_study_{sample}"], f"disclosure-centered event study, {SAMPLE_LABELS[sample]}"
            diff = es[(es["sample"] == sample) & (es["series"] == "difference")].set_index("relative_trading_day")
            out += [(f"Ex{sname}AppEventStudyPurchasePaths", fmt_int(meta["purchase_paths"]), f"{label}: purchase signals with a complete event-time path ({src}/report.json)"),
                    (f"Ex{sname}AppEventStudySalePaths", fmt_int(meta["sale_paths"]), f"{label}: sale signals with a complete event-time path ({src}/report.json)"),
                    (f"Ex{sname}AppEventStudyDraws", fmt_int(meta["n_draws"]), f"{label}: multiplier draws ({src}/report.json)"),
                    (f"Ex{sname}AppEventStudyCritical", fmt_num(meta["difference_critical"]), f"{label}: simultaneous 95% critical value of the difference path ({src}/report.json)"),
                    (f"Ex{sname}AppEventStudyDaysExcludingZero", fmt_int(meta["difference_days_band_excludes_zero"]), f"{label}: displayed days on which the band of the difference path excludes zero ({src}/report.json)"),
                    (f"Ex{sname}AppEventStudyDiffMinBps", fmt_num(diff["estimate"].min() * rerun.BP), f"{label}: minimum of the cumulative purchase-minus-sale path over the window, bp ({src}/event_study_path.csv)"),
                    (f"Ex{sname}AppEventStudyDiffMaxBps", fmt_num(diff["estimate"].max() * rerun.BP), f"{label}: maximum of the cumulative purchase-minus-sale path over the window, bp ({src}/event_study_path.csv)")]
            for day, dname in EVENT_DAYS.items():
                out += [(f"Ex{sname}AppEventStudyDiffAt{dname}Bps", fmt_num(diff.loc[day, "estimate"] * rerun.BP), f"{label}: cumulative purchase-minus-sale path at day {day}, bp ({src}/event_study_path.csv)"),
                        (f"Ex{sname}AppEventStudyBandLowAt{dname}Bps", fmt_num(diff.loc[day, "lower"] * rerun.BP), f"{label}: simultaneous band low at day {day}, bp ({src}/event_study_path.csv)"),
                        (f"Ex{sname}AppEventStudyBandHighAt{dname}Bps", fmt_num(diff.loc[day, "upper"] * rerun.BP), f"{label}: simultaneous band high at day {day}, bp ({src}/event_study_path.csv)")]
    return out


def collect(tables_path: Path = rerun.TABLES_PATH, summary_path: Path = ps.SUMMARY_PATH,
            counts_path: Path = rerun.EVENT_COUNTS_PATH, table_one_path: Path = table1.OUT_PATH,
            portfolio_path: Path = pf.SUMMARY_PATH, appendix_dir: Path = ap.OUT_DIR) -> list[tuple[str, str, str]]:
    import json

    tables = pd.read_csv(tables_path)
    report = json.loads(tables_path.with_name("rerun_report.json").read_text())
    macros = sample_macros(report) + rerun_macros(tables) + unit_macros(tables)
    if summary_path.exists():
        macros += placebo_macros(pd.read_csv(summary_path))
    if counts_path.exists():
        macros += event_level_macros(pd.read_csv(counts_path))
    if table_one_path.exists():
        macros += table_one_macros(pd.read_csv(table_one_path))
    if portfolio_path.exists():
        macros += portfolio_macros(pd.read_csv(portfolio_path))
    if (appendix_dir / "full_results.csv").exists():
        macros += appendix_macros(appendix_dir)
    names = [m[0] for m in macros]
    if len(set(names)) != len(names):
        raise ValueError("duplicate macro names")
    return macros


def write_tex(macros: list[tuple[str, str, str]], path: Path = TEX_PATH) -> Path:
    lines = ["% generated by trading_on_disclosure_v4.existing.macros; numbers only, do not edit"]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v, _ in macros]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_md(macros: list[tuple[str, str, str]], path: Path = MD_PATH) -> Path:
    head = [
        "# Macros from the existing-results re-run (`paper/generated/existing_rerun_macros.tex`)",
        "",
        "Emitter: `python -m trading_on_disclosure_v4.existing.macros`. "
        "Inputs: `data/v4/existing/rerun_tables.csv`, `data/v4/existing/rerun_report.json`, "
        "`data/v4/existing/spread_placebo_size.csv`, `data/v4/existing/event_level_counts.csv`, "
        "`data/v4/existing/table1_dedup.csv`, `data/v4/existing/portfolio_dedup_summary.csv`. Every value is a "
        "number in bp, a count, a t, a p or a rate; no words. Name scheme `\\Ex<Sample><Test><Quantity>` with "
        "Sample `Orig` / `Dedup` / `DedupOfficialOnly` (de-duplicated, official-source records only) / "
        "`DedupExclTwo` (de-duplicated without the two vendor-only households), Test as below, "
        "Quantity `Bps`, `SeDay`, `SeBlock`, `TDay`, `TBlock`, `PDay`, `PBlock`, `CiLowDay`, `CiHighDay`, "
        "`CiLowBlock`, `CiHighBlock`, `N`, `Households`, `Days`, `Blocks`. 'Day' = household x anchor-day "
        "clusters (the existing paper's convention); 'Block' = household x 20-trading-day block of the anchor. "
        "Placebo macros `\\ExPlaceboSpread<Quantity><Day|Block><Timing|Centered>` come from the 1,000-draw "
        "both-direction block-shift placebo set on the MARKET-ADJUSTED outcome (the headline's; Timing = raw "
        "shifted outcomes, Centered = household-period centered); `\\ExPlaceboSpreadFactor<...>` are the same "
        "quantities on the FF5+MOM-adjusted outcome, kept for the CAPM and FF5+MOM spread rows. "
        "`\\ExDedup<Test>Placebo[Factor]P<Timing|Centered>` = the observed de-duplicated spread's two-sided "
        "placebo p-value. `\\ExEventLevel<Quantity>` = counts of the released event-level test, "
        "`\\ExDedupEventLevel<Quantity>` the same on the de-duplicated sample. Final wave (item 3): "
        "`\\Ex<Orig|Dedup>TableOne<Field>` = every Table 1 field on both samples (`table1_dedup.csv`); "
        "`\\Ex<Sample><Test><Quantity>` with Test `SignedPayoff` / `RawRows` / `SourceNativeIds` (clustered "
        "quantities as above), `EqualHousehold` / `EventLevel` (`Bps`, `Se`, `T`, `P`, `CiLow`, `CiHigh`, `N`) and "
        "`PlaceboDirection` / `PlaceboDates` (`Bps`, `P`, `Draws`, `MeanBps`, `SdBps`); `\\Ex<Sample>Port<Field>` = "
        "the funded long-after-purchase portfolio (`EligiblePurchases`, `SignalsAccepted`, `SignalsRejected`, `Days`, "
        "`AnnualizedReturn`, `BenchmarkReturn`, `AvgInvested`, `MaxExposure`, `Turnover`, `Sharpe`, and "
        "`Mean` / `Capm` / `FiveFactorMomentum` x `Pct`, `SEPct`, `T`, `P`, `CiLowPct`, `CiHighPct`, `N`; "
        "annualized, percent, as the released `\\Port...` macros). Appendix wave (`existing/appendix.py`; files in "
        "`data/v4/existing/appendix_dedup/`, named in each meaning): `\\Ex<Sample><Test>BootCi{Low,High}` = the released "
        "20-date moving-block bootstrap interval (4,999 draws, freeze-hash seeds) on both samples; "
        "`\\Ex<Sample><Test>HolmP{Day,Block}` and `\\Ex<Sample>Port<Test>HolmP` = Holm within the released families "
        "(day clusters = the released convention; block clusters; a cluster-free test carries its one p in both), printed at "
        "three decimals like the released tables, with `\\Ex<Sample><Test>P{Day,Block}Three` and `\\Ex<Sample>Port<Test>PThree` "
        "the raw p at the same precision; "
        "`\\ExCompPort<Field>{Released,Dedup}` = the portfolio rows of the two-sample comparison table "
        "(`Positions`, `Rejected`, `EligiblePurchases`, `InvestedFraction`, `MaxExposure`, `Turnover`, `AnnualizedReturn`, "
        "`Benchmark`, `Sharpe`, `Days`, `MeanOpenLots`, `DaysNoPosition`, `DaysCashBound`, `DaysFullyInvested`, `DaysAtLotCap`, "
        "`NwMean[Se|P]`, `CapmAlpha[Se|P]`, `FfFiveMomAlpha[Se|P]`, `SameInvestedDays`); "
        "`\\ExDedupAppAttrition{Mixed,Duplicates,CompleteTwenty}{In,Out,Removed}` = the three signal-level attrition stages "
        "of the de-duplicated stream (the six released record-level stages are identical on both samples); "
        "`\\ExDedupAppAssembly<Field>` = source-assembly counts from the dated frame and the records of the removed signals; "
        "`\\ExDedupAppDateGap<Zero..Seven|Total><Overlap|InSignals|Dedup><N|Share>` (+ `<Set>Mean`, `<Set>Median`, "
        "`RemovedByRule`) = the vendor-minus-official gap on the retained-overlap records, all / parents of a signal / "
        "parents of a de-duplicated non-mixed signal; `\\Ex<Orig|Dedup>AppEventStudy<Cell>` = the disclosure-centered "
        "event study with its simultaneous band (4,999 draws) on both samples (`PurchasePaths`, `SalePaths`, `Draws`, "
        "`Critical`, `DaysExcludingZero`, `DiffMinBps`, `DiffMaxBps`, `DiffAt<Day>Bps`, `BandLowAt<Day>Bps`, "
        "`BandHighAt<Day>Bps` for days MinusTen, Zero, Twenty, Forty, Sixty). `\\ExBootDraws` = 4,999.",
        "",
        "Renames in wave 2 (the factor-adjusted placebo was the only placebo before; its macros now carry the "
        "`Factor` infix and the un-infixed names hold the market-adjusted placebo):",
        "",
        "| Old name | New name | Quantities |", "|---|---|---|",
    ]
    head += [f"| `{a}` | `{b}` | {c} |" for a, b, c in RENAMES]
    head += ["", "| Test key | Meaning |", "|---|---|"]
    head += [f"| `{t}` | {d} (`{k}`) |" for k, (t, d) in {**TEST_NAMES, **UNIT_NAMES}.items()]
    head += ["", "| Macro | Value | Meaning |", "|---|---|---|"]
    rows = [f"| `\\{n}` | {v} | {d} |" for n, v, d in macros]
    path.write_text("\n".join(head + rows) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="emit existing_rerun_macros.tex and the macro list")
    ap.add_argument("--tex", type=Path, default=TEX_PATH)
    ap.add_argument("--md", type=Path, default=MD_PATH)
    args = ap.parse_args(argv)
    macros = collect()
    write_tex(macros, args.tex)
    write_md(macros, args.md)
    print(f"{len(macros)} macros -> {args.tex}, {args.md}")


if __name__ == "__main__":
    main()
