"""Appendix counterparts on the de-duplicated sample.

Every released appendix number that depends on the sample gets its de-duplicated twin here, with the
released executors called by import wherever a released number exists to check against:

* attrition: the released record-level stages are reproduced from the row counts of the frozen frames
  (`outputs/frames/*.csv`, asserted against the canonical ledger) and continued through the three
  signal-level stages the main text now uses (mixed-direction exclusion, the vendor-duplicate rule,
  complete 20-day outcomes);
* source assembly: the retained vendor records split into scan-linked and vendor-only from the dated
  frame, and the records that belong to the signals the rule removes;
* the vendor-minus-official date gap on the retained-overlap records, restricted to the records that
  reach a signal and to the de-duplicated non-mixed signals (the rule removes none of them);
* the 20-date moving-block bootstrap interval of every released spread test (same freeze-hash seeds,
  4,999 draws): the released `calendar_block_bootstrap` statistic is a difference of means, so the
  draws are evaluated from per-date sufficient statistics on the released index matrix
  (`draw_block_indices`), which reproduces the registry intervals to 1e-16 in a fraction of a second
  instead of forty minutes per test;
* the complete results table (all 26 released analyses) on both samples with Holm within the released
  families, for day clusters (the released convention) and 20-day block clusters;
* the disclosure-centered event study with the released simultaneous band (4,999 draws, same seeds);
* the horizon profile, and the portfolio rows of the two-sample comparison (with the share of days at
  the 50-lot cap, the reason the invested fraction hardly moves).

Items that cannot be recomputed on the de-duplicated sample are listed in `not_recomputed.csv` with
the reason; they are all pre-assembly counts that the signal-level rule cannot touch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.inference.block_bootstrap import (
    DEFAULT_FINAL_DRAWS,
    PRIMARY_BLOCK_LENGTH,
    draw_block_indices,
)
from trading_on_disclosure_v2.inference.event_bands import supt_band_day_multiplier
from trading_on_disclosure_v2.inference.multiplicity import holm
from trading_on_disclosure_v2.pipeline import locked_estimands as le
from trading_on_disclosure_v2.pipeline.dryrun_outputs import build_event_time_car
from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL
from trading_on_disclosure_v4.existing import portfolio as pf
from trading_on_disclosure_v4.existing import rerun, units
from trading_on_disclosure_v4.outcomes import panel

BP = rerun.BP
FRAMES = panel.FRAMES
CANONICAL = panel.REPO_ROOT / "outputs" / "canonical"
ATTRITION_PATH = CANONICAL / "attrition_ledger.csv"
FIG3_PATH = CANONICAL / "figures" / "figure3_event_study_path.csv"
OUT_DIR = rerun.OUT_DIR / "appendix_dedup"
OUTCOME = le.PRIMARY_OUTCOME
PARENT_SEP = ";"
EVENT_WINDOW = (-10, 60)
BAND_DRAWS = 4999
BOOT_DRAWS = DEFAULT_FINAL_DRAWS
PORT_TESTS = tuple(pf.REGRESSIONS)
SPREAD_TESTS = tuple(t for t in rerun.TESTS if not t.paired) + (
    rerun.SpreadTest("TW_S1_RAWROWS", OUTCOME, "E"),
    rerun.SpreadTest("TW_S1_SRCID", OUTCOME, "E"),
)
BOOT_PURPOSE = {t.test_id: f"locked|{t.outcome}|{t.anchor}|" + ("source_native_household_id" if t.test_id == "TW_S1_SRCID" else "filer_uid")
                for t in SPREAD_TESTS}
STAGE_FRAMES = [  # released attrition stages -> the frozen frame whose row count is the stage output
    ("public_date_and_sample_window", "windowed_transactions.csv"),
    ("amendment_exact_duplicates", "deduplicated_transactions.csv"),
    ("purchase_sale_filter", "directional_transactions.csv"),
    ("security_matching", "matched_transactions.csv"),
    ("common_stock_screen", "matched_transactions.csv"),
    ("collapse_to_signals", "signals.csv"),
]
SIGNAL_STAGES = [  # (stage, reason column) appended on the de-duplicated stream
    ("mixed_direction_exclusion", "mixed_direction_signal"),
    ("vendor_duplicate_rule", "duplicate_of_official_transaction"),
    ("complete_h20_outcome", "incomplete_h20_outcome"),
]
NOT_RECOMPUTED = [
    ("Source assembly", "Capitol Trades snapshot rows, vendor duplicates of parsed official filings, repeat-scrape duplicates",
     "pre-assembly counts of the frozen snapshot (not in outputs/frames); rule-independent: the duplicate rule acts on signals after assembly"),
    ("Source assembly", "official filing census, image-only filings, amendment lineage counts, later exact cross-document duplicates",
     "official-store counts (not in outputs/frames); rule-independent"),
    ("Capitol Trades publication date minus official filing date", "record-weighted and filing-weighted linked universes",
     "pre-assembly linkage of the snapshot to the official store; rule-independent; the retained-overlap column is recomputed and restricted"),
    ("Sample attrition", "the six released record-level stages",
     "reproduced from the frozen frames' row counts; identical on both samples because the rule acts on signals"),
    ("Analysis and inference map", "every cell", "text; the draw counts (999, 19) are the released ones and are reused"),
]


def _dup_uids_and_frames():
    signals_raw = pd.read_csv(FRAMES / "signals.csv", low_memory=False)
    dup = rerun.duplicate_signal_uids()
    return signals_raw, dup


# ---- counts ---------------------------------------------------------------------------------------------

def released_stage_counts(frames: Path = FRAMES) -> dict[str, int]:
    """Row count of each stage's output frame (the released attrition ledger's n_out)."""
    return {stage: int(len(pd.read_csv(frames / name, usecols=[0]))) for stage, name in STAGE_FRAMES}


def attrition(signals_raw: pd.DataFrame, duplicate_uids: set[str], released: pd.DataFrame,
              stage_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """The released stages (checked against ``stage_counts`` when given) continued through the three
    signal-level stages; reason columns as in the released ledger plus one per new stage."""
    out = released.copy()
    if stage_counts is not None:
        for r in out.itertuples(index=False):
            if int(r.n_out) != stage_counts[r.stage]:
                raise AssertionError(f"{r.stage}: ledger n_out {r.n_out} vs frame rows {stage_counts[r.stage]}")
    mixed = signals_raw["mixed_direction"].astype(bool)
    prim = signals_raw[~mixed]
    is_dup = prim["signal_uid"].isin(duplicate_uids)
    dedup = prim[~is_dup]
    h20 = dedup.dropna(subset=[OUTCOME])
    steps = [(len(signals_raw), len(prim)), (len(prim), len(dedup)), (len(dedup), len(h20))]
    if int(out["n_out"].iloc[-1]) != len(signals_raw):
        raise AssertionError("signal frame does not continue the released ledger")
    rows = []
    for (stage, reason), (n_in, n_out) in zip(SIGNAL_STAGES, steps, strict=True):
        rows.append({"stage": stage, "n_in": int(n_in), "n_out": int(n_out), reason: float(n_in - n_out)})
    out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    reason_cols = [c for c in out.columns if c not in ("stage", "n_in", "n_out")]
    out[reason_cols] = out[reason_cols].fillna(0.0)
    out["sample"] = ["both"] * len(released) + ["dedup"] * len(rows)
    return out


def parent_records(signals_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (signal_uid, transaction_uid) from `parent_transaction_uids`."""
    ex = signals_raw[["signal_uid", "mixed_direction", "parent_transaction_uids"]].copy()
    ex["transaction_uid"] = ex.pop("parent_transaction_uids").str.split(PARENT_SEP)
    return ex.explode("transaction_uid").reset_index(drop=True)


def _overlap(dated: pd.DataFrame) -> pd.DataFrame:
    """Retained vendor records carrying both an official public date and the raw vendor date."""
    v = dated[dated["record_source"] != "official"]
    return v[v["official_public_date"].notna() & v["capitol_trades_publication_date_raw"].notna()]


def source_assembly(dated: pd.DataFrame, signals_raw: pd.DataFrame, duplicate_uids: set[str]) -> pd.DataFrame:
    """Record-level rows of the source-assembly table from the dated frame, plus the records that
    belong to the signals the rule removes (long: field, value, description)."""
    vendor = dated[dated["record_source"] != "official"]
    overlap_ids = set(_overlap(dated)["transaction_uid"])
    par = parent_records(signals_raw)
    removed = par[par["signal_uid"].isin(duplicate_uids)]
    removed_ids = set(removed["transaction_uid"])
    rows = [
        ("official_records", int((dated["record_source"] == "official").sum()), "official-source transaction records"),
        ("vendor_records", int(len(vendor)), "retained Capitol Trades records"),
        ("scan_linked_records", int(len(overlap_ids)), "retained records linked to unparsed official filings (both dates observed)"),
        ("vendor_only_records", int(len(vendor) - len(overlap_ids)), "retained vendor-only records (vendor-date proxy)"),
        ("assembled_records", int(len(dated)), "assembled transaction records"),
        ("removed_signals", int(removed["signal_uid"].nunique()), "non-mixed signals removed by the duplicate rule"),
        ("removed_signal_records", int(len(removed_ids)), "retained vendor records that belong to a removed signal"),
        ("removed_signal_records_scan_linked", int(len(removed_ids & overlap_ids)), "of which linked to unparsed official filings"),
        ("removed_signal_records_vendor_only", int(len(removed_ids - overlap_ids)), "of which vendor-only"),
        ("vendor_records_after_rule", int(len(vendor) - len(removed_ids)), "retained Capitol Trades records not in a removed signal"),
    ]
    if not removed_ids <= set(vendor["transaction_uid"]):
        raise AssertionError("a removed signal has a non-vendor parent record")
    return pd.DataFrame(rows, columns=["field", "value", "description"])


def date_gap(dated: pd.DataFrame, signals_raw: pd.DataFrame, duplicate_uids: set[str], max_gap: int = 7) -> pd.DataFrame:
    """Vendor-minus-official calendar-day gap on the retained-overlap records: all of them (the
    released column), those that are parents of a signal, and those that are parents of a
    de-duplicated non-mixed signal."""
    ov = _overlap(dated).copy()
    ov["gap_days"] = (pd.to_datetime(ov["capitol_trades_publication_date_raw"]) - pd.to_datetime(ov["official_public_date"])).dt.days
    par = parent_records(signals_raw)
    in_signal = set(par["transaction_uid"])
    kept = par[(~par["mixed_direction"].astype(bool)) & (~par["signal_uid"].isin(duplicate_uids))]
    removed = set(par.loc[par["signal_uid"].isin(duplicate_uids), "transaction_uid"])
    sets = {"retained_overlap": ov, "in_signals": ov[ov["transaction_uid"].isin(in_signal)],
            "dedup_nonmixed_signals": ov[ov["transaction_uid"].isin(set(kept["transaction_uid"]))]}
    lo = min(0, int(ov["gap_days"].min()))
    rows = [{"gap_days": g} for g in range(lo, max_gap + 1)] + [{"gap_days": "total"}]
    for name, frame in sets.items():
        counts = frame["gap_days"].value_counts()
        n = len(frame)
        for r in rows:
            c = n if r["gap_days"] == "total" else int(counts.get(r["gap_days"], 0))
            r[f"{name}_n"], r[f"{name}_share"] = c, (c / n if n else np.nan)
    out = pd.DataFrame(rows)
    out.attrs["removed_by_rule"] = int(ov["transaction_uid"].isin(removed).sum())
    out.attrs["summary"] = {name: {"n": int(len(f)), "mean_gap_days": float(f["gap_days"].mean()),
                                   "median_gap_days": float(f["gap_days"].median()),
                                   "share_gap_one": float((f["gap_days"] == 1).mean())} for name, f in sets.items()}
    return out


# ---- block bootstrap ------------------------------------------------------------------------------------

def spread_block_bootstrap(data: pd.DataFrame, outcome: str, entry_col: str, freeze_hash: str, purpose: str,
                           n_draws: int = BOOT_DRAWS, weight_col: str | None = None,
                           block_length: int = PRIMARY_BLOCK_LENGTH) -> tuple[float, float, np.ndarray]:
    """`calendar_block_bootstrap` of the purchase-minus-sale spread on the released index matrix,
    evaluated from per-date sums: (2.5%, 97.5% quantiles, the draw statistics)."""
    dates = pd.DatetimeIndex(pd.unique(data[entry_col])).sort_values()
    idx = draw_block_indices(len(dates), n_draws, block_length, freeze_hash, purpose, "moving_block")
    pos = dates.get_indexer(pd.DatetimeIndex(data[entry_col]))
    y = data[outcome].to_numpy(float)
    w = np.ones(len(y)) if weight_col is None else data[weight_col].fillna(1).astype(int).clip(lower=1).to_numpy(float)
    buy = (data["direction"] == "purchase").to_numpy()
    D = len(dates)
    sums = [np.bincount(pos[m], weights=v[m], minlength=D) for m in (buy, ~buy) for v in (y * w, w)]
    sb, nb, ss, ns = (s[idx].sum(axis=1) for s in sums)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.where((nb > 0) & (ns > 0), sb / nb - ss / ns, np.nan)
    lo, hi = np.nanquantile(stat, [0.025, 0.975])
    return float(lo), float(hi), stat


def block_intervals(frames: dict[str, pd.DataFrame], freeze_hash: str, n_draws: int = BOOT_DRAWS,
                    tests: tuple[rerun.SpreadTest, ...] = SPREAD_TESTS) -> pd.DataFrame:
    """The moving-block interval of every spread test on every frame (bp)."""
    rows = []
    for test in tests:
        for sample, frame in frames.items():
            data = frame.dropna(subset=[test.outcome, test.entry_col])
            if test.mask is not None:
                data = data[le._segment_masks(data)[test.mask]]
            lo, hi, stat = spread_block_bootstrap(data, test.outcome, test.entry_col, freeze_hash, BOOT_PURPOSE[test.test_id],
                                                  n_draws, "lot_count" if test.test_id == "TW_S1_RAWROWS" else None)
            rows.append({"test_id": test.test_id, "sample": sample, "boot_ci_low": lo * BP, "boot_ci_high": hi * BP,
                         "boot_sd": float(np.nanstd(stat, ddof=1)) * BP, "n_draws": int(n_draws),
                         "n_dates": int(data[test.entry_col].nunique()), "block_length": PRIMARY_BLOCK_LENGTH, "units": "bp"})
    return pd.DataFrame(rows)


def check_block_intervals(boot: pd.DataFrame, registry_path: Path = rerun.REGISTRY_PATH, tol: float = 1e-10) -> float:
    """Original-sample intervals against the registry's `block_ci_*` (decimal units)."""
    reg = pd.read_csv(registry_path).set_index("test_id")
    sub = boot[boot["sample"] == "original"].set_index("test_id")
    d = np.maximum((sub["boot_ci_low"] / BP - reg.loc[sub.index, "block_ci_low"]).abs(),
                   (sub["boot_ci_high"] / BP - reg.loc[sub.index, "block_ci_high"]).abs())
    if not (d < tol).all():
        raise AssertionError(f"block intervals not reproduced: {d[~(d < tol)].to_dict()}")
    return float(d.max())


# ---- full results ---------------------------------------------------------------------------------------

def full_results(tables: pd.DataFrame, portfolio: pd.DataFrame, boot: pd.DataFrame, registry: pd.DataFrame,
                 samples: tuple[str, ...] = rerun.SAMPLES) -> pd.DataFrame:
    """Every released analysis on every sample: the day-cluster (released convention) and block-cluster
    quantities, the moving-block interval, and Holm within the released families for both p columns
    (a cluster-free test carries its one p in both columns). Signal rows in bp; portfolio rows in
    annual percent."""
    fam = registry.set_index("test_id")[["family", "inference_eligible"]]
    port = portfolio.set_index("sample")
    rows = []
    for sample in samples:
        for test_id in registry["test_id"]:
            r = {"test_id": test_id, "sample": sample, "family": fam.loc[test_id, "family"],
                 "inference_eligible": fam.loc[test_id, "inference_eligible"]}
            if test_id in PORT_TESTS:
                g = {k: port.loc[sample, f"{test_id}.{k}"] for k in pf.REG_FIELDS}
                r.update({"estimate": 100 * g["estimate"], "se_day": 100 * g["se_multiway"], "ci_low_day": 100 * g["ci_low"],
                          "ci_high_day": 100 * g["ci_high"], "p_day": g["raw_p"], "n": int(g["n"]), "units": "pct_annual"})
            else:
                sub = tables[(tables["test_id"] == test_id) & (tables["sample"] == sample)].set_index("clusters")
                if sub.empty:
                    raise KeyError(f"{test_id} missing on sample {sample}")
                day = sub.loc["day"] if "day" in sub.index else sub.iloc[0]
                r.update({"estimate": day["estimate"], "se_day": day["se"], "ci_low_day": day["ci_low"], "ci_high_day": day["ci_high"],
                          "p_day": day["p"], "n": int(day["n"]), "n_households": day["n_households"], "n_dates": day["n_dates_or_blocks"],
                          "dof_day": day["dof"], "placebo_mean": day.get("placebo_mean", np.nan), "placebo_sd": day.get("placebo_sd", np.nan),
                          "n_draws": day.get("n_draws", np.nan), "units": "bp", "clusters_day": day.name})
                if "block20" in sub.index:
                    blk = sub.loc["block20"]
                    r.update({"se_block": blk["se"], "ci_low_block": blk["ci_low"], "ci_high_block": blk["ci_high"], "p_block": blk["p"],
                              "n_blocks": blk["n_dates_or_blocks"], "dof_block": blk["dof"]})
                elif "day" not in sub.index:  # cluster-free test (household t, Welch, placebo): one set of numbers for both conventions
                    r.update({"se_block": day["se"], "ci_low_block": day["ci_low"], "ci_high_block": day["ci_high"], "p_block": day["p"],
                              "dof_block": day["dof"]})
                b = boot[(boot["test_id"] == test_id) & (boot["sample"] == sample)]
                if len(b):
                    r.update({"boot_ci_low": float(b["boot_ci_low"].iloc[0]), "boot_ci_high": float(b["boot_ci_high"].iloc[0])})
            rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm_day"], out["p_holm_block"] = np.nan, np.nan
    eligible = out["inference_eligible"].astype(str).str.lower().eq("yes")
    for (_, _), idx in out[eligible].groupby(["sample", "family"]).groups.items():
        for col in ("p_day", "p_block"):
            p = out.loc[idx, col].to_numpy(float)
            if np.isfinite(p).all():
                out.loc[idx, f"p_holm_{col[2:]}"] = holm(p)
    cols = ["test_id", "sample", "family", "inference_eligible", "estimate", "se_day", "ci_low_day", "ci_high_day", "p_day", "p_holm_day",
            "se_block", "ci_low_block", "ci_high_block", "p_block", "p_holm_block", "boot_ci_low", "boot_ci_high", "n", "n_households",
            "n_dates", "n_blocks", "dof_day", "dof_block", "placebo_mean", "placebo_sd", "n_draws", "units"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def check_full_results(full: pd.DataFrame, registry: pd.DataFrame, tol: float = 1e-10) -> float:
    """Original-sample day-cluster rows (estimate, raw p, Holm p, n) against the registry."""
    reg = registry.set_index("test_id")
    sub = full[full["sample"] == "original"].set_index("test_id")
    worst = 0.0
    for tid, r in sub.iterrows():
        scale = 100.0 if r["units"] == "pct_annual" else BP
        diffs = [abs(r["estimate"] / scale - reg.loc[tid, "estimate"]), abs(r["p_day"] - reg.loc[tid, "raw_p"])]
        if pd.notna(reg.loc[tid, "p_holm"]):
            diffs.append(abs(r["p_holm_day"] - reg.loc[tid, "p_holm"]))
        elif pd.notna(r["p_holm_day"]):
            raise AssertionError(f"{tid}: Holm p emitted where the registry has none")
        if tid != "TW_S1_EVENT" and int(r["n"]) != int(reg.loc[tid, "n"]):
            raise AssertionError(f"{tid}: n {r['n']} vs registry {reg.loc[tid, 'n']}")
        worst = max(worst, max(diffs))
    if not worst < tol:
        raise AssertionError(f"full-results reproduction failed: max diff {worst}")
    return worst


# ---- event study, horizons, comparison ------------------------------------------------------------------

def event_study(signals: pd.DataFrame, daily: pd.DataFrame, benchmark: pd.Series, calendar: pd.DatetimeIndex,
                freeze_hash: str, n_draws: int = BAND_DRAWS, window: tuple[int, int] = EVENT_WINDOW) -> tuple[pd.DataFrame, dict]:
    """The released disclosure-centered paths and simultaneous bands (`_s15_event_bands` by import)."""
    car, keys, direction = build_event_time_car(signals.dropna(subset=[FOLLOWER_ENTRY_COL]), daily, benchmark, calendar,
                                                pre=abs(window[0]), post=window[1])
    parts, meta = [], {"purchase_paths": int((direction == "purchase").sum()), "sale_paths": int((direction == "sale").sum()),
                       "n_draws": int(n_draws)}
    for series in ("purchase", "sale", "difference"):
        b = supt_band_day_multiplier(car, keys, direction, series, freeze_hash, purpose=f"locked_event_band|{series}",
                                     n_draws=n_draws, bandwidth=PRIMARY_BLOCK_LENGTH, centering="null_imposed", trading_days=calendar)
        parts.append(pd.DataFrame({"series": series, "relative_trading_day": np.asarray(b.ks, dtype=int), "estimate": b.estimate,
                                   "se": b.se, "lower": b.lower, "upper": b.upper}))
        meta[f"{series}_critical"] = float(b.critical)
        meta[f"{series}_days_band_excludes_zero"] = int(((b.lower > 0) | (b.upper < 0)).sum())
    return pd.concat(parts, ignore_index=True), meta


def check_event_study(path: pd.DataFrame, released_path: Path = FIG3_PATH, tol: float = 1e-10) -> float:
    rel = pd.read_csv(released_path)
    key = ["series", "relative_trading_day"]
    m = path.merge(rel, on=key, suffixes=("", "_rel"))
    if len(m) != len(rel) or len(m) != len(path):
        raise AssertionError("event-study rows differ from the released path")
    d = max(float((m[c] - m[f"{c}_rel"]).abs().max()) for c in ("estimate", "se", "lower", "upper"))
    if not d < tol:
        raise AssertionError(f"event-study path not reproduced: {d}")
    return d


def horizons(full: pd.DataFrame) -> pd.DataFrame:
    """The horizon profile (Figure 'horizons'): both samples, day and block intervals, bp."""
    ids = {5: "TW_P2_E5", 10: "TW_P2_E10", 20: "TW_P1_E20", 40: "TW_P2_E40", 60: "TW_P2_E60", 127: "TW_P2_E127"}
    sub = full.set_index(["sample", "test_id"])
    rows = []
    for sample in full["sample"].unique():
        for h, tid in ids.items():
            r = sub.loc[(sample, tid)]
            rows.append({"sample": sample, "horizon": h, "test_id": tid, **{c: r[c] for c in (
                "estimate", "se_day", "ci_low_day", "ci_high_day", "p_day", "p_holm_day", "se_block", "ci_low_block", "ci_high_block",
                "p_block", "p_holm_block", "boot_ci_low", "boot_ci_high", "n")}, "units": "bp"})
    return pd.DataFrame(rows)


COMPARISON_FIELDS = [  # (field, source column or (test, field), kind, description)
    ("EligiblePurchases", "n_eligible_purchases", "int", "eligible purchase signals offered to the account"),
    ("Positions", "signals_accepted", "int", "positions opened"),
    ("Rejected", "signals_rejected", "int", "candidate signals not admitted"),
    ("InvestedFraction", "avg_invested_fraction", "pct", "average invested fraction"),
    ("MaxExposure", "max_security_exposure", "pct", "maximum realized single-security exposure"),
    ("Turnover", "annual_turnover", "pct", "annualized one-way turnover"),
    ("AnnualizedReturn", "annualized_return", "pct", "annualized total-account return"),
    ("Benchmark", "benchmark_annualized_return", "pct", "exposure-matched benchmark return, annualized"),
    ("Sharpe", "sharpe", "num", "annualized Sharpe ratio"),
    ("RejectedLotCap", "daily.rejected_max_open_lots", "int", "candidate signals not admitted because 50 positions were open"),
    ("RejectedCashOrCap", "daily.rejected_cash_or_cap", "int", "candidate signals not admitted for want of cash or by the 5% security cap"),
    ("RejectedTiming", "daily.rejected_timing", "int", "candidate signals not admitted because no entry price was available"),
    ("Days", "n_days", "int", "trading days in the account path"),
    ("MeanOpenLots", "daily.mean_open_lots", "num", "mean number of open positions per trading day"),
    ("DaysNoPosition", "daily.share_days_no_position", "pct", "share of trading days with no open position"),
    ("DaysCashBound", "daily.share_days_cash_bound", "pct", "share of trading days with at least 1 / (target fraction) = 20 open positions, where the no-leverage rule leaves no cash"),
    ("DaysFullyInvested", "daily.share_days_fully_invested", "pct", "share of trading days with at least 99% of NAV invested"),
    ("DaysAtLotCap", "daily.share_days_at_lot_cap", "pct", "share of trading days with 50 open positions (the lot cap binding)"),
    ("NwMean", ("TW_IMPL_PORT_NW", "estimate"), "pct", "mean excess return, annualized (Newey-West lag 20)"),
    ("NwMeanSe", ("TW_IMPL_PORT_NW", "se_multiway"), "pct", "its standard error"),
    ("NwMeanP", ("TW_IMPL_PORT_NW", "raw_p"), "p", "its two-sided p"),
    ("CapmAlpha", ("TW_IMPL_PORT_CAPM", "estimate"), "pct", "CAPM alpha, annualized"),
    ("CapmAlphaSe", ("TW_IMPL_PORT_CAPM", "se_multiway"), "pct", "its standard error"),
    ("CapmAlphaP", ("TW_IMPL_PORT_CAPM", "raw_p"), "p", "its two-sided p"),
    ("FfFiveMomAlpha", ("TW_IMPL_PORT_FF5MOM", "estimate"), "pct", "FF5+MOM alpha, annualized"),
    ("FfFiveMomAlphaSe", ("TW_IMPL_PORT_FF5MOM", "se_multiway"), "pct", "its standard error"),
    ("FfFiveMomAlphaP", ("TW_IMPL_PORT_FF5MOM", "raw_p"), "p", "its two-sided p"),
]


ROUTING_STATES = {"max_open_lots_constraint": "rejected_max_open_lots", "cash_or_cap_constraint": "rejected_cash_or_cap",
                  "timing_unavailable": "rejected_timing"}


def daily_stats(daily: pd.DataFrame, params: pf.PortfolioParams, routing: pd.DataFrame | None = None) -> dict[str, float]:
    """Why the invested fraction hardly moves: the days on which the constraints bind, and (with the
    constructor's routing frame) the rejections by constraint."""
    lots = daily["n_open_lots"]
    out = {"mean_open_lots": float(lots.mean()), "share_days_no_position": float((lots == 0).mean()),
           "share_days_cash_bound": float((lots >= int(round(1 / params.signal_target_fraction))).mean()),
           "share_days_fully_invested": float((daily["risky_fraction"] >= 0.99).mean()),
           "share_days_at_lot_cap": float((lots >= params.max_open_lots).mean())}
    if routing is not None:
        counts = routing["state"].value_counts()
        out.update({key: int(counts.get(state, 0)) for state, key in ROUTING_STATES.items()})
        if sum(out[k] for k in ROUTING_STATES.values()) + int(counts.get("accepted", 0)) != len(routing):
            raise ValueError("unrecognised routing state")
    return out


def same_invested_share(released_daily: pd.DataFrame, dedup_daily: pd.DataFrame, tol: float = 1e-6) -> float:
    """Share of trading days on which the two account paths hold the same invested fraction."""
    if len(released_daily) != len(dedup_daily):
        raise ValueError("account paths differ in length")
    return float((np.abs(released_daily["risky_fraction"].to_numpy() - dedup_daily["risky_fraction"].to_numpy()) < tol).mean())


def comparison_portfolio(summary: pd.DataFrame, daily: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Portfolio rows of the two-sample comparison table (long: field, released, dedup, kind, description)."""
    s = summary.set_index("sample")
    rows = []
    for field, src, kind, desc in COMPARISON_FIELDS:
        vals = {}
        for sample in ("original", "dedup"):
            if isinstance(src, str) and src.startswith("daily."):
                vals[sample] = daily[sample][src[6:]]
            elif isinstance(src, tuple):
                vals[sample] = float(s.loc[sample, f"{src[0]}.{src[1]}"])
            else:
                vals[sample] = float(s.loc[sample, src])
        rows.append({"field": field, "released": vals["original"], "dedup": vals["dedup"], "kind": kind, "description": desc})
    return pd.DataFrame(rows)


# ---- driver ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="appendix counterparts on the de-duplicated sample")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--boot-draws", type=int, default=BOOT_DRAWS)
    ap.add_argument("--band-draws", type=int, default=BAND_DRAWS)
    ap.add_argument("--skip-event-study", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"boot_draws": args.boot_draws, "band_draws": args.band_draws}

    signals_raw, dup = _dup_uids_and_frames()
    signals = rerun.load_signals()
    dedup, rep = rerun.dedup_filter(signals, dup)
    report.update(rep)
    dated = pd.read_csv(FRAMES / "dated_transactions.csv", low_memory=False)
    registry = pd.read_csv(rerun.REGISTRY_PATH)
    fh = units.freeze_hash()

    att = attrition(signals_raw, dup, pd.read_csv(ATTRITION_PATH), released_stage_counts())
    att.to_csv(args.out / "attrition.csv", index=False)
    source_assembly(dated, signals_raw, dup).to_csv(args.out / "source_assembly.csv", index=False)
    gap = date_gap(dated, signals_raw, dup)
    gap.to_csv(args.out / "date_gap.csv", index=False)
    report["date_gap"] = {"removed_by_rule": gap.attrs["removed_by_rule"], **gap.attrs["summary"]}
    print(f"counts written, {time.time() - t0:.0f} s", flush=True)

    boot = block_intervals({"original": signals, "dedup": dedup}, fh, args.boot_draws)
    report["max_abs_block_interval_diff_decimal"] = check_block_intervals(boot)
    boot.to_csv(args.out / "block_intervals.csv", index=False)
    print(f"block intervals written ({report['max_abs_block_interval_diff_decimal']:.2e}), {time.time() - t0:.0f} s", flush=True)

    tables = pd.read_csv(rerun.TABLES_PATH)
    port = pd.read_csv(pf.SUMMARY_PATH)
    full = full_results(tables, port, boot, registry)
    report["max_abs_full_results_diff"] = check_full_results(full, registry)
    full.to_csv(args.out / "full_results.csv", index=False)
    horizons(full).to_csv(args.out / "horizons.csv", index=False)
    params, paths = pf.released_params(), {"original": pd.read_csv(pf.RELEASED_DAILY_PATH), "dedup": pd.read_csv(pf.DAILY_PATH)}
    market = pf.load_market()
    routing = {k: pf.simulate(f, market, params).routing for k, f in (("original", signals), ("dedup", dedup))}
    stats = {k: daily_stats(v, params, routing[k]) for k, v in paths.items()}
    report["portfolio_daily"] = {**stats, "share_days_same_invested_fraction": same_invested_share(paths["original"], paths["dedup"])}
    comp = comparison_portfolio(port, stats)
    comp.loc[len(comp)] = {"field": "SameInvestedDays", "released": report["portfolio_daily"]["share_days_same_invested_fraction"],
                           "dedup": report["portfolio_daily"]["share_days_same_invested_fraction"], "kind": "pct",
                           "description": "share of trading days on which the two account paths hold the same invested fraction"}
    comp.to_csv(args.out / "comparison_portfolio.csv", index=False)
    pd.DataFrame(NOT_RECOMPUTED, columns=["table", "item", "reason"]).to_csv(args.out / "not_recomputed.csv", index=False)
    print(f"full results written, {time.time() - t0:.0f} s", flush=True)

    if not args.skip_event_study:
        daily, bench, cal = panel.load_local_daily(), panel.load_benchmark(), panel.v2_trading_calendar()
        parts = []
        for sample, frame in (("original", signals), ("dedup", dedup)):
            path, meta = event_study(frame, daily, bench, cal, fh, args.band_draws)
            if sample == "original" and args.band_draws == BAND_DRAWS:
                report["max_abs_event_study_diff"] = check_event_study(path)
            report[f"event_study_{sample}"] = meta
            parts.append(path.assign(sample=sample))
        pd.concat(parts, ignore_index=True).to_csv(args.out / "event_study_path.csv", index=False)
        print(f"event study written, {time.time() - t0:.0f} s", flush=True)

    report["seconds"] = round(time.time() - t0, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
