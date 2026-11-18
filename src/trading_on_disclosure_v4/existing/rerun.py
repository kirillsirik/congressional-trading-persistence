"""Re-run of the existing paper's spread tables.

Every test is the paper's own estimand on the paper's own viewed outcomes
(`outputs/frames/signals.csv`): OLS of a signal outcome on a purchase dummy,
two-way Cameron-Gelbach-Miller covariance by household x anchor cluster, t(dof)
with dof = min cluster count - 1 (`locked_estimands.two_way_spread`). Four
columns per test: original / de-duplicated sample x day / 20-trading-day-block
clusters. Day clusters on the original sample reproduce
`outputs/canonical/result_registry.csv` (asserted in tests).

Block = position of the test's own anchor date in the paper's trading calendar,
integer-divided by 20 (the rule of `calibration.run_common.add_block_clusters`).
De-duplicated sample = signals whose public-clock event carries
`dup_primary_all_lines` (every line duplicates an official transaction) are
dropped from every test, including the transaction-clock ones. Two further
samples of the primary test only: the official-source segment of the
de-duplicated sample, and the de-duplicated sample without the two vendor-only
households (the two households of the `vendor_without_official_2023plus` class
with the most vendor lines; ids are built inside code and never written).
Event-level counts: the released `TW_S1_EVENT` collapse of the signals frame to
(permno, follower_entry_date) events, counted here from the signals file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v2.inference.cgm3 import cgm_cluster_cov
from trading_on_disclosure_v2.pipeline.locked_estimands import (
    ENTRY_COLS,
    _segment_masks,
    primary_sample,
)
from trading_on_disclosure_v4.data.calendar import load_calendar
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import panel

BP = 1e4
BLOCK_L = 20
HOUSEHOLD = "filer_uid"
SIGNALS_PATH = panel.FRAMES / "signals.csv"
REGISTRY_PATH = panel.REPO_ROOT / "outputs" / "canonical" / "result_registry.csv"
OUT_DIR = panel.REPO_ROOT / "data" / "v4" / "existing"
TABLES_PATH = OUT_DIR / "rerun_tables.csv"
EVENT_COUNTS_PATH = OUT_DIR / "event_level_counts.csv"
SAMPLES = ("original", "dedup")
EXTRA_SAMPLES = ("dedup_official_only", "dedup_excl_two_vendor_households")
PRIMARY_TEST_ID = "TW_P1_E20"
CLUSTERS = ("day", "block20")
COLUMNS = ["test_id", "sample", "clusters", "estimate", "se", "t", "p", "ci_low", "ci_high",
           "n", "n_households", "n_dates_or_blocks", "dof", "psd_corrected", "units"]


@dataclass(frozen=True)
class SpreadTest:
    test_id: str
    outcome: str
    anchor: str  # key of ENTRY_COLS
    mask: str | None = None  # key of _segment_masks
    paired: bool = False  # intercept-only mean of ar_T_h20 - ar_E_h20

    @property
    def entry_col(self) -> str:
        return ENTRY_COLS[self.anchor]


TESTS: tuple[SpreadTest, ...] = (
    SpreadTest("TW_P1_E20", "ar_E_h20", "E"),
    SpreadTest("TW_P2_E5", "ar_E_h5", "E"),
    SpreadTest("TW_P2_E10", "ar_E_h10", "E"),
    SpreadTest("TW_P2_E40", "ar_E_h40", "E"),
    SpreadTest("TW_P2_E60", "ar_E_h60", "E"),
    SpreadTest("TW_P2_E127", "ar_E_h127", "E"),
    SpreadTest("TW_P3_RAW_CT", "ar_ERAW_h20", "ERAW"),
    SpreadTest("TW_P3_OFFICIAL_ONLY", "ar_E_h20", "E", mask="official_date_only"),
    SpreadTest("TW_P3_OFFICIAL_SEGMENT", "ar_E_h20", "E", mask="official_segment"),
    SpreadTest("TW_P3_VENDOR_SEGMENT", "ar_E_h20", "E", mask="vendor_segment"),
    SpreadTest("TW_P3_ENTRY_PLUS1", "ar_EPLUS1_h20", "EPLUS1"),
    SpreadTest("TW_C1_T20", "ar_T_h20", "T"),
    SpreadTest("TW_C1_TRAW20", "ar_TRAW_h20", "TRAW"),
    SpreadTest("TW_C1_PAIRED", "paired_diff", "E", paired=True),
    SpreadTest("TW_S2_CAPM", "car_capm_h20", "E"),
    SpreadTest("TW_S2_FF5MOM", "car_ff5mom_h20", "E"),
)
TEST_IDS = tuple(t.test_id for t in TESTS)


def load_signals(path: Path = SIGNALS_PATH) -> pd.DataFrame:
    """The paper's non-mixed signals with parsed entry dates and the paired-clock difference."""
    s = pd.read_csv(path, low_memory=False)
    prim = primary_sample(s).copy()
    for col in set(ENTRY_COLS.values()):
        prim[col] = pd.to_datetime(prim[col])
    prim["paired_diff"] = prim["ar_T_h20"] - prim["ar_E_h20"]
    return prim.reset_index(drop=True)


def duplicate_signal_uids(flags_path: Path = ef.FLAGS_PATH,
                          events_path: Path = ef.EVENTS_DIR / "events_P.parquet") -> set[str]:
    """signal_uids of public-clock events whose every line is a vendor duplicate."""
    flags = pd.read_parquet(flags_path, columns=["event_id", "clock", "dup_primary_all_lines"])
    flagged = flags.loc[(flags["clock"] == "P") & flags["dup_primary_all_lines"], "event_id"]
    events = pd.read_parquet(events_path, columns=["event_id", "signal_uid"])
    return set(events.loc[events["event_id"].isin(flagged), "signal_uid"])


def dedup_filter(signals: pd.DataFrame, duplicate_uids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Drop the flagged signals; count-only report."""
    is_dup = signals["signal_uid"].isin(duplicate_uids)
    out = signals.loc[~is_dup].reset_index(drop=True)
    return out, {"n_in": int(len(signals)), "n_dropped": int(is_dup.sum()), "n_out": int(len(out))}


def two_vendor_household_ids(matched: pd.DataFrame | None = None) -> frozenset:
    """The two `vendor_without_official_2023plus` households with the most vendor lines (ids stay in
    memory; callers use them only to build masks)."""
    from trading_on_disclosure_v4.data import coverage as cv
    from trading_on_disclosure_v4.data import load

    matched = load.load_matched_transactions() if matched is None else matched
    classes = cv.classify_households(matched)
    blind = classes.index[classes == cv.CLASS_WITHOUT]
    vendor = matched[(matched["record_source"] == "capitol_trades") & matched["filer_uid"].isin(blind)]
    return frozenset(vendor.groupby("filer_uid").size().nlargest(2).index)


def extra_frames(dedup: pd.DataFrame, two_households: frozenset) -> dict[str, pd.DataFrame]:
    """The two item-4 samples of the de-duplicated frame."""
    return {"dedup_official_only": dedup[dedup["record_source"] == "official"].reset_index(drop=True),
            "dedup_excl_two_vendor_households": dedup[~dedup[HOUSEHOLD].isin(two_households)].reset_index(drop=True)}


def event_level_counts(signals_raw: pd.DataFrame) -> dict:
    """`TW_S1_EVENT` counts: every signal (mixed ones included, by lot direction) with a 20-day
    public-clock outcome, collapsed to (permno, follower_entry_date) as `build_security_entry_events`
    does; purchase-only / sale-only / mixed events by distinct households per direction."""
    from trading_on_disclosure_v2.signals.build import build_security_entry_events

    ev = build_security_entry_events(signals_raw.dropna(subset=["ar_E_h20"]))
    buy, sell = ev["purchase_household_count"] > 0, ev["sale_household_count"] > 0
    out = {"purchase_only_events": int((buy & ~sell).sum()), "sale_only_events": int((sell & ~buy).sum()),
           "mixed_events_excluded": int((buy & sell).sum()), "all_outcome_bearing_events": int(len(ev))}
    out["events_in_estimate"] = out["purchase_only_events"] + out["sale_only_events"]
    out["signals_with_outcome"] = int(signals_raw["ar_E_h20"].notna().sum())
    return out


def block_of(dates: pd.Series, calendar: pd.DatetimeIndex, L: int = BLOCK_L) -> np.ndarray:
    """Calendar position // L for every date (raises when a date is not a session)."""
    idx = calendar.get_indexer(pd.DatetimeIndex(pd.to_datetime(dates)))
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} dates are not trading sessions")
    return idx // L


def clustered_mean_or_spread(
    y: np.ndarray, purchase: np.ndarray | None, household: np.ndarray, date_cluster: np.ndarray,
    cluster_name: str, household_name: str = HOUSEHOLD,
) -> dict:
    """Two-way CGM (household x date_cluster) on the purchase-dummy slope, or on the mean when
    ``purchase`` is None; Act I conventions (t with dof = min cluster count - 1).
    ``household_name`` labels the household dimension (source-native ids in the S1 row)."""
    X = np.ones((len(y), 1)) if purchase is None else np.column_stack([np.ones(len(y)), purchase])
    j = X.shape[1] - 1
    clusters = pd.DataFrame({household_name: household, cluster_name: date_cluster})
    res = cgm_cluster_cov(y, X, clusters)
    est, se = float(res.beta[j]), float(np.sqrt(res.cov[j, j]))
    dof = max(int(res.dof), 1)
    t = est / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=dof)) if np.isfinite(t) else np.nan
    crit = float(stats.t.ppf(0.975, df=dof))
    return {"estimate": est, "se": se, "t": float(t) if np.isfinite(t) else np.nan, "p": p,
            "ci_low": est - crit * se, "ci_high": est + crit * se, "n": int(len(y)),
            "n_households": int(res.cluster_counts[household_name]),
            "n_dates_or_blocks": int(res.cluster_counts[cluster_name]), "dof": dof,
            "psd_corrected": bool(res.psd_corrected)}


def run_test(frame: pd.DataFrame, test: SpreadTest, clusters: str, calendar: pd.DatetimeIndex) -> dict:
    """One test on one frame with day or block clusters; decimal units (registry convention)."""
    data = frame.dropna(subset=[test.outcome, test.entry_col])
    if test.mask is not None:
        data = data[_segment_masks(data)[test.mask]]
    if clusters == "day":
        cl, name = data[test.entry_col].to_numpy(), test.entry_col
    elif clusters == "block20":
        cl, name = block_of(data[test.entry_col], calendar), f"{test.entry_col}_blk{BLOCK_L}"
    else:
        raise ValueError(f"unknown clusters {clusters!r}")
    purchase = None if test.paired else (data["direction"] == "purchase").to_numpy(float)
    return clustered_mean_or_spread(data[test.outcome].to_numpy(float), purchase,
                                    data[HOUSEHOLD].to_numpy(), cl, name)


def _rows(frames: dict[str, pd.DataFrame], tests: tuple[SpreadTest, ...], calendar: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for test in tests:
        for sample, frame in frames.items():
            for clusters in CLUSTERS:
                r = run_test(frame, test, clusters, calendar)
                for k in ("estimate", "se", "ci_low", "ci_high"):
                    r[k] *= BP
                rows.append({"test_id": test.test_id, "sample": sample, "clusters": clusters, **r, "units": "bp"})
    return rows


def run_all(signals: pd.DataFrame, duplicate_uids: set[str], calendar: pd.DatetimeIndex | None = None,
            tests: tuple[SpreadTest, ...] = TESTS, two_households: frozenset | None = None) -> tuple[pd.DataFrame, dict]:
    """Every test x sample x clusters (bp); with ``two_households`` also the primary test on the two
    extra samples."""
    calendar = load_calendar() if calendar is None else calendar
    dedup, report = dedup_filter(signals, duplicate_uids)
    rows = _rows({"original": signals, "dedup": dedup}, tests, calendar)
    if two_households is not None:
        extra = extra_frames(dedup, two_households)
        report["n_dedup_official_only"] = int(len(extra["dedup_official_only"]))
        report["n_dedup_excl_two_vendor_households"] = int(len(extra["dedup_excl_two_vendor_households"]))
        rows += _rows(extra, tuple(t for t in tests if t.test_id == PRIMARY_TEST_ID), calendar)
    return pd.DataFrame(rows)[COLUMNS], report


def check_reproduction(tables: pd.DataFrame, registry_path: Path = REGISTRY_PATH, tol: float = 1e-10) -> pd.DataFrame:
    """Original-sample day-cluster rows against the canonical registry (decimal units)."""
    reg = pd.read_csv(registry_path).set_index("test_id")
    sub = tables[(tables["sample"] == "original") & (tables["clusters"] == "day")].set_index("test_id")
    out = pd.DataFrame({
        "d_estimate": sub["estimate"] / BP - reg.loc[sub.index, "estimate"],
        "d_se": sub["se"] / BP - reg.loc[sub.index, "se_multiway"],
        "d_n": sub["n"] - reg.loc[sub.index, "n"],
        "d_households": sub["n_households"] - reg.loc[sub.index, "n_households"],
        "d_dates": sub["n_dates_or_blocks"] - reg.loc[sub.index, "n_entry_dates"],
    })
    bad = out[(out[["d_estimate", "d_se"]].abs() > tol).any(axis=1) | (out[["d_n", "d_households", "d_dates"]] != 0).any(axis=1)]
    if not bad.empty:
        raise AssertionError(f"registry reproduction failed for {bad.index.tolist()}")
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="re-run the existing paper's spread tables")
    ap.add_argument("--out", type=Path, default=TABLES_PATH)
    args = ap.parse_args(argv)
    signals = load_signals()
    tables, report = run_all(signals, duplicate_signal_uids(), two_households=two_vendor_household_ids())
    diffs = check_reproduction(tables)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():  # keep rows other modules merged in (existing.units), their extra columns included
        kept = pd.read_csv(args.out)
        kept = kept[~kept["test_id"].isin(set(tables["test_id"]))]
        tables = pd.concat([tables, kept], ignore_index=True) if len(kept) else tables
    tables.to_csv(args.out, index=False)
    report["max_abs_reproduction_diff_decimal"] = float(diffs[["d_estimate", "d_se"]].abs().to_numpy().max())
    (args.out.with_name("rerun_report.json")).write_text(json.dumps(report, indent=2))
    counts = event_level_counts(pd.read_csv(SIGNALS_PATH, low_memory=False))
    counts_path = args.out.with_name(EVENT_COUNTS_PATH.name)
    counts_frame = pd.DataFrame([{"sample": "original", **counts}])
    if counts_path.exists():  # keep the other samples' rows (existing.units)
        old = pd.read_csv(counts_path)
        if "sample" in old.columns:
            counts_frame = pd.concat([counts_frame, old[old["sample"] != "original"]], ignore_index=True)
    counts_frame.to_csv(counts_path, index=False)
    print(json.dumps({**report, "event_level": counts}, indent=2))
    return tables


if __name__ == "__main__":
    main()
