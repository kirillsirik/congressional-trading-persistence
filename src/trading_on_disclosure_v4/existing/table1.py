"""Table 1 of the existing paper on the released and the de-duplicated sample.

Every count is computed from `outputs/frames/{dated,matched}_transactions.csv` and `signals.csv` with
the all-lines duplicate rule of `rerun.duplicate_signal_uids` (a non-mixed public-clock signal whose
every parent line duplicates an official transaction is dropped). The record-level rows describe the
assembled and matched files and do not depend on the rule; the matched vendor lines that duplicate an
official line are counted from D's `duplicate_audit_rows.parquet`. Concentration = the canonical
`household_concentration` (signal counts per household) on the signals with a complete 20-day
public-clock outcome. The released column reproduces the `t1.*` ledger keys (asserted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v2.dates import (
    ANALYSIS_PUBLIC_PROXY_COL,
    ANALYSIS_PUBLIC_SOURCE_COL,
    SOURCE_OFFICIAL_OBSERVED,
)
from trading_on_disclosure_v2.inference.influence import household_concentration
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import panel

FRAMES = panel.FRAMES
LEDGER_PATH = panel.REPO_ROOT / "outputs" / "canonical" / "numerical_ledger.json"
AUDIT_ROWS_PATH = ef.EVENTS_DIR / "duplicate_audit_rows.parquet"
OUT_PATH = rerun.OUT_DIR / "table1_dedup.csv"
OUTCOME = "ar_E_h20"

# field -> (ledger key or None, description); order = the table's order
FIELDS = {
    "official_records": ("t1.official_records", "official-source transaction records"),
    "vendor_records": ("t1.vendor_records", "retained Capitol Trades records"),
    "assembled_records": ("t1.parsed_records_total", "assembled transaction records"),
    "observed_official_dates": ("t1.observed_official_dates", "records with observed official public dates"),
    "vendor_proxy_dates": ("t1.vendor_one_day_proxies", "records using the vendor-date proxy"),
    "matched_records": ("t1.matched_transactions", "CRSP-matched common-stock transaction records"),
    "matched_vendor_duplicates": (None, "matched vendor records that duplicate an official transaction (all-lines rule input)"),
    "signals": ("t1.unique_signals", "distinct public directional signals"),
    "mixed_signals": ("t1.mixed_direction_excluded", "mixed-direction signals excluded"),
    "nonmixed_signals": (None, "non-mixed signals"),
    "duplicate_signals_removed": (None, "non-mixed signals removed by the duplicate rule"),
    "signals_after_rule": (None, "non-mixed signals after the duplicate rule"),
    "signals_complete_h20": ("t1.complete_h20_outcomes", "signals with complete 20-day outcomes (the estimation sample)"),
    "households_all_signals": ("t1.households", "households among the non-mixed signals (after the rule)"),
    "households_h20": ("t1.concentration.n_households", "households in the 20-day sample"),
    "securities": ("t1.securities", "securities among the non-mixed signals (after the rule)"),
    "purchases_h20": (None, "purchase signals in the 20-day sample"),
    "sales_h20": (None, "sale signals in the 20-day sample"),
    "official_source_h20": (None, "official-source signals in the 20-day sample"),
    "vendor_source_h20": (None, "vendor-source signals in the 20-day sample"),
    "observed_date_h20": (None, "20-day-sample signals with an observed official public date"),
    "proxy_date_h20": (None, "20-day-sample signals with the vendor-date proxy"),
    "top_1_share": ("t1.concentration.top_1_share", "largest household share of the 20-day sample"),
    "top_5_share": ("t1.concentration.top_5_share", "top five household share"),
    "top_10_share": ("t1.concentration.top_10_share", "top ten household share"),
    "herfindahl": ("t1.concentration.herfindahl", "Herfindahl index of household shares"),
    "effective_households": ("t1.concentration.effective_household_count", "Herfindahl-implied effective household count"),
}
BLOCKS_FIELD = "blocks_h20"


def record_rows(dated: pd.DataFrame, matched: pd.DataFrame, audit_rows: pd.DataFrame | None) -> dict:
    src = dated["record_source"]
    out = {"official_records": int((src == "official").sum()), "vendor_records": int((src != "official").sum()),
           "assembled_records": int(len(dated)),
           "observed_official_dates": int((dated[ANALYSIS_PUBLIC_SOURCE_COL] == SOURCE_OFFICIAL_OBSERVED).sum()),
           "vendor_proxy_dates": int(dated[ANALYSIS_PUBLIC_PROXY_COL].fillna(False).astype(bool).sum()),
           "matched_records": int(len(matched))}
    if audit_rows is not None:
        out["matched_vendor_duplicates"] = int(audit_rows["dup_primary"].sum())
    return out


def signal_rows(signals_raw: pd.DataFrame, duplicate_uids: set[str], apply_rule: bool) -> dict:
    """Signal counts and concentration; ``apply_rule`` drops the flagged signals."""
    mixed = signals_raw["mixed_direction"].astype(bool)
    prim = signals_raw[~mixed]
    is_dup = prim["signal_uid"].isin(duplicate_uids)
    out = {"signals": int(len(signals_raw)), "mixed_signals": int(mixed.sum()), "nonmixed_signals": int(len(prim)),
           "duplicate_signals_removed": int(is_dup.sum()) if apply_rule else 0}
    if apply_rule:
        prim = prim[~is_dup]
    out["signals_after_rule"] = int(len(prim))
    h20 = prim.dropna(subset=[OUTCOME])
    out.update({"signals_complete_h20": int(len(h20)), "households_all_signals": int(prim["filer_uid"].nunique()),
                "households_h20": int(h20["filer_uid"].nunique()), "securities": int(prim["permno"].nunique()),
                "purchases_h20": int((h20["direction"] == "purchase").sum()),
                "sales_h20": int((h20["direction"] == "sale").sum()),
                "official_source_h20": int((h20["record_source"] == "official").sum()),
                "vendor_source_h20": int((h20["record_source"] != "official").sum()),
                "observed_date_h20": int((h20[ANALYSIS_PUBLIC_SOURCE_COL] == SOURCE_OFFICIAL_OBSERVED).sum()),
                "proxy_date_h20": int(h20[ANALYSIS_PUBLIC_PROXY_COL].astype(bool).sum())})
    conc = household_concentration(h20, household_col="filer_uid")
    out.update({"top_1_share": conc["top_1_share"], "top_5_share": conc["top_5_share"],
                "top_10_share": conc["top_10_share"], "herfindahl": conc["herfindahl"],
                "effective_households": conc["effective_household_count"]})
    return out


def build(signals_raw: pd.DataFrame, dated: pd.DataFrame, matched: pd.DataFrame, duplicate_uids: set[str],
          audit_rows: pd.DataFrame | None = None, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Long table: field, original, dedup, ledger_key, description."""
    rec = record_rows(dated, matched, audit_rows)
    orig, dedup = signal_rows(signals_raw, duplicate_uids, False), signal_rows(signals_raw, duplicate_uids, True)
    rows = []
    for field, (key, desc) in FIELDS.items():
        if field in rec:
            o = d = rec[field]
        elif field in orig:
            o, d = orig[field], dedup[field]
        else:
            continue
        rows.append({"field": field, "original": o, "dedup": d, "ledger_key": key or "", "description": desc})
    if calendar is not None:
        mixed = signals_raw["mixed_direction"].astype(bool)
        blocks = {}
        for label, rule in (("original", False), ("dedup", True)):
            prim = signals_raw[~mixed]
            if rule:
                prim = prim[~prim["signal_uid"].isin(duplicate_uids)]
            h20 = prim.dropna(subset=[OUTCOME, rerun.ENTRY_COLS["E"]])
            blocks[label] = int(len(set(rerun.block_of(h20[rerun.ENTRY_COLS["E"]], calendar))))
        rows.append({"field": BLOCKS_FIELD, "original": blocks["original"], "dedup": blocks["dedup"],
                     "ledger_key": "", "description": "20-trading-day blocks of the 20-day sample's anchors"})
    return pd.DataFrame(rows)


def check_ledger(table: pd.DataFrame, ledger_path: Path = LEDGER_PATH, tol: float = 1e-12) -> float:
    """The released column against the canonical ledger; returns the max absolute difference."""
    ledger = json.loads(ledger_path.read_text())
    worst = 0.0
    for r in table.itertuples(index=False):
        if isinstance(r.ledger_key, str) and r.ledger_key:  # empty keys read back from CSV as NaN
            d = abs(float(r.original) - float(ledger[r.ledger_key]))
            worst = max(worst, d)
            if not d <= tol:
                raise AssertionError(f"{r.field}: {r.original} vs ledger {ledger[r.ledger_key]}")
    return worst


def main(argv: list[str] | None = None) -> pd.DataFrame:
    import argparse

    ap = argparse.ArgumentParser(description="Table 1 on the released and de-duplicated samples")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    signals_raw = pd.read_csv(FRAMES / "signals.csv", low_memory=False)
    signals_raw[rerun.ENTRY_COLS["E"]] = pd.to_datetime(signals_raw[rerun.ENTRY_COLS["E"]])
    dated = pd.read_csv(FRAMES / "dated_transactions.csv", low_memory=False)
    matched = pd.read_csv(FRAMES / "matched_transactions.csv", low_memory=False)
    audit_rows = pd.read_parquet(AUDIT_ROWS_PATH, columns=["dup_primary"])
    table = build(signals_raw, dated, matched, rerun.duplicate_signal_uids(), audit_rows, panel.v2_trading_calendar())
    worst = check_ledger(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"max |original - ledger| = {worst:.3g}; -> {args.out}")
    return table


if __name__ == "__main__":
    main()
