"""The existing paper's Section 3.4 rows on the de-duplicated sample.

Rows: signed mimicking payoff (`TW_D_SAR20`), raw-row weighting (`TW_S1_RAWROWS`), equal-household
spread (`TW_S1_EQHH`, household t), event-level spread (`TW_S1_EVENT`, Welch), source-native household
ids (`TW_S1_SRCID`), the within-month direction permutation (`TW_PLC_DIRECTION`, 999 draws) and the
within-security-year pseudo-entry placebo (`TW_PLC_DATES`, 19 draws). Each is the released estimand:
the clustered rows use the two-way CGM covariance of `rerun.clustered_mean_or_spread` with day and
20-session block clusters; the household-t, Welch and placebo rows call the v2 executors
(`locked_estimands`) on a context whose signal frame is substituted, seeds derived from the release
freeze hash exactly as released. Original-sample day-cluster rows reproduce the canonical registry
(asserted). Rows are merged into `rerun_tables.csv`; event counts per sample into
`event_level_counts.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.pipeline import locked_estimands as le
from trading_on_disclosure_v2.signals.build import build_security_entry_events
from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.outcomes import panel

BP = rerun.BP
OUTCOME = le.PRIMARY_OUTCOME
RELEASE_RECORD_PATH = panel.REPO_ROOT / "outputs" / "RELEASE_RECORD.json"
REPORT_PATH = rerun.OUT_DIR / "units_report.json"
CLUSTERED_TESTS = ("TW_D_SAR20", "TW_S1_RAWROWS", "TW_S1_SRCID")
EXECUTOR_TESTS = ("TW_S1_EQHH", "TW_S1_EVENT")
PLACEBO_TESTS = ("TW_PLC_DIRECTION", "TW_PLC_DATES")
UNIT_TEST_IDS = CLUSTERED_TESTS + EXECUTOR_TESTS + PLACEBO_TESTS
EXTRA_COLUMNS = ["placebo_mean", "placebo_sd", "n_draws"]
COLUMNS = rerun.COLUMNS + EXTRA_COLUMNS
N_PERM = 999
N_PSEUDO = 19


def freeze_hash(path: Path = RELEASE_RECORD_PATH) -> str:
    return json.loads(path.read_text())["freeze_hash"]


def _nan_row(test_id: str, sample: str, clusters: str) -> dict:
    return {"test_id": test_id, "sample": sample, "clusters": clusters, **{c: np.nan for c in COLUMNS[3:]},
            "units": "bp"}


def _finish(row: dict, r: dict, scale: tuple[str, ...] = ("estimate", "se", "ci_low", "ci_high")) -> dict:
    row.update(r)
    for k in scale:
        if k in row and pd.notna(row[k]):
            row[k] = float(row[k]) * BP
    return row


def clustered_row(frame: pd.DataFrame, test_id: str, clusters: str, calendar: pd.DatetimeIndex) -> dict:
    """Signed payoff (intercept mean), raw-row weighted spread, or source-native-id spread."""
    data = frame.dropna(subset=[OUTCOME, FOLLOWER_ENTRY_COL])
    household_name = rerun.HOUSEHOLD
    if test_id == "TW_S1_RAWROWS":  # released rule: repeat each signal lot_count times
        reps = data["lot_count"].fillna(1).astype(int).clip(lower=1)
        data = data.loc[data.index.repeat(reps)].reset_index(drop=True)
    elif test_id == "TW_S1_SRCID":
        household_name = "source_native_household_id"
    if clusters == "day":
        cl, name = data[FOLLOWER_ENTRY_COL].to_numpy(), FOLLOWER_ENTRY_COL
    elif clusters == "block20":
        cl, name = rerun.block_of(data[FOLLOWER_ENTRY_COL], calendar), f"{FOLLOWER_ENTRY_COL}_blk{rerun.BLOCK_L}"
    else:
        raise ValueError(f"unknown clusters {clusters!r}")
    y = data[OUTCOME].to_numpy(float)
    if test_id == "TW_D_SAR20":
        y, purchase = y * np.where(data["direction"] == "purchase", 1.0, -1.0), None
    else:
        purchase = (data["direction"] == "purchase").to_numpy(float)
    r = rerun.clustered_mean_or_spread(y, purchase, data[household_name].to_numpy(), cl, name, household_name)
    return _finish({"test_id": test_id, "clusters": clusters, "units": "bp"}, r)


def event_frame(signals_raw: pd.DataFrame) -> pd.DataFrame:
    """The released event frame: every signal (mixed included) collapsed to (permno, entry date),
    with the event's mean 20-day outcome merged as `canonical_locked._s10_returns` does."""
    events = build_security_entry_events(signals_raw)
    key = ["permno", FOLLOWER_ENTRY_COL]
    return events.merge(signals_raw.groupby(key)[OUTCOME].mean().reset_index(), on=key, how="left")


def executor_row(test_id: str, ctx: le.EstimandContext) -> dict:
    """Equal-household (household t) and event-level (Welch) rows via the v2 executors."""
    r = {"TW_S1_EQHH": le._equal_household, "TW_S1_EVENT": le._event_level}[test_id](ctx)
    if r.get("status") != "executed":
        raise RuntimeError(f"{test_id}: {r.get('status')}")
    row = _nan_row(test_id, "", {"TW_S1_EQHH": "household_t", "TW_S1_EVENT": "welch"}[test_id])
    row.update({"estimate": r["estimate"] * BP, "se": r["se_multiway"] * BP, "t": r["t_stat"], "p": r["raw_p"],
                "ci_low": r["ci_low"] * BP, "ci_high": r["ci_high"] * BP, "dof": r.get("dof", r["n"] - 1)})
    if test_id == "TW_S1_EQHH":
        row["n"], row["n_households"] = r["n"], r["n"]
    else:  # events entering the estimate (purchase-only + sale-only), the paper's printed count
        ev = ctx.events.dropna(subset=[OUTCOME])
        buy, sell = ev["purchase_household_count"] > 0, ev["sale_household_count"] > 0
        row["n"] = int((buy ^ sell).sum())
    return row


def placebo_row(test_id: str, ctx: le.EstimandContext, n_perm: int = N_PERM, n_pseudo: int = N_PSEUDO) -> dict:
    if test_id == "TW_PLC_DIRECTION":
        r = le.placebo_direction_permutation(ctx, n_perm=n_perm)
    else:
        r = le.placebo_pseudo_entry(ctx, n_draws=n_pseudo)
    if r.get("status") != "executed":
        raise RuntimeError(f"{test_id}: {r.get('status')}")
    row = _nan_row(test_id, "", {"TW_PLC_DIRECTION": "permutation", "TW_PLC_DATES": "pseudo_entry"}[test_id])
    row.update({"estimate": r["estimate"] * BP, "p": r["raw_p"], "n": r["n"], "placebo_mean": r["placebo_mean"] * BP,
                "placebo_sd": r["placebo_sd"] * BP, "n_draws": r["n_draws"]})
    return row


def event_counts(events: pd.DataFrame, signals_raw: pd.DataFrame) -> dict:
    """`rerun.event_level_counts` arithmetic on a prebuilt event frame."""
    ev = events.dropna(subset=[OUTCOME])
    buy, sell = ev["purchase_household_count"] > 0, ev["sale_household_count"] > 0
    out = {"purchase_only_events": int((buy & ~sell).sum()), "sale_only_events": int((sell & ~buy).sum()),
           "mixed_events_excluded": int((buy & sell).sum()), "all_outcome_bearing_events": int(len(ev))}
    out["events_in_estimate"] = out["purchase_only_events"] + out["sale_only_events"]
    out["signals_with_outcome"] = int(signals_raw[OUTCOME].notna().sum())
    return out


def run_sample(sample: str, signals: pd.DataFrame, signals_raw: pd.DataFrame, calendar: pd.DatetimeIndex,
               fh: str, market: dict | None, n_perm: int = N_PERM, n_pseudo: int = N_PSEUDO,
               placebos: bool = True) -> tuple[list[dict], dict]:
    """Every unit row for one sample (non-mixed frame `signals`; `signals_raw` includes mixed signals
    and feeds the event frame); ``market`` = {daily, benchmark} for the pseudo-entry placebo."""
    rows = [dict(clustered_row(signals, t, c, calendar), sample=sample)
            for t in CLUSTERED_TESTS for c in rerun.CLUSTERS]
    events = event_frame(signals_raw)
    ctx = le.EstimandContext(signals=signals, events=events, portfolio_daily=None, factors=None, freeze_hash=fh,
                             daily=None if market is None else market["daily"],
                             benchmark=None if market is None else market["benchmark"], calendar=calendar)
    rows += [dict(executor_row(t, ctx), sample=sample) for t in EXECUTOR_TESTS]
    if placebos:
        rows += [dict(placebo_row(t, ctx, n_perm, n_pseudo), sample=sample) for t in PLACEBO_TESTS]
    return [{c: r.get(c, np.nan) for c in COLUMNS} for r in rows], {"sample": sample, **event_counts(events, signals_raw)}


def check_reproduction(rows: pd.DataFrame, registry_path: Path = rerun.REGISTRY_PATH, tol: float = 1e-10) -> dict:
    """Original-sample rows (day clusters where clustered) against the canonical registry."""
    reg = pd.read_csv(registry_path).set_index("test_id")
    orig = rows[(rows["sample"] == "original") & (~rows["clusters"].eq("block20"))].set_index("test_id")
    diffs = {}
    for tid in CLUSTERED_TESTS + EXECUTOR_TESTS:
        r, x = orig.loc[tid], reg.loc[tid]
        diffs[tid] = max(abs(r["estimate"] / BP - x["estimate"]), abs(r["se"] / BP - x["se_multiway"]))
        if tid != "TW_S1_EVENT" and int(r["n"]) != int(x["n"]):
            raise AssertionError(f"{tid}: n {r['n']} vs registry {x['n']}")
    for tid in PLACEBO_TESTS:
        if tid not in orig.index:
            continue
        r, x = orig.loc[tid], reg.loc[tid]
        diffs[tid] = max(abs(r["estimate"] / BP - x["estimate"]), abs(r["placebo_mean"] / BP - x["placebo_mean"]),
                         abs(r["placebo_sd"] / BP - x["placebo_sd"]), abs(r["p"] - x["raw_p"]))
        if int(r["n_draws"]) != int(x["n_draws"]):
            raise AssertionError(f"{tid}: draws {r['n_draws']} vs registry {x['n_draws']}")
    bad = {k: v for k, v in diffs.items() if not v < tol}
    if bad:
        raise AssertionError(f"registry reproduction failed: {bad}")
    return diffs


def merge_into_tables(rows: pd.DataFrame, path: Path = rerun.TABLES_PATH) -> pd.DataFrame:
    """Replace this module's rows in `rerun_tables.csv` (other rows untouched; extra columns added)."""
    tables = pd.read_csv(path)
    tables = tables[~tables["test_id"].isin(UNIT_TEST_IDS)]
    out = pd.concat([tables, rows], ignore_index=True)
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[COLUMNS]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Section 3.4 unit / placebo rows on both samples")
    ap.add_argument("--tables", type=Path, default=rerun.TABLES_PATH)
    ap.add_argument("--perm-draws", type=int, default=N_PERM)
    ap.add_argument("--pseudo-draws", type=int, default=N_PSEUDO)
    ap.add_argument("--skip-placebos", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    signals_raw = pd.read_csv(rerun.SIGNALS_PATH, low_memory=False)
    signals_raw[FOLLOWER_ENTRY_COL] = pd.to_datetime(signals_raw[FOLLOWER_ENTRY_COL])
    signals = rerun.load_signals()
    dup = rerun.duplicate_signal_uids()
    dedup, report = rerun.dedup_filter(signals, dup)
    dedup_raw = signals_raw[~signals_raw["signal_uid"].isin(dup)].reset_index(drop=True)
    calendar = panel.v2_trading_calendar()
    market = None if args.skip_placebos else {"daily": panel.load_local_daily(), "benchmark": panel.load_benchmark()}
    fh = freeze_hash()
    rows, counts = [], []
    for sample, frame, raw in (("original", signals, signals_raw), ("dedup", dedup, dedup_raw)):
        r, c = run_sample(sample, frame, raw, calendar, fh, market, args.perm_draws, args.pseudo_draws,
                          placebos=not args.skip_placebos)
        rows += r
        counts.append(c)
        print(f"{sample}: {len(r)} rows, {time.time() - t0:.0f} s", flush=True)
    rows = pd.DataFrame(rows)[COLUMNS]
    diffs = check_reproduction(rows)
    merged = merge_into_tables(rows, args.tables)
    merged.to_csv(args.tables, index=False)
    pd.DataFrame(counts).to_csv(args.tables.with_name(rerun.EVENT_COUNTS_PATH.name), index=False)
    report.update({"unit_rows": int(len(rows)), "max_abs_reproduction_diff_decimal": max(diffs.values()),
                   "reproduction_diffs": diffs, "perm_draws": args.perm_draws, "pseudo_draws": args.pseudo_draws,
                   "seconds": round(time.time() - t0, 1)})
    args.tables.with_name(REPORT_PATH.name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return merged


if __name__ == "__main__":
    main()
