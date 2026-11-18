"""Build the v4 data objects: restricted event tables plus committable count aggregates.

Run: ``PYTHONPATH=src python -m trading_on_disclosure_v4.data.build``.
Prints a manifest (row counts, SHA-256) and writes it next to the aggregates.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.data import concentration as co
from trading_on_disclosure_v4.data import coverage as cv
from trading_on_disclosure_v4.data import duplicates as du
from trading_on_disclosure_v4.data import eligibility as el
from trading_on_disclosure_v4.data import events as ev
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import streams as st
from trading_on_disclosure_v4.data.calendar import load_calendar

REPO_ROOT = load.REPO_ROOT
RESTRICTED_DIR = REPO_ROOT / "data" / "v4" / "restricted" / "data"
DIAG_DIR = REPO_ROOT / "data" / "v4" / "diagnostics"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    load.assert_no_outcome_columns(frame, where=path.name)
    frame.to_parquet(path, index=False, engine="pyarrow")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    load.assert_no_outcome_columns(frame, where=path.name)
    frame.to_csv(path, index=False, float_format="%.6g")


def events_by_year_table(events: pd.DataFrame) -> pd.DataFrame:
    out = (
        events.groupby(["anchor_year", "clock", "direction", "record_source", "period"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    return out


def events_by_chamber_table(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["clock", "period", "chamber", "direction"])
        .agg(n_events=("event_id", "size"), n_households=("household_id", "nunique"))
        .reset_index()
    )


def period_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Events per clock x period x direction, plus the 2014/2026 anchors that fall outside 2015-2025."""
    out = events.groupby(["clock", "period", "direction"]).size().rename("n_events").reset_index()
    outside = events[~events["anchor_year"].between(2015, 2025)]
    extra = (
        outside.groupby(["clock", "anchor_year", "direction"]).size().rename("n_events").reset_index()
        .assign(period=lambda d: "X_outside_" + d["anchor_year"].astype(str))
        .drop(columns="anchor_year")
    )
    return pd.concat([out, extra[out.columns]], ignore_index=True)


def source_composition(events: pd.DataFrame, direction: str = "purchase") -> pd.DataFrame:
    """Vendor vs official share by clock x period x anchor year (the 2023 regime break inside B)."""
    sub = events[events["direction"] == direction]
    out = (
        sub.groupby(["clock", "period", "anchor_year", "record_source"]).size().rename("n_events").reset_index()
        .pivot_table(index=["clock", "period", "anchor_year"], columns="record_source", values="n_events", fill_value=0)
        .reset_index()
    )
    for col in ("official", "capitol_trades"):
        if col not in out:
            out[col] = 0
    out["n_events"] = out["official"] + out["capitol_trades"]
    out["vendor_share"] = out["capitol_trades"] / out["n_events"]
    return out


def event_duplicate_flags(rows: pd.DataFrame, events_t: pd.DataFrame, events_p: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Restricted: event_id -> per audit variant, ``dup_<v>`` (>= 1 line flagged) and
    ``dup_<v>_all_lines`` (every line flagged, i.e. the event vanishes if flagged lines are dropped)."""
    frames = []
    p_lines = pairs.groupby("signal_uid").size()
    for variant in du.VARIANTS:
        dup = set(rows.loc[rows[f"dup_{variant}"], "transaction_uid"])
        t_n = du.flagged_line_counts(events_t, dup)
        p_hits = pairs[pairs["transaction_uid"].isin(dup)].groupby("signal_uid").size()
        p_n = events_p["signal_uid"].map(p_hits).fillna(0).astype("int64")
        p_total = events_p["signal_uid"].map(p_lines).fillna(0).astype("int64")
        any_flag = pd.concat([t_n > 0, p_n > 0], ignore_index=True).rename(f"dup_{variant}")
        all_flag = pd.concat(
            [(t_n > 0) & (t_n >= events_t["n_lines"]), (p_n > 0) & (p_n >= p_total)], ignore_index=True
        ).rename(f"dup_{variant}_all_lines")
        frames += [any_flag, all_flag]
    ids = pd.concat([events_t[["event_id", "clock"]], events_p[["event_id", "clock"]]], ignore_index=True)
    return pd.concat([ids] + frames, axis=1)


def build(restricted_dir: Path = RESTRICTED_DIR, diag_dir: Path = DIAG_DIR) -> dict:
    """Build every D artifact and return the manifest."""
    t0 = time.time()
    restricted_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    calendar = load_calendar()
    signals = load.load_signals()
    timed = load.load_timed_transactions()
    matched = load.load_matched_transactions()

    events_t, t_report = ev.build_events_T(timed, calendar)
    events_p, p_report = ev.build_events_P(signals, calendar)
    events = pd.concat([events_t, events_p], ignore_index=True)
    counts = el.household_period_counts(events)

    audit, audit_counts = du.prepare_audit_frame(matched)
    rows = du.audit_rows(audit)
    pairs = du.explode_signal_parents(signals)
    dup_flags = event_duplicate_flags(rows, events_t, events_p, pairs)
    classes = cv.classify_households(matched)

    restricted = {
        "events_T.parquet": events_t,
        "events_P.parquet": events_p,
        "household_period_counts.parquet": counts,
        "duplicate_audit_rows.parquet": rows,
        "event_duplicate_flags.parquet": dup_flags,
    }
    for name, frame in restricted.items():
        _write_parquet(frame, restricted_dir / name)

    quantiles = pd.concat(
        [el.count_quantiles(el.eligibility_flags(counts, c), r).assign(clock=c) for c in ("T", "P") for r in el.RULES],
        ignore_index=True,
    )
    eligible, concentration = [], []
    for name in st.DUP_FILTERS:
        sub = st.filter_events(events, dup_flags, name)
        sub_counts = counts if name == "none" else el.household_period_counts(sub)
        eligible.append(el.eligible_household_table(sub_counts, dup_filter=name))
        concentration.append(co.concentration_table(sub, sub_counts, dup_filter=name))
    aggregates = {
        "events_by_year_clock_direction_source.csv": events_by_year_table(events),
        "events_by_chamber.csv": events_by_chamber_table(events),
        "eligible_households_by_rule_clock.csv": pd.concat(eligible, ignore_index=True),
        "eligible_count_quantiles.csv": quantiles,
        "concentration_by_clock_period.csv": pd.concat(concentration, ignore_index=True),
        "continuation_rates.csv": pd.concat([el.continuation_rates(counts, c) for c in ("T", "P")], ignore_index=True),
        "duplicate_audit_summary.csv": du.audit_summary(rows, pairs, events_p),
        "duplicate_audit_by_year.csv": pd.concat(
            [du.affected_signals_by_year(rows, pairs, events_p, v) for v in du.VARIANTS], ignore_index=True
        ),
        "duplicate_audit_t_episodes.csv": pd.DataFrame([du.t_episode_exposure(rows, events_t, v) for v in du.VARIANTS]),
        "duplicate_audit_pairing.csv": du.pairing_diagnostic(audit, rows, "primary"),
        "vendor_household_classes.csv": cv.coverage_classes(matched, rows, classes),
        "vendor_household_coverage.csv": cv.coverage_by_period(events, classes),
        "stream_sizes.csv": st.stream_sizes(events, dup_flags),
        "household_activity_distribution.csv": el.activity_distribution(events),
        "period_summary.csv": period_summary(events),
        "source_composition_by_period.csv": source_composition(events),
    }
    for name, frame in aggregates.items():
        _write_csv(frame, diag_dir / name)

    manifest = {
        "calendar": {"n_sessions": int(len(calendar)), "first": str(calendar[0].date()), "last": str(calendar[-1].date())},
        "inputs": {"signals_rows": int(len(signals)), "timed_rows": int(len(timed)), "matched_rows": int(len(matched))},
        "events_T_report": t_report,
        "events_P_report": p_report,
        "lag_days_report": ev.lag_days_report(events),
        "duplicate_audit_counts": audit_counts,
        "t_episode_exposure_primary": du.t_episode_exposure(rows, events_t, "primary"),
        "restricted": {
            name: {"rows": int(len(frame)), "sha256": sha256_of(restricted_dir / name)}
            for name, frame in restricted.items()
        },
        "aggregates": {
            name: {"rows": int(len(frame)), "sha256": sha256_of(diag_dir / name)}
            for name, frame in aggregates.items()
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(diag_dir / "build_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    manifest = build()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
