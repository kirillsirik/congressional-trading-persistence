#!/usr/bin/env python
"""Household estimates on actual outcomes (plan §4-§7), run once, written to data/v4/results.csv.

    python scripts/v4/run_results.py                      # the run: results.csv, rank sets, years to detect, macros
    python scripts/v4/run_results.py --placebo-draw 7     # dry run on centered placebo draw 7 -> data/v4/restricted/e/dryrun_results_7.csv
    python scripts/v4/run_results.py --clock T            # one clock only

No gate, no environment variable, no hash: the safeguard is the pre-analysis plan committed with a date
before this script's first run. The dry run replaces the actual outcomes of both
clocks by one centered placebo draw and writes under data/v4/restricted/e/ only. Placebo distributions
(2,000 draws per clock and flavour) are cached under data/v4/restricted/e/placebo_stats/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trading_on_disclosure_v4.estimators import event_frames as evf  # noqa: E402
from trading_on_disclosure_v4.estimators import results as rs  # noqa: E402
from trading_on_disclosure_v4.estimators import specs as sp  # noqa: E402
from trading_on_disclosure_v4.ledger import macros as mc  # noqa: E402
from trading_on_disclosure_v4.outcomes import panel as panel_mod  # noqa: E402

RESTRICTED_E = ROOT / "data" / "v4" / "restricted" / "e"
CACHE_DIR = RESTRICTED_E / "placebo_stats"
RESULTS = ROOT / "data" / "v4" / "results.csv"


def _log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")

    return write


def head_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_context(clocks, placebo_draw: int | None, log) -> rs.Context:
    calendar = panel_mod.v2_trading_calendar()
    # the two vendor-only households (ids stay in memory; only the count is logged)
    excluded = evf.blind_vendor_households(2)
    sp.register_household_exclusion(sp.EXCL_TWO_VENDOR, excluded)
    log(f"household exclusion '{sp.EXCL_TWO_VENDOR}' registered: {len(excluded)} households")
    meta = {c: evf.event_metadata(c) for c in clocks}
    frames = {}
    for c in clocks:
        if placebo_draw is None:
            frames[c] = evf.load_event_frame(c, calendar=calendar)
            log(f"actual frame {c}: {len(frames[c])} events (purchases and sales, periods A + B)")
        else:
            frames[c] = evf.placebo_draw_frame(c, placebo_draw, "N2", meta[c])
            log(f"DRY RUN frame {c}: centered placebo draw {placebo_draw}, {len(frames[c])} events")

    def source(clock: str, flavour: str, dup: str):
        return evf.placebo_event_frames(clock, flavour, meta[clock], dup=dup, batches=True)

    def set_name(clock: str, dup: str) -> str:
        return evf.PLACEBO_SETS[(clock, "keepdups" if dup == "keepdups" else "default")]

    levels = rs.read_calibrated_levels() if rs.CALIBRATION_CHOICE.exists() else {}
    log(f"calibrated bootstrap level of the upper bound (target {rs.CALIBRATION_TARGET[0]}, injection {rs.CALIBRATION_TARGET[1]}): "
        f"{levels or 'none on file'}")
    bounds = rs.read_inverted_bounds() if rs.INVERTED_BOUND.exists() else None
    log(f"inverted placebo-injection bounds on file: {'none' if bounds is None else len(bounds)} rows "
        f"({', '.join('/'.join(k) for k in bounds.index) if bounds is not None else rs.INVERTED_BOUND.name + ' absent'})")
    return rs.Context(frames=frames, calendar=calendar, cutoffs=evf.pooled_cutoffs, placebo_source=source, placebo_set_name=set_name,
                      cache_dir=CACHE_DIR, log=log, calibrated_levels=levels, inverted_bounds=bounds)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebo-draw", type=int, default=None, help="dry run on this centered placebo draw (writes under data/v4/restricted/e/)")
    ap.add_argument("--clock", choices=["T", "P"], action="append", default=None)
    ap.add_argument("--no-macros", action="store_true")
    ap.add_argument("--n-boot", type=int, default=rs.N_BOOT)
    ap.add_argument("--n-placebo", type=int, default=rs.N_PLACEBO)
    a = ap.parse_args(argv)
    clocks = tuple(a.clock) if a.clock else ("T", "P")
    dry = a.placebo_draw is not None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = head_sha()
    if dry:
        out = RESTRICTED_E / f"dryrun_results_{a.placebo_draw}.csv"
        log = _log(RESTRICTED_E / "logs" / f"dryrun_{a.placebo_draw}_{stamp}.log")
    else:
        out = RESULTS
        log = _log(RESTRICTED_E / "logs" / f"results_{head}_{stamp}.log")
    t0 = time.time()
    ctx = build_context(clocks, a.placebo_draw, log)
    ctx.n_boot, ctx.n_placebo = a.n_boot, a.n_placebo
    results, rank_sets, ttd = rs.run_all(ctx, clocks)
    run_label = f"dry run on centered placebo draw {a.placebo_draw}" if dry else "actual outcomes"
    results = rs.results_frame(results.to_dict("records") + [
        rs.row("", "run", "default", "head_sha", float("nan"), "", notes=head),
        rs.row("", "run", "default", "date_utc", float("nan"), "", notes=stamp[:8]),
        rs.row("", "run", "default", "input", float("nan"), "", notes=run_label)])
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, lineterminator="\n")
    rank_sets.to_csv(out.with_name(out.stem + "_rank_sets.csv"), index=False, lineterminator="\n")
    ttd.to_csv(out.with_name(out.stem + "_time_to_detect.csv"), index=False, lineterminator="\n")
    scatter = rs.scatter_frame(ctx)
    scatter.to_csv(out.with_name(out.stem + "_household_scatter.csv"), index=False, lineterminator="\n")
    log(f"written: {out} ({len(results)} rows), rank sets ({len(rank_sets)}), time to detect ({len(ttd)}), "
        f"household scatter ({len(scatter)})")
    if not a.no_macros and not dry:
        path, n = mc.write_macros(results, header=f"run {head} {stamp}")
        log(f"macros written: {path} ({n} macros)")
    log(f"total runtime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
