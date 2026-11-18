"""A3 false winners on the production null sets (plan §7.1); pre-tag, null panels only.

For each (clock, flavour, set, min_n): per-draw winner statistics are written batch by batch
(resumable) under `data/v4/restricted/nullside/false_winners/<set>/<flavour>_n<min_n>_batch_XXXX.parquet`
(these rows carry winner identities, hence restricted); the count-only tidy summary goes to
`data/v4/nullside/false_winners_null.csv`. Log: `data/v4/restricted/nullside/logs/false_winners.log`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.null import run_null_panels as rnp
from trading_on_disclosure_v4.nullside import false_winners as fw
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel

RESTRICTED = panel.REPO_ROOT / "data" / "v4" / "restricted" / "nullside"
OUT_DIR = panel.REPO_ROOT / "data" / "v4" / "nullside"
SUMMARY_PATH = OUT_DIR / "false_winners_null.csv"
LOG_PATH = RESTRICTED / "logs" / "false_winners.log"
# (clock, set_name, drop_duplicates, n_draws, role)
PRIMARY_SETS = (("T", "A0_T_purchase_AB_L20_both_keepdups", True, 2000, "primary"),
                ("P", "A0_P_purchase_AB_L20_both_nodup", True, 2000, "primary"))
SENSITIVITY_SETS = (("P", "A0_P_purchase_AB_L20_both_keepdups", False, 500, "cp3_keepdups"),
                    ("T", "A0_T_purchase_AB_L40_both_keepdups", True, 500, "block_L40"),
                    ("T", "A0_T_purchase_AB_L60_both_keepdups", True, 500, "block_L60"),
                    ("T", "A0_T_purchase_AB_L20_backward_keepdups", True, 500, "backward"),
                    ("T", "A0_T_purchase_AB_L20_forward_keepdups", True, 500, "forward"))


def run_set(clock: str, flavour: str, set_name: str, min_n: int, households: pd.Index, max_draws: int,
            out_dir: Path, log: Path | None, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Per-draw rows for one (set, flavour, min_n); one parquet per stored batch, skipped when present."""
    src = rnp.set_dir(clock, flavour, set_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, t0, n_done = [], time.time(), 0
    for p in sorted(src.glob(f"{flavour.lower()}_batch_*.parquet")):
        b = int(p.stem.split("_")[-1])
        target = out_dir / f"{flavour}_n{min_n}_batch_{b:04d}.parquet"
        if not target.exists():
            batch = pd.read_parquet(p, columns=fw.READ_COLUMNS)
            rows = fw.batch_winner_rows(batch, households, calendar=calendar)
            rows.to_parquet(target, index=False)
            fc._log(log, f"{set_name} {flavour} n>={min_n} batch {b} ({len(rows)} draws) elapsed {time.time() - t0:.1f}s")
        frames.append(pd.read_parquet(target))
        n_done += len(frames[-1])
        if n_done >= max_draws:
            break
    per_draw = pd.concat(frames, ignore_index=True).sort_values("draw").reset_index(drop=True)
    return per_draw.iloc[:max_draws]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="A3 false-winner null distributions on stored null panels")
    ap.add_argument("--sets", choices=("primary", "sensitivity", "all"), default="all")
    ap.add_argument("--min-n", default=",".join(str(m) for m in fw.MIN_N_GRID))
    ap.add_argument("--max-draws", type=int, default=0, help="cap draws per set (0 = all stored)")
    ap.add_argument("--out", type=Path, default=SUMMARY_PATH)
    ap.add_argument("--log", type=Path, default=LOG_PATH)
    args = ap.parse_args(argv)
    sets = {"primary": PRIMARY_SETS, "sensitivity": SENSITIVITY_SETS, "all": PRIMARY_SETS + SENSITIVITY_SETS}[args.sets]
    min_ns = [int(m) for m in args.min_n.split(",")]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    summaries, sets_info = [], {}
    calendar = panel.v2_trading_calendar()
    for clock, set_name, drop, n_draws, role in sets:
        n_use = min(n_draws, args.max_draws) if args.max_draws else n_draws
        for min_n in min_ns:
            if role != "primary" and min_n != fw.MIN_N_GRID[0]:
                continue
            households, info = fw.household_set(clock, min_n, drop)
            sets_info[f"{set_name}|n{min_n}"] = info
            for flavour in rnp.FLAVOURS:
                t0 = time.time()
                per_draw = run_set(clock, flavour, set_name, min_n, households, n_use,
                                   RESTRICTED / "false_winners" / set_name, args.log, calendar)
                meta = {"clock": clock, "flavour": flavour, "set_name": set_name, "role": role, "min_n": min_n,
                        "n_households_set": info["n_households_set"], "outcome": "y_h20"}
                summaries.append(fw.summarize_draws(per_draw, meta))
                fc._log(args.log, f"{set_name} {flavour} n>={min_n}: {len(per_draw)} draws summarized in {time.time() - t0:.1f}s")
    table = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.8g")
    (args.out.with_name("false_winners_sets.json")).write_text(json.dumps(sets_info, indent=2))
    fc._log(args.log, f"summary written: {args.out} ({len(table)} rows)")
    return table


if __name__ == "__main__":
    main()
