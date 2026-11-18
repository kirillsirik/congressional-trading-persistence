"""Production block-shift null panels (plan §4 A0).

CLI: one clock, one flavour (N1 raw / N2 centered), purchases only, periods
A+B, B draws with the wave-1 generator (`null.block_shift`). Batches of
`--batch-size` draws are written under
`data/v4/restricted/n/panels/<clock>/<flavour>/<set_name>/` together with a
per-batch checkpoint, so an interrupted run resumes at the first missing
batch. N2 panels are the N1 panels of the SAME draws minus the household-period
centering mean over the whole draw set, so `--flavour N2`
first completes the N1 set and then writes the centered copies.

Files per set directory:
  n1_batch_XXXX.parquet / n2_batch_XXXX.parquet   panel rows (schema n_null_panels.md)
  sums_batch_XXXX.parquet                          per-(household, period, draw) sum/count accumulators (N1 only)
  droplog_batch_XXXX.parquet, droplog.parquet      per-(draw, period, block) shift log (N1 only)
  draw_droprates.csv                               per-draw drop-rate log (count-only)
  centering_means.parquet                          household-period centering means (N1 dir)
  checkpoint.json                                  completed batches + config (resume key)
  manifest.json                                    seeds, counts, drop rates, sha256 per file
Duplicate filter (plan §11): `--dup-rule all_lines` (default; drops P
events whose EVERY line is `dup_primary`, set suffix `_alllines`; 0 T episodes),
`--dup-rule any_line`, or
`--keep-duplicates` (`_keepdups`). `--tag` appends `_<tag>` to the set name
(new seeds) for a regenerated set that would otherwise collide with an older one.
With `--carry resid_sd` every row also carries `sigma_event` = resid_sd x sqrt(20).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trading_on_disclosure_v4.null import block_shift as bs
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel

PANELS_ROOT = panel.RESTRICTED_N / "panels"
LOG_DIR = panel.RESTRICTED_N / "logs"
FULL_OUTCOMES = ("y_h20", "y_h10", "y_h60", "y_noalpha_h20", "y_mkt_h20")
FLAVOURS = ("N1", "N2")


def set_dir(clock: str, flavour: str, set_name: str, root: Path = PANELS_ROOT) -> Path:
    if flavour not in FLAVOURS:
        raise ValueError(f"flavour must be one of {FLAVOURS}")
    return root / clock / flavour / set_name


def batch_range(n_draws: int, batch_size: int) -> list[tuple[int, int, int]]:
    """(batch index, first draw, last draw + 1) for every batch."""
    return [(i, s, min(s + batch_size, n_draws)) for i, s in enumerate(range(0, n_draws, batch_size))]


def _read_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def resume_key(cfg: bs.ShiftConfig, n_draws: int, batch_size: int, n_events: int, cache_sha: str) -> dict:
    """Everything a stored batch depends on; a checkpoint with another key is refused."""
    key = {"cfg": asdict(cfg), "n_draws": n_draws, "batch_size": batch_size,
           "n_events_in": n_events, "cache_sha256": cache_sha}
    return json.loads(json.dumps(key))  # JSON-normalized so a stored checkpoint compares equal


def generate_n1(
    events: pd.DataFrame, grid: bs.CacheGrid, cfg: bs.ShiftConfig, n_draws: int, out: Path,
    batch_size: int, cache_sha: str, log_path: Path | None = None,
) -> dict:
    """Resumable N1 generation: one parquet batch + accumulator + droplog per batch."""
    out.mkdir(parents=True, exist_ok=True)
    ck_path = out / "checkpoint.json"
    key = resume_key(cfg, n_draws, batch_size, len(events), cache_sha)
    ck = _read_checkpoint(ck_path)
    if ck and ck.get("key") != key:
        raise ValueError(f"{ck_path} was written under another configuration; use a new set directory")
    done = set(ck.get("completed_batches", []))
    geom = bs.Geometry(events, grid)
    t0 = time.time()
    for b, start, stop in batch_range(n_draws, batch_size):
        files = [out / f"{k}_batch_{b:04d}.parquet" for k in ("n1", "sums", "droplog")]
        if b in done and all(f.exists() for f in files):
            continue
        panels, logs = [], []
        for d in range(start, stop):
            p, lg = bs.shift_one_draw(events, grid, cfg, d, geom)
            panels.append(p)
            logs.append(lg)
        batch = pd.concat(panels, ignore_index=True)
        batch.to_parquet(files[0], index=False)
        bs._per_draw_sums(batch, cfg.outcomes).to_parquet(files[1], index=False)
        pd.concat(logs, ignore_index=True).to_parquet(files[2], index=False)
        done.add(b)
        _write_checkpoint(ck_path, {"key": key, "completed_batches": sorted(done),
                                    "n_batches": len(batch_range(n_draws, batch_size)),
                                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        fc._log(log_path, f"{cfg.test_name} N1 batch {b} draws {start}-{stop - 1} "
                          f"({len(batch)} rows) elapsed {time.time() - t0:.1f}s")
    return {"seconds": time.time() - t0, "n_batches": len(done)}


def finalize_n1(out: Path, cfg: bs.ShiftConfig, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Droplog, per-draw drop-rate CSV, centering means and count-only summary for a complete N1 set."""
    logs = sorted(out.glob("droplog_batch_*.parquet"))
    droplog = pd.concat([pd.read_parquet(p) for p in logs], ignore_index=True)
    droplog.to_parquet(out / "droplog.parquet", index=False)
    per_draw = droplog.groupby("draw").agg(n_events=("n_events", "sum"), n_infeasible=("n_infeasible", "sum"),
                                           n_overlap_dropped=("n_overlap_dropped", "sum"),
                                           n_redraws=("n_redraws", "sum"), cap_hits=("cap_hit", "sum"),
                                           n_blocks_redrawn=("n_redraws", lambda s: int((s > 0).sum())))
    per_draw["n_kept"] = per_draw["n_events"] - per_draw["n_infeasible"] - per_draw["n_overlap_dropped"]
    per_draw["drop_rate"] = per_draw["n_infeasible"] / per_draw["n_events"]
    per_draw.reset_index().to_csv(out / "draw_droprates.csv", index=False)
    sums = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("sums_batch_*.parquet"))], ignore_index=True)
    means = bs.centering_from_sums(sums, cfg.outcomes)
    means.to_parquet(out / "centering_means.parquet", index=False)
    n_rows = int(per_draw["n_kept"].sum())
    flags = _flag_shares(out, n_rows)
    summary = bs.summarize_run(events, n_rows, droplog, cfg, 0.0, flag_shares=flags)
    return summary, means


def _flag_shares(out: Path, n_rows: int) -> dict:
    counts = np.zeros(len(bs.FLAG_COLUMNS))
    for p in sorted(out.glob("n1_batch_*.parquet")):
        counts += pd.read_parquet(p, columns=bs.FLAG_COLUMNS).sum().to_numpy(float)
    return {c: float(v / n_rows) if n_rows else 0.0 for c, v in zip(bs.FLAG_COLUMNS, counts, strict=True)}


def write_n2(n1_dir: Path, n2_dir: Path, means: pd.DataFrame, cfg: bs.ShiftConfig,
             log_path: Path | None = None) -> dict:
    """Centered copies of every N1 batch (resumable; keyed on the centering means' sha256)."""
    n2_dir.mkdir(parents=True, exist_ok=True)
    means_sha = fc.sha256_file(n1_dir / "centering_means.parquet")
    ck_path = n2_dir / "checkpoint.json"
    ck = _read_checkpoint(ck_path)
    done = set(ck.get("completed_batches", [])) if ck.get("centering_sha256") == means_sha else set()
    t0 = time.time()
    n1_paths = sorted(n1_dir.glob("n1_batch_*.parquet"))
    for p in n1_paths:
        b = int(p.stem.split("_")[-1])
        target = n2_dir / p.name.replace("n1_", "n2_")
        if b in done and target.exists():
            continue
        bs.center_panel(pd.read_parquet(p), means, cfg.outcomes).to_parquet(target, index=False)
        done.add(b)
        _write_checkpoint(ck_path, {"centering_sha256": means_sha, "completed_batches": sorted(done),
                                    "n_batches": len(n1_paths), "source_n1_dir": str(n1_dir),
                                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        fc._log(log_path, f"{cfg.test_name} N2 batch {b} written, elapsed {time.time() - t0:.1f}s")
    return {"seconds": time.time() - t0, "n_batches": len(done), "centering_sha256": means_sha}


def n2_centering_check(n2_dir: Path, cfg: bs.ShiftConfig) -> dict:
    """Max |draw-averaged household-period mean| of N2 per outcome (must be ~0 by construction)."""
    sums = [bs._per_draw_sums(pd.read_parquet(p), cfg.outcomes) for p in sorted(n2_dir.glob("n2_batch_*.parquet"))]
    means = bs.centering_from_sums(pd.concat(sums, ignore_index=True), cfg.outcomes)
    return {c: float(means[f"center_{c}"].abs().max()) for c in cfg.outcomes} | {
        "n_household_period_cells": int(len(means))}


def write_manifest(out: Path, summary: dict, extra: dict) -> dict:
    files = sorted(p for p in out.glob("*.parquet")) + sorted(out.glob("*.csv"))
    manifest = {**summary, **extra,
                "seed_scheme": "seed_from(test_name, draw) = int(sha256('<test_name>|<draw>')[:8], 16)",
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {p.name: {"sha256": fc.sha256_file(p), "bytes": p.stat().st_size} for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def time_draws(events: pd.DataFrame, grid: bs.CacheGrid, cfg: bs.ShiftConfig, n: int = 20) -> dict:
    """Time n in-memory draws; nothing written."""
    geom = bs.Geometry(events, grid)
    t0 = time.time()
    rows = 0
    for d in range(n):
        p, _ = bs.shift_one_draw(events, grid, cfg, d, geom)
        rows += len(p)
    sec = time.time() - t0
    return {"n_timed": n, "seconds": round(sec, 2), "seconds_per_draw": round(sec / n, 4),
            "rows_per_draw": rows / n, "bytes_per_row_est": 8 * (len(cfg.outcomes) + len(cfg.carry)) + 40}


def load_inputs(args) -> tuple[pd.DataFrame, bs.CacheGrid, bs.ShiftConfig, dict, str]:
    """Events (filtered, block-assigned), cache grid, config and duplicate report for the CLI args."""
    drop = not args.keep_duplicates
    name = bs.test_name_for(args.clock, "purchase", "A,B", args.block_length, args.direction, drop, False, False,
                            args.dup_rule, args.tag)
    cfg = bs.ShiftConfig(test_name=name, block_length=args.block_length, direction=args.direction,
                         outcomes=tuple(args.outcomes.split(",")),
                         carry=tuple(c for c in args.carry.split(",") if c))
    cols = ["permno", "anchor_idx", "anchor_date", "reason_h20"] + [bs.Y_COLUMNS[o] for o in cfg.outcomes]
    cols += [bs.CARRY_COLUMNS[c] for c in cfg.carry]
    grid = bs.CacheGrid(fc.load_cache(columns=cols), outcomes=cfg.outcomes, carry=cfg.carry)
    events, report = bs.load_events_for_null(args.clock, "purchase", "A,B", drop, False, cfg, args.dup_rule)
    cache_sha = json.loads(fc.MANIFEST_PATH.read_text())["sha256"]
    return events, grid, cfg, report, cache_sha


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(description="production block-shift null panels (A0), one clock and flavour")
    ap.add_argument("--clock", choices=("T", "P"), required=True)
    ap.add_argument("--flavour", choices=FLAVOURS, default="N2", help="N2 also completes the N1 set")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--block-length", type=int, default=20)
    ap.add_argument("--direction", choices=("both", "backward", "forward"), default="both")
    ap.add_argument("--outcomes", default=",".join(FULL_OUTCOMES))
    ap.add_argument("--carry", default="resid_sd")
    ap.add_argument("--keep-duplicates", action="store_true", help="_keepdups sensitivity (P clock)")
    ap.add_argument("--dup-rule", choices=("all_lines", "any_line"), default="all_lines",
                    help="duplicate-event flag: all_lines (plan §11 default, suffix _alllines) or any_line (_nodup)")
    ap.add_argument("--tag", default="", help="suffix appended to the set name (new seeds); e.g. w3")
    ap.add_argument("--time-only", type=int, default=0, metavar="N", help="time N draws in memory and exit")
    ap.add_argument("--root", type=Path, default=PANELS_ROOT)
    ap.add_argument("--log", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    events, grid, cfg, report, cache_sha = load_inputs(args)
    if args.time_only:
        est = time_draws(events, grid, cfg, args.time_only)
        est["extrapolated_seconds_for_draws"] = round(est["seconds_per_draw"] * args.draws, 1)
        est["extrapolated_gb_per_flavour"] = round(est["rows_per_draw"] * args.draws * est["bytes_per_row_est"] / 1e9, 2)
        print(json.dumps({"test_name": cfg.test_name, **est}, indent=2))
        return est
    n1_dir = set_dir(args.clock, "N1", cfg.test_name, args.root)
    gen = generate_n1(events, grid, cfg, args.draws, n1_dir, args.batch_size, cache_sha, args.log)
    summary, means = finalize_n1(n1_dir, cfg, events)
    summary["seconds"] = round(gen["seconds"], 2)
    extra = {"flavour": "N1", "clock": args.clock, "batch_size": args.batch_size, "cache_sha256": cache_sha,
             "duplicate_filter": report, "n_households_in": int(events["household_id"].nunique())}
    manifest = write_manifest(n1_dir, summary, extra)
    fc._log(args.log, f"{cfg.test_name} N1 manifest written ({manifest['n_draws']} draws)")
    if args.flavour == "N2":
        n2_dir = set_dir(args.clock, "N2", cfg.test_name, args.root)
        w = write_n2(n1_dir, n2_dir, means, cfg, args.log)
        check = n2_centering_check(n2_dir, cfg)
        manifest = write_manifest(n2_dir, summary, {**extra, "flavour": "N2", "source_n1_dir": str(n1_dir),
                                                    "centering_sha256": w["centering_sha256"],
                                                    "n2_centering_check_max_abs": check,
                                                    "seconds_n2": round(w["seconds"], 2)})
        fc._log(args.log, f"{cfg.test_name} N2 manifest written; centering check {check}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    main()


BP = 1e4


def harness_view(batch: pd.DataFrame, outcome: str = "y_h20", horizon: int = 20) -> pd.DataFrame:
    """One stored batch in E's panel interface (`e_panel_interface.md`): `y_h20` in bp,
    `anchor_date` = the SHIFTED anchor, `sigma_event` = resid_sd x sqrt(horizon) in bp."""
    out = pd.DataFrame({"draw": batch["draw"].astype("int64"), "event_id": batch["event_id"],
                        "household_id": batch["household_id"], "clock": batch["clock"],
                        "period": batch["period"], "y_h20": batch[outcome].astype(float) * BP,
                        "anchor_date": pd.to_datetime(batch["shifted_anchor_date"])})
    if "sigma_event" in batch.columns:  # stored: resid_sd x sqrt(20), decimal
        out["sigma_event"] = batch["sigma_event"].astype(float) * BP
    elif "resid_sd" in batch.columns:
        out["sigma_event"] = batch["resid_sd"].astype(float) * np.sqrt(horizon) * BP
    return out


def iter_panels_from_disk(clock: str, flavour: str, set_name: str, n: int, seed: int | None = None,
                          outcome: str = "y_h20", root: Path = PANELS_ROOT):
    """`iter_panels(n, seed)` for E's harness: the first n stored draws of a set, one frame per draw.
    Deterministic in n alone (seed is ignored, as the interface allows for pre-built panels)."""
    d = set_dir(clock, flavour, set_name, root)
    prefix = flavour.lower()
    cols = bs.PANEL_COLUMNS + [outcome, "shifted_anchor_date"]
    yielded = 0
    for p in sorted(d.glob(f"{prefix}_batch_*.parquet")):
        avail = pq.read_schema(p).names
        use = cols + [c for c in ("resid_sd", "sigma_event") if c in avail]
        batch = pd.read_parquet(p, columns=use)
        for _, g in batch.groupby("draw", sort=True):
            if yielded >= n:
                return
            yield harness_view(g.reset_index(drop=True), outcome)
            yielded += 1
    if yielded < n:
        raise ValueError(f"{d} holds only {yielded} draws; {n} requested")
