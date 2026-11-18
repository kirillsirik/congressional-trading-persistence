"""Figure 2 power grid with every test on ONE common sample.

The published power surfaces give each test its own analysis set: the variance-component tests use the
persistence-rule households, the best-household test the households with at least 20 purchases, and the
mean and tail tests every household. The frontiers are then not comparable across tests, because the
tests differ in sample as well as in what they look for. This module re-runs both Figure 2 alternatives
with EVERY test evaluated on one sample: the purchases of the persistence-rule households, which is the
sample the headline estimand is defined on. Each placebo dataset is restricted to those households
BEFORE the effect is injected, so the informed households of panel (a) and the informed purchases of
panel (b) are drawn from that sample too.

Cells (both clocks, >= 500 centered placebo datasets each, draws 0-499):
  panel (a)  k informed households of the persistence set, k in K_GRID (k = H is a pure mean shift)
             x mu in MU_GRID bp per informed household per 20 trading days
  panel (b)  a share PI_GRID of the persistence households' purchases informed x mu per informed purchase
Tests: block-clustered mean test (two-sided, raw Y), placebo-test tau2 and household bootstrap tau2
(event-share weights, winsorized Y), best-household test (winsorized Y, >= 20 purchases inside the common
sample) and the tail test z > 2 — the five tests of Figure 2, each with its critical value recomputed on
the common sample (zero-effect datasets 1-1000 for the size of each test, 1001-2000 for the critical
values, never a dataset used for power).

CLI: python -m trading_on_disclosure_v4.calibration.power_common [--clocks T,P] [--n-panels 500]
     [--workers 8] [--time-cells 3] [--aggregate-only] [--sample-only]
Outputs: data/v4/calibration/power_surface_common.csv, frontiers_80_common.csv, common_sample_size.csv;
checkpoints: data/v4/restricted/c/checkpoints/{power_common,size_common}/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration import run_power as rp
from trading_on_disclosure_v4.calibration import run_power_v2 as rp2
from trading_on_disclosure_v4.calibration.run_size_v2 import N_ALL, N_TEST

GRID = "power_common"
SIZE_GRID = "size_common"
SAMPLE = "persistence_households_purchases"
K_GRID = (1, 2, 4, 8, 16, 38)  # capped at H of the draw inside `inject_k_households` (P has fewer households)
PI_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
MU_GRID = (10.0, 20.0, 35.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
WEIGHTS = ("ES",)
SURFACE_CSV = rc.CALIB_DIR / "power_surface_common.csv"
FRONTIER_CSV = rc.CALIB_DIR / "frontiers_80_common.csv"
SAMPLE_CSV = rc.CALIB_DIR / "common_sample_size.csv"
SIZE_CSV = rc.CALIB_DIR / "size_common.csv"


# ---------------------------------------------------------------- the common sample

def restrict(panel: pd.DataFrame, r1: set[str]) -> pd.DataFrame:
    """The common sample of one placebo dataset: the persistence households' rows only."""
    return panel.loc[panel["household_id"].isin(r1)].reset_index(drop=True)


def sample_description(clocks, n_draws: int = 100, min_n: int = 20) -> pd.DataFrame:
    """Households and events of the common sample per clock, averaged over the first `n_draws` datasets."""
    rows = []
    for clock in clocks:
        r1, _, _ = rc._shared(clock)
        hh, ev, n20 = [], [], []
        for p in rc.iter_draws(clock, 0, n_draws, set_name=rc.DEFAULT_SET[clock]):
            s = restrict(p, r1)
            c = s.groupby("household_id").size()
            hh.append(int(len(c)))
            ev.append(int(len(s)))
            n20.append(int((c >= min_n).sum()))
        rows.append({"clock": clock, "sample": SAMPLE, "n_households_r1": len(r1), "n_households_mean": float(np.mean(hh)),
                     "n_events_mean": float(np.mean(ev)), "n_events_min": int(np.min(ev)), "n_events_max": int(np.max(ev)),
                     "n_households_ge20_mean": float(np.mean(n20)), "n_draws": n_draws})
    out = pd.DataFrame(rows)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAMPLE_CSV, index=False)
    return out


# ---------------------------------------------------------------- work units

def specs() -> list[dict]:
    a = [{"kind": "sparse_k", "k": k, "mu_bp": m, "base": "R1", "informed": "uniform"} for k in K_GRID for m in MU_GRID]
    # "base" is inert for the sparse-trade engine; it records that the informed purchases come from the common sample
    b = [{"kind": "sparse_trades", "pi_t": p, "mu_bp": m, "household_specific": False, "base": "R1"}
         for p in PI_GRID for m in MU_GRID]
    return a + b


def power_units(clocks, n_panels: int, boot: int = 499) -> list[dict]:
    out = []
    for clock in clocks:
        for spec in specs():
            key = f"{clock}_wins_{rc.cell_id(spec)}".replace("|", "_").replace("=", "-")
            out.append({"grid": GRID, "key": key, "clock": clock, "variant": "wins", "spec": spec, "first": 0,
                        "n": n_panels, "hh_draws": boot, "boot_draws": 0, "weights": WEIGHTS, "t1_raw": True,
                        "set_name": rc.DEFAULT_SET[clock]})
    return out


def size_units(clocks, boot: int = 499) -> list[dict]:
    """Zero-effect datasets on the common sample: 0-999 for measured size, 1000-1999 for the critical values."""
    out = []
    for clock in clocks:
        for variant in ("raw", "wins"):
            for first in range(0, N_ALL, rc.BATCH_SIZE):
                out.append({"grid": SIZE_GRID, "key": f"{clock}_{variant}_{first:04d}", "clock": clock, "variant": variant,
                            "spec": {"kind": "none"}, "first": first, "n": rc.BATCH_SIZE, "boot_draws": 0,
                            "hh_draws": boot if first < N_TEST else 0, "t1_raw": True, "weights": WEIGHTS,
                            "set_name": rc.DEFAULT_SET[clock]})
    return out


def run_unit(unit: dict) -> dict:
    """`rc.run_unit` with each dataset restricted to the common sample before injection and testing."""
    grid, key = unit["grid"], unit["key"]
    path = rc.unit_path(grid, key)
    if path.exists():
        return {"key": key, "status": "cached", "seconds": 0.0, "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, spec, variant = unit["clock"], unit["spec"], unit["variant"]
    set_name = unit.get("set_name") or rc.DEFAULT_SET[clock]
    r1, cal, cuts = rc._shared(clock)
    name = f"{grid}|{rc.cell_id(spec)}"
    rows = []
    for p in rc.iter_draws(clock, unit["first"], unit["n"], set_name=set_name):
        q = rc.inject(restrict(p, r1), spec, name, r1=r1)
        raw = q
        if variant == "wins":
            q = rc.winsorize(q, cuts)
        s = rc.panel_statistics(q, r1, cal, weights=tuple(unit.get("weights", WEIGHTS)), boot_draws=unit.get("boot_draws", 0),
                                hh_draws=unit.get("hh_draws", 0), raw_for_t4=raw)
        if variant == "wins" and unit.get("t1_raw", False):
            clusters = {"I1": "anchor_date", **{f"I1blk{L}": f"anchor_blk{L}" for L in rc.BLOCK_CLUSTERS}}
            s.update({"T1raw_" + k[3:]: v for k, v in rc._t1_rows(rc.add_block_clusters(raw, cal), 0.05, clusters).items()})
        s["T1_target"] = float(q["theta_h"].mean())
        for k, v in q.attrs.get("inject_info", {}).items():
            s[f"inj_{k}"] = v
        rows.append(s)
    frame = pd.DataFrame(rows)
    for k, v in (("clock", clock), ("variant", variant), ("spec_json", json.dumps(spec, sort_keys=True)), ("grid", grid),
                 ("panel_set", set_name), ("sample", SAMPLE)):
        frame[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": key, "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def run_units(units_: list[dict], workers: int, log: Path | None) -> None:
    import multiprocessing as mp

    pending = [u for u in units_ if not rc.unit_path(u["grid"], u["key"]).exists()]
    rc.log_line(log, f"{len(units_)} units, {len(pending)} pending, {workers} workers")
    if not pending:
        return
    if workers <= 1:
        for u in pending:
            r = run_unit(u)
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")
        return
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(run_unit, pending):
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")


# ---------------------------------------------------------------- aggregation

def aggregate(clocks, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    size = rc.load_grid(SIZE_GRID)
    stats["superseded"] = False
    parts = []
    for clock in clocks:
        ref = rp2.reference_from_size_v2(size, clock)
        for variant in ("raw", "wins"):
            g = stats[(stats["clock"] == clock) & (stats["variant"] == variant)]
            if len(g):
                parts.append(rc.apply_reference(g, ref, variant))
    cur = rc.add_i1cal(pd.concat(parts, ignore_index=True), size, weights=WEIGHTS, first=N_TEST, n=N_TEST)
    surf = rp2.surface(cur)
    surf["sample"] = SAMPLE
    # k is capped at H of the draw inside the injector, so the share is the REALIZED k over H (k = 38 is all 30 on P)
    m = surf["k_effective_mean"].notna() & surf["H_mean"].notna()
    surf.loc[m, "k"] = surf.loc[m, "k_effective_mean"]
    surf.loc[m, "fraction"] = surf.loc[m, "k_effective_mean"] / surf.loc[m, "H_mean"]
    fr = rp2.frontiers(surf)
    fr["sample"] = SAMPLE
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    surf.to_csv(SURFACE_CSV, index=False)
    fr.to_csv(FRONTIER_CSV, index=False)
    st = size_table(clocks)
    st.to_csv(SIZE_CSV, index=False)
    rc.log_line(log, f"{SURFACE_CSV.name} {len(surf)} rows over {surf['params'].nunique()} cells; "
                     f"{FRONTIER_CSV.name} {len(fr)} rows; {SIZE_CSV.name} {len(st)} rows")
    return surf, fr


def size_table(clocks, alpha: float = 0.05) -> pd.DataFrame:
    """Measured rejection at nominal 5% of each Figure 2 test on the common sample (zero-effect datasets 1-1000)."""
    size = rc.load_grid(SIZE_GRID)
    rows = []
    for clock in clocks:
        ref = rp2.reference_from_size_v2(size, clock)
        parts = [rc.apply_reference(size[(size["clock"] == clock) & (size["variant"] == v) & (size["draw"] < N_TEST)], ref, v)
                 for v in ("raw", "wins")]
        g = rc.add_i1cal(pd.concat(parts, ignore_index=True), size, weights=WEIGHTS, first=N_TEST, n=N_TEST)
        for test, col in rp2.TEST_COLUMNS.items():
            if test not in ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3"):
                continue
            variant = "raw" if test.startswith(("T1", "T4")) else "wins"
            sub = g[g["variant"] == variant]
            c = col
            if test.startswith("T1"):  # the mean test is judged on raw Y; `wins` units carry it as T1raw_*
                alt = "T1raw_" + col[3:]
                c = alt if alt in sub.columns and sub[alt].notna().any() else col
            if c not in sub.columns or sub[c].isna().all():
                continue
            r = sub[c].to_numpy(dtype=float)
            n = int(np.isfinite(r).sum())
            p = float(np.nanmean(r))
            rows.append({"clock": clock, "sample": SAMPLE, "test": test, "test_name": rp2_plain(test), "variant": variant,
                         "reject_rate": p, "mc_se": rc.mc_se(p, n), "n_panels": n, "alpha": alpha})
    return pd.DataFrame(rows)


def rp2_plain(test: str) -> str:
    from trading_on_disclosure_v4.calibration.figure2_v3 import PLAIN

    return PLAIN.get(test, test)


def time_cells(clocks, n_cells: int, n_panels: int, log: Path | None) -> None:
    picks = power_units(clocks, n_panels)
    picks = [picks[0], picks[len(K_GRID) * len(MU_GRID) // 2], picks[-1]][:n_cells]
    per = []
    for u in picks:
        r = run_unit({**u, "grid": "power_common_timing", "key": u["key"] + f"_n{n_panels}", "n": n_panels})
        per.append(r["seconds"] / max(r["n"], 1))
        rc.log_line(log, f"timing {u['key']}: {r['seconds']:.1f}s for {r['n']} datasets = {per[-1]:.3f} s/dataset")
    n_power = len(power_units(clocks, 500))
    est = float(np.mean(per)) * (500 * n_power + rc.BATCH_SIZE * len(size_units(clocks)))
    rc.log_line(log, f"extrapolation: {n_power} power units x 500 + {len(size_units(clocks))} size units x 100 = "
                     f"{est:.0f} CPU-s ({est / 3600:.2f} CPU-h); /8 workers = {est / 8 / 60:.1f} min")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2 power grid on one common sample")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "power_common.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if args.sample_only:
        rc.log_line(args.log, f"common sample: {sample_description(clocks).to_dict('records')}")
        return
    if args.time_cells:
        time_cells(clocks, args.time_cells, 20, args.log)
        return
    if not args.aggregate_only:
        run_units(size_units(clocks, args.boot), args.workers, args.log)
        run_units(power_units(clocks, args.n_panels, args.boot), args.workers, args.log)
    sample_description(clocks)
    aggregate(clocks, args.log)
    rc.log_line(args.log, f"power_common finished in {time.time() - t0:.0f}s ({rp.POWER_TARGET:.0%} frontiers)")


if __name__ == "__main__":
    main()
