"""Extension 2 (plan section 6): detection by household activity on the common sample of calibration
report section 15. Cells: clock x k informed households x selection (random, most_active, least_active)
x mu bp per purchase; the five Figure 2 tests in their common-sample form; critical values from the
size_common reference draws (1000-1999), power on centered placebo draws 0-499.

CLI: python -m trading_on_disclosure_v4.ext.activity_power [--clocks T,P] [--n-panels 500] [--workers 8]
     [--boot 499] [--time-cells 3] [--aggregate-only]
`--time-cells K` times K cells on 20 datasets and logs the extrapolated total before the full run.
Outputs: data/v4/ext/activity_power_{surface,frontiers,size}.csv; checkpoints
data/v4/restricted/c/checkpoints/activity_power/; log data/v4/restricted/c/logs/activity_power.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import power_common as pc
from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration import run_power as rp
from trading_on_disclosure_v4.calibration import run_power_v2 as rp2
from trading_on_disclosure_v4.calibration.inject_k import base_households, inject_k_households
from trading_on_disclosure_v4.calibration.run_size_v2 import N_TEST
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.inject.effects import HH

GRID = "activity_power"
TIMING_GRID = "activity_power_timing"
SIZE_GRID = pc.SIZE_GRID
K_GRID = (1, 2, 4, 8)
SELECTIONS = ("random", "most_active", "least_active")
MU_GRID = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
WEIGHTS = ("ES",)
TESTS = ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")
SURFACE_CSV = ec.OUT_DIR / "activity_power_surface.csv"
FRONTIER_CSV = ec.OUT_DIR / "activity_power_frontiers.csv"
SIZE_CSV = ec.OUT_DIR / "activity_power_size.csv"
LOG = rc.LOG_DIR / "activity_power.log"
SIZE_SOURCE = "size_common.csv, zero-effect draws 0-999"


# ---------------------------------------------------------------- injection

def _affected_share(out: pd.DataFrame) -> float:
    """Informed rows (theta_h != 0) over the rows of the panel's last draw (a work unit passes one draw)."""
    hit = out["theta_h"].to_numpy(dtype=float) != 0.0
    ix = list(out.groupby("draw", sort=True).indices.values())
    return float(hit[ix[-1]].mean()) if ix else float("nan")


def inject_k_by_activity(panel: pd.DataFrame, k: int, mu: float, name: str, r1: set[str], selection: str,
                         y_col: str = "y_h20") -> tuple[pd.DataFrame, dict]:
    """k informed households per draw (theta_h = mu on each of their purchases): random through inject_k's
    uniform draw and seed convention, else the k most / least active R1 households, ties by sorted id."""
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
    if int(k) < 0:
        raise ValueError("k must be non-negative")
    if selection == "random":
        out, raw = inject_k_households(panel, int(k), float(mu), name, "R1", r1, selection="uniform", y_col=y_col)
        info = {"n_base": int(raw.get("n_base", 0)), "k_effective": int(raw.get("k_effective", 0))}
    else:
        out = panel.copy()
        y = out[y_col].to_numpy(dtype=float).copy()
        theta_h = np.zeros(len(out))
        info = {"n_base": 0, "k_effective": 0}
        for _draw, ix in out.groupby("draw", sort=True).indices.items():
            sub = out.iloc[ix]
            cand = base_households(sub, "R1", r1)
            m = min(int(k), int(len(cand)))
            counts = cand.to_numpy(dtype=float)
            order = np.argsort(-counts if selection == "most_active" else counts, kind="stable")
            chosen = cand.index.to_numpy()[order[:m]]
            hit = sub[HH].isin(set(chosen)).to_numpy()
            y[ix[hit]] += float(mu)
            theta_h[ix[hit]] = float(mu)
            info = {"n_base": int(len(cand)), "k_effective": int(m)}
        out[y_col] = y
        out["theta_h"] = theta_h
    info.update({"selection": selection, "mu": float(mu), "affected_share": _affected_share(out)})
    return out, info


# ---------------------------------------------------------------- work units

def specs() -> list[dict]:
    return [{"kind": "sparse_k_activity", "k": k, "mu_bp": m, "selection": s, "base": "R1"}
            for k in K_GRID for s in SELECTIONS for m in MU_GRID]


def power_units(clocks, n_panels: int, boot: int = 499) -> list[dict]:
    out = []
    for clock in clocks:
        for spec in specs():
            key = f"{clock}_wins_{rc.cell_id(spec)}".replace("|", "_").replace("=", "-")
            out.append({"grid": GRID, "key": key, "clock": clock, "variant": "wins", "spec": spec, "first": 0,
                        "n": n_panels, "hh_draws": boot, "boot_draws": 0, "weights": WEIGHTS, "t1_raw": True,
                        "set_name": rc.DEFAULT_SET[clock]})
    return out


def run_unit(unit: dict) -> dict:
    """power_common.run_unit with the activity injector: restrict to the common sample, inject, winsorize, test."""
    grid, key = unit["grid"], unit["key"]
    path = rc.unit_path(grid, key)
    if path.exists():
        return {"key": key, "status": "cached", "seconds": 0.0, "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, spec = unit["clock"], unit["spec"]
    set_name = unit.get("set_name") or rc.DEFAULT_SET[clock]
    r1, cal, cuts = rc._shared(clock)
    name = f"{GRID}|{rc.cell_id(spec)}"
    clusters = {"I1": "anchor_date", **{f"I1blk{L}": f"anchor_blk{L}" for L in rc.BLOCK_CLUSTERS}}
    rows = []
    for p in rc.iter_draws(clock, unit["first"], unit["n"], set_name=set_name):
        raw, info = inject_k_by_activity(pc.restrict(p, r1), int(spec["k"]), float(spec["mu_bp"]), name, r1, spec["selection"])
        q = rc.winsorize(raw, cuts)
        s = rc.panel_statistics(q, r1, cal, weights=tuple(unit.get("weights", WEIGHTS)), boot_draws=unit.get("boot_draws", 0),
                                hh_draws=unit.get("hh_draws", 0), raw_for_t4=raw)
        s.update({"T1raw_" + c[3:]: v for c, v in rc._t1_rows(rc.add_block_clusters(raw, cal), 0.05, clusters).items()})
        s["T1_target"] = float(q["theta_h"].mean())
        s["inj_mean_bp"] = float(raw["theta_h"].mean())
        for c in ("affected_share", "k_effective", "n_base"):
            s[f"inj_{c}"] = info[c]
        rows.append(s)
    frame = pd.DataFrame(rows)
    for c, v in (("clock", clock), ("variant", "wins"), ("spec_json", json.dumps(spec, sort_keys=True)), ("grid", grid),
                 ("panel_set", set_name), ("sample", pc.SAMPLE)):
        frame[c] = v
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


def time_cells(clocks, n_cells: int, n_panels: int, log: Path | None) -> float:
    """Time n_cells representative cells on n_panels datasets each; log the extrapolated cost of the grid."""
    units = power_units(clocks, n_panels)
    picks = [units[0], units[len(units) // 2], units[-1]][:n_cells]
    per = []
    for u in picks:
        r = run_unit({**u, "grid": TIMING_GRID, "key": u["key"] + f"_n{n_panels}", "n": n_panels})
        if r["status"] == "done":
            per.append(r["seconds"] / max(r["n"], 1))
        rc.log_line(log, f"timing {u['key']}: {r['status']} {r['seconds']:.1f}s for {r['n']} datasets")
    if not per:
        rc.log_line(log, "timing: every timing cell was cached, nothing measured")
        return float("nan")
    n_units = len(power_units(clocks, 500))
    est = float(np.mean(per)) * 500 * n_units
    rc.log_line(log, f"timing: {np.mean(per):.3f} s/dataset over {len(per)} cells; extrapolation {n_units} units x 500 = "
                     f"{est:.0f} CPU-s ({est / 3600:.2f} CPU-h); /8 workers = {est / 8 / 60:.1f} min")
    return est


# ---------------------------------------------------------------- aggregation

def surface(stats: pd.DataFrame, clocks=("T", "P")) -> pd.DataFrame:
    """One row per cell x test in grid order: power, MC SE, affected share, injected mean, realized tau2."""
    groups = {key: g for key, g in stats.groupby(["clock", "spec_json"], sort=False)}
    rows = []
    for clock in clocks:
        for spec in specs():
            g = groups.get((clock, json.dumps(spec, sort_keys=True)))
            if g is None:
                continue
            share = g["inj_affected_share"].to_numpy(dtype=float)
            for test in TESTS:
                col = rp2.TEST_COLUMNS[test]
                if test.startswith("T1"):
                    col = "T1raw_" + col[3:]
                if col not in g or g[col].isna().all():
                    continue
                r = g[col].to_numpy(dtype=float)
                n = int(np.isfinite(r).sum())
                p = float(np.nanmean(r))
                rows.append({"clock": clock, "k": int(spec["k"]), "selection": spec["selection"], "mu_bp": float(spec["mu_bp"]),
                             "test": test, "test_name": pc.rp2_plain(test), "power": p, "mc_se": rc.mc_se(p, n), "n_panels": n,
                             "affected_share_mean": float(np.mean(share)), "affected_share_min": float(np.min(share)),
                             "affected_share_max": float(np.max(share)), "injected_mean_bp": float(g["inj_mean_bp"].mean()),
                             "realized_tau2_mean": float(g["T2_target_ES"].mean()), "size_correct": test in rp2.SIZE_CORRECT_TESTS,
                             "grid": GRID})
    return pd.DataFrame(rows)


def share_at_frontier(mu: np.ndarray, power: np.ndarray, share: np.ndarray, status: str, target: float) -> float:
    """The cell's affected share read at the frontier with the same bracketing fraction as the mu interpolation."""
    order = np.argsort(mu)
    mu, power, share = np.asarray(mu, float)[order], np.asarray(power, float)[order], np.asarray(share, float)[order]
    above = np.flatnonzero(power >= target)
    if status == "not_reached" or len(above) == 0:
        return float("nan")
    i = int(above[0])
    if i == 0 or power[i] <= power[i - 1] or mu[i - 1] <= 0:
        return float(share[i])
    f = (target - power[i - 1]) / (power[i] - power[i - 1])
    return float(share[i - 1] + f * (share[i] - share[i - 1]))


def frontiers(surf: pd.DataFrame, target: float = rp.POWER_TARGET) -> pd.DataFrame:
    """Per (clock, k, selection, test): smallest grid mu at >= target power, the log-linear interpolated mu
    with its delta-method SE, and the affected purchase share at that frontier ("> grid" when never reached)."""
    rows = []
    for (clock, k, sel, test), g in surf.groupby(["clock", "k", "selection", "test"], sort=False):
        gg = g.sort_values("mu_bp")
        mu, pw = gg["mu_bp"].to_numpy(dtype=float), gg["power"].to_numpy(dtype=float)
        e, status = rp.interpolate_frontier(mu, pw, target)
        rows.append({"clock": clock, "k": int(k), "selection": sel, "test": test, "test_name": gg["test_name"].iloc[0],
                     "target_power": target,
                     "grid_min_mu_bp": float(mu[pw >= target].min()) if (pw >= target).any() else np.nan,
                     "mu_bp_at_target": e, "mu_se_bp": rp2.frontier_se(mu, pw, gg["mc_se"].to_numpy(dtype=float), e, status, target),
                     "affected_share_at_target": share_at_frontier(mu, pw, gg["affected_share_mean"].to_numpy(dtype=float), status, target),
                     "status": "> grid" if status == "not_reached" else status, "max_power_on_grid": float(pw.max()),
                     "n_grid_points": int(len(gg)), "n_panels": int(gg["n_panels"].min()), "size_correct": bool(gg["size_correct"].iloc[0]),
                     "grid": GRID})
    return pd.DataFrame(rows)


def size_table(clocks=("T", "P")) -> pd.DataFrame:
    """The measured size of these tests on the common sample, restated from size_common.csv."""
    st = pd.read_csv(pc.SIZE_CSV)
    st = st[st["clock"].isin(clocks) & st["test"].isin(TESTS)].copy()
    st["clock_order"] = st["clock"].map({c: i for i, c in enumerate(clocks)})
    st["test_order"] = st["test"].map({t: i for i, t in enumerate(TESTS)})
    st = st.sort_values(["clock_order", "test_order"]).drop(columns=["clock_order", "test_order"]).reset_index(drop=True)
    st["source"] = SIZE_SOURCE
    return st


def aggregate(clocks, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    size = rc.load_grid(SIZE_GRID)
    parts = [rc.apply_reference(stats[(stats["clock"] == c) & (stats["variant"] == "wins")], rp2.reference_from_size_v2(size, c), "wins")
             for c in clocks if ((stats["clock"] == c) & (stats["variant"] == "wins")).any()]
    cur = rc.add_i1cal(pd.concat(parts, ignore_index=True), size, weights=WEIGHTS, first=N_TEST, n=N_TEST)
    surf = surface(cur, clocks)
    fr = frontiers(surf)
    st = size_table(clocks)
    ec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    surf.to_csv(SURFACE_CSV, index=False, lineterminator="\n")
    fr.to_csv(FRONTIER_CSV, index=False, lineterminator="\n")
    st.to_csv(SIZE_CSV, index=False, lineterminator="\n")
    n_cells = surf.groupby(["clock", "k", "selection", "mu_bp"]).ngroups if len(surf) else 0
    rc.log_line(log, f"{SURFACE_CSV.name} {len(surf)} rows over {n_cells} cells (expected {len(clocks) * len(specs())}); "
                     f"{FRONTIER_CSV.name} {len(fr)} rows; {SIZE_CSV.name} {len(st)} rows")
    return surf, fr, st


# ---------------------------------------------------------------- manifest

def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ec.REPO_ROOT))
    except ValueError:
        return str(path)


def manifest_entries(clocks) -> list[dict]:
    """Inputs read: the size_common checkpoints (count + sha of the sorted names), the placebo sets, the cutoffs, size_common.csv."""
    d = rc.CHECKPOINT_DIR / SIZE_GRID
    names = sorted(p.name for p in d.glob("*.parquet"))
    entries = [{"path": _rel(d), "n_files": len(names), "sha256_of_names": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                "use": "T3 / T4 / I1cal references from draws 1000-1999"}]
    for clock in clocks:
        sd = rc.set_dir(clock)
        entries.append({"path": _rel(sd), "set_name": rc.DEFAULT_SET[clock], "clock": clock,
                        "n_files": len(list(sd.glob("*.parquet"))), "use": "centered placebo draws 0-499"})
    entries.append({**ec.file_entry(rc.POOLED_CSV), "use": "pooled p1/p99 winsorization cutoffs"})
    entries.append({**ec.file_entry(pc.SIZE_CSV), "use": SIZE_SOURCE})
    return entries


def write_manifest(clocks, n_panels: int, boot: int, seconds: float | None = None) -> None:
    extra = {"grid": GRID, "clocks": list(clocks), "k_grid": list(K_GRID), "selections": list(SELECTIONS), "mu_grid_bp": list(MU_GRID),
             "n_panels": int(n_panels), "hh_draws": int(boot), "draws": "0-499", "tests": list(TESTS),
             "outputs": [_rel(p) for p in (SURFACE_CSV, FRONTIER_CSV, SIZE_CSV)], "checkpoints": _rel(rc.CHECKPOINT_DIR / GRID)}
    if seconds is not None:
        extra["run_seconds"] = round(float(seconds), 1)
    ec.write_manifest(GRID, manifest_entries(clocks), extra=extra)


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extension 2: detection by household activity on the common sample")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--log", type=Path, default=LOG)
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if not args.aggregate_only:
        if args.time_cells:
            time_cells(clocks, args.time_cells, 20, args.log)
        run_units(power_units(clocks, args.n_panels, args.boot), args.workers, args.log)
    aggregate(clocks, args.log)
    write_manifest(clocks, args.n_panels, args.boot, None if args.aggregate_only else time.time() - t0)
    rc.log_line(args.log, f"{GRID} finished in {time.time() - t0:.0f}s ({rp.POWER_TARGET:.0%} frontiers)")


if __name__ == "__main__":
    main()
