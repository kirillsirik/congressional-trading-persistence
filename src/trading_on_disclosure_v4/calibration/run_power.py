"""Detection region (plan §6): the plan's injection grid on N2 panels.

Grid (both clocks; dense also raw vs winsorized):
  dense            tau in {0, 10, 25, 50, 75, 100, 150, 200} bp
  sparse households pi_H in {2, 5, 10, 20}% x mu_H in {50, 100, 200, 400, 800} bp x {uniform, activity}
  sparse trades    pi_T in {0.5, 1, 2, 5, 10}% x mu_T in {250, 500, 1000, 2000, 3600} bp x {common, household}
Tests per panel: T1 (I1), T2 (I1 with ES/EH/PW; I2b household bootstrap 499 draws; I2 calendar-block
bootstrap 499 draws in the dense cells only), T3 max-t (critical value from N2 draws 1000-1999),
T4 shares z > 2 / z > 3 (raw Y; add-one Monte Carlo p against the same reference draws).
>= 500 injected panels per cell (N2 draws 0-499); one resumable checkpoint per cell.

CLI: python -m trading_on_disclosure_v4.calibration.run_power [--clocks T,P] [--n-panels 500]
     [--workers 10] [--time-cells 5] [--aggregate-only]
Outputs: data/v4/calibration/power_surface.csv, data/v4/calibration/frontiers_80.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc

GRID = "power"
DENSE_TAU = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0)
PI_H = (0.02, 0.05, 0.10, 0.20)
MU_H = (50.0, 100.0, 200.0, 400.0, 800.0)
PI_T = (0.005, 0.01, 0.02, 0.05, 0.10)
MU_T = (250.0, 500.0, 1000.0, 2000.0, 3600.0)
POWER_TARGET = 0.8


def grid_specs() -> list[dict]:
    specs = [{"kind": "dense", "tau_bp": t} for t in DENSE_TAU]
    specs += [{"kind": "sparse_households", "pi_h": p, "mu_bp": m, "informed": s}
              for s in ("uniform", "activity") for p in PI_H for m in MU_H]
    specs += [{"kind": "sparse_trades", "pi_t": p, "mu_bp": m, "household_specific": hs}
              for hs in (False, True) for p in PI_T for m in MU_T]
    return specs


def power_units(clocks, n_panels: int, boot: int = 499) -> list[dict]:
    units = []
    for clock in clocks:
        for spec in grid_specs():
            variants = ("raw", "wins") if spec["kind"] == "dense" else ("raw",)
            for variant in variants:
                key = f"{clock}_{variant}_{rc.cell_id(spec)}".replace("|", "_").replace("=", "-")
                units.append({"grid": GRID, "key": key, "clock": clock, "variant": variant, "spec": spec,
                              "first": 0, "n": n_panels, "hh_draws": boot,
                              "boot_draws": boot if spec["kind"] == "dense" else 0})
    return units


def _family(spec: dict) -> tuple[str, float, float, str]:
    """(alternative, fraction, effect_bp, selection) from a spec."""
    k = spec["kind"]
    if k == "dense":
        return "dense", np.nan, float(spec["tau_bp"]), "all"
    if k == "sparse_households":
        return "sparse_households", float(spec["pi_h"]), float(spec["mu_bp"]), spec["informed"]
    return "sparse_trades", float(spec["pi_t"]), float(spec["mu_bp"]), ("household" if spec["household_specific"] else "common")


TEST_COLUMNS = {
    "T1": "T1_I1_rej1", "T1_blk20": "T1_I1blk20_rej1", "T1_blk60": "T1_I1blk60_rej1",
    "T3": "T3_rej", "T4_z2": "T4_z2_rej", "T4_z3": "T4_z3_rej",
}
for _w in rc.WEIGHTS:
    TEST_COLUMNS[f"T2_I1_{_w}"] = f"T2_I1_{_w}_rej1"
    TEST_COLUMNS[f"T2_I1blk20_{_w}"] = f"T2_I1blk20_{_w}_rej1"
    TEST_COLUMNS[f"T2_I2b_{_w}"] = f"T2_I2b_{_w}_rej1"
    TEST_COLUMNS[f"T2_I2_{_w}"] = f"T2_I2L20_{_w}_rej1"
    TEST_COLUMNS[f"T2_I1cal_{_w}"] = f"T2_I1cal_{_w}_rej1"
PRIMARY_TESTS = ("T1", "T2_I1_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")


def surface(stats: pd.DataFrame) -> pd.DataFrame:
    """Tidy power surface: one row per clock x variant x cell x test."""
    rows = []
    for (clock, variant, sj), g in stats.groupby(["clock", "variant", "spec_json"], sort=True):
        spec = json.loads(sj)
        alt, frac, eff, sel = _family(spec)
        for test, col in TEST_COLUMNS.items():
            if col not in g or g[col].isna().all():
                continue
            r = g[col].to_numpy(dtype=float)
            n = int(np.isfinite(r).sum())
            p = float(np.nanmean(r))
            row = {"clock": clock, "variant": variant, "alternative": alt, "fraction": frac, "effect_bp": eff,
                   "selection": sel, "params": sj, "test": test, "power": p, "mc_se": rc.mc_se(p, n), "n_panels": n,
                   "truth_tau2": rc.truth_of(spec)["tau2"], "truth_mean": rc.truth_of(spec)["mean"]}
            if test.startswith("T2_"):
                tag = test[3:].rsplit("_", 1)
                base = {"I2": "I2L20", "I1cal": "I1"}.get(tag[0], tag[0])
                k = f"T2_{base}_{tag[1]}"
                row.update({"mean_estimate": g[f"{k}_tau2"].mean(), "coverage": g[f"{k}_cov"].mean(),
                            "mean_target": g[f"T2_target_{tag[1]}"].mean()})
            elif test.startswith("T1"):
                row.update({"mean_estimate": g["T1_I1_est"].mean(), "mean_target": g["T1_target"].mean()})
            rows.append(row)
    return pd.DataFrame(rows)


def interpolate_frontier(eff: np.ndarray, power: np.ndarray, target: float = POWER_TARGET) -> tuple[float, str]:
    """Smallest effect with power >= target on the grid, refined by LINEAR interpolation of power in
    log(effect) between the last grid point below target and the first at/above it (scanning upward).
    Returns (effect, status): status in {'interpolated', 'at_grid_min', 'not_reached'}."""
    order = np.argsort(eff)
    eff, power = np.asarray(eff, float)[order], np.asarray(power, float)[order]
    above = np.flatnonzero(power >= target)
    if len(above) == 0:
        return np.nan, "not_reached"
    i = int(above[0])
    if i == 0:
        return float(eff[0]), "at_grid_min"
    lo, hi, plo, phi = eff[i - 1], eff[i], power[i - 1], power[i]
    if phi <= plo or lo <= 0:
        return float(hi), "interpolated"
    f = (target - plo) / (phi - plo)
    return float(np.exp(np.log(lo) + f * (np.log(hi) - np.log(lo)))), "interpolated"


def frontiers(surf: pd.DataFrame, target: float = POWER_TARGET) -> pd.DataFrame:
    rows = []
    keys = ["clock", "variant", "alternative", "selection", "test", "fraction"]
    for key, g in surf.groupby(keys, sort=True, dropna=False):
        e, status = interpolate_frontier(g["effect_bp"].to_numpy(), g["power"].to_numpy(), target)
        row = dict(zip(keys, key, strict=True))
        gg = g.sort_values("effect_bp")
        row.update({"target_power": target, "effect_bp_at_target": e, "status": status,
                    "grid_min_effect_reaching_target": float(gg.loc[gg["power"] >= target, "effect_bp"].min()) if (gg["power"] >= target).any() else np.nan,
                    "max_power_on_grid": float(gg["power"].max()), "n_grid_points": int(len(gg)), "n_panels": int(gg["n_panels"].min())})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(clocks, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    parts = []
    for clock in clocks:
        ref = rc.load_reference(clock)
        for variant in ("raw", "wins"):
            g = stats[(stats["clock"] == clock) & (stats["variant"] == variant)]
            if len(g):
                parts.append(rc.apply_reference(g, ref, variant))
    surf = surface(rc.add_i1cal(pd.concat(parts, ignore_index=True), rc.load_grid("size")))
    fr = frontiers(surf)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    surf.to_csv(rc.CALIB_DIR / "power_surface.csv", index=False)
    fr.to_csv(rc.CALIB_DIR / "frontiers_80.csv", index=False)
    rc.log_line(log, f"power_surface.csv {len(surf)} rows over {surf['params'].nunique()} cells; frontiers_80.csv {len(fr)} rows")
    return surf, fr


def time_cells(clocks, n_cells: int, n_panels: int, log: Path | None) -> None:
    """Time n_cells representative units on n_panels panels each into a scratch grid; extrapolate."""
    units = power_units(clocks, n_panels)
    picks = [u for u in units if u["spec"]["kind"] == "dense"][:2] + \
            [u for u in units if u["spec"]["kind"] == "sparse_households"][:2] + \
            [u for u in units if u["spec"]["kind"] == "sparse_trades"][:1]
    picks = picks[:n_cells]
    tot = {"dense": 0.0, "sparse": 0.0}
    for u in picks:
        u = {**u, "grid": "power_timing", "key": u["key"] + f"_n{n_panels}"}
        r = rc.run_unit(u)
        per = r["seconds"] / max(r["n"], 1)
        tot["dense" if u["spec"]["kind"] == "dense" else "sparse"] += per
        rc.log_line(log, f"timing {u['key']}: {r['seconds']:.1f}s for {r['n']} panels = {per:.3f} s/panel")
    n_dense = sum(u["spec"]["kind"] == "dense" for u in units)
    n_sparse = len(units) - n_dense
    est = 500 * (n_dense * tot["dense"] / max(sum(u["spec"]["kind"] == "dense" for u in picks), 1)
                 + n_sparse * tot["sparse"] / max(sum(u["spec"]["kind"] != "dense" for u in picks), 1))
    rc.log_line(log, f"extrapolation: {n_dense} dense + {n_sparse} sparse units x 500 panels = {est:.0f} CPU-s; "
                     f"/10 workers = {est / 600:.1f} min")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="detection-region power surface on injected N2 panels")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K", help="time K cells on 20 panels and exit")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "run_power.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if args.time_cells:
        time_cells(clocks, args.time_cells, 20, args.log)
        return
    if not args.aggregate_only:
        rc.run_reference(clocks, args.workers, args.log)
        rc.run_units(power_units(clocks, args.n_panels, args.boot), args.workers, args.log)
    aggregate(clocks, args.log)
    rc.log_line(args.log, f"run_power finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
