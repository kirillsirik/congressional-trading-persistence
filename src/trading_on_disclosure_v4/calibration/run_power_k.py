"""Wave-3 power grids (plan §11 rows "Sparse-household alternative" and "Duplicate filter semantics").

Grid K (`power_k`, both clocks): k in {1, 2, 4, 8} informed households drawn from the test's own analysis
set — base "R1" (T2's household set; I2b run) or "n20" (T3's set: households with >= 20 kept events) —
mu_H in {50, 100, 200, 400, 800} bp, uniform or activity-proportional selection; winsorized Y for T2/T3
with T1 on raw Y (`t1_raw`) and T4 raw by construction; ES weights; 500 injected N2 panels (draws 0-499).
Grid W3 (`power_w3`, P clock only): the dense cells (raw and winsorized, all weights, I2b) and the
sparse-trade cells (raw, ES, I2b) of wave 2 re-run on the all-lines P set. I2 (calendar-block bootstrap)
is not re-run (plan §11 moved it to a note). T cells are not re-run (the T set is unchanged).

CLI: python -m trading_on_disclosure_v4.calibration.run_power_k [--grids k,w3] [--clocks T,P] [--n-panels 500]
     [--workers 10] [--time-cells K]
Aggregation into power_surface.csv / frontiers_80.csv: `run_power_v2.py`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration import run_power as rp
from trading_on_disclosure_v4.calibration.inject_k import K_GRID

GRID_K = "power_k"
GRID_W3 = "power_w3"
BASES = ("R1", "n20")
SELECTIONS = ("uniform", "activity")


def k_specs() -> list[dict]:
    return [{"kind": "sparse_k", "k": k, "mu_bp": m, "base": b, "informed": s}
            for b in BASES for s in SELECTIONS for k in K_GRID for m in rp.MU_H]


def _key(clock: str, variant: str, spec: dict) -> str:
    return f"{clock}_{variant}_{rc.cell_id(spec)}".replace("|", "_").replace("=", "-")


def k_units(clocks, n_panels: int, boot: int = 499, variant: str = "wins") -> list[dict]:
    units = []
    for clock in clocks:
        for spec in k_specs():
            units.append({"grid": GRID_K, "key": _key(clock, variant, spec), "clock": clock, "variant": variant, "spec": spec,
                          "first": 0, "n": n_panels, "hh_draws": boot if spec["base"] == "R1" else 0, "boot_draws": 0,
                          "weights": ("ES",), "t1_raw": True, "set_name": rc.DEFAULT_SET[clock]})
    return units


def w3_units(n_panels: int, boot: int = 499, clocks=("P",)) -> list[dict]:
    units = []
    for clock in clocks:
        for spec in rp.grid_specs():
            if spec["kind"] == "dense":
                for variant in ("raw", "wins"):
                    units.append({"grid": GRID_W3, "key": _key(clock, variant, spec), "clock": clock, "variant": variant, "spec": spec,
                                  "first": 0, "n": n_panels, "hh_draws": boot, "boot_draws": 0, "t1_raw": True,
                                  "set_name": rc.DEFAULT_SET[clock]})
            elif spec["kind"] == "sparse_trades":
                units.append({"grid": GRID_W3, "key": _key(clock, "raw", spec), "clock": clock, "variant": "raw", "spec": spec,
                              "first": 0, "n": n_panels, "hh_draws": boot, "boot_draws": 0, "weights": ("ES",),
                              "set_name": rc.DEFAULT_SET[clock]})
    return units


def time_cells(units: list[dict], n_cells: int, n_panels: int, log: Path | None) -> float:
    """Time n_cells representative units on n_panels panels each (scratch grid) and extrapolate."""
    picks, seen = [], set()
    for u in units:
        sig = (u["clock"], u["grid"], u["spec"].get("kind"), u["spec"].get("base"), u["variant"], u["hh_draws"] > 0)
        if sig not in seen:
            seen.add(sig)
            picks.append(u)
    picks = picks[:n_cells]
    per = {}
    for u in picks:
        v = {**u, "grid": "power_timing_w3", "key": u["key"] + f"_n{n_panels}", "n": n_panels}
        r = rc.run_unit(v)
        sig = (u["clock"], u["grid"], u["spec"].get("kind"), u["spec"].get("base"), u["variant"], u["hh_draws"] > 0)
        per[sig] = r["seconds"] / max(r["n"], 1)
        rc.log_line(log, f"timing {v['key']}: {r['seconds']:.1f}s for {r['n']} panels = {per[sig]:.3f} s/panel")
    est = 0.0
    for u in units:
        sig = (u["clock"], u["grid"], u["spec"].get("kind"), u["spec"].get("base"), u["variant"], u["hh_draws"] > 0)
        est += u["n"] * per.get(sig, max(per.values()))
    rc.log_line(log, f"extrapolation: {len(units)} units -> {est:.0f} CPU-s; /10 workers = {est / 600:.1f} min")
    return est


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="wave-3 power grids (k informed households; P re-run)")
    ap.add_argument("--grids", default="k,w3")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K", help="time K representative units on 20 panels and exit")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "run_power_k.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    grids = set(args.grids.split(","))
    units = []
    if "k" in grids:
        units += k_units(clocks, args.n_panels, args.boot)
    if "w3" in grids:
        units += w3_units(args.n_panels, args.boot)
    t0 = time.time()
    if args.time_cells:
        time_cells(units, args.time_cells, 20, args.log)
        return
    rc.run_units(units, args.workers, args.log)
    rc.log_line(args.log, f"run_power_k finished in {time.time() - t0:.0f}s ({len(units)} units)")


if __name__ == "__main__":
    main()
