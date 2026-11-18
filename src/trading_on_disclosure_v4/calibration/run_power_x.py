"""Figure 2 v3 cells: extend both x-axes to 100%.

Grid X (`power_x`, both clocks, 500 centered placebo panels per cell, draws 0-499):
  sparse households  k = H/2 and k = H (`k_frac` 0.5 / 1.0) of each test's own analysis set (plus 10% and 20% of the
                     best-household set) — base "R1"
                     (the tau2 set), "n20" (the best-household set), "all" (the mean and tail tests) —
                     informed households drawn at random; mu in MU_X; winsorized Y with T1 on raw Y.
  sparse trades      20%, 50% and 100% of purchases informed with the same probability (100% = a pure
                     mean shift of mu per purchase); raw Y, as the existing 0.5-10% cells.
Tests read from the checkpoints: block-clustered mean test, placebo-test tau2 (event-share weights),
household bootstrap tau2 (499 draws), best-household test, tail test.

CLI: python -m trading_on_disclosure_v4.calibration.run_power_x [--clocks T,P] [--n-panels 500] [--workers 8]
     [--time-cells 3]
Aggregation into power_surface.csv / frontiers_80.csv: `run_power_v2.py` (grid `power_x` is included).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.run_power_k import _key, time_cells

GRID_X = "power_x"
MU_X = (10.0, 20.0, 35.0, 50.0, 100.0, 200.0, 400.0, 800.0)
K_FRAC = (0.5, 1.0)
K_FRAC_N20_EXTRA = (0.1, 0.2)  # shorten the best-household segment between k = 8 (6.6%) and 50%
BASES_X = ("R1", "n20", "all")
PI_T_X = (0.2, 0.5, 1.0)


def x_specs() -> list[dict]:
    specs = [{"kind": "sparse_k", "k_frac": f, "mu_bp": m, "base": b, "informed": "uniform"}
             for b in BASES_X for f in K_FRAC for m in MU_X]
    specs += [{"kind": "sparse_k", "k_frac": f, "mu_bp": m, "base": "n20", "informed": "uniform"} for f in K_FRAC_N20_EXTRA for m in MU_X]
    specs += [{"kind": "sparse_trades", "pi_t": p, "mu_bp": m, "household_specific": False} for p in PI_T_X for m in MU_X]
    return specs


def x_units(clocks, n_panels: int, boot: int = 499) -> list[dict]:
    units = []
    for clock in clocks:
        for spec in x_specs():
            variant = "wins" if spec["kind"] == "sparse_k" else "raw"
            units.append({"grid": GRID_X, "key": _key(clock, variant, spec), "clock": clock, "variant": variant, "spec": spec,
                          "first": 0, "n": n_panels, "hh_draws": boot, "boot_draws": 0, "weights": ("ES",),
                          "t1_raw": True, "set_name": rc.DEFAULT_SET[clock]})
    return units


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Figure 2 v3 power cells (x-axis to 100%)")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K", help="time K representative units on 20 panels and exit")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "run_power_x.log")
    args = ap.parse_args(argv)
    units = x_units(tuple(args.clocks.split(",")), args.n_panels, args.boot)
    t0 = time.time()
    if args.time_cells:
        time_cells(units, args.time_cells, 20, args.log)
        return
    rc.run_units(units, args.workers, args.log)
    rc.log_line(args.log, f"run_power_x finished in {time.time() - t0:.0f}s ({len(units)} units)")


if __name__ == "__main__":
    main()
