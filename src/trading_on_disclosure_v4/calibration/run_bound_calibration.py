"""Calibration of the one-sided household-bootstrap upper bound on tau.

On centered placebo (N2) panels of each clock, dense ability theta_h ~ N(0, tau^2) with tau in
{0, 50, 75, 100} bp is injected (N's `inject/effects.py`, before the pooled p1/p99 winsorization
as in every power cell of §6v2), the persistence (R1) households are kept, and the household
bootstrap of tau2 (`inference/block_bootstrap.tau2_household_bootstrap`, event-share weights,
4,999 draws = the actual run's count) is drawn once per panel. For each nominal level q in LEVELS
the bound is the q-th percentile of the bootstrap tau2; coverage is the share of panels on which
that percentile is >= tau^2 (identical to "signed root of the percentile >= tau" on the tau
scale; the actual run's `tau_upper` clips the root at 0, which only matters at tau = 0). The
median bound and the mean gap (bound - tau) are on the tau scale with the clipped root, as the
paper prints the bound. A second pass injects after winsorization (sensitivity).

Calibrated level (per clock): the smallest grid level whose coverage is >= 0.95 at every tau
of the grid; beside it, the level at which the minimum-over-tau coverage crosses 0.95 and the
level at which the grid-average coverage crosses 0.95, both by linear interpolation of coverage
between adjacent grid levels. Levels above 0.99 and taus 125 and 150 are run beyond the
requested grid; the extra taus feed the test-inversion table (`inversion`): the alpha-quantile
of tau2_hat at each injected tau, and the tau at which it equals the actual headline tau2_hat.

CLI: python -m trading_on_disclosure_v4.calibration.run_bound_calibration [--clocks T,P]
     [--n-panels 1000] [--boot 4999] [--workers 6] [--aggregate-only]
Outputs: data/v4/calibration/upper_bound_calibration.csv (primary, injection before winsorization),
         upper_bound_calibration_inject_after_winsor.csv (sensitivity),
         upper_bound_calibration_choice.csv (one row per clock x injection order),
         upper_bound_calibration_inversion.csv (tau2_hat quantiles by injected tau; inverted bound);
checkpoints: data/v4/restricted/c/checkpoints/bound_cal/<clock>_<order>_tau<tau>_<first>.parquet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.synthetic import realized_tau2_target, seed_from_name
from trading_on_disclosure_v4.inference import block_bootstrap as bb

GRID = "bound_cal"
TAUS = (0.0, 50.0, 75.0, 100.0)  # the researcher's grid: the level rule is judged on these
TAUS_RUN = (0.0, 50.0, 75.0, 100.0, 125.0, 150.0)  # two more for the test-inversion bound
LEVELS = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99, 0.995, 0.9975, 0.999)  # 0.995+ beyond the requested grid
INVERSION_ALPHAS = (0.05, 0.10)
RESULTS_CSV = rc.REPO_ROOT / "data" / "v4" / "results.csv"
ORDERS = ("before_winsor", "after_winsor")
WEIGHTS = "ES"
TARGET = 0.95
N_BOOT = 4999
OUT_COLUMNS = ("clock", "tau_true", "level", "coverage", "mc_se", "median_bound", "mean_gap", "n_panels", "n_boot")


def _lvl(level: float) -> str:
    return f"{level:.3f}".rstrip("0").rstrip(".")


def units(clocks, n_panels: int, boot: int, orders=ORDERS) -> list[dict]:
    out = []
    for clock in clocks:
        for order in orders:
            for tau in TAUS_RUN:
                for first in range(0, n_panels, rc.BATCH_SIZE):
                    out.append({"grid": GRID, "key": f"{clock}_{order}_tau{tau:g}_{first:04d}", "clock": clock, "order": order,
                                "tau": float(tau), "first": first, "n": min(rc.BATCH_SIZE, n_panels - first), "boot": boot})
    return out


def panel_bound_stats(sub: pd.DataFrame, draw: int, boot: int, levels=LEVELS) -> dict:
    """tau2_hat, realized target, bootstrap percentiles at `levels` and bootstrap SD for one R1 panel."""
    res = bb.tau2_household_bootstrap(sub, WEIGHTS, boot, seed_from_name(f"I2b|{draw}"))
    d = np.asarray(res["statistics"], dtype=float)
    f = d[np.isfinite(d)]
    out = {"draw": draw, "n_households": int(res["n_households"]), "n_events": int(len(sub)),
           "tau2_hat": float(res["tau2_hat"]), "target_tau2": float(realized_tau2_target(sub, WEIGHTS)),
           "boot_sd": float(np.std(f, ddof=1)) if len(f) > 1 else np.nan, "n_valid": int(res["n_valid"]),
           "n_invalid": int(res["n_invalid_draws"])}
    qs = np.quantile(f, levels) if len(f) else np.full(len(levels), np.nan)
    for lv, q in zip(levels, qs, strict=True):
        out[f"q_{_lvl(lv)}"] = float(q)
    return out


def run_unit(unit: dict) -> dict:
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0, "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, order, tau = unit["clock"], unit["order"], unit["tau"]
    r1, _, cuts = rc._shared(clock)
    spec = {"kind": "dense", "tau_bp": tau}
    name = f"{GRID}|{rc.cell_id(spec)}"
    rows = []
    for p in rc.iter_draws(clock, unit["first"], unit["n"], set_name=rc.DEFAULT_SET[clock]):
        if order == "before_winsor":
            q = rc.winsorize(rc.inject(p, spec, name, r1=r1), cuts)
        else:
            q = rc.inject(rc.winsorize(p, cuts), spec, name, r1=r1)
        sub = q.loc[q["household_id"].isin(r1)].reset_index(drop=True)
        rows.append(panel_bound_stats(sub, int(p["draw"].iloc[0]), unit["boot"]))
    frame = pd.DataFrame(rows)
    for k, v in (("clock", clock), ("order", order), ("tau_true", tau), ("n_boot", unit["boot"]), ("weights", WEIGHTS),
                 ("spec_json", json.dumps(spec, sort_keys=True)), ("panel_set", rc.DEFAULT_SET[clock])):
        frame[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": unit["key"], "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


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

def _bound_bp(q: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(q, 0.0))


def summarize(stats: pd.DataFrame, levels=LEVELS, taus=TAUS) -> pd.DataFrame:
    """One row per clock x order x tau x level: coverage of tau (and of the realized target), bound moments."""
    rows = []
    for (clock, order, tau), g in stats.groupby(["clock", "order", "tau_true"], sort=True):
        tau = float(tau)
        if tau not in taus:
            continue
        for lv in levels:
            q = g[f"q_{_lvl(lv)}"].to_numpy(dtype=float)
            ok = np.isfinite(q)
            n = int(ok.sum())
            cov = float(np.mean(q[ok] >= tau * tau))
            cov_r = float(np.mean(q[ok] >= g["target_tau2"].to_numpy(dtype=float)[ok]))
            b = _bound_bp(q[ok])
            rows.append({"clock": clock, "tau_true": tau, "level": lv, "coverage": cov, "mc_se": rc.mc_se(cov, n),
                         "median_bound": float(np.median(b)), "mean_gap": float(np.mean(b - tau)), "n_panels": n,
                         "n_boot": int(g["n_boot"].iloc[0]), "injection": order, "coverage_realized_target": cov_r,
                         "mean_bound": float(np.mean(b)), "share_bound_root_clipped": float(np.mean(q[ok] < 0.0)),
                         "mean_tau2_hat": float(g["tau2_hat"].mean()), "mean_realized_tau2": float(g["target_tau2"].mean()),
                         "mean_boot_sd": float(g["boot_sd"].mean()), "mc_sd_tau2_hat": float(g["tau2_hat"].std(ddof=1)),
                         "mean_invalid_draws": float(g["n_invalid"].mean()), "n_households": int(g["n_households"].iloc[0])})
    return pd.DataFrame(rows)


def _cross(levels: np.ndarray, cov: np.ndarray, target: float = TARGET) -> float:
    """Smallest level at which the piecewise-linear interpolation of cov(level) reaches `target`
    (nan if never; the lowest grid level if already above target there)."""
    if cov[0] >= target:
        return float(levels[0])
    for i in range(1, len(levels)):
        if cov[i] >= target:
            c0, c1 = cov[i - 1], cov[i]
            if c1 == c0:
                return float(levels[i])
            return float(levels[i - 1] + (target - c0) / (c1 - c0) * (levels[i] - levels[i - 1]))
    return float("nan")


def choose(summary: pd.DataFrame, levels=LEVELS, target: float = TARGET, cov_col: str = "coverage") -> pd.DataFrame:
    """Per clock x injection order: the calibrated level and the two interpolated companions, judged on
    `cov_col` ("coverage" = the injected population tau; "coverage_realized_target" = the panel's own
    realized dispersion)."""
    rows = []
    lv = np.asarray(levels, dtype=float)
    for (clock, order), g in summary.groupby(["clock", "injection"], sort=True):
        piv = g.pivot(index="tau_true", columns="level", values=cov_col).reindex(columns=levels)
        cmin, cmean = piv.min(axis=0).to_numpy(dtype=float), piv.mean(axis=0).to_numpy(dtype=float)
        grid_ok = [float(lv[i]) for i in range(len(lv)) if cmin[i] >= target]
        chosen = min(grid_ok) if grid_ok else float("nan")
        cov_at = {f"{t:g}": float(piv.loc[t, chosen]) for t in piv.index} if grid_ok else {}
        rows.append({"clock": clock, "injection": order, "target": "population_tau" if cov_col == "coverage" else "realized_dispersion",
                     "calibrated_level": chosen,
                     "rule": f"smallest grid level with coverage >= {target:.2f} at every tau in {{{', '.join(f'{t:g}' for t in TAUS)}}} bp",
                     "coverage_at_choice_by_tau": json.dumps(cov_at),
                     "level_min_coverage_crosses_target": _cross(lv, cmin, target),
                     "level_mean_coverage_crosses_target": _cross(lv, cmean, target),
                     "min_coverage_at_nominal_95": float(piv[0.95].min()), "mean_coverage_at_nominal_95": float(piv[0.95].mean()),
                     "coverage_at_nominal_95_by_tau": json.dumps({f"{t:g}": float(piv.loc[t, 0.95]) for t in piv.index}),
                     "min_coverage_at_0.99": float(piv[0.99].min()), "min_coverage_at_max_level": float(cmin[-1]), "max_level": float(lv[-1]),
                     "binding_tau": float(piv.idxmin(axis=0)[chosen]) if grid_ok else float(piv.idxmin(axis=0)[lv[-1]]),
                     "n_panels_per_tau": int(g["n_panels"].min()), "n_boot": int(g["n_boot"].iloc[0])})
    return pd.DataFrame(rows)


def observed_tau2(clock: str, path: Path = RESULTS_CSV) -> float:
    """The actual headline tau2_hat (event-share weights) of `clock` from data/v4/results.csv."""
    r = pd.read_csv(path)
    r = r[(r["clock"] == clock) & (r["analysis"] == "tau") & (r["variant"] == "event_share") & (r["statistic"] == "tau2")]
    if len(r) != 1:
        raise ValueError(f"expected one headline tau2 row for clock {clock}, found {len(r)}")
    return float(r["value"].iloc[0])


def _invert(taus: np.ndarray, q: np.ndarray, observed: float) -> float:
    """Largest tau at which the alpha-quantile of tau2_hat is <= the observed tau2_hat: linear
    interpolation of the quantile in tau^2 between the two bracketing grid taus (nan when the observed
    value is above the quantile at the largest tau; 0 when below the quantile at tau = 0)."""
    t2 = taus * taus
    if observed < q[0]:
        return 0.0
    for i in range(1, len(taus)):
        if q[i] > observed >= q[i - 1]:
            return float(np.sqrt(t2[i - 1] + (observed - q[i - 1]) / (q[i] - q[i - 1]) * (t2[i] - t2[i - 1])))
    return float("nan")


def inversion(stats: pd.DataFrame, taus=TAUS_RUN, alphas=INVERSION_ALPHAS, observed: dict[str, float] | None = None) -> pd.DataFrame:
    """Test-inversion bound: per clock x order, the alpha-quantile of tau2_hat at each injected tau, and the
    tau at which that quantile equals the actual headline tau2_hat (a Monte Carlo (1 - alpha) upper bound for
    the injected tau by construction, up to the interpolation and the dense-N(0, tau^2) model)."""
    rows = []
    for (clock, order), g in stats.groupby(["clock", "order"], sort=True):
        obs = observed.get(clock, np.nan) if observed else np.nan
        tv = np.asarray([t for t in taus if t in set(g["tau_true"])], dtype=float)
        by = {t: g[g["tau_true"] == t]["tau2_hat"].to_numpy(dtype=float) for t in tv}
        for t in tv:
            h = by[t]
            rows.append({"clock": clock, "injection": order, "tau_true": t, "n_panels": int(len(h)), "mean_tau2_hat": float(h.mean()),
                         "sd_tau2_hat": float(h.std(ddof=1)), "median_tau2_hat": float(np.median(h)),
                         **{f"q{int(a * 100):02d}_tau2_hat": float(np.quantile(h, a)) for a in alphas},
                         "observed_tau2_hat": obs, "share_tau2_hat_below_observed": float(np.mean(h <= obs)) if np.isfinite(obs) else np.nan,
                         **{f"inverted_bound_{int((1 - a) * 100)}": _invert(tv, np.array([np.quantile(by[x], a) for x in tv]), obs) if np.isfinite(obs) else np.nan
                            for a in alphas}})
    return pd.DataFrame(rows)


def aggregate(clocks, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    stats = stats[stats["clock"].isin(clocks)]
    summary = summarize(stats)
    choice = pd.concat([choose(summary), choose(summary, cov_col="coverage_realized_target")], ignore_index=True)
    obs = {c: observed_tau2(c) for c in clocks} if RESULTS_CSV.exists() else None
    inversion(stats, observed=obs).to_csv(rc.CALIB_DIR / "upper_bound_calibration_inversion.csv", index=False)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(OUT_COLUMNS) + [c for c in summary.columns if c not in OUT_COLUMNS]
    primary = summary[summary["injection"] == "before_winsor"][cols]
    primary.to_csv(rc.CALIB_DIR / "upper_bound_calibration.csv", index=False)
    summary[summary["injection"] == "after_winsor"][cols].to_csv(rc.CALIB_DIR / "upper_bound_calibration_inject_after_winsor.csv", index=False)
    choice.to_csv(rc.CALIB_DIR / "upper_bound_calibration_choice.csv", index=False)
    rc.log_line(log, f"upper_bound_calibration.csv {len(primary)} rows; choice {len(choice)} rows; panels per cell "
                     f"{stats.groupby(['clock', 'order', 'tau_true']).size().to_dict()}")
    return summary, choice


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="coverage calibration of the one-sided household-bootstrap upper bound on tau")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "run_bound_calibration.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if not args.aggregate_only:
        run_units(units(clocks, args.n_panels, args.boot, tuple(args.orders.split(","))), args.workers, args.log)
    aggregate(clocks, args.log)
    rc.log_line(args.log, f"run_bound_calibration finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
