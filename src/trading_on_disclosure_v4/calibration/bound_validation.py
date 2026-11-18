"""Out-of-sample validation of the inverted placebo-injection upper bound.

Section 13 builds the inversion curve q05(tau) on centered placebo datasets 1-1000 (draws 0-999) and
reads the bound off that same curve, so its 95% coverage at the grid points is arithmetic, not evidence.
This module separates the two samples and changes the shape of the injected household effect:

  * the curve is built on datasets 1-1000 exactly as in section 13 (`inverted_bound.py`, same estimator,
    same winsorization, same seeds);
  * a known dispersion tau is injected into the HELD-OUT datasets 1001-2000, each of those datasets is
    given the bound implied by the first half's curve, and the share with bound >= tau is the coverage;
  * the household effect is drawn from four distributions, each scaled to the same dispersion tau:
    normal (the shape section 13 assumes), two-point half-and-half (+/- tau), two-point skewed (a fifth of
    households high) and Student t with 3 degrees of freedom (heavy-tailed). Coverage is measured both
    with the curve built under the same shape ("matched") and with the normal curve the paper would use
    ("normal curve"), which is what the bound would actually deliver if the truth were not normal.

The bound at the observed tau2 is inverted against each shape's curve as well: the bound is only as good
as the shape assumed when the curve was built, and the table says by how much.

CLI: python -m trading_on_disclosure_v4.calibration.bound_validation [--clocks T,P]
     [--shapes gaussian,two_point_half,two_point_skew,student_t3] [--workers 8] [--time-cells 2]
     [--aggregate-only] [--verify]
Outputs: data/v4/calibration/bound_validation.csv;
checkpoints: data/v4/restricted/c/checkpoints/bound_validation/<clock>_<shape>_tau<tau>_<first>.parquet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import inverted_bound as ib
from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.synthetic import realized_tau2_target
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.inject.effects import HH, panel_id_of
from trading_on_disclosure_v4.outcomes.seeds import seed_from

GRID = "bound_validation"
OUTCOME = "headline"  # the headline stream only: y_h20, pooled p1/p99 cutoffs, event-share weights
WEIGHTS = "ES"
TAUS = ib.TAUS + ib.TAUS_EXTRA["headline"]  # the section-13 fine grid, 0 to 250 bp
CURVE_FIRST, CURVE_N = 0, 1000  # placebo datasets 1-1000
HELD_FIRST, HELD_N = 1000, 1000  # placebo datasets 1001-2000
SHAPES = ("gaussian", "two_point_half", "two_point_skew", "student_t3")
SKEW_P = 0.2  # two-point skewed: this share of households takes the high value
T_DF = 3  # Student t degrees of freedom (variance df / (df - 2) = 3)
SHAPE_LABEL = {"gaussian": "normal", "two_point_half": "two-point, half and half",
               "two_point_skew": f"two-point, {SKEW_P:.0%} high", "student_t3": f"Student t({T_DF})"}
OUT_CSV = rc.CALIB_DIR / "bound_validation.csv"
COLUMNS = ("clock", "effect_shape", "tau_true", "level_or_method", "coverage", "mc_se", "median_bound", "n_panels")


# ---------------------------------------------------------------- household-effect shapes

def draw_theta(shape: str, tau: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n household effects with mean 0 and standard deviation `tau`, in the named shape."""
    if tau == 0.0:
        return np.zeros(n)
    if shape == "gaussian":
        return rng.normal(0.0, tau, size=n)
    if shape == "two_point_half":
        return tau * np.where(rng.random(n) < 0.5, 1.0, -1.0)
    if shape == "two_point_skew":
        high = rng.random(n) < SKEW_P
        return np.where(high, tau * np.sqrt((1.0 - SKEW_P) / SKEW_P), -tau * np.sqrt(SKEW_P / (1.0 - SKEW_P)))
    if shape == "student_t3":
        return tau * rng.standard_t(T_DF, size=n) / np.sqrt(T_DF / (T_DF - 2.0))
    raise ValueError(f"unknown effect shape {shape!r}; expected one of {SHAPES}")


def inject_shape(panel: pd.DataFrame, shape: str, tau: float, y_col: str = "y_h20") -> pd.DataFrame:
    """theta_h of the named shape added to every event of household h, fresh per draw.

    `gaussian` goes through N's `inject/effects.inject_dense` under section 13's injection name, so the
    injected values (and therefore tau2_hat) of draws 0-999 reproduce `inverted_bound.py` exactly."""
    if shape == "gaussian":
        spec = {"kind": "dense", "tau_bp": float(tau)}
        return rc.inject(panel, spec, f"{ib.INJECT_GRID}|{rc.cell_id(spec)}", y=y_col)
    name = f"{GRID}|shape={shape}|tau_bp={float(tau)}"
    pid = panel_id_of(panel)
    out = panel.copy()
    y = out[y_col].to_numpy(dtype=float).copy()
    theta_row = np.zeros(len(out))
    for draw, ix in out.groupby("draw", sort=True).indices.items():
        hh, inv = np.unique(out[HH].to_numpy()[ix], return_inverse=True)
        theta = draw_theta(shape, float(tau), len(hh), np.random.default_rng(seed_from(name, int(draw), shape, pid)))
        y[ix] += theta[inv]
        theta_row[ix] = theta[inv]
    out[y_col] = y
    out["theta_h"] = theta_row
    return out


# ---------------------------------------------------------------- work units

def units(clocks, shapes, taus=TAUS) -> list[dict]:
    """One unit per clock x shape x tau x batch of 100 datasets, over both halves (draws 0-1999).

    The normal shape on draws 0-999 is already stored under the `inverted_bound` grid and is skipped."""
    out = []
    for clock in clocks:
        for shape in shapes:
            for tau in taus:
                for first in range(CURVE_FIRST, HELD_FIRST + HELD_N, rc.BATCH_SIZE):
                    if shape == "gaussian" and first < HELD_FIRST:
                        continue
                    out.append({"grid": GRID, "key": f"{clock}_{shape}_tau{tau:g}_{first:04d}", "clock": clock,
                                "shape": shape, "tau": float(tau), "first": first, "n": rc.BATCH_SIZE})
    return out


def run_unit(unit: dict) -> dict:
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0,
                "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, shape, tau = unit["clock"], unit["shape"], unit["tau"]
    r1, _, _ = rc._shared(clock)
    cuts = ib._cutoffs(clock, OUTCOME)
    rows = []
    for p in rc.iter_draws(clock, unit["first"], unit["n"], set_name=rc.DEFAULT_SET[clock],
                           outcome=ib.OUTCOMES[OUTCOME][0]):
        q = rc.winsorize(inject_shape(p, shape, tau), cuts)
        sub = q.loc[q["household_id"].isin(r1)].reset_index(drop=True)
        table = tau_mod.household_table(sub, WEIGHTS)
        rows.append({"draw": int(p["draw"].iloc[0]), "n_events": int(len(sub)),
                     "n_households": int(len(table.household_id)),
                     f"tau2_hat_{WEIGHTS}": tau_mod.tau2_household_covariance(table),
                     f"target_tau2_{WEIGHTS}": float(realized_tau2_target(sub, WEIGHTS))})
    frame = pd.DataFrame(rows)
    for k, v in (("clock", clock), ("effect_shape", shape), ("tau_true", tau), ("outcome", OUTCOME),
                 ("panel_set", rc.DEFAULT_SET[clock]), ("spec_json", json.dumps({"shape": shape, "tau_bp": tau}))):
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


# ---------------------------------------------------------------- reading the two halves

def _stored(clock: str, shape: str) -> pd.DataFrame:
    g = rc.load_grid(GRID)
    return g[(g["clock"] == clock) & (g["effect_shape"] == shape)]


def tau2_by_tau(clock: str, shape: str, first: int, n: int) -> dict[float, np.ndarray]:
    """{tau: tau2_hat over draws [first, first + n), ordered by draw} for one clock and effect shape.

    The normal shape on the first half is read from the stored `inverted_bound` checkpoints (headline
    outcome), which the injection code path reproduces exactly."""
    if shape == "gaussian" and first < HELD_FIRST:
        g = rc.load_grid(ib.GRID)
        g = g[(g["clock"] == clock) & (g["outcome"] == OUTCOME)]
    else:
        g = _stored(clock, shape)
    g = g[(g["draw"] >= first) & (g["draw"] < first + n)]
    out = {}
    for t, h in g.groupby("tau_true"):
        v = h.sort_values("draw")[f"tau2_hat_{WEIGHTS}"].to_numpy(dtype=float)
        if len(v):
            out[float(t)] = v
    return out


# ---------------------------------------------------------------- coverage and bounds

def held_out_coverage(curve: pd.DataFrame, held: dict[float, np.ndarray]) -> pd.DataFrame:
    """Coverage of the held-out datasets: each one's bound comes from `curve`, built on the other half."""
    taus = curve["tau_true"].to_numpy(dtype=float)
    q05 = curve["q05_tau2_hat"].to_numpy(dtype=float)
    rows = []
    for t in sorted(held):
        v = held[t]
        v = v[np.isfinite(v)]
        b = ib.inverted_bounds_many(v, taus, q05)
        cov = float(np.mean(np.isnan(b) | (b >= float(t) - 1e-9)))
        rows.append({"tau_true": float(t), "coverage": cov, "mc_se": rc.mc_se(cov, len(v)),
                     "median_bound": float(np.nanmedian(b)) if np.isfinite(b).any() else np.nan,
                     "n_panels": int(len(v)), "share_bound_beyond_grid": float(np.mean(np.isnan(b))),
                     "mean_tau2_hat": float(v.mean())})
    return pd.DataFrame(rows)


def aggregate(clocks, shapes, log: Path | None = None, results_csv: Path = ib.RESULTS_CSV) -> pd.DataFrame:
    rows = []
    for clock in clocks:
        obs = ib.observed_tau2(clock, OUTCOME, WEIGHTS, results_csv)
        curves = {s: ib.curve_from_panels(tau2_by_tau(clock, s, CURVE_FIRST, CURVE_N)) for s in shapes}
        for shape in shapes:
            held = tau2_by_tau(clock, shape, HELD_FIRST, HELD_N)
            if not held:
                continue
            for source, label in ((shape, "out_of_sample_matched_curve"), ("gaussian", "out_of_sample_normal_curve")):
                if source == shape and label.endswith("normal_curve"):
                    continue
                cov = held_out_coverage(curves[source], held)
                for r in cov.itertuples():
                    rows.append({"clock": clock, "effect_shape": shape, "tau_true": r.tau_true,
                                 "level_or_method": label, "coverage": r.coverage, "mc_se": r.mc_se,
                                 "median_bound": r.median_bound, "n_panels": r.n_panels,
                                 "curve_shape": source, "curve_draws": f"{CURVE_FIRST + 1}-{CURVE_FIRST + CURVE_N}",
                                 "held_draws": f"{HELD_FIRST + 1}-{HELD_FIRST + HELD_N}",
                                 "share_bound_beyond_grid": r.share_bound_beyond_grid,
                                 "mean_tau2_hat_held": r.mean_tau2_hat,
                                 "q05_curve": float(curves[source].set_index("tau_true")["q05_tau2_hat"].get(r.tau_true, np.nan)),
                                 "observed_tau2": obs})
            # in-sample comparison: the same-half curve applied to its own datasets (section 13's 95% by construction)
            own = tau2_by_tau(clock, shape, CURVE_FIRST, CURVE_N)
            ins = held_out_coverage(curves[shape], own)
            for r in ins.itertuples():
                rows.append({"clock": clock, "effect_shape": shape, "tau_true": r.tau_true,
                             "level_or_method": "in_sample_matched_curve", "coverage": r.coverage, "mc_se": r.mc_se,
                             "median_bound": r.median_bound, "n_panels": r.n_panels, "curve_shape": shape,
                             "curve_draws": f"{CURVE_FIRST + 1}-{CURVE_FIRST + CURVE_N}",
                             "held_draws": f"{CURVE_FIRST + 1}-{CURVE_FIRST + CURVE_N}",
                             "share_bound_beyond_grid": r.share_bound_beyond_grid,
                             "mean_tau2_hat_held": r.mean_tau2_hat,
                             "q05_curve": float(curves[shape].set_index("tau_true")["q05_tau2_hat"].get(r.tau_true, np.nan)),
                             "observed_tau2": obs})
            # the bound the paper would report if the household effect had this shape
            b = ib.inverted_injection_bound(obs, curves[shape])
            se = ib.bound_mc_se(tau2_by_tau(clock, shape, CURVE_FIRST, CURVE_N), obs)
            rows.append({"clock": clock, "effect_shape": shape, "tau_true": np.nan,
                         "level_or_method": "bound_at_observed_tau2", "coverage": np.nan,
                         "mc_se": se.get("bound_mc_se_bp", np.nan), "median_bound": b["bound_bp"],
                         "n_panels": CURVE_N, "curve_shape": shape,
                         "curve_draws": f"{CURVE_FIRST + 1}-{CURVE_FIRST + CURVE_N}", "held_draws": "",
                         "share_bound_beyond_grid": np.nan, "mean_tau2_hat_held": np.nan,
                         "q05_curve": np.nan, "observed_tau2": obs, "bound_status": b["status"],
                         "bound_bracket_lower": b["crossing_lower_grid"], "bound_bracket_upper": b["crossing_upper_grid"],
                         "bound_annualized_pct": b["bound_bp"] * tau_mod.ANNUALIZED_SCALE / 100.0})
    out = pd.DataFrame(rows)
    out = out[list(COLUMNS) + [c for c in out.columns if c not in COLUMNS]]
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    rc.log_line(log, f"{OUT_CSV.name} {len(out)} rows")
    return out


def verify_reproduction(clock: str = "T", tau: float = 100.0, n: int = 20) -> dict:
    """Injecting the normal shape here reproduces `inverted_bound.py` on the same draws (max |diff|)."""
    r1, _, _ = rc._shared(clock)
    cuts = ib._cutoffs(clock, OUTCOME)
    mine = []
    for p in rc.iter_draws(clock, 0, n, set_name=rc.DEFAULT_SET[clock], outcome=ib.OUTCOMES[OUTCOME][0]):
        q = rc.winsorize(inject_shape(p, "gaussian", tau), cuts)
        sub = q.loc[q["household_id"].isin(r1)].reset_index(drop=True)
        mine.append({"draw": int(p["draw"].iloc[0]),
                     "mine": tau_mod.tau2_household_covariance(tau_mod.household_table(sub, WEIGHTS))})
    g = rc.load_grid(ib.GRID)
    g = g[(g["clock"] == clock) & (g["outcome"] == OUTCOME) & (g["tau_true"] == tau) & (g["draw"] < n)]
    m = pd.DataFrame(mine).merge(g[["draw", f"tau2_hat_{WEIGHTS}"]], on="draw")
    d = float(np.max(np.abs(m["mine"] - m[f"tau2_hat_{WEIGHTS}"])))
    return {"clock": clock, "tau": tau, "n_draws": int(len(m)), "max_abs_diff": d}


def time_cells(clocks, shapes, n_cells: int, n_panels: int, log: Path | None) -> None:
    """Time n_cells representative units on n_panels datasets each; extrapolate to the whole grid."""
    picks = units(clocks, shapes)[:n_cells]
    per = []
    for u in picks:
        r = run_unit({**u, "grid": "bound_validation_timing", "key": u["key"] + f"_n{n_panels}", "n": n_panels})
        per.append(r["seconds"] / max(r["n"], 1))
        rc.log_line(log, f"timing {u['key']}: {r['seconds']:.1f}s for {r['n']} datasets = {per[-1]:.3f} s/dataset")
    n_units = len(units(clocks, shapes))
    est = float(np.mean(per)) * rc.BATCH_SIZE * n_units
    rc.log_line(log, f"extrapolation: {n_units} units x 100 datasets = {est:.0f} CPU-s "
                     f"({est / 3600:.2f} CPU-h); /8 workers = {est / 8 / 60:.1f} min")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="out-of-sample validation of the inverted injection bound")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "bound_validation.log")
    args = ap.parse_args(argv)
    clocks, shapes = tuple(args.clocks.split(",")), tuple(args.shapes.split(","))
    t0 = time.time()
    if args.verify:
        rc.log_line(args.log, f"reproduction check: {verify_reproduction()}")
        return
    if args.time_cells:
        time_cells(clocks, shapes, args.time_cells, 20, args.log)
        return
    if not args.aggregate_only:
        run_units(units(clocks, shapes), args.workers, args.log)
    aggregate(clocks, shapes, args.log)
    rc.log_line(args.log, f"bound_validation finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
