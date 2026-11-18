"""Inverted placebo-injection upper bound on tau.

The bound is the largest dispersion tau that would still produce an estimate as small as the observed
one 5% of the time on placebo data: on centered placebo (N2) panels of each clock, dense ability
theta_h ~ N(0, tau^2) is injected (N's `inject/effects.inject_dense`, the §12 seeds, before the pooled
p1/p99 winsorization), the persistence (R1) households are kept and ONE tau2_hat per panel is computed
(`estimators/tau.tau2_household_covariance`; no bootstrap). For each tau of a fine grid the 5th percentile
of tau2_hat over 1,000 panels (draws 0-999) is the curve q05(tau); the bound is the largest tau with
q05(tau) <= observed tau2_hat, with q05 interpolated linearly in tau between the bracketing grid points
(Neyman inversion: coverage of the injected tau is 95% at the grid points by construction).

Also computed on the same injected panels: the curve for the equal-household (EH) and pair (PW)
weights, and for the two robustness outcomes Table 4 prints (`no_alpha` = y_noalpha_h20, `market_adjusted`
= y_mkt_h20, each winsorized at its own pooled cutoffs), with their observed tau2 from `results.csv`.

CLI: python -m trading_on_disclosure_v4.calibration.inverted_bound [--clocks T,P] [--n-panels 1000]
     [--workers 6] [--outcomes headline,no_alpha,market_adjusted] [--aggregate-only]
Outputs: data/v4/calibration/inverted_bound_curve.csv, inverted_bound.csv;
checkpoints: data/v4/restricted/c/checkpoints/inverted_bound/<clock>_<outcome>_tau<tau>_<first>.parquet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.synthetic import realized_tau2_target
from trading_on_disclosure_v4.estimators import tau as tau_mod

GRID = "inverted_bound"
INJECT_GRID = "bound_cal"  # same injection name (and seeds) as §12, so shared taus reproduce C5's tau2_hat exactly
TAUS = (0.0, 25.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 125.0, 150.0)
TAUS_EXTRA = {"headline": (175.0, 200.0, 225.0, 250.0)}  # beyond the original grid: the equal-household bound on P sits above 150
ALPHA = 0.05
WEIGHTS = ("ES", "EH", "PW")
WEIGHT_NAMES = {"ES": "event_share", "EH": "equal_household", "PW": "pair"}
# outcome variant -> (column of the stored N2 batches, pooled-cutoff outcome label, results.csv (analysis, variant))
OUTCOMES = {
    "headline": ("y_h20", "ff5mom_h20", ("tau", None)),
    "no_alpha": ("y_noalpha_h20", "ff5mom_noalpha_h20", ("robustness", "no_alpha")),
    "market_adjusted": ("y_mkt_h20", "mkt_h20", ("robustness", "market_adjusted")),
}
RESULTS_CSV = rc.REPO_ROOT / "data" / "v4" / "results.csv"
CURVE_CSV = rc.CALIB_DIR / "inverted_bound_curve.csv"
BOUND_CSV = rc.CALIB_DIR / "inverted_bound.csv"
CURVE_COLUMNS = ("clock", "weights", "tau_true", "q05_tau2_hat", "q50", "mean", "n_panels", "mc_se_q05")
BOUND_COLUMNS = ("clock", "weights", "observed_tau2", "bound_bp", "bound_annualized_pct", "crossing_lower_grid",
                 "crossing_upper_grid", "coverage_check_min", "coverage_check_max")


# ---------------------------------------------------------------- inversion (pure functions)

def rank_se_quantile(x: np.ndarray, alpha: float = ALPHA, z: float = 1.959964) -> float:
    """Distribution-free Monte Carlo SE of the empirical alpha-quantile of x: the order statistics at ranks
    n*alpha -/+ z*sqrt(n*alpha*(1-alpha)) bracket the quantile with probability ~ 2*Phi(z) - 1; SE = width / (2z)."""
    s = np.sort(np.asarray(x, dtype=float))
    n = len(s)
    if n < 2:
        return float("nan")
    k = n * alpha
    h = z * np.sqrt(n * alpha * (1.0 - alpha))
    lo = int(np.clip(np.floor(k - h), 0, n - 1))
    hi = int(np.clip(np.ceil(k + h), 0, n - 1))
    return float((s[hi] - s[lo]) / (2.0 * z))


def curve_from_panels(tau2_by_tau: dict[float, np.ndarray], alpha: float = ALPHA) -> pd.DataFrame:
    """One row per grid tau: the alpha-quantile of tau2_hat (the curve), its rank-based MC SE, median and mean."""
    rows = []
    for t in sorted(tau2_by_tau):
        v = np.asarray(tau2_by_tau[t], dtype=float)
        v = v[np.isfinite(v)]
        rows.append({"tau_true": float(t), "q05_tau2_hat": float(np.quantile(v, alpha)), "q50": float(np.median(v)),
                     "mean": float(v.mean()), "n_panels": int(len(v)), "mc_se_q05": rank_se_quantile(v, alpha),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan})
    return pd.DataFrame(rows)


def inverted_injection_bound(observed_tau2: float, grid: np.ndarray | pd.DataFrame, q05: np.ndarray | None = None) -> dict:
    """Largest tau on `grid` whose 5th-percentile tau2_hat is <= `observed_tau2`, refined by linear interpolation
    of the q05 curve in tau to the crossing with the next grid point (where q05 > observed).

    `grid` is either the curve frame from `curve_from_panels` or an array of taus with `q05` beside it. Returns
    bound_bp (nan if the observed value is above q05 at the largest grid tau: bound beyond the grid; 0.0 if it is
    below q05 at tau = 0), the bracketing grid taus, the interpolation weight, the same crossing interpolated in
    tau^2 (§12.5's rule), and whether the curve is monotone."""
    if isinstance(grid, pd.DataFrame):
        taus = grid["tau_true"].to_numpy(dtype=float)
        q05 = grid["q05_tau2_hat"].to_numpy(dtype=float)
    else:
        taus = np.asarray(grid, dtype=float)
        q05 = np.asarray(q05, dtype=float)
    order = np.argsort(taus)
    taus, q05 = taus[order], q05[order]
    obs = float(observed_tau2)
    out = {"observed_tau2": obs, "monotone_curve": bool(np.all(np.diff(q05) > 0)), "grid_max": float(taus[-1])}
    below = np.flatnonzero(q05 <= obs)
    if len(below) == 0:
        out.update({"bound_bp": 0.0, "bound_bp_interp_tau2": 0.0, "crossing_lower_grid": np.nan,
                    "crossing_upper_grid": float(taus[0]), "interp_weight": np.nan, "status": "below_grid"})
        return out
    i = int(below[-1])
    if i == len(taus) - 1:
        out.update({"bound_bp": np.nan, "bound_bp_interp_tau2": np.nan, "crossing_lower_grid": float(taus[i]),
                    "crossing_upper_grid": np.nan, "interp_weight": np.nan, "status": "above_grid"})
        return out
    lo, hi = taus[i], taus[i + 1]
    w = (obs - q05[i]) / (q05[i + 1] - q05[i])  # q05[i + 1] > obs >= q05[i] by construction of i
    out.update({"bound_bp": float(lo + w * (hi - lo)), "bound_bp_interp_tau2": float(np.sqrt(lo * lo + w * (hi * hi - lo * lo))),
                "crossing_lower_grid": float(lo), "crossing_upper_grid": float(hi), "interp_weight": float(w),
                "status": "interpolated"})
    return out


def inverted_bounds_many(values: np.ndarray, taus: np.ndarray, q05: np.ndarray) -> np.ndarray:
    """`inverted_injection_bound` for many observed values at once (same rule; nan = beyond the grid, which
    counts as >= every grid tau in the coverage check)."""
    taus = np.asarray(taus, dtype=float)
    q05 = np.asarray(q05, dtype=float)
    order = np.argsort(taus)
    taus, q05 = taus[order], q05[order]
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v))
    le = q05[None, :] <= v[:, None]  # (n, k)
    has = le.any(axis=1)
    idx = np.where(has, le.shape[1] - 1 - np.argmax(le[:, ::-1], axis=1), -1)
    top = idx == len(taus) - 1
    mid = has & ~top
    i = idx[mid]
    w = (v[mid] - q05[i]) / (q05[i + 1] - q05[i])
    out[mid] = taus[i] + w * (taus[i + 1] - taus[i])
    out[top] = np.nan
    return out


def bound_mc_se(tau2_by_tau: dict[float, np.ndarray], observed_tau2: float, n_boot: int = 1000, seed: int = 20260921,
                alpha: float = ALPHA) -> dict:
    """Monte Carlo SE of the bound from a bootstrap over the placebo draws: the draw index is resampled with
    replacement JOINTLY across the grid taus (the same N2 draws underlie every tau, so their noise is shared), the
    q05 curve and the bound are recomputed on each resample. Arrays must be aligned by draw across taus. Returns
    the SD of the resampled bounds (nan bounds, beyond the grid, are dropped and counted) and the 2.5/97.5 percentiles."""
    taus = np.array(sorted(tau2_by_tau), dtype=float)
    m = np.column_stack([np.asarray(tau2_by_tau[t], dtype=float) for t in taus])
    rng = np.random.default_rng(seed)
    n = m.shape[0]
    out = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        q = np.quantile(m[idx], alpha, axis=0)
        out[i] = inverted_injection_bound(observed_tau2, taus, q)["bound_bp"]
    f = out[np.isfinite(out)]
    return {"bound_mc_se_bp": float(f.std(ddof=1)) if len(f) > 1 else np.nan,
            "bound_mc_p025": float(np.quantile(f, 0.025)) if len(f) else np.nan,
            "bound_mc_p975": float(np.quantile(f, 0.975)) if len(f) else np.nan,
            "bound_mc_n_beyond_grid": int(np.sum(~np.isfinite(out))), "bound_mc_n_boot": n_boot}


def coverage_check(tau2_by_tau: dict[float, np.ndarray], curve: pd.DataFrame) -> pd.DataFrame:
    """Share of injected panels at each grid tau whose own inverted bound (from the same curve, nothing left
    out) is >= that tau (nan bounds, beyond the grid, count as covering). 95% at grid points by construction."""
    taus = curve["tau_true"].to_numpy(dtype=float)
    q05 = curve["q05_tau2_hat"].to_numpy(dtype=float)
    rows = []
    for t in sorted(tau2_by_tau):
        v = np.asarray(tau2_by_tau[t], dtype=float)
        v = v[np.isfinite(v)]
        b = inverted_bounds_many(v, taus, q05)
        cov = float(np.mean(np.isnan(b) | (b >= float(t) - 1e-9)))
        rows.append({"tau_true": float(t), "coverage": cov, "mc_se": rc.mc_se(cov, len(v)), "n_panels": int(len(v)),
                     "share_bound_beyond_grid": float(np.mean(np.isnan(b)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- panels (checkpointed work units)

def units(clocks, n_panels: int, outcomes=tuple(OUTCOMES)) -> list[dict]:
    out = []
    for clock in clocks:
        for oc in outcomes:
            for tau in TAUS + TAUS_EXTRA.get(oc, ()):
                for first in range(0, n_panels, rc.BATCH_SIZE):
                    out.append({"grid": GRID, "key": f"{clock}_{oc}_tau{tau:g}_{first:04d}", "clock": clock, "outcome": oc,
                                "tau": float(tau), "first": first, "n": min(rc.BATCH_SIZE, n_panels - first)})
    return out


def panel_tau2(sub: pd.DataFrame, draw: int, weights=WEIGHTS) -> dict:
    """tau2_hat (household-covariance form, activity control) and the realized target per weighting for one R1 panel."""
    out = {"draw": draw, "n_events": int(len(sub))}
    for w in weights:
        table = tau_mod.household_table(sub, w)
        out[f"tau2_hat_{w}"] = tau_mod.tau2_household_covariance(table)
        out[f"target_tau2_{w}"] = float(realized_tau2_target(sub, w))
        out["n_households"] = int(len(table.household_id))
        out["n_events_finite"] = int(table.n_a.sum() + table.n_b.sum())
    return out


_CUTS: dict = {}


def _cutoffs(clock: str, outcome: str) -> dict:
    key = (clock, outcome)
    if key not in _CUTS:
        _CUTS[key] = rc.pooled_cutoffs(clock, OUTCOMES[outcome][1])
    return _CUTS[key]


def run_unit(unit: dict) -> dict:
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0, "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, oc, tau = unit["clock"], unit["outcome"], unit["tau"]
    r1, _, _ = rc._shared(clock)
    cuts = _cutoffs(clock, oc)
    spec = {"kind": "dense", "tau_bp": tau}
    name = f"{INJECT_GRID}|{rc.cell_id(spec)}"
    rows = []
    for p in rc.iter_draws(clock, unit["first"], unit["n"], set_name=rc.DEFAULT_SET[clock], outcome=OUTCOMES[oc][0]):
        q = rc.winsorize(rc.inject(p, spec, name, r1=r1), cuts)
        sub = q.loc[q["household_id"].isin(r1)].reset_index(drop=True)
        rows.append(panel_tau2(sub, int(p["draw"].iloc[0])))
    frame = pd.DataFrame(rows)
    for k, v in (("clock", clock), ("outcome", oc), ("tau_true", tau), ("spec_json", json.dumps(spec, sort_keys=True)),
                 ("panel_set", rc.DEFAULT_SET[clock]), ("injection", "before_winsor")):
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


# ---------------------------------------------------------------- observed values and aggregation

def observed_tau2(clock: str, outcome: str, weights: str, path: Path = RESULTS_CSV) -> float:
    """The actual tau2_hat of clock x outcome x weights from data/v4/results.csv (nan when the row does not exist)."""
    analysis, variant = OUTCOMES[outcome][2]
    variant = variant or WEIGHT_NAMES[weights]
    if analysis == "robustness" and weights != "ES":
        return float("nan")  # the robustness rows are event-share only
    r = pd.read_csv(path)
    r = r[(r["clock"] == clock) & (r["analysis"] == analysis) & (r["variant"] == variant) & (r["statistic"] == "tau2")]
    if len(r) != 1:
        raise ValueError(f"expected one tau2 row for {clock}/{analysis}/{variant}, found {len(r)}")
    return float(r["value"].iloc[0])


def aggregate(clocks, log: Path | None = None, results_csv: Path = RESULTS_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    stats = stats[stats["clock"].isin(clocks)]
    curves, bounds = [], []
    for (clock, oc), g in stats.groupby(["clock", "outcome"], sort=True):
        for w in WEIGHTS:
            by = {float(t): h.sort_values("draw")[f"tau2_hat_{w}"].to_numpy(dtype=float) for t, h in g.groupby("tau_true")}
            curve = curve_from_panels(by)
            cov = coverage_check(by, curve)
            c = curve.merge(cov[["tau_true", "coverage", "mc_se", "share_bound_beyond_grid"]], on="tau_true")
            c = c.rename(columns={"coverage": "coverage_check", "mc_se": "coverage_check_mc_se"})
            c.insert(0, "weights", WEIGHT_NAMES[w])
            c.insert(0, "clock", clock)
            c["outcome"] = oc
            c["n_households"] = int(g["n_households"].iloc[0])
            c["mean_realized_tau2"] = [float(g[g["tau_true"] == t][f"target_tau2_{w}"].mean()) for t in c["tau_true"]]
            curves.append(c)
            obs = observed_tau2(clock, oc, w, results_csv) if results_csv.exists() else float("nan")
            if not np.isfinite(obs):
                continue
            b = inverted_injection_bound(obs, curve)
            bb = b["bound_bp"]
            se = bound_mc_se(by, obs) if len({len(v) for v in by.values()}) == 1 else {}
            bounds.append({"clock": clock, "weights": WEIGHT_NAMES[w], "observed_tau2": obs, "bound_bp": bb,
                           "bound_annualized_pct": bb * tau_mod.ANNUALIZED_SCALE / 100.0,
                           "crossing_lower_grid": b["crossing_lower_grid"], "crossing_upper_grid": b["crossing_upper_grid"],
                           "coverage_check_min": float(cov["coverage"].min()), "coverage_check_max": float(cov["coverage"].max()),
                           "outcome": oc, "status": b["status"], "interp_weight": b["interp_weight"],
                           "bound_bp_interp_tau2": b["bound_bp_interp_tau2"], "monotone_curve": b["monotone_curve"],
                           "q05_at_lower_grid": float(curve.set_index("tau_true")["q05_tau2_hat"].get(b["crossing_lower_grid"], np.nan)),
                           "q05_at_upper_grid": float(curve.set_index("tau_true")["q05_tau2_hat"].get(b["crossing_upper_grid"], np.nan)),
                           "share_tau2_hat_below_observed_at_tau0": float(np.mean(by[0.0] <= obs)) if 0.0 in by else np.nan,
                           "n_panels_per_tau": int(curve["n_panels"].min()), "alpha": ALPHA, **se,
                           "grid": " ".join(f"{t:g}" for t in curve["tau_true"])})
    curve_df = pd.concat(curves, ignore_index=True)
    cols = list(CURVE_COLUMNS) + [c for c in curve_df.columns if c not in CURVE_COLUMNS]
    curve_df = curve_df[cols]
    bound_df = pd.DataFrame(bounds)
    if len(bound_df):
        bound_df = bound_df[list(BOUND_COLUMNS) + [c for c in bound_df.columns if c not in BOUND_COLUMNS]]
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(CURVE_CSV, index=False)
    bound_df.to_csv(BOUND_CSV, index=False)
    rc.log_line(log, f"{CURVE_CSV.name} {len(curve_df)} rows; {BOUND_CSV.name} {len(bound_df)} rows; panels per cell "
                     f"{stats.groupby(['clock', 'outcome', 'tau_true']).size().min()}-{stats.groupby(['clock', 'outcome', 'tau_true']).size().max()}")
    return curve_df, bound_df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="inverted placebo-injection upper bound on tau")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outcomes", default=",".join(OUTCOMES))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "inverted_bound.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if not args.aggregate_only:
        run_units(units(clocks, args.n_panels, tuple(args.outcomes.split(","))), args.workers, args.log)
    aggregate(clocks, args.log)
    rc.log_line(args.log, f"inverted_bound finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
