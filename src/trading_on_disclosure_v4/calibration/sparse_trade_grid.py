"""Informed-purchase alternative at a stated prevalence and effect.

A share pi of purchases is informed and each informed purchase earns mu basis points over the holding
horizon; the informed purchases are either spread uniformly over every purchase or confined to a
fraction of households. The module reports, over centered placebo datasets injected with that
alternative, the power of the three tests the paper runs on the whole purchase record — the
block-clustered mean test, the tail test (share of standardized outcomes above 3 and below -3) and the
best-household test — together with the distribution of each test statistic under the alternative and
how compatible the OBSERVED statistics are with it (the share of injected datasets whose statistic is at
least as extreme as the observed one, in the direction of the observed value).

It exists so that an external estimate of prevalence and effect size can be run through the paper's own
tests in one command. The grid below is a bracketing grid chosen here; it carries no claim about any
particular paper's numbers, and `bcc_cell(pi, mu, horizon)` runs an arbitrary cell when a published pair
is in hand. Nothing in this module supplies a prevalence or an effect size of its own.

Design: all households, both periods, raw outcome (never winsorized: the tail test is defined on raw Y
and the mean and best-household statistics are reported here on the same Y so the three are comparable);
horizon 10, 20 or 60 trading sessions from the stored caches, with the standardizing scale recomputed at
that horizon; effects are quoted per 20 sessions and scaled by horizon/20 unless `--no-scale-mu`.
Critical values and null tail shares come from zero-effect placebo datasets 1001-2000, never a dataset
used for power.

CLI: python -m trading_on_disclosure_v4.calibration.sparse_trade_grid [--clocks T,P] [--n-panels 500]
     [--horizon 20] [--workers 8] [--time-cells 2] [--aggregate-only]
     [--cell PI,MU] [--selection uniform|concentrated]
Outputs: data/v4/calibration/sparse_trade_grid.csv (power and compatibility per cell x test, with the
results.csv row behind each observed statistic in observed_source) and sparse_trade_statistics.csv (the
distribution of each statistic per cell); checkpoints:
data/v4/restricted/c/checkpoints/{sparse_trade,sparse_trade_ref}/.
The observed statistics are read from raw-outcome rows of results.csv, because the injected ones are computed
on raw Y; results.csv publishes none of them away from 20 sessions, where they are emitted as missing with a
note (see OBSERVED_ROWS).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.inject.effects import HH, panel_id_of
from trading_on_disclosure_v4.null import run_null_panels as rnp
from trading_on_disclosure_v4.outcomes.seeds import seed_from

GRID = "sparse_trade"
REF_GRID = "sparse_trade_ref"
PI_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
MU_GRID = (100.0, 250.0, 500.0, 1000.0, 2000.0)  # bp per informed purchase per 20 trading sessions
SELECTIONS = ("uniform", "concentrated")
CONCENTRATED_FRACTION = 0.25  # share of households that hold every informed purchase in the concentrated variant
HORIZONS = {10: "y_h10", 20: "y_h20", 60: "y_h60"}
REF_FIRST, REF_N = 1000, 1000
ALPHA = 0.05
Z_TAIL = 3.0
OUT_CSV = rc.CALIB_DIR / "sparse_trade_grid.csv"
STAT_CSV = rc.CALIB_DIR / "sparse_trade_statistics.csv"
TESTS = ("mean_block", "tail_up", "tail_down", "tail_both", "best_household")
TEST_NAME = {"mean_block": "block-clustered mean test", "tail_up": f"tail test, z > {Z_TAIL:g}",
             "tail_down": f"tail test, z < -{Z_TAIL:g}", "tail_both": f"tail test, |z| > {Z_TAIL:g}",
             "best_household": "best-household test"}
STAT_OF = {"mean_block": "mean_bp", "tail_up": "share_gt_3", "tail_down": "share_lt_minus_3",
           "tail_both": "share_abs_gt_3", "best_household": "max_t"}


# ---------------------------------------------------------------- panels at a horizon

def iter_draws(clock: str, first: int, n: int, horizon: int = 20, flavour: str = "N2"):
    """Placebo datasets with the outcome and the standardizing scale of `horizon` trading sessions.

    `flavour` picks the placebo stream: N2, the centered one (the default everywhere in this module),
    or N1, the timing-only one, whose outcomes are the raw shifted returns and therefore keep each
    household's persistent security choice."""
    outcome = HORIZONS[horizon]
    last = first + n - 1
    for b in range(first // rc.BATCH_SIZE, last // rc.BATCH_SIZE + 1):
        batch = rc.load_batch(clock, b, flavour, rc.DEFAULT_SET[clock], outcome)
        for d, g in batch.groupby("draw", sort=True):
            if first <= int(d) <= last:
                g = g.reset_index(drop=True)
                for c in ("event_id", "household_id", "period"):
                    g[c] = g[c].astype(str)
                yield rnp.harness_view(g, outcome, horizon=horizon)


def scaled_mu(mu_bp_per20: float, horizon: int, scale: bool = True) -> float:
    """The effect actually injected: `mu` is quoted per 20 sessions and pro-rated to the horizon."""
    return float(mu_bp_per20) * (horizon / 20.0 if scale else 1.0)


# ---------------------------------------------------------------- injection

def inject_informed_purchases(panel: pd.DataFrame, pi: float, mu: float, selection: str, name: str,
                              frac: float = CONCENTRATED_FRACTION, y_col: str = "y_h20") -> pd.DataFrame:
    """Each purchase informed with probability pi (uniform) or only inside a fraction `frac` of households,
    there with probability pi / frac so that the overall prevalence is pi; informed purchases get +mu."""
    if not 0.0 <= pi <= 1.0:
        raise ValueError("pi must lie in [0, 1]")
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
    pid = panel_id_of(panel)
    out = panel.copy()
    y = out[y_col].to_numpy(dtype=float).copy()
    informed = np.zeros(len(out), dtype=bool)
    theta = np.zeros(len(out))
    for draw, ix in out.groupby("draw", sort=True).indices.items():
        rng = np.random.default_rng(seed_from(name, int(draw), f"informed_{selection}", pid))
        hh, inv = np.unique(out[HH].to_numpy()[ix], return_inverse=True)
        if selection == "uniform":
            prob = np.full(len(hh), float(pi))
        else:
            m = max(int(round(frac * len(hh))), 1)
            chosen = rng.choice(len(hh), size=m, replace=False)
            prob = np.zeros(len(hh))
            prob[chosen] = min(float(pi) / (m / len(hh)), 1.0)
        flag = rng.random(len(ix)) < prob[inv]
        informed[ix] = flag
        y[ix] += np.where(flag, float(mu), 0.0)
        theta[ix] = prob[inv] * float(mu)
    out[y_col] = y
    out["informed"] = informed
    out["theta_h"] = theta
    return out


# ---------------------------------------------------------------- per-dataset statistics

def panel_statistics(panel: pd.DataFrame, calendar: pd.DatetimeIndex,
                     min_n: int = winners.MIN_EVENTS_FULL_SAMPLE) -> dict:
    """The three tests' statistics on one (injected, raw) dataset."""
    p = rc.add_block_clusters(panel, calendar)
    m = tt.t1_clustered_mean(p, anchor="anchor_blk20", alpha=ALPHA)
    t4 = tt.t4_tail_shares(p)
    mt = tt.max_t_statistic(p, min_n=min_n)
    return {"draw": int(p["draw"].iloc[0]), "n_rows": int(len(p)), "mean_bp": m["estimate"], "mean_se": m["se"],
            "mean_t": m["t_stat"], "mean_p_two": m["p_two_sided"], "mean_dof": m["dof"],
            "share_gt_3": t4[f"share_gt_{Z_TAIL:g}"], "share_lt_minus_3": t4[tt.lower_tail_stat(Z_TAIL)],
            "share_abs_gt_3": t4[f"share_gt_{Z_TAIL:g}"] + t4[tt.lower_tail_stat(Z_TAIL)],
            "max_t": mt["max_t"], "n_households_ge20": mt["n_households"],
            "share_informed": float(p["informed"].mean()) if "informed" in p else 0.0,
            "mean_theta_bp": float(p["theta_h"].mean()) if "theta_h" in p else 0.0}


# ---------------------------------------------------------------- work units

def cell_spec(pi: float, mu: float, selection: str, horizon: int, scale: bool = True) -> dict:
    return {"pi": float(pi), "mu_bp_per20": float(mu), "selection": selection, "horizon": int(horizon),
            "mu_injected_bp": scaled_mu(mu, horizon, scale), "frac": CONCENTRATED_FRACTION if selection == "concentrated" else 1.0}


def unit_key(clock: str, spec: dict) -> str:
    return (f"{clock}_h{spec['horizon']}_{spec['selection']}_pi{spec['pi']:g}_mu{spec['mu_bp_per20']:g}"
            f"_inj{spec['mu_injected_bp']:g}")


def grid_units(clocks, n_panels: int, horizon: int, scale: bool = True, pis=PI_GRID, mus=MU_GRID,
               selections=SELECTIONS) -> list[dict]:
    out = []
    for clock in clocks:
        for sel in selections:
            for pi in pis:
                for mu in mus:
                    spec = cell_spec(pi, mu, sel, horizon, scale)
                    out.append({"grid": GRID, "key": unit_key(clock, spec), "clock": clock, "spec": spec,
                                "first": 0, "n": n_panels})
    return out


def run_unit(unit: dict) -> dict:
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0,
                "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    spec, clock = unit["spec"], unit["clock"]
    _, cal, _ = rc._shared(clock)
    name = f"{GRID}|{unit['key']}"
    rows = []
    for p in iter_draws(clock, unit["first"], unit["n"], spec["horizon"]):
        q = inject_informed_purchases(p, spec["pi"], spec["mu_injected_bp"], spec["selection"], name, spec["frac"])
        rows.append(panel_statistics(q, cal))
    frame = pd.DataFrame(rows)
    frame["clock"] = clock
    frame["spec_json"] = json.dumps(spec, sort_keys=True)
    for k, v in spec.items():
        frame[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": unit["key"], "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def ref_units(clocks, horizon: int) -> list[dict]:
    return [{"grid": REF_GRID, "key": f"{clock}_h{horizon}_{first:04d}", "clock": clock, "horizon": horizon,
             "first": first, "n": rc.BATCH_SIZE} for clock in clocks
            for first in range(REF_FIRST, REF_FIRST + REF_N, rc.BATCH_SIZE)]


def run_ref_unit(unit: dict) -> dict:
    """Zero-effect statistics on the reference datasets (1001-2000) at one horizon."""
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0,
                "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    _, cal, _ = rc._shared(unit["clock"])
    rows = []
    for p in iter_draws(unit["clock"], unit["first"], unit["n"], unit["horizon"]):
        q = p.copy()
        q["informed"] = False
        q["theta_h"] = 0.0
        rows.append(panel_statistics(q, cal))
    frame = pd.DataFrame(rows)
    frame["clock"], frame["horizon"] = unit["clock"], unit["horizon"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": unit["key"], "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def run_units(units_: list[dict], workers: int, log: Path | None, fn=run_unit) -> None:
    import multiprocessing as mp

    pending = [u for u in units_ if not rc.unit_path(u["grid"], u["key"]).exists()]
    rc.log_line(log, f"{len(units_)} units, {len(pending)} pending, {workers} workers")
    if not pending:
        return
    if workers <= 1:
        for u in pending:
            r = fn(u)
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")
        return
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(fn, pending):
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")


# ---------------------------------------------------------------- observed statistics and scoring

# Every row below is taken on the RAW outcome, because the injected statistics are: `purchase_mean/vs_zero` is
# the raw purchase mean, the tail rows standardize the raw outcome, and the best-household row is the
# `raw_outcome_` variant. The winsorized `best_household/vs_centered_placebo` rows are NOT comparable with an
# injected statistic computed on raw Y (transaction clock: largest household t 3.067 winsorized against 3.049
# raw, largest household mean 309 bp against 747 bp).
OBSERVED_ROWS = {"mean_bp": ("purchase_mean", "vs_zero", "mean"), "mean_t": ("purchase_mean", "vs_zero", "t_block"),
                 "share_gt_3": ("tail", "vs_centered_placebo", "share_gt_3"),
                 "share_lt_minus_3": ("tail", "vs_centered_placebo", "share_lt_minus_3"),
                 "max_t": ("best_household", "raw_outcome_vs_centered_placebo", "max_t")}
# `results.csv` publishes none of these five statistics away from 20 sessions: the `robustness/horizon_{h}` rows
# carry the variance component tau2 on the persistence households only, and their `t_block` (dof 37) is that
# component's t, not the purchase-mean t (138 blocks, dof 137). At horizons 10 and 60 the observed columns are
# therefore emitted as missing with the note below rather than filled with a statistic of another test.
OBSERVED_ROWS_H: dict[str, tuple[str, str, str]] = {}
OBSERVED_STATISTICS = ("mean_bp", "mean_t", "share_gt_3", "share_lt_minus_3", "share_abs_gt_3", "max_t")
NO_OBSERVED_NOTE = ("data/v4/results.csv publishes no purchase-mean, tail or best-household statistic at horizon "
                    "{h} sessions (robustness/horizon_{h} carries the variance component only); observed omitted")


def observed_statistics(clock: str, horizon: int = 20, path: Path = rc.REPO_ROOT / "data" / "v4" / "results.csv") -> dict:
    """The actual statistics from `data/v4/results.csv` (NaN where that horizon has no row).

    Returns the statistics themselves, `sources` (the results.csv row each one came from) and `note` (why a
    statistic is missing, empty when none is)."""
    r = pd.read_csv(path)
    sources: dict[str, str] = {}

    def pick(key, analysis, variant, statistic):
        g = r[(r["clock"] == clock) & (r["analysis"] == analysis) & (r["variant"] == variant) & (r["statistic"] == statistic)]
        if len(g) != 1:
            return np.nan
        sources[key] = f"{analysis}/{variant}/{statistic}"
        return float(g["value"].iloc[0])

    out = {k: np.nan for k in OBSERVED_STATISTICS}
    rows = OBSERVED_ROWS if horizon == 20 else OBSERVED_ROWS_H
    for k, (a, v, s) in rows.items():
        out[k] = pick(k, a, v.format(h=horizon), s)
    if horizon == 20:
        out["share_abs_gt_3"] = out["share_gt_3"] + out["share_lt_minus_3"]
        sources["share_abs_gt_3"] = f"{sources.get('share_gt_3', '')} + {sources.get('share_lt_minus_3', '')}"
    out["sources"] = sources
    out["note"] = "" if rows else NO_OBSERVED_NOTE.format(h=horizon)
    return out


def _power_and_compat(stat: np.ndarray, null: np.ndarray, observed: float, direction: str) -> dict:
    """Rejection share against the null reference (add-one Monte Carlo p, or the two-sided t for the mean)
    and the compatibility of `observed` with the injected distribution of the statistic."""
    stat = np.asarray(stat, dtype=float)
    f = stat[np.isfinite(stat)]
    if direction == "greater":
        p = np.array([placebo_p_value(v, null, "greater")["p"] for v in f])
        power = float(np.mean(p < ALPHA))
        crit = float(np.nanquantile(null, 1.0 - ALPHA))
    else:
        power = np.nan
        crit = np.nan
    out = {"power": power, "mc_se": rc.mc_se(power, len(f)), "critical_value": crit, "n_panels": int(len(f)),
           "stat_mean": float(f.mean()), "stat_sd": float(f.std(ddof=1)) if len(f) > 1 else np.nan,
           "stat_q05": float(np.quantile(f, 0.05)), "stat_q50": float(np.median(f)), "stat_q95": float(np.quantile(f, 0.95))}
    if np.isfinite(observed):
        lo = float(np.mean(f <= observed))
        hi = float(np.mean(f >= observed))
        out.update({"observed": observed, "share_at_or_below_observed": lo, "share_at_or_above_observed": hi,
                    "compatibility_p": float(min(1.0, 2.0 * min(lo, hi))),
                    "compatible_at_5pct": bool(min(1.0, 2.0 * min(lo, hi)) >= ALPHA)})
    else:
        out.update({"observed": np.nan, "share_at_or_below_observed": np.nan, "share_at_or_above_observed": np.nan,
                    "compatibility_p": np.nan, "compatible_at_5pct": None})
    return out


def _mean_power(g: pd.DataFrame, sided: str = "two") -> float:
    p = g["mean_p_two"].to_numpy(dtype=float)
    if sided == "two":
        return float(np.mean(p < ALPHA))
    return float(np.mean((p / 2.0 < ALPHA) & (g["mean_t"].to_numpy(dtype=float) > 0)))


def aggregate(clocks, horizon: int, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = rc.load_grid(GRID)
    stats = stats[(stats["clock"].isin(clocks)) & (stats["horizon"] == horizon)]
    ref = rc.load_grid(REF_GRID)
    ref = ref[ref["horizon"] == horizon]
    rows, dists = [], []
    for clock in clocks:
        obs = observed_statistics(clock, horizon)
        rf = ref[ref["clock"] == clock]
        if not len(rf):
            continue
        for sj, g in stats[stats["clock"] == clock].groupby("spec_json", sort=True):
            spec = json.loads(sj)
            base = {"clock": clock, "sample": "all_households_all_purchases", "horizon": horizon,
                    "selection": spec["selection"], "prevalence": spec["pi"], "mu_bp_per20": spec["mu_bp_per20"],
                    "mu_injected_bp": spec["mu_injected_bp"], "concentration_fraction": spec["frac"],
                    "realized_share_informed": float(g["share_informed"].mean()),
                    "implied_mean_bp": float(g["mean_theta_bp"].mean()), "params": sj,
                    "observed_note": obs["note"]}
            for test in TESTS:
                col = STAT_OF[test]
                if test == "mean_block":
                    r = _power_and_compat(g[col].to_numpy(), rf[col].to_numpy(), obs.get(col, np.nan), "none")
                    r["power"] = _mean_power(g, "two")
                    r["power_one_sided"] = _mean_power(g, "one")
                    r["mc_se"] = rc.mc_se(r["power"], r["n_panels"])
                    rt = _power_and_compat(g["mean_t"].to_numpy(), rf["mean_t"].to_numpy(), obs.get("mean_t", np.nan), "none")
                    r["observed_t"] = rt["observed"]
                    r["compatibility_p_t"] = rt["compatibility_p"]
                else:
                    r = _power_and_compat(g[col].to_numpy(), rf[col].to_numpy(), obs.get(col, np.nan), "greater")
                rows.append({**base, "test": test, "test_name": TEST_NAME[test], "statistic": col,
                             "observed_source": obs["sources"].get(col, ""), **r})
            d = g[["draw", "mean_bp", "mean_t", "share_gt_3", "share_lt_minus_3", "share_abs_gt_3", "max_t"]].copy()
            for k, v in base.items():
                if k not in ("params", "observed_note"):  # cell identity only: the note is about the observed statistic
                    d[k] = v
            dists.append(d)
    out = pd.DataFrame(rows)
    dist = pd.concat(dists, ignore_index=True) if dists else pd.DataFrame()
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    dist.to_csv(STAT_CSV, index=False)
    rc.log_line(log, f"{OUT_CSV.name} {len(out)} rows over {out['params'].nunique()} cells; {STAT_CSV.name} {len(dist)} rows")
    return out, dist


# ---------------------------------------------------------------- one-call helper

def bcc_cell(pi: float, mu: float, horizon: int = 20, clocks=("T", "P"), selections=SELECTIONS, n_panels: int = 500,
             workers: int = 8, scale: bool = True, log: Path | None = None, quiet: bool = False) -> pd.DataFrame:
    """Run ONE (prevalence, effect, horizon) cell and print the three powers and the compatibility of the
    observed statistics with it.

    `pi` is the share of purchases that are informed, `mu` the effect per informed purchase in basis points
    per 20 trading sessions (pro-rated to `horizon` unless `scale=False`). Supply a published pair here; the
    module does not assume one. Returns the rows of `sparse_trade_grid.csv` for that cell.
    """
    units_ = grid_units(clocks, n_panels, horizon, scale, pis=(pi,), mus=(mu,), selections=selections)
    run_units(ref_units(clocks, horizon), workers, log, run_ref_unit)
    run_units(units_, workers, log)
    keys = {u["key"] for u in units_}
    stats = rc.load_grid(GRID)
    stats = stats[stats.apply(lambda r: unit_key(r["clock"], json.loads(r["spec_json"])) in keys, axis=1)]
    ref = rc.load_grid(REF_GRID)
    ref = ref[ref["horizon"] == horizon]
    rows = []
    for clock in clocks:
        obs = observed_statistics(clock, horizon)
        rf = ref[ref["clock"] == clock]
        for sj, g in stats[stats["clock"] == clock].groupby("spec_json", sort=True):
            spec = json.loads(sj)
            for test in ("mean_block", "tail_up", "tail_down", "best_household"):
                col = STAT_OF[test]
                if test == "mean_block":
                    r = _power_and_compat(g[col].to_numpy(), rf[col].to_numpy(), obs.get(col, np.nan), "none")
                    r["power"] = _mean_power(g, "two")
                else:
                    r = _power_and_compat(g[col].to_numpy(), rf[col].to_numpy(), obs.get(col, np.nan), "greater")
                rows.append({"clock": clock, "selection": spec["selection"], "prevalence": spec["pi"],
                             "mu_bp_per20": spec["mu_bp_per20"], "horizon": horizon, "test": TEST_NAME[test],
                             "power": r["power"], "mc_se": r["mc_se"], "observed": r["observed"],
                             "observed_source": obs["sources"].get(col, ""), "observed_note": obs["note"],
                             "compatibility_p": r["compatibility_p"], "stat_q05": r["stat_q05"],
                             "stat_q50": r["stat_q50"], "stat_q95": r["stat_q95"], "n_panels": r["n_panels"]})
    out = pd.DataFrame(rows)
    if not quiet:
        print(f"\nprevalence {pi:.3%} of purchases informed, {mu:g} bp per informed purchase per 20 sessions, "
              f"horizon {horizon} sessions ({n_panels} centered placebo datasets per cell)")
        for (clock, sel), g in out.groupby(["clock", "selection"], sort=True):
            print(f"\n  clock {clock}, informed purchases {sel}")
            for r in g.itertuples():
                obs = "" if not np.isfinite(r.observed) else (
                    f"; observed {r.observed:.4g}, compatibility p {r.compatibility_p:.3f} "
                    f"(5-50-95% of the injected statistic {r.stat_q05:.4g} / {r.stat_q50:.4g} / {r.stat_q95:.4g})")
                print(f"    {r.test:<28} power {r.power:5.1%} (MC SE {r.mc_se:.3f}){obs}")
    return out


def time_cells(clocks, horizon: int, n_cells: int, n_panels: int, log: Path | None) -> None:
    picks = grid_units(clocks, n_panels, horizon)[:n_cells]
    per = []
    for u in picks:
        r = run_unit({**u, "grid": "sparse_trade_timing", "key": u["key"] + f"_n{n_panels}", "n": n_panels})
        per.append(r["seconds"] / max(r["n"], 1))
        rc.log_line(log, f"timing {u['key']}: {r['seconds']:.1f}s for {r['n']} datasets = {per[-1]:.3f} s/dataset")
    n_units = len(grid_units(clocks, 500, horizon))
    est = float(np.mean(per)) * (500 * n_units + rc.BATCH_SIZE * len(ref_units(clocks, horizon)))
    rc.log_line(log, f"extrapolation: {n_units} cells x 500 + {len(ref_units(clocks, horizon))} reference units x 100 = "
                     f"{est:.0f} CPU-s ({est / 3600:.2f} CPU-h); /8 workers = {est / 8 / 60:.1f} min")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="informed-purchase alternative at a stated prevalence and effect")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-panels", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=20, choices=sorted(HORIZONS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-scale-mu", action="store_true", help="inject mu as given instead of pro-rating it to the horizon")
    ap.add_argument("--cell", default="", metavar="PI,MU", help="run one cell (prevalence, bp per 20 sessions) and print it")
    ap.add_argument("--selection", default=",".join(SELECTIONS))
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "sparse_trade_grid.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    sels = tuple(args.selection.split(","))
    t0 = time.time()
    if args.cell:
        pi, mu = (float(x) for x in args.cell.split(","))
        bcc_cell(pi, mu, args.horizon, clocks, sels, args.n_panels, args.workers, not args.no_scale_mu, args.log)
        return
    if args.time_cells:
        time_cells(clocks, args.horizon, args.time_cells, 20, args.log)
        return
    if not args.aggregate_only:
        run_units(ref_units(clocks, args.horizon), args.workers, args.log, run_ref_unit)
        run_units(grid_units(clocks, args.n_panels, args.horizon, not args.no_scale_mu, selections=sels),
                  args.workers, args.log)
    aggregate(clocks, args.horizon, args.log)
    rc.log_line(args.log, f"sparse_trade_grid finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
