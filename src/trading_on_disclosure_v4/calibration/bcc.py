"""The Blonien, Crane & Crotty (2025) alternative injected at its published prevalence and effect.

Their alternative, read off the paper (SSRN Working Paper 5524863):

- prevalence: 2.5% of TRADES are informed -- pooled posterior mean of p_i = 0.025 (SD 0.038; 5% 0.000,
  50% 0.009, 95% 0.104), Table 4, PDF p. 37 (printed 36); text PDF p. 19 (printed 18). It is a share of
  a trader's trades, not of traders and not of member-periods. The headline 92.5% is a different
  parameter, pi, the extensive margin (share of traders ever informed, Eq. 2 PDF p. 8, Table 4 PDF p. 37),
  and reading it as a prevalence of informed trades would be off by a factor of 37;
- effect: +31.5% signed market-adjusted buy-and-hold abnormal return per INFORMED trade over 63 trading
  days -- posterior mean of mu = 0.315 (5% 0.260, 95% 0.378), Table 4, PDF p. 37. The text prints the same
  number as 31% (PDF p. 18) and 32% (PDF p. 4); the table value is used here and the discrepancy is stated;
- horizon: 63 trading days from t+1 on the transaction clock, CRSP value-weighted buy-and-hold return
  subtracted, no factor model and no matched control firm (Eq. 5, PDF p. 15); signed +1 for purchases and
  -1 for disposals (Eq. 6, PDF p. 15). The paper carries no filing-date object anywhere, so nothing in it
  bridges to the public clock.

Rescaling to our cached horizons is additive (the ledger's house convention), because 0.315 is a total
simple return over 63 sessions: 3,150 bp x h / 63. The closest cached horizon is 60 sessions (4.8%
rescaling), which carries the effect at 3,000 bp; the ledger and Figure 2 unit, 20 sessions, carries it at
1,000 bp. The geometric alternative is reported beside it and never injected.

What this module runs. Informed purchases are injected into centered placebo datasets at that pair --
either spread uniformly over every purchase, or confined to the share of traders their paper implies
(92.5% ever informed, Table 4 PDF p. 37; 9.8% with a 95% HDI excluding zero, Figure 7 caption PDF p. 47;
67.2% in the survey prior, Table 4 PDF p. 37) -- and every test the paper runs on the whole purchase
record is scored at nominal 5%: the block-clustered mean test, the tail test (share of standardized
outcomes above 3 and below -3 against the placebo distribution), the best-household test (largest
household mean and largest household t against the placebo critical value) and, for completeness, the
variance-component test. It then asks the opposite question -- whether the OBSERVED statistics of
`data/v4/results.csv` are compatible with their alternative -- as the share of injected datasets whose
statistic is at least as extreme as the observed one.

Two placebo streams. The same eleven cells are injected, on the same draws and with the same
seeds, into both the CENTERED placebo datasets (N2, the household-period mean removed from every shifted
outcome) and the TIMING-ONLY ones (N1, the raw shifted outcomes, which keep each household's persistent
security choice). Power is scored on the centered stream alone, against its own null. Compatibility is
scored on both, and `bcc_compatibility.csv` carries `compatibility_p_centered` beside
`compatibility_p_timing` with `compatibility_p` kept as an alias of the centered column.

Horizons. Power is reported at 60 sessions (closest to their 63) and at 20 sessions (the horizon of the
paper's headline statistics). `data/v4/results.csv` carries the observed mean, tail and best-household
statistics at 20 sessions only; at 60 sessions it carries the variance component alone. Compatibility is
therefore computed wherever an observed statistic exists at that horizon, and is silent elsewhere.

Every statistic is computed on the raw outcome, never winsorized, except the variance component, whose
published value is winsorized; both variants of it are reported. Critical values and null distributions
come from zero-effect placebo datasets 1001-2000, disjoint from the 1-1000 used for power.

CLI: python -m trading_on_disclosure_v4.calibration.bcc [--clocks T,P] [--horizons 60,20]
     [--n-panels 1000] [--workers 6] [--time-cells 2] [--aggregate-only]
Outputs: data/v4/calibration/bcc_power.csv, bcc_compatibility.csv, paper/generated/bcc_macros.tex;
checkpoints data/v4/restricted/c/checkpoints/{bcc,bcc_ref,bcc_timing_only}/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.sparse_trade_grid import (
    inject_informed_purchases,
    iter_draws,
)
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value

GRID = "bcc"
REF_GRID = "bcc_ref"
TIMING_GRID = "bcc_timing_only"  # the same cells injected into the timing-only (N1) placebo datasets
CENTERED_FLAVOUR = "N2"  # centered: the household-period mean is removed from every shifted outcome
TIMING_FLAVOUR = "N1"  # timing-only: the raw shifted outcome, which keeps the persistent security choice
HORIZONS = (60, 20)
BCC_HORIZON_DAYS = 63  # Eq. (5), PDF p. 15
CACHE_HORIZON = 60  # closest cached horizon to theirs
LEDGER_HORIZON = 20  # the ledger / Figure 2 unit and the horizon of the observed statistics
REF_FIRST, REF_N = 1000, 1000
N_PANELS = 1000
ALPHA = 0.05
Z_TAIL = 3.0
WEIGHTS = "ES"  # event-share weights, the published tau2 convention
ANCHOR = "anchor_blk20"  # 20-session calendar blocks: the clustering of the published mean and tau2 tests
OUT_POWER = rc.CALIB_DIR / "bcc_power.csv"
OUT_COMPAT = rc.CALIB_DIR / "bcc_compatibility.csv"
OUT_MACROS = rc.REPO_ROOT / "paper" / "generated" / "bcc_macros.tex"

TABLE4 = "Table 4, PDF p. 37 (printed 36)"


def effect_bp(effect_63: float, horizon: int, convention: str = "arithmetic") -> float:
    """Their 63-session total simple return carried to `horizon` sessions, in basis points.

    'arithmetic' is additive pro-rating (the ledger's house convention, used for every injection);
    'geometric' compounds, ((1 + r) ** (h / 63) - 1), and is reported beside it, never injected."""
    if convention == "arithmetic":
        return float(effect_63) * 1e4 * horizon / BCC_HORIZON_DAYS
    if convention == "geometric":
        return ((1.0 + float(effect_63)) ** (horizon / BCC_HORIZON_DAYS) - 1.0) * 1e4
    raise ValueError(f"unknown convention {convention!r}")


# ---------------------------------------------------------------- the injected pairs
# `effect_63` is their return per informed trade over 63 trading days; `frac` is the share of households
# holding every informed purchase (1.0 = spread over all of them); `source` carries the page citation.

PAIRS: tuple[dict, ...] = (
    {"cell": "headline", "prevalence": 0.025, "effect_63": 0.315, "frac": 1.0, "role": "headline",
     "label": "posterior mean prevalence x posterior mean effect, informed purchases spread over all households",
     "source": f"p_i pooled posterior mean 0.025 and mu posterior mean 0.315, {TABLE4}"},
    {"cell": "extensive_925", "prevalence": 0.025, "effect_63": 0.315, "frac": 0.925, "role": "headline",
     "label": "same pair, informed purchases confined to the 92.5% of traders their extensive margin implies",
     "source": f"pi posterior mean 0.925, {TABLE4}; text PDF p. 18 (printed 17)"},
    {"cell": "concentrated_hdi", "prevalence": 0.025, "effect_63": 0.315, "frac": 0.098, "role": "concentration",
     "label": "same pair, confined to the 9.8% of traders whose 95% HDI excludes zero",
     "source": "Figure 7 caption, PDF p. 47; text PDF p. 21 (printed 20) rounds it to 10%"},
    {"cell": "survey_prior", "prevalence": 0.025, "effect_63": 0.358, "frac": 0.672, "role": "sensitivity",
     "label": "survey beliefs: 67.2% of traders ever informed x 36% per informed trade, trade frequency "
              "ASSUMED at their posterior 2.5% because the survey elicits none",
     "source": f"prior of pi mean 0.672 and prior of mu mean 0.358, {TABLE4}; text PDF p. 14 (printed 13)"},
    {"cell": "effect_p05", "prevalence": 0.025, "effect_63": 0.260, "frac": 1.0, "role": "sensitivity",
     "label": "effect at the 5th percentile of its posterior, prevalence held at 2.5%",
     "source": f"mu 5th percentile 0.260, {TABLE4}"},
    {"cell": "effect_p95", "prevalence": 0.025, "effect_63": 0.378, "frac": 1.0, "role": "sensitivity",
     "label": "effect at the 95th percentile of its posterior, prevalence held at 2.5%",
     "source": f"mu 95th percentile 0.378, {TABLE4}"},
    {"cell": "effect_median_alpha", "prevalence": 0.025, "effect_63": 0.201, "frac": 1.0, "role": "sensitivity",
     "label": "effect at the pooled posterior MEDIAN of the per-trader effect rather than its mean",
     "source": f"alpha_i pooled posterior median 0.201 (mean 0.314), {TABLE4}"},
    {"cell": "prevalence_median", "prevalence": 0.009, "effect_63": 0.315, "frac": 1.0, "role": "sensitivity",
     "label": "prevalence at the pooled posterior MEDIAN of p_i, effect held at 31.5%",
     "source": f"p_i pooled posterior median 0.009, {TABLE4}"},
    {"cell": "prevalence_p95", "prevalence": 0.104, "effect_63": 0.315, "frac": 1.0, "role": "sensitivity",
     "label": "prevalence at the 95th percentile of p_i, effect held at 31.5%",
     "source": f"p_i 95th percentile 0.104, {TABLE4}"},
    {"cell": "effect_half", "prevalence": 0.025, "effect_63": 0.1575, "frac": 1.0, "role": "sensitivity",
     "label": "published effect halved, prevalence held at 2.5%", "source": "0.315 / 2, ours"},
    {"cell": "effect_double", "prevalence": 0.025, "effect_63": 0.630, "frac": 1.0, "role": "sensitivity",
     "label": "published effect doubled, prevalence held at 2.5%", "source": "0.315 x 2, ours"},
)
BOTH_CLOCKS = ("headline", "extensive_925")  # the rest run on the transaction clock, theirs


def pair_of(cell: str) -> dict:
    for p in PAIRS:
        if p["cell"] == cell:
            return p
    raise KeyError(cell)


def cell_spec(pair: dict, horizon: int) -> dict:
    """The injected cell: the pair, the horizon and the effect actually added, in basis points."""
    return {"cell": pair["cell"], "prevalence": float(pair["prevalence"]), "effect_63": float(pair["effect_63"]),
            "frac": float(pair["frac"]), "horizon": int(horizon),
            "mu_injected_bp": round(effect_bp(pair["effect_63"], horizon), 6),
            "mu_geometric_bp": round(effect_bp(pair["effect_63"], horizon, "geometric"), 6)}


def unit_key(clock: str, spec: dict) -> str:
    return f"{clock}_h{spec['horizon']}_{spec['cell']}_pi{spec['prevalence']:g}_mu{spec['mu_injected_bp']:g}"


def grid_units(clocks, horizons, n_panels: int, grid: str = GRID,
               flavour: str = CENTERED_FLAVOUR) -> list[dict]:
    """The injected cells. `grid` and `flavour` pick the placebo stream; the injection seed is keyed on
    GRID and the unit key alone, so the same cell draws the same informed purchases in both streams."""
    out = []
    for clock in clocks:
        for pair in PAIRS:
            if clock != "T" and pair["cell"] not in BOTH_CLOCKS:
                continue
            for h in horizons:
                spec = cell_spec(pair, h)
                out.append({"grid": grid, "key": unit_key(clock, spec), "clock": clock, "spec": spec,
                            "first": 0, "n": n_panels, "flavour": flavour})
    return out


def ref_units(clocks, horizons) -> list[dict]:
    return [{"grid": REF_GRID, "key": f"{clock}_h{h}_{first:04d}", "clock": clock, "horizon": h,
             "first": first, "n": rc.BATCH_SIZE}
            for clock in clocks for h in horizons
            for first in range(REF_FIRST, REF_FIRST + REF_N, rc.BATCH_SIZE)]


# ---------------------------------------------------------------- per-dataset statistics

def panel_statistics(panel: pd.DataFrame, calendar: pd.DatetimeIndex, r1: set[str],
                     cutoffs: dict[str, tuple[float, float]]) -> dict:
    """Every statistic of one (injected) dataset: the mean test, the tail test, the best-household test
    and the variance component, the last on the persistence households in both raw and winsorized form."""
    p = rc.add_block_clusters(panel, calendar)
    m = tt.t1_clustered_mean(p, anchor=ANCHOR, alpha=ALPHA)
    t4 = tt.t4_tail_shares(p)
    hh = winners.household_stats(p, min_n=winners.MIN_EVENTS_FULL_SAMPLE)
    sub = p.loc[p["household_id"].isin(r1)].reset_index(drop=True)
    subw = rc.winsorize(sub, cutoffs)
    out = {"draw": int(p["draw"].iloc[0]), "n_rows": int(len(p)),
           "mean_bp": m["estimate"], "mean_se": m["se"], "mean_t": m["t_stat"], "mean_p_two": m["p_two_sided"],
           "share_gt_3": t4[f"share_gt_{Z_TAIL:g}"], "share_lt_minus_3": t4[tt.lower_tail_stat(Z_TAIL)],
           "share_gt_2": t4["share_gt_2"],
           "max_mean": float(hh["mean"].max()), "max_t": float(hh["t"].max()),
           "n_households_ge20": int(len(hh)),
           "tau2_raw": cgm.tau2_i1(sub, WEIGHTS, anchor=ANCHOR)["tau2_hat"],
           "tau2_wins": cgm.tau2_i1(subw, WEIGHTS, anchor=ANCHOR)["tau2_hat"],
           "n_households_persistence": int(sub["household_id"].nunique()),
           "share_informed": float(p["informed"].mean()) if "informed" in p else 0.0,
           "implied_mean_bp": float(p["theta_h"].mean()) if "theta_h" in p else 0.0}
    return out


GREATER_STATS = ("share_gt_3", "share_lt_minus_3", "share_gt_2", "max_mean", "max_t", "tau2_raw", "tau2_wins")
STAT_COLUMNS = ("mean_bp", "mean_t", *GREATER_STATS)


def run_unit(unit: dict) -> dict:
    """One resumable unit: `n` injected datasets of one clock, cell and horizon -> one parquet checkpoint."""
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0,
                "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    spec, clock = unit["spec"], unit["clock"]
    flavour = unit.get("flavour", CENTERED_FLAVOUR)
    r1, cal, _ = rc._shared(clock)
    cuts = rc.pooled_cutoffs(clock, f"ff5mom_h{spec['horizon']}")
    name = f"{GRID}|{unit['key']}"  # never the grid of the unit: both placebo streams share the seed
    selection = "uniform" if spec["frac"] >= 1.0 else "concentrated"
    rows = []
    for p in iter_draws(clock, unit["first"], unit["n"], spec["horizon"], flavour):
        q = inject_informed_purchases(p, spec["prevalence"], spec["mu_injected_bp"], selection, name, spec["frac"])
        rows.append(panel_statistics(q, cal, r1, cuts))
    frame = pd.DataFrame(rows)
    frame["clock"] = clock
    frame["placebo"] = flavour
    frame["spec_json"] = json.dumps(spec, sort_keys=True)
    for k, v in spec.items():
        frame[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": unit["key"], "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def run_ref_unit(unit: dict) -> dict:
    """The same statistics on zero-effect datasets, for the critical values and the null tail shares."""
    path = rc.unit_path(unit["grid"], unit["key"])
    if path.exists():
        return {"key": unit["key"], "status": "cached", "seconds": 0.0,
                "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    r1, cal, _ = rc._shared(unit["clock"])
    cuts = rc.pooled_cutoffs(unit["clock"], f"ff5mom_h{unit['horizon']}")
    rows = []
    for p in iter_draws(unit["clock"], unit["first"], unit["n"], unit["horizon"]):
        q = p.copy()
        q["informed"] = False
        q["theta_h"] = 0.0
        rows.append(panel_statistics(q, cal, r1, cuts))
    frame = pd.DataFrame(rows)
    frame["clock"], frame["horizon"] = unit["clock"], unit["horizon"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": unit["key"], "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def run_units(units: list[dict], workers: int, log: Path | None, fn=run_unit) -> None:
    import multiprocessing as mp

    pending = [u for u in units if not rc.unit_path(u["grid"], u["key"]).exists()]
    rc.log_line(log, f"{len(units)} units, {len(pending)} pending, {workers} workers")
    if not pending:
        return
    if workers <= 1:
        for u in pending:
            r = fn(u)
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")
        return
    with mp.get_context("spawn").Pool(min(workers, len(pending)), maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(fn, pending):
            rc.log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")


# ---------------------------------------------------------------- the observed statistics

OBSERVED_ROWS = {
    "mean_bp": ("purchase_mean", "vs_zero", "mean"),
    "mean_t": ("purchase_mean", "vs_zero", "t_block"),
    "share_gt_3": ("tail", "vs_centered_placebo", "share_gt_3"),
    "share_lt_minus_3": ("tail", "vs_centered_placebo", "share_lt_minus_3"),
    "share_gt_2": ("tail", "vs_centered_placebo", "share_gt_2"),
    # the injected best-household statistics are raw, so the raw-outcome row is the comparable one
    "max_mean": ("best_household", "raw_outcome_vs_centered_placebo", "max_mean"),
    "max_t": ("best_household", "raw_outcome_vs_centered_placebo", "max_t"),
    "tau2_wins": ("tau", "event_share", "tau2"),
}
OBSERVED_ROWS_H60 = {"tau2_wins": ("robustness", "horizon_60", "tau2")}


def observed_statistics(clock: str, horizon: int,
                        path: Path = rc.REPO_ROOT / "data" / "v4" / "results.csv") -> dict:
    """The observed statistics from `data/v4/results.csv` at this horizon, NaN where no row exists.

    At 20 sessions every statistic is published; at 60 sessions only the variance component is."""
    r = pd.read_csv(path)

    def pick(analysis, variant, statistic):
        g = r[(r["clock"] == clock) & (r["analysis"] == analysis) & (r["variant"] == variant)
              & (r["statistic"] == statistic)]
        return (float(g["value"].iloc[0]), f"{analysis}/{variant}/{statistic}") if len(g) == 1 else (np.nan, "")

    rows = OBSERVED_ROWS if horizon == LEDGER_HORIZON else OBSERVED_ROWS_H60
    out = {k: (np.nan, "") for k in STAT_COLUMNS}
    out.update({k: pick(*v) for k, v in rows.items()})
    return out


# ---------------------------------------------------------------- power and compatibility

def _distribution(stat: np.ndarray) -> dict:
    f = np.asarray(stat, dtype=float)
    f = f[np.isfinite(f)]
    return {"n_panels": int(len(f)), "stat_mean": float(f.mean()), "stat_sd": float(f.std(ddof=1)),
            "stat_q05": float(np.quantile(f, 0.05)), "stat_q50": float(np.median(f)),
            "stat_q95": float(np.quantile(f, 0.95))}


def power_greater(stat: np.ndarray, null: np.ndarray) -> dict:
    """Rejection share of a one-sided placebo test (add-one Monte Carlo p against the null datasets)."""
    f = np.asarray(stat, dtype=float)
    f = f[np.isfinite(f)]
    p = np.array([placebo_p_value(v, null, "greater")["p"] for v in f])
    power = float(np.mean(p < ALPHA))
    return {"power": power, "mc_se": rc.mc_se(power, len(f)),
            "critical_value": float(np.nanquantile(np.asarray(null, dtype=float), 1.0 - ALPHA))}


def power_mean(g: pd.DataFrame) -> dict:
    """The block-clustered mean test: two-sided at 5% (as published) and one-sided positive beside it."""
    p = g["mean_p_two"].to_numpy(dtype=float)
    t = g["mean_t"].to_numpy(dtype=float)
    two = float(np.mean(p < ALPHA))
    return {"power": two, "mc_se": rc.mc_se(two, len(p)), "critical_value": np.nan,
            "power_one_sided": float(np.mean((p / 2.0 < ALPHA) & (t > 0)))}


def compatibility(stat: np.ndarray, observed: float) -> dict:
    """Share of injected datasets whose statistic is at least as extreme as the observed one.

    `share_at_or_below_observed` and `share_at_or_above_observed` bracket the observed value inside the
    injected distribution; the two-sided compatibility p is twice the smaller of the two, capped at 1."""
    f = np.asarray(stat, dtype=float)
    f = f[np.isfinite(f)]
    if not np.isfinite(observed) or not len(f):
        return {"observed": np.nan, "share_at_or_below_observed": np.nan, "share_at_or_above_observed": np.nan,
                "compatibility_p": np.nan, "compatible_at_5pct": None}
    lo, hi = float(np.mean(f <= observed)), float(np.mean(f >= observed))
    p = float(min(1.0, 2.0 * min(lo, hi)))
    return {"observed": float(observed), "share_at_or_below_observed": lo, "share_at_or_above_observed": hi,
            "compatibility_p": p, "compatible_at_5pct": bool(p >= ALPHA)}


TEST_NAME = {"mean_bp": "block-clustered mean test", "share_gt_3": f"tail test, z > {Z_TAIL:g}",
             "share_lt_minus_3": f"tail test, z < -{Z_TAIL:g}", "share_gt_2": "tail test, z > 2",
             "max_mean": "best-household test, largest household mean",
             "max_t": "best-household test, largest household t",
             "tau2_raw": "variance-component test, raw outcome",
             "tau2_wins": "variance-component test, winsorized outcome (published convention)"}


def cell_base(spec: dict, g: pd.DataFrame, clock: str, h: int, sj: str) -> dict:
    """The descriptive columns every scored row of one cell carries."""
    pair = pair_of(spec["cell"])
    return {"clock": clock, "horizon": h, "cell": spec["cell"], "role": pair["role"],
            "prevalence": spec["prevalence"], "effect_63_decimal": spec["effect_63"],
            "effect_bp_at_horizon": spec["mu_injected_bp"],
            "effect_bp_geometric": spec["mu_geometric_bp"],
            "concentration_fraction": spec["frac"],
            "realized_share_informed": float(g["share_informed"].mean()),
            "implied_mean_bp": float(g["implied_mean_bp"].mean()),
            "label": pair["label"], "source": pair["source"], "params": sj}


COMPAT_KEYS = ["clock", "horizon", "cell", "statistic"]
COMPAT_SCORES = ("share_at_or_below_observed", "share_at_or_above_observed", "compatibility_p",
                 "compatible_at_5pct", "n_panels", "stat_mean", "stat_sd", "stat_q05", "stat_q50", "stat_q95")


def compat_frame(stats: pd.DataFrame, clocks, horizons) -> pd.DataFrame:
    """Compatibility of every observed statistic with every injected cell of one placebo stream."""
    rows = []
    for clock in clocks:
        for h in horizons:
            obs = observed_statistics(clock, h)
            sel = stats[(stats["clock"] == clock) & (stats["horizon"] == h)]
            if not len(sel):
                continue
            for sj, g in sel.groupby("spec_json", sort=True):
                base = cell_base(json.loads(sj), g, clock, h, sj)
                for col in STAT_COLUMNS:
                    value, row = obs.get(col, (np.nan, ""))
                    c = compatibility(g[col].to_numpy(), value)
                    rows.append({**base, "statistic": col, "results_row": row,
                                 **c, **_distribution(g[col].to_numpy())})
    return pd.DataFrame(rows)


def merge_placebos(centered: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """The centered compatibility columns with the timing-only ones beside them.

    `compatibility_p` stays the centered value (readers that predate the second stream keep working) and
    is repeated as `compatibility_p_centered`; every timing-only column carries the `_timing` suffix.
    `verdict_same_at_5pct` says whether the two streams agree on compatibility at the nominal 5%."""
    out = centered.copy()
    for col in COMPAT_SCORES:
        out[col + "_centered"] = out[col]
    if timing is None or not len(timing):
        return out
    right = timing[COMPAT_KEYS + list(COMPAT_SCORES)].rename(
        columns={c: c + "_timing" for c in COMPAT_SCORES})
    out = out.merge(right, on=COMPAT_KEYS, how="left", validate="one_to_one")
    same = out["compatible_at_5pct_centered"].astype("object") == out["compatible_at_5pct_timing"].astype("object")
    out["verdict_same_at_5pct"] = np.where(out["compatible_at_5pct_centered"].isna()
                                           | out["compatible_at_5pct_timing"].isna(), None, same)
    return out


def aggregate(clocks, horizons, log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every injected cell against the null datasets and against the observed statistics."""
    stats = rc.load_grid(GRID)
    ref = rc.load_grid(REF_GRID)
    power_rows = []
    for clock in clocks:
        for h in horizons:
            rf = ref[(ref["clock"] == clock) & (ref["horizon"] == h)]
            sel = stats[(stats["clock"] == clock) & (stats["horizon"] == h)]
            if not len(rf) or not len(sel):
                continue
            for sj, g in sel.groupby("spec_json", sort=True):
                base = cell_base(json.loads(sj), g, clock, h, sj)
                for col in ("mean_bp", *GREATER_STATS):
                    if col == "share_gt_2":
                        continue
                    d = _distribution(g[col].to_numpy())
                    r = power_mean(g) if col == "mean_bp" else power_greater(g[col].to_numpy(),
                                                                            rf[col].to_numpy())
                    power_rows.append({**base, "test": col, "test_name": TEST_NAME[col], **r, **d})
    power = pd.DataFrame(power_rows)
    timing = None
    if (rc.CHECKPOINT_DIR / TIMING_GRID).exists():
        timing = compat_frame(rc.load_grid(TIMING_GRID), clocks, horizons)
    compat = merge_placebos(compat_frame(stats, clocks, horizons), timing)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    power.to_csv(OUT_POWER, index=False)
    compat.to_csv(OUT_COMPAT, index=False)
    rc.log_line(log, f"{OUT_POWER.name} {len(power)} rows; {OUT_COMPAT.name} {len(compat)} rows "
                     f"({'with' if timing is not None else 'without'} the timing-only columns)")
    return power, compat


# ---------------------------------------------------------------- macros

def _pct(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{100.0 * x:.1f}"


def _bp(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:,.0f}"


def _p(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


MACRO_TESTS = {"Mean": "mean_bp", "Tail": "share_gt_3", "Best": "max_t", "Tau": "tau2_wins"}
COMPAT_MACROS = {"Mean": "mean_bp", "Tail": "share_gt_3", "Best": "max_t"}
# the compatibility macros are emitted twice: bare names from the centered placebo (unchanged), and
# `...Timing` names from the timing-only one, so a table can print the two side by side
COMPAT_MACRO_SOURCE = {"": "compatibility_p_centered", "Timing": "compatibility_p_timing"}
SENSITIVITY_MACROS = {"effect_p05": "EffectLow", "effect_p95": "EffectHigh", "effect_half": "EffectHalf",
                      "effect_double": "EffectDouble", "effect_median_alpha": "EffectMedian",
                      "prevalence_median": "PrevalenceMedian", "prevalence_p95": "PrevalenceHigh",
                      "survey_prior": "Survey", "concentrated_hdi": "Concentrated",
                      "extensive_925": "Extensive"}


def macros(power: pd.DataFrame, compat: pd.DataFrame, clock: str = "T") -> str:
    """The macros the Section 5 paragraph needs; power at the closest horizon, compatibility where the
    observed statistic exists (60 sessions for the variance component, 20 for the other three)."""
    def pw(cell, test, horizon):
        g = power[(power["clock"] == clock) & (power["horizon"] == horizon) & (power["cell"] == cell)
                  & (power["test"] == test)]
        return float(g["power"].iloc[0]) if len(g) == 1 else np.nan

    def cp(cell, stat, horizon, column="compatibility_p"):
        if column == "compatibility_p_centered" and column not in compat.columns:
            column = "compatibility_p"  # a frame from before the second placebo stream
        if column not in compat.columns:
            return np.nan
        g = compat[(compat["clock"] == clock) & (compat["horizon"] == horizon) & (compat["cell"] == cell)
                   & (compat["statistic"] == stat)]
        return float(g[column].iloc[0]) if len(g) == 1 else np.nan

    head = pair_of("headline")
    lines = ["% generated by trading_on_disclosure_v4.calibration.bcc; numbers only, do not edit",
             f"\\newcommand{{\\BccPrevalencePct}}{{{100 * head['prevalence']:.1f}}}",
             f"\\newcommand{{\\BccEffectPct}}{{{100 * head['effect_63']:.1f}}}",
             f"\\newcommand{{\\BccEffectBp}}{{{_bp(effect_bp(head['effect_63'], CACHE_HORIZON))}}}",
             f"\\newcommand{{\\BccEffectBpTwenty}}{{{_bp(effect_bp(head['effect_63'], LEDGER_HORIZON))}}}",
             f"\\newcommand{{\\BccHorizon}}{{{CACHE_HORIZON}}}",
             f"\\newcommand{{\\BccHorizonTheirs}}{{{BCC_HORIZON_DAYS}}}",
             f"\\newcommand{{\\BccCompatHorizon}}{{{LEDGER_HORIZON}}}",
             f"\\newcommand{{\\BccCompatTauHorizon}}{{{CACHE_HORIZON}}}",
             f"\\newcommand{{\\BccImpliedMeanBp}}{{{_bp(head['prevalence'] * effect_bp(head['effect_63'], CACHE_HORIZON))}}}",
             f"\\newcommand{{\\BccNPanels}}{{{N_PANELS:,}}}"]
    for name, test in MACRO_TESTS.items():
        lines.append(f"\\newcommand{{\\BccPower{name}}}{{{_pct(pw('headline', test, CACHE_HORIZON))}}}")
        lines.append(f"\\newcommand{{\\BccPower{name}Twenty}}{{{_pct(pw('headline', test, LEDGER_HORIZON))}}}")
    for suffix, column in COMPAT_MACRO_SOURCE.items():
        for name, stat in COMPAT_MACROS.items():
            lines.append(f"\\newcommand{{\\BccCompat{name}{suffix}}}"
                         f"{{{_p(cp('headline', stat, LEDGER_HORIZON, column))}}}")
        # the variance component has no observed value at 20 sessions in the published ledger's units,
        # so its compatibility is named at both horizons and `\BccCompatTau` keeps the 60-session one
        lines.append(f"\\newcommand{{\\BccCompatTau{suffix}}}"
                     f"{{{_p(cp('headline', 'tau2_wins', CACHE_HORIZON, column))}}}")
        lines.append(f"\\newcommand{{\\BccCompatTauSixty{suffix}}}"
                     f"{{{_p(cp('headline', 'tau2_wins', CACHE_HORIZON, column))}}}")
        lines.append(f"\\newcommand{{\\BccCompatTauTwenty{suffix}}}"
                     f"{{{_p(cp('headline', 'tau2_wins', LEDGER_HORIZON, column))}}}")
        lines.append(f"\\newcommand{{\\BccCompatBestMean{suffix}}}"
                     f"{{{_p(cp('headline', 'max_mean', LEDGER_HORIZON, column))}}}")
    for cell, tag in SENSITIVITY_MACROS.items():
        p = pair_of(cell)
        lines.append(f"\\newcommand{{\\Bcc{tag}PrevalencePct}}{{{100 * p['prevalence']:.1f}}}")
        lines.append(f"\\newcommand{{\\Bcc{tag}EffectBp}}{{{_bp(effect_bp(p['effect_63'], CACHE_HORIZON))}}}")
        for name, test in MACRO_TESTS.items():
            lines.append(f"\\newcommand{{\\Bcc{tag}Power{name}}}{{{_pct(pw(cell, test, CACHE_HORIZON))}}}")
        for suffix, column in COMPAT_MACRO_SOURCE.items():
            for name, stat in COMPAT_MACROS.items():
                lines.append(f"\\newcommand{{\\Bcc{tag}Compat{name}{suffix}}}"
                             f"{{{_p(cp(cell, stat, LEDGER_HORIZON, column))}}}")
            lines.append(f"\\newcommand{{\\Bcc{tag}CompatTau{suffix}}}"
                         f"{{{_p(cp(cell, 'tau2_wins', CACHE_HORIZON, column))}}}")
    return "\n".join(lines) + "\n"


def write_macros(power: pd.DataFrame, compat: pd.DataFrame, path: Path = OUT_MACROS,
                 log: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = macros(power, compat)
    path.write_text(text)
    rc.log_line(log, f"{path.name} {len(text.splitlines())} macros")
    return path


# ---------------------------------------------------------------- CLI

def time_cells(clocks, horizons, n_cells: int, n_panels: int, log: Path | None) -> None:
    picks = grid_units(clocks, horizons, n_panels)[:n_cells]
    per = []
    for u in picks:
        r = run_unit({**u, "grid": "bcc_timing", "key": u["key"] + f"_n{n_panels}"})
        per.append(r["seconds"] / max(r["n"], 1))
        rc.log_line(log, f"timing {u['key']}: {r['seconds']:.1f}s for {r['n']} datasets = {per[-1]:.3f} s/dataset")
    n_units = len(grid_units(clocks, horizons, N_PANELS))
    n_ref = len(ref_units(clocks, horizons))
    est = float(np.mean(per)) * (N_PANELS * n_units + rc.BATCH_SIZE * n_ref)
    rc.log_line(log, f"extrapolation: {n_units} cells x {N_PANELS} + {n_ref} reference units x {rc.BATCH_SIZE} "
                     f"= {est:.0f} CPU-s ({est / 3600:.2f} CPU-h)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="the Blonien-Crane-Crotty alternative through our own tests")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--n-panels", type=int, default=N_PANELS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-cells", type=int, default=0, metavar="K")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-timing-only", dest="timing_only", action="store_false",
                    help="skip the second placebo stream (the timing-only, N1, datasets)")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "bcc.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    horizons = tuple(int(h) for h in args.horizons.split(","))
    t0 = time.time()
    if args.time_cells:
        time_cells(clocks, horizons, args.time_cells, 20, args.log)
        return
    if not args.aggregate_only:
        run_units(ref_units(clocks, horizons), args.workers, args.log, run_ref_unit)
        run_units(grid_units(clocks, horizons, args.n_panels), args.workers, args.log)
        if args.timing_only:
            run_units(grid_units(clocks, horizons, args.n_panels, TIMING_GRID, TIMING_FLAVOUR),
                      args.workers, args.log)
    power, compat = aggregate(clocks, horizons, args.log)
    write_macros(power, compat, log=args.log)
    rc.log_line(args.log, f"bcc finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
