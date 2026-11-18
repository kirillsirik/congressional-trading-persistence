"""Shared pieces of the calibration runs (plan §6, A2): lean N2 panel reader, R1 household
set, pooled winsorization cutoffs, the injection adapter over N's `inject/effects.py`, the
per-panel test bundle (T1, T2 with I1/I2/I2b, T3, T4 and the block-cluster I1 diagnostics),
and per-cell checkpoint files under `data/v4/restricted/c/checkpoints/`.

Everything here operates on shifted placebo (N2) panels or effect-injected copies of them;
no actual household-level outcome statistic is computed (pre-tag boundary).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trading_on_disclosure_v4.calibration.synthetic import realized_tau2_target, seed_from_name
from trading_on_disclosure_v4.data.eligibility import eligibility_flags, household_period_counts
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.inference import block_bootstrap as bb
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.inject import effects
from trading_on_disclosure_v4.null import block_shift as bs
from trading_on_disclosure_v4.null import run_null_panels as rnp
from trading_on_disclosure_v4.outcomes import panel as panel_mod

REPO_ROOT = panel_mod.REPO_ROOT
CALIB_DIR = REPO_ROOT / "data" / "v4" / "calibration"
RESTRICTED_C = REPO_ROOT / "data" / "v4" / "restricted" / "c"
CHECKPOINT_DIR = RESTRICTED_C / "checkpoints"
LOG_DIR = RESTRICTED_C / "logs"
POOLED_CSV = REPO_ROOT / "data" / "v4" / "pooled" / "pooled_outcome_distribution.csv"
# Wave 3: the P default is the all-lines set (plan §11); the wave-2 any-line set is kept for labelling old rows.
DEFAULT_SET = {"T": "A0_T_purchase_AB_L20_both_keepdups", "P": "A0_P_purchase_AB_L20_both_alllines"}
WAVE2_SET = {"T": "A0_T_purchase_AB_L20_both_keepdups", "P": "A0_P_purchase_AB_L20_both_nodup"}
# label of the pooled-cutoff rows in POOLED_CSV (still the any-line table)
DUP_LABEL = {"T": "keepdups", "P": "nodup"}
BP = 1e4
WEIGHTS = ("ES", "EH", "PW")
BATCH_SIZE = 100
PERIOD_START = {"A": "2015-01-01", "B": "2021-01-01"}


def log_line(path: Path | None, msg: str) -> None:
    """Append one timestamped line (and print it)."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------- panels

def set_dir(clock: str, flavour: str = "N2", set_name: str | None = None) -> Path:
    return rnp.set_dir(clock, flavour, set_name or DEFAULT_SET[clock])


def load_batch(clock: str, batch: int, flavour: str = "N2", set_name: str | None = None,
               outcome: str = "y_h20") -> pd.DataFrame:
    """One stored batch (draws batch*100 .. +99) with only the columns the tests need."""
    path = set_dir(clock, flavour, set_name) / f"{flavour.lower()}_batch_{batch:04d}.parquet"
    avail = pq.read_schema(path).names
    cols = ["draw", "event_id", "household_id", "period", outcome, "shifted_anchor_date"]
    cols += [c for c in ("resid_sd", "sigma_event") if c in avail]
    tbl = pq.read_table(path, columns=cols, read_dictionary=["event_id", "household_id", "period"])
    df = tbl.to_pandas()
    df["clock"] = clock
    return df


def iter_draws(clock: str, first: int, n: int, flavour: str = "N2", set_name: str | None = None,
               outcome: str = "y_h20"):
    """Harness-form panels for draws first .. first+n-1 (bp units, shifted anchor, sigma_event)."""
    last = first + n - 1
    for b in range(first // BATCH_SIZE, last // BATCH_SIZE + 1):
        batch = load_batch(clock, b, flavour, set_name, outcome)
        for d, g in batch.groupby("draw", sort=True):
            if first <= int(d) <= last:
                g = g.reset_index(drop=True)
                for c in ("event_id", "household_id", "period"):
                    g[c] = g[c].astype(str)
                yield rnp.harness_view(g, outcome)


def r1_households(clock: str) -> set[str]:
    """R1-eligible households recomputed on D's events after the plan-default duplicate filter."""
    ev, _ = bs.load_events_for_null(clock, "purchase", "A,B", drop_duplicates=True, drop_duplicates_on_T=False)
    flags = eligibility_flags(household_period_counts(ev), clock)
    return set(flags.loc[flags["R1"], "household_id"].astype(str))


def trading_calendar() -> pd.DatetimeIndex:
    return panel_mod.v2_trading_calendar()


# ---------------------------------------------------------------- winsorization

def pooled_cutoffs(clock: str, outcome: str = "ff5mom_h20") -> dict[str, tuple[float, float]]:
    """(p1, p99) in bp per period from the pooled distribution file, for the plan-default stream."""
    t = pd.read_csv(POOLED_CSV)
    t = t[(t["clock"] == clock) & (t["direction"] == "purchase") & (t["horizon"] == outcome)
          & (t["duplicates"] == DUP_LABEL[clock])]
    if len(t) != 2:
        raise ValueError(f"expected 2 pooled rows for {clock}/{outcome}, found {len(t)}")
    return {str(r.period): (float(r.winsor_lo) * BP, float(r.winsor_hi) * BP) for r in t.itertuples()}


def winsorize(panel: pd.DataFrame, cutoffs: dict[str, tuple[float, float]], y: str = "y_h20") -> pd.DataFrame:
    """Clip y at the period's (lo, hi); rows of a period without cutoffs are left as they are."""
    out = panel.copy()
    yv = out[y].to_numpy(dtype=float).copy()
    per = out["period"].to_numpy()
    for p, (lo, hi) in cutoffs.items():
        m = per == p
        yv[m] = np.clip(yv[m], lo, hi)
    out[y] = yv
    return out


# ---------------------------------------------------------------- injection adapter (N's engine)

def cell_id(spec: dict) -> str:
    return "|".join(f"{k}={spec[k]}" for k in sorted(spec))


def inject(panel: pd.DataFrame, spec: dict, name: str, y: str = "y_h20", r1: set[str] | None = None) -> pd.DataFrame:
    """Harness-style injection through N's `inject/effects.py` (units: bp, as `y_h20` is in bp).

    Seeds are N's: seed_from(name, draw, kind, panel_id) with name = the cell id, so every cell,
    draw and clock receives its own effect stream. Adds `theta_h` (per-event true household
    effect: theta for dense/sparse households, pi_h x mu_T for sparse trades) for the realized
    tau2 target of T2 coverage."""
    kind = spec.get("kind", "none")
    if kind == "none":
        out = panel.copy()
        out["theta_h"] = 0.0
        return out
    if kind == "dense":
        inj = effects.inject_dense(panel, float(spec["tau_bp"]), name, y_col=y)
        th = inj.truth.set_index("household_id")["theta"]
    elif kind == "sparse_households":
        inj = effects.inject_sparse_households(panel, float(spec["pi_h"]), float(spec["mu_bp"]), name,
                                               selection=spec.get("informed", "uniform"), y_col=y)
        th = inj.truth.set_index("household_id")["theta"]
    elif kind == "sparse_k":  # wave 3: k informed households drawn from the test's own analysis set
        from trading_on_disclosure_v4.calibration.inject_k import inject_k_households

        out, info = inject_k_households(panel, int(spec.get("k", -1)), float(spec["mu_bp"]), name, spec["base"], r1,
                                        selection=spec.get("informed", "uniform"), y_col=y, k_frac=spec.get("k_frac"))
        out.attrs["inject_info"] = info
        return out
    elif kind == "sparse_trades":
        inj = effects.inject_sparse_trades(panel, float(spec["pi_t"]), float(spec["mu_bp"]), name,
                                           household_specific=bool(spec.get("household_specific", False)), y_col=y)
        hp = inj.params["household_pi"].set_index("household_id")["pi_h"]
        th = hp * float(spec["mu_bp"])
    else:
        raise ValueError(f"unknown injection kind {kind!r}")
    out = inj.panel
    out["theta_h"] = out["household_id"].map(th).to_numpy(dtype=float)
    return out


def truth_of(spec: dict) -> dict:
    """Closed-form population targets (bp^2, bp); coverage is scored on the realized target."""
    kind = spec.get("kind", "none")
    if kind == "dense":
        return {"tau2": float(spec["tau_bp"]) ** 2, "mean": 0.0}
    if kind == "sparse_households":
        p, m = float(spec["pi_h"]), float(spec["mu_bp"])
        return {"tau2": p * (1 - p) * m * m, "mean": p * m}
    if kind == "sparse_trades":
        p, m = float(spec["pi_t"]), float(spec["mu_bp"])
        cv = 1.0 if spec.get("household_specific", False) else 0.0
        return {"tau2": (cv * p * m) ** 2, "mean": p * m}
    if kind == "sparse_k":  # population values depend on H of the draw; coverage uses the realized target
        return {"tau2": np.nan, "mean": np.nan}
    return {"tau2": 0.0, "mean": 0.0}


# ---------------------------------------------------------------- per-panel statistics

BLOCK_CLUSTERS = (20, 60)


def add_block_clusters(panel: pd.DataFrame, calendar: pd.DatetimeIndex, lengths=BLOCK_CLUSTERS) -> pd.DataFrame:
    """Diagnostic date clusters: calendar position of the (shifted) anchor // L, for L in lengths."""
    pos = calendar.get_indexer(pd.DatetimeIndex(panel["anchor_date"]))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors are not calendar sessions")
    out = panel.copy()
    for L in lengths:
        out[f"anchor_blk{L}"] = pos // L
    return out


def _t1_rows(panel: pd.DataFrame, alpha: float, clusters: dict[str, str]) -> dict:
    out = {}
    for tag, anchor in clusters.items():
        r = tt.t1_clustered_mean(panel, anchor=anchor, alpha=alpha)
        out.update({f"T1_{tag}_est": r["estimate"], f"T1_{tag}_se": r["se"], f"T1_{tag}_lo": r["ci_low"],
                    f"T1_{tag}_hi": r["ci_high"], f"T1_{tag}_rej1": float(r["reject_one_sided"]),
                    f"T1_{tag}_rej2": float(r["reject_two_sided"]), f"T1_{tag}_dof": r["dof"]})
    out["T1_n_events"] = r["n_events"]
    return out


def _i1_rows(sub: pd.DataFrame, w: str, alpha: float, tag: str, anchor: str, target: float) -> dict:
    r = cgm.tau2_i1(sub, w, anchor=anchor)
    k = f"T2_{tag}_{w}"
    two = bool(r["tau2_low"] > 0.0 or r["tau2_high"] < 0.0)
    return {f"{k}_tau2": r["tau2_hat"], f"{k}_lo": r["tau2_low"], f"{k}_hi": r["tau2_high"], f"{k}_se_b": r["se_b"],
            f"{k}_rej1": float(r["p_one_sided_tau2_positive"] < alpha), f"{k}_rej2": float(two),
            f"{k}_cov": float(r["tau2_low"] <= target <= r["tau2_high"]), f"{k}_psd": float(r["psd_corrected"]),
            f"{k}_dof": r["dof"], f"{k}_up": r["tau2_upper"], f"{k}_t": r["t_stat"]}


def _boot_rows(res: dict, w: str, alpha: float, tag: str, target: float) -> dict:
    k = f"T2_{tag}_{w}"
    d = np.asarray(res["statistics"], dtype=float)
    f = d[np.isfinite(d)]
    two = bool(res["tau2_low"] > 0.0 or res["tau2_high"] < 0.0)
    return {f"{k}_tau2": res["tau2_hat"], f"{k}_lo": res["tau2_low"], f"{k}_hi": res["tau2_high"],
            f"{k}_bsd": float(np.std(f, ddof=1)) if len(f) > 1 else np.nan,
            f"{k}_rej1": float(tt.i2_one_sided_reject(d, alpha)), f"{k}_rej2": float(two),
            f"{k}_cov": float(res["tau2_low"] <= target <= res["tau2_high"]),
            f"{k}_ninv": res["n_invalid_draws"], f"{k}_up": res["tau2_upper_one_sided"]}


def panel_statistics(
    panel: pd.DataFrame, r1: set[str], calendar: pd.DatetimeIndex, weights=WEIGHTS, alpha: float = 0.05,
    boot_draws: int = 0, hh_draws: int = 0, block_length: int = 20, raw_for_t4: pd.DataFrame | None = None,
    block_clusters: bool = True, min_n: int = winners.MIN_EVENTS_FULL_SAMPLE,
) -> dict:
    """Every statistic of ONE (injected, possibly winsorized) panel as a flat dict.

    T1 on all rows; T2 on the R1 households (I1; I2 with `boot_draws`; I2b with `hh_draws`);
    T3 max-t and T4 tail shares as statistics (rejections need the null reference, applied
    later); `raw_for_t4` supplies the un-winsorized rows for T4 (plan §3: T4 always raw)."""
    draw = int(panel["draw"].iloc[0])
    p = add_block_clusters(panel, calendar) if block_clusters else panel
    clusters = {"I1": "anchor_date"}
    if block_clusters:
        clusters.update({f"I1blk{L}": f"anchor_blk{L}" for L in BLOCK_CLUSTERS})
    out = {"draw": draw, "n_rows": int(len(p))}
    out.update(_t1_rows(p, alpha, clusters))
    sub = p.loc[p["household_id"].isin(r1)].reset_index(drop=True)
    out["T2_n_households_r1"] = int(sub["household_id"].nunique())
    for w in weights:
        target = realized_tau2_target(sub, w) if "theta_h" in sub.columns else 0.0
        out[f"T2_target_{w}"] = target
        for tag, anchor in clusters.items():
            out.update(_i1_rows(sub, w, alpha, tag, anchor, target))
        if boot_draws > 0:
            res = bb.tau2_block_bootstrap(sub, calendar, w, block_length, boot_draws, seed_from_name(f"I2|{draw}"))
            out.update(_boot_rows(res, w, alpha, f"I2L{block_length}", target))
        if hh_draws > 0:
            res = bb.tau2_household_bootstrap(sub, w, hh_draws, seed_from_name(f"I2b|{draw}"))
            out.update(_boot_rows(res, w, alpha, "I2b", target))
    mt = tt.max_t_statistic(p, min_n=min_n)
    out["T3_max_t"], out["T3_n_households"] = mt["max_t"], mt["n_households"]
    t4 = tt.t4_tail_shares(raw_for_t4 if raw_for_t4 is not None else p)
    out["T4_share_gt_2"], out["T4_share_gt_3"], out["T4_scale"] = t4["share_gt_2"], t4["share_gt_3"], t4["scale"]
    return out


def reference_statistics(clock: str, first: int, n: int, cutoffs: dict | None = None,
                         min_n: int = winners.MIN_EVENTS_FULL_SAMPLE) -> pd.DataFrame:
    """Null max-t (raw and winsorized) and raw T4 shares over n disjoint N2 draws."""
    rows = []
    for p in iter_draws(clock, first, n):
        r = {"draw": int(p["draw"].iloc[0]), "max_t_raw": tt.max_t_statistic(p, min_n=min_n)["max_t"]}
        if cutoffs is not None:
            r["max_t_wins"] = tt.max_t_statistic(winsorize(p, cutoffs), min_n=min_n)["max_t"]
        t4 = tt.t4_tail_shares(p)
        r["share_gt_2"], r["share_gt_3"] = t4["share_gt_2"], t4["share_gt_3"]
        rows.append(r)
    return pd.DataFrame(rows)


def apply_reference(stats: pd.DataFrame, ref: pd.DataFrame, variant: str, alpha: float = 0.05) -> pd.DataFrame:
    """T3 / T4 rejections from the disjoint null reference (add-one Monte Carlo p for T4)."""
    out = stats.copy()
    col = "max_t_wins" if variant == "wins" else "max_t_raw"
    crit = float(np.nanquantile(ref[col].to_numpy(dtype=float), 1.0 - alpha))
    out["T3_crit"] = crit
    out["T3_rej"] = (out["T3_max_t"] > crit).astype(float)
    for z in ("2", "3"):
        null = ref[f"share_gt_{z}"].to_numpy(dtype=float)
        pv = [placebo_p_value(v, null, "greater")["p"] for v in out[f"T4_share_gt_{z}"]]
        out[f"T4_z{z}_p"] = pv
        out[f"T4_z{z}_rej"] = (np.asarray(pv) < alpha).astype(float)
    return out


# ---------------------------------------------------------------- work units, checkpoints, pool

_CACHE: dict = {}


def _shared(clock: str) -> tuple[set[str], pd.DatetimeIndex, dict]:
    """Per-process cache of the R1 set, the calendar and the winsorization cutoffs."""
    if clock not in _CACHE:
        _CACHE[clock] = (r1_households(clock), trading_calendar(), pooled_cutoffs(clock))
    return _CACHE[clock]


def unit_path(grid: str, key: str) -> Path:
    return CHECKPOINT_DIR / grid / f"{key}.parquet"


def run_unit(unit: dict) -> dict:
    """One resumable work unit: draws [first, first+n) of one clock, one injection spec and one
    variant (raw | wins), every statistic per panel -> one parquet checkpoint. Returns a summary."""
    grid, key = unit["grid"], unit["key"]
    path = unit_path(grid, key)
    if path.exists():
        return {"key": key, "status": "cached", "seconds": 0.0, "n": int(len(pd.read_parquet(path, columns=["draw"])))}
    t0 = time.perf_counter()
    clock, spec, variant = unit["clock"], unit["spec"], unit["variant"]
    set_name = unit.get("set_name") or DEFAULT_SET[clock]
    r1, cal, cuts = _shared(clock)
    name = f"{grid}|{cell_id(spec)}"
    weights = tuple(unit.get("weights", WEIGHTS))
    rows = []
    for p in iter_draws(clock, unit["first"], unit["n"], set_name=set_name):
        q = inject(p, spec, name, r1=r1)
        raw = q
        if variant == "wins":
            q = winsorize(q, cuts)
        s = panel_statistics(q, r1, cal, weights=weights, boot_draws=unit.get("boot_draws", 0), hh_draws=unit.get("hh_draws", 0),
                             raw_for_t4=raw, block_clusters=unit.get("block_clusters", True))
        if variant == "wins" and unit.get("t1_raw", False):  # plan §11: T1 is judged on raw Y even when tau2 is winsorized
            clusters = {"I1": "anchor_date", **{f"I1blk{L}": f"anchor_blk{L}" for L in BLOCK_CLUSTERS}}
            s.update({"T1raw_" + k[3:]: v for k, v in _t1_rows(add_block_clusters(raw, cal), 0.05, clusters).items()})
        s["T1_target"] = float(q["theta_h"].mean())
        for k, v in q.attrs.get("inject_info", {}).items():
            s[f"inj_{k}"] = v
        rows.append(s)
    frame = pd.DataFrame(rows)
    for k, v in (("clock", clock), ("variant", variant), ("spec_json", json.dumps(spec, sort_keys=True)), ("grid", grid),
                 ("panel_set", set_name)):
        frame[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"key": key, "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def run_units(units: list[dict], workers: int, log: Path | None = None) -> list[dict]:
    """Run units in a process pool (spawn-safe); each finished unit is logged with its time."""
    import multiprocessing as mp

    results = []
    pending = [u for u in units if not unit_path(u["grid"], u["key"]).exists()]
    log_line(log, f"{len(units)} units, {len(pending)} pending, {workers} workers")
    if not pending:
        return [run_unit(u) for u in units]
    if workers <= 1:
        for u in pending:
            r = run_unit(u)
            log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")
            results.append(r)
        return results
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(run_unit, pending):
            log_line(log, f"unit {r['key']} {r['status']} n={r['n']} {r['seconds']:.1f}s")
            results.append(r)
    return results


def load_grid(grid: str) -> pd.DataFrame:
    files = sorted((CHECKPOINT_DIR / grid).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no checkpoints under {CHECKPOINT_DIR / grid}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------- null reference (T3 / T4)

REFERENCE_FIRST = 1000
REFERENCE_N = 1000


def reference_unit(unit: dict) -> dict:
    """max-t (raw, winsorized) and raw T4 shares for one batch of reference draws."""
    clock, first, n = unit["clock"], unit["first"], unit["n"]
    path = CHECKPOINT_DIR / "reference" / f"{clock}_{first:04d}.parquet"
    if path.exists():
        return {"key": path.stem, "status": "cached", "seconds": 0.0, "n": n}
    t0 = time.perf_counter()
    _, _, cuts = _shared(clock)
    frame = reference_statistics(clock, first, n, cuts)
    frame["clock"] = clock
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {"key": path.stem, "status": "done", "seconds": time.perf_counter() - t0, "n": int(len(frame))}


def reference_units(clocks=("T", "P"), first: int = REFERENCE_FIRST, n: int = REFERENCE_N) -> list[dict]:
    return [{"clock": c, "first": f, "n": BATCH_SIZE} for c in clocks for f in range(first, first + n, BATCH_SIZE)]


def load_reference(clock: str) -> pd.DataFrame:
    files = sorted((CHECKPOINT_DIR / "reference").glob(f"{clock}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no reference checkpoints for clock {clock}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_reference(clocks, workers: int, log: Path | None = None) -> None:
    import multiprocessing as mp

    units = [u for u in reference_units(clocks) if not (CHECKPOINT_DIR / "reference" / f"{u['clock']}_{u['first']:04d}.parquet").exists()]
    log_line(log, f"reference: {len(units)} pending batches")
    if not units:
        return
    if workers <= 1:
        for u in units:
            r = reference_unit(u)
            log_line(log, f"reference {r['key']} {r['seconds']:.1f}s")
        return
    with mp.get_context("spawn").Pool(min(workers, len(units))) as pool:
        for r in pool.imap_unordered(reference_unit, units):
            log_line(log, f"reference {r['key']} {r['seconds']:.1f}s")


def mc_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1.0 - p) / n)) if n > 0 and np.isfinite(p) else np.nan


# ---------------------------------------------------------------- I1cal: placebo-calibrated I1 critical value (checkpoint-5 candidate)

CAL_FIRST, CAL_N = 500, 500  # size draws used for the critical value in the power grid (disjoint from draws 0-499)


def i1_t_stat(frame: pd.DataFrame, w: str, tag: str = "I1") -> pd.Series:
    """Recover the I1 t-statistic of b_hat from the stored tau2 interval: t = tau2 x 2 c / (hi - lo)."""
    from scipy import stats as st

    k = f"T2_{tag}_{w}"
    if f"{k}_t" in frame.columns and frame[f"{k}_t"].notna().all():  # stored directly since wave 3
        return frame[f"{k}_t"].astype(float)
    crit = st.t.ppf(0.975, frame[f"{k}_dof"].to_numpy(dtype=float))
    return frame[f"{k}_tau2"] * 2.0 * crit / (frame[f"{k}_hi"] - frame[f"{k}_lo"])


def calibrated_critical(size_stats: pd.DataFrame, clock: str, variant: str, w: str, first: int = CAL_FIRST,
                        n: int = CAL_N, alpha: float = 0.05) -> float:
    """(1 - alpha) quantile of the null I1 t-statistic over size draws [first, first + n) of clock x variant."""
    g = size_stats[(size_stats["clock"] == clock) & (size_stats["variant"] == variant)
                   & (size_stats["draw"] >= first) & (size_stats["draw"] < first + n)]
    if len(g) < n:
        raise ValueError(f"only {len(g)} size draws for {clock}/{variant} in [{first}, {first + n})")
    return float(np.quantile(i1_t_stat(g, w).to_numpy(dtype=float), 1.0 - alpha))


def add_i1cal(stats: pd.DataFrame, size_stats: pd.DataFrame, weights=WEIGHTS, alpha: float = 0.05,
              first: int = CAL_FIRST, n: int = CAL_N) -> pd.DataFrame:
    """Add T2_I1cal_<w>_rej1 (t above the N2-calibrated critical value from size draws [first, first + n))
    per clock x variant; weights whose I1 columns are absent from a group are skipped."""
    out = stats.copy()
    for (clock, variant), idx in out.groupby(["clock", "variant"]).groups.items():
        for w in weights:
            if f"T2_I1_{w}_tau2" not in out.columns or out.loc[idx, f"T2_I1_{w}_tau2"].isna().all():
                continue
            c = calibrated_critical(size_stats, clock, variant, w, first=first, n=n, alpha=alpha)
            t = i1_t_stat(out.loc[idx], w)
            out.loc[idx, f"T2_I1cal_{w}_crit"] = c
            out.loc[idx, f"T2_I1cal_{w}_rej1"] = (t > c).astype(float)
    return out
