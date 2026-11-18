"""Size at zero effect, version 2 (plan §11 procedures on the final null sets).

Per clock (T: the original set, unchanged; P: the all-lines set) and outcome variant (raw | wins):
N2 draws 0-999 are the TEST panels, N2 draws 1000-1999 the REFERENCE panels (I1cal critical value,
T3 critical value, T4 null shares). Procedures scored:
  T1   raw Y, household x 20-session-block CGM (one-/two-sided, coverage of 0), and the N1 placebo p
       (T1 mean of N1 draws 0-999 against the N1 distribution of draws 1000-1999; add-one p);
  T2   ES/EH/PW: I1cal = I1 t-statistic vs its N2 95th percentile (reference draws) — decision rule —
       and the add-one p over the other 1,999 draws (the actual run's p); I2b household bootstrap
       (499 draws here; the actual run uses 4,999) one-/two-sided and coverage of the realized target
       (0 at the null); the I1 t-interval's coverage of 0 is reported as the descriptive interval;
  T3   max-t vs the reference 95th percentile (winsorized primary, raw robustness);
  T4   raw Y, shares z > 2 / z > 3 vs the reference shares (add-one p).
Wave-2 diagnostics (day clusters, I2 calendar-block bootstrap) are not re-run; day-cluster T1/T2 rows
are kept in the CSV as `plan_procedure = False` references.

CLI: python -m trading_on_disclosure_v4.calibration.run_size_v2 [--clocks T,P] [--boot 499] [--workers 10]
     [--aggregate-only]
Outputs: data/v4/calibration/size_null_v2.csv; checkpoints data/v4/restricted/c/checkpoints/size_v2/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.inference.placebo import placebo_p_value

GRID = "size_v2"
N1_GRID = "size_v2_n1"
N_TEST, N_ALL = 1000, 2000
SIZE_BAND = (0.03, 0.07)
COVERAGE_MIN = 0.93
ALPHA = 0.05


def size_units(clocks, boot: int) -> list[dict]:
    units = []
    for clock in clocks:
        for variant in ("raw", "wins"):
            for first in range(0, N_ALL, rc.BATCH_SIZE):
                test = first < N_TEST
                units.append({"grid": GRID, "key": f"{clock}_{variant}_{first:04d}", "clock": clock, "variant": variant,
                              "spec": {"kind": "none"}, "first": first, "n": rc.BATCH_SIZE, "boot_draws": 0,
                              "hh_draws": boot if test else 0, "t1_raw": True, "set_name": rc.DEFAULT_SET[clock]})
    return units


# ---------------------------------------------------------------- N1 placebo distribution of the T1 mean

def n1_means_path(clock: str) -> Path:
    return rc.CHECKPOINT_DIR / N1_GRID / f"{clock}_n1_means.parquet"


def n1_means(clock: str, set_name: str | None = None, n_draws: int = N_ALL) -> pd.DataFrame:
    """Per-draw mean of y_h20 (bp) over all purchase rows of the N1 (uncentered) panels — the T1 statistic
    under the timing null; pooled placebo statistic, no household identity."""
    path = n1_means_path(clock)
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for b in range(0, n_draws // rc.BATCH_SIZE):
        batch = rc.load_batch(clock, b, "N1", set_name)
        m = batch.groupby("draw", sort=True)["y_h20"].agg(["mean", "size"])
        rows.append(pd.DataFrame({"draw": m.index.astype(int), "t1_mean_bp": m["mean"].to_numpy() * rc.BP, "n_rows": m["size"].to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["clock"] = clock
    out["panel_set"] = set_name or rc.DEFAULT_SET[clock]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def t1_placebo_rows(clock: str, alpha: float = ALPHA) -> list[dict]:
    m = n1_means(clock)
    test = m[m["draw"] < N_TEST].sort_values("draw")
    ref = m[m["draw"] >= N_TEST]["t1_mean_bp"].to_numpy(dtype=float)
    allv = m.set_index("draw")["t1_mean_bp"]
    rows = []
    for alt, sided in (("two-sided", "two-sided"), ("greater", "one-sided")):
        p_ref = np.array([placebo_p_value(v, ref, alt)["p"] for v in test["t1_mean_bp"]])
        p_loo = np.array([placebo_p_value(v, allv.drop(int(d)).to_numpy(dtype=float), alt)["p"]
                          for d, v in zip(test["draw"], test["t1_mean_bp"], strict=True)])
        for inf, pv in (("N1_placebo_p", p_ref), ("N1_placebo_p_loo", p_loo)):
            rate = float(np.mean(pv < alpha))
            rows.append({"clock": clock, "variant": "raw", "test": "T1", "weights": "none", "inference": inf, "sidedness": sided,
                         "reject_rate": rate, "reject_mc_se": rc.mc_se(rate, len(pv)), "n_panels": int(len(pv)),
                         "in_size_band": bool(SIZE_BAND[0] <= rate <= SIZE_BAND[1]), "coverage": np.nan, "coverage_mc_se": np.nan,
                         "coverage_ok": None, "plan_procedure": inf == "N1_placebo_p", "role": "registered decision (T1 vs timing null)" if inf == "N1_placebo_p" else "p over the other 1,999 draws",
                         "mean_estimate": float(m["t1_mean_bp"].mean()), "sd_estimate": float(m["t1_mean_bp"].std(ddof=1)),
                         "null_mean_reference": float(ref.mean()), "null_sd_reference": float(ref.std(ddof=1)), "n_reference": int(len(ref))})
    return rows


# ---------------------------------------------------------------- summary

def _row(clock, variant, test, weights, inference, sided, rej, cov, plan: bool, role: str, extra=None) -> dict:
    rej = np.asarray(rej, dtype=float)
    n = int(np.isfinite(rej).sum())
    rate = float(np.nanmean(rej)) if n else np.nan
    row = {"clock": clock, "variant": variant, "test": test, "weights": weights, "inference": inference, "sidedness": sided,
           "reject_rate": rate, "reject_mc_se": rc.mc_se(rate, n), "n_panels": n,
           "in_size_band": bool(SIZE_BAND[0] <= rate <= SIZE_BAND[1]) if n else None, "plan_procedure": plan, "role": role}
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        c = float(np.nanmean(cov))
        row.update({"coverage": c, "coverage_mc_se": rc.mc_se(c, int(np.isfinite(cov).sum())), "coverage_ok": bool(c >= COVERAGE_MIN)})
    else:
        row.update({"coverage": np.nan, "coverage_mc_se": np.nan, "coverage_ok": None})
    row.update(extra or {})
    return row


def _t1(g: pd.DataFrame, clock: str, variant: str, prefix: str, tag: str, plan: bool, role: str) -> list[dict]:
    k = f"{prefix}_{tag}"
    cov = ((g[f"{k}_lo"] <= 0.0) & (0.0 <= g[f"{k}_hi"])).astype(float)
    ex = {"mean_estimate": g[f"{k}_est"].mean(), "sd_estimate": g[f"{k}_est"].std(ddof=1), "mean_se": g[f"{k}_se"].mean(),
          "sd_over_se": g[f"{k}_est"].std(ddof=1) / g[f"{k}_se"].mean(), "mean_dof": g[f"{k}_dof"].mean()}
    return [_row(clock, variant, "T1", "none", tag, "one-sided", g[f"{k}_rej1"], cov, plan, role, ex),
            _row(clock, variant, "T1", "none", tag, "two-sided", g[f"{k}_rej2"], cov, plan, role, ex)]


def summarize(stats: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    rows = []
    for (clock, variant), g_all in stats.groupby(["clock", "variant"], sort=True):
        g = g_all[g_all["draw"] < N_TEST]
        ref = g_all[g_all["draw"] >= N_TEST]
        if len(g) != N_TEST or len(ref) != N_ALL - N_TEST:
            raise ValueError(f"{clock}/{variant}: {len(g)} test and {len(ref)} reference draws")
        if variant == "raw":  # T1 is a raw-Y procedure; the wins rows carry T1raw_* = the same numbers
            rows += _t1(g, clock, variant, "T1", "I1blk20", True, "registered T1 inference (plan §11)")
            rows += _t1(g, clock, variant, "T1", "I1", False, "brief's day clusters (reference)")
            rows += _t1(g, clock, variant, "T1", "I1blk60", False, "60-session blocks (reference)")
            rows += t1_placebo_rows(clock, alpha)
        role_t2 = "primary outcome variant" if variant == "wins" else "robustness (raw tau2)"
        for w in rc.WEIGHTS:
            k = f"T2_I1_{w}"
            t_test, t_ref = rc.i1_t_stat(g, w).to_numpy(dtype=float), rc.i1_t_stat(ref, w).to_numpy(dtype=float)
            t_all = rc.i1_t_stat(g_all, w)
            t_all.index = g_all["draw"].to_numpy()
            crit = float(np.quantile(t_ref, 1.0 - alpha))
            rej_crit = (t_test > crit).astype(float)
            p_loo = np.array([placebo_p_value(v, t_all.drop(int(d)).to_numpy(dtype=float), "greater")["p"]
                              for d, v in zip(g["draw"], t_test, strict=True)])
            ex = {"mean_estimate": g[f"{k}_tau2"].mean(), "sd_estimate": g[f"{k}_tau2"].std(ddof=1), "crit_value": crit,
                  "t_stat_sd_test": float(np.std(t_test, ddof=1)), "t_stat_sd_reference": float(np.std(t_ref, ddof=1)),
                  "reject_rate_p_rule": float(np.mean(p_loo < alpha)), "mean_dof": g[f"{k}_dof"].mean(),
                  "psd_corrected_share": g[f"{k}_psd"].mean(),
                  # calibrated one-sided upper bound tau2 + c_cal x SE(tau2), SE(tau2) = (hi - lo) / (2 t_crit(dof))
                  "mean_upper_bound_tau2_cal": float(np.mean(g[f"{k}_tau2"] + crit * (g[f"{k}_hi"] - g[f"{k}_lo"]) / (2.0 * _t_crit(g[f"{k}_dof"]))))}
            rows.append(_row(clock, variant, "T2", w, "I1cal", "one-sided", rej_crit, None, True, f"T2 decision rule; {role_t2}", ex))
            cov = g[f"{k}_cov"]
            ex = {"mean_estimate": g[f"{k}_tau2"].mean(), "sd_estimate": g[f"{k}_tau2"].std(ddof=1),
                  "mean_interval_width": (g[f"{k}_hi"] - g[f"{k}_lo"]).mean(), "mean_dof": g[f"{k}_dof"].mean(),
                  "psd_corrected_share": g[f"{k}_psd"].mean()}
            rows.append(_row(clock, variant, "T2", w, "I1", "one-sided", g[f"{k}_rej1"], cov, False, f"descriptive t-interval (nominal test shown for reference); {role_t2}", ex))
            rows.append(_row(clock, variant, "T2", w, "I1", "two-sided", g[f"{k}_rej2"], cov, False, f"descriptive t-interval; {role_t2}", ex))
            kb = f"T2_I2b_{w}"
            if f"{kb}_rej1" in g:
                ex = {"mean_estimate": g[f"{kb}_tau2"].mean(), "sd_estimate": g[f"{kb}_tau2"].std(ddof=1),
                      "mean_interval_width": (g[f"{kb}_hi"] - g[f"{kb}_lo"]).mean(), "boot_sd_over_mc_sd": g[f"{kb}_bsd"].mean() / g[f"{kb}_tau2"].std(ddof=1),
                      "mean_invalid_draws": g[f"{kb}_ninv"].mean(), "mean_upper_bound_tau2": g[f"{kb}_up"].mean()}
                rows.append(_row(clock, variant, "T2", w, "I2b", "one-sided", g[f"{kb}_rej1"], g[f"{kb}_cov"], True, f"unconditional companion (499 draws here; 4,999 in the actual run); {role_t2}", ex))
                rows.append(_row(clock, variant, "T2", w, "I2b", "two-sided", g[f"{kb}_rej2"], g[f"{kb}_cov"], True, f"unconditional companion; {role_t2}", ex))
        crit3 = float(np.nanquantile(ref["T3_max_t"].to_numpy(dtype=float), 1.0 - alpha))
        ex = {"mean_estimate": g["T3_max_t"].mean(), "sd_estimate": g["T3_max_t"].std(ddof=1), "critical_value": crit3,
              "mean_n_households": g["T3_n_households"].mean()}
        rows.append(_row(clock, variant, "T3", "none", "N2_critical_value", "one-sided", (g["T3_max_t"] > crit3).astype(float), None, True,
                         "registered T3 (winsorized primary; raw robustness)" if variant == "wins" else "robustness (raw)", ex))
        if variant == "raw":  # T4 is always raw (raw_for_t4); the wins rows carry the same shares
            for z in ("2", "3"):
                null = ref[f"T4_share_gt_{z}"].to_numpy(dtype=float)
                pv = np.array([placebo_p_value(v, null, "greater")["p"] for v in g[f"T4_share_gt_{z}"]])
                ex = {"mean_estimate": g[f"T4_share_gt_{z}"].mean(), "sd_estimate": g[f"T4_share_gt_{z}"].std(ddof=1),
                      "null_mean_reference": float(null.mean())}
                rows.append(_row(clock, variant, f"T4_z{z}", "none", "N2_placebo_p", "one-sided", (pv < alpha).astype(float), None, True, "registered T4 (raw Y)", ex))
    return pd.DataFrame(rows)


def _t_crit(dof: pd.Series) -> np.ndarray:
    from scipy import stats as st

    return st.t.ppf(0.975, dof.to_numpy(dtype=float))


def aggregate(clocks, log: Path | None = None) -> pd.DataFrame:
    stats = rc.load_grid(GRID)
    stats = stats[stats["clock"].isin(clocks)]
    long = summarize(stats)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(rc.CALIB_DIR / "size_null_v2.csv", index=False)
    rc.log_line(log, f"size_null_v2.csv {len(long)} rows; draws per clock x variant {stats.groupby(['clock', 'variant']).size().to_dict()}")
    return long


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="size calibration v2 at zero effect on N2 panels (plan §11 procedures)")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--boot", type=int, default=499)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--log", type=Path, default=rc.LOG_DIR / "run_size_v2.log")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    t0 = time.time()
    if not args.aggregate_only:
        for c in clocks:
            m = n1_means(c)
            rc.log_line(args.log, f"N1 T1 means {c}: {len(m)} draws, mean {m['t1_mean_bp'].mean():.2f} bp, SD {m['t1_mean_bp'].std(ddof=1):.2f}")
        rc.run_units(size_units(clocks, args.boot), args.workers, args.log)
    aggregate(clocks, args.log)
    rc.log_line(args.log, f"run_size_v2 finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
