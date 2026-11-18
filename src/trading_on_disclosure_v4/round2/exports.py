"""Exports 1, 2, 3 and 5 of the Round 2 closeout and the central-figure input, built from the original panels.

python -m trading_on_disclosure_v4.round2.exports
Writes data/v4/round2/closeout/<run>/inputs/ (household-level, anonymous keys) and the purchase-level design under
data/v4/restricted/round2_closeout/<run>/inputs/; every frozen comparator is checked before a file is written.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.round2 import common as cm

CLOCKS = ("T", "P")
#: Table 3 rows -> (spec name in analysis_specs, results.csv analysis, results.csv variant, benchmark, outcome cutoffs)
TABLE3 = {
    "principal": ("tau/event_share", "tau", "event_share", "fitted_intercept", "ff5mom_h20"),
    "equal_household": ("tau/equal_household", "tau", "equal_household", "fitted_intercept", "ff5mom_h20"),
    "product": ("tau/pair", "tau", "pair", "fitted_intercept", "ff5mom_h20"),
    "no_activity_adjustment": ("tau_no_control/event_share", "tau_no_control", "event_share", "fitted_intercept", "ff5mom_h20"),
    "intercept_omitted": ("robustness/no_alpha", "robustness", "no_alpha", "intercept_omitted_fixed_loadings", "ff5mom_noalpha_h20"),
    "market_adjusted": ("robustness/market_adjusted", "robustness", "market_adjusted", "market_adjusted", "mkt_h20"),
}
REL_TOL = 1e-9


def weighted_projection_residual(m: np.ndarray, C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Residual of m on the columns of C under weights w (the original lstsq convention)."""
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(C * sw[:, None], m * sw, rcond=None)
    return m - C @ beta


def household_covariance(m_a, m_b, n_a, w, activity: bool) -> float:
    """kappa = sum w a b / sum w with both periods residualized on [1, log(1+n_A)] (or [1])."""
    C = tau_mod.activity_design(n_a) if activity else np.ones((len(n_a), 1))
    a = weighted_projection_residual(np.asarray(m_a, float), C, w)
    b = weighted_projection_residual(np.asarray(m_b, float), C, w)
    return float(np.sum(w * a * b) / np.sum(w))


def check(value: float, frozen: float, what: str, log) -> float:
    gap = abs(value - frozen)
    if gap > REL_TOL * max(1.0, abs(frozen)):
        raise AssertionError(f"{what}: reproduced {value!r} differs from frozen {frozen!r} by {gap:.3e}")
    log(f"ok {what}: {value:.10f} vs frozen {frozen:.10f} (|diff| {gap:.2e})")
    return gap


# ---------------------------------------------------------------- export 1

def principal_means(ctx: rs.Context, clock: str, key: pd.Series, log) -> tuple[pd.DataFrame, dict]:
    """Household-period means of Y1 (fitted intercept) and Y0 (intercept omitted, loadings fixed) on the
    principal cohort's purchases, raw and separately winsorized at each outcome's own pooled cutoffs."""
    base = dc.multi_outcome_panel(ctx, clock)
    ok = np.isfinite(base[["Y1", "Y0"]].to_numpy(dtype=float)).all(axis=1)
    if not ok.all():
        raise AssertionError(f"{clock}: {int((~ok).sum())} principal purchases lack one of the two outcomes")
    panel = base.copy()
    panel["Y1w"] = dc.winsorized_column(base, "Y1", clock)
    panel["Y0w"] = dc.winsorized_column(base, "Y0", clock)
    raw = dc.household_means(panel, ["Y1", "Y0"])
    wins = dc.household_means(panel, ["Y1w", "Y0w"])
    if not (raw.index.equals(wins.index) and np.array_equal(raw["n_A"], wins["n_A"]) and np.array_equal(raw["n_B"], wins["n_B"])):
        raise AssertionError(f"{clock}: raw and winsorized cohorts differ")
    rows = []
    for scale, hh, s in (("raw", raw, ""), ("winsorized", wins, "w")):
        out = pd.DataFrame({"clock": clock, "scale": scale, "household_index": key.reindex(hh.index.astype(str)).to_numpy(),
                            "n_A": hh["n_A"].astype(np.int64).to_numpy(), "n_B": hh["n_B"].astype(np.int64).to_numpy(),
                            "Y1_A": hh[f"Y1{s}_A"].to_numpy(), "Y1_B": hh[f"Y1{s}_B"].to_numpy(),
                            "Y0_A": hh[f"Y0{s}_A"].to_numpy(), "Y0_B": hh[f"Y0{s}_B"].to_numpy()})
        rows.append(out)
    frame = pd.concat(rows, ignore_index=True)
    if frame["household_index"].isna().any():
        raise AssertionError("household without an anonymous key")
    # frozen comparators at full precision
    stats = {}
    for scale in ("raw", "winsorized"):
        g = frame[frame["scale"] == scale]
        w = np.sqrt(g["n_A"].to_numpy(float) * g["n_B"].to_numpy(float))
        for k in ("Y1", "Y0"):
            stats[(scale, k)] = household_covariance(g[f"{k}_A"], g[f"{k}_B"], g["n_A"].to_numpy(float), w, True)
    tag = {"raw": "raw", "winsorized": "wins"}
    gaps = []
    for scale in ("raw", "winsorized"):
        gaps.append(check(stats[(scale, "Y1")], cm.frozen_value(clock, "decomposition", "no_alpha_omit", f"tau2_total_{tag[scale]}", path=cm.FROZEN_DECOMP),
                          f"{clock} {scale} K11 vs decomposition.csv", log))
        gaps.append(check(stats[(scale, "Y0")], cm.frozen_value(clock, "decomposition", "no_alpha_omit", f"tau2_base_{tag[scale]}", path=cm.FROZEN_DECOMP),
                          f"{clock} {scale} K00 vs decomposition.csv", log))
    gaps.append(check(stats[("winsorized", "Y1")], cm.frozen_value(clock, "tau", "event_share", "tau2"), f"{clock} K11 winsorized vs results.csv tau/event_share", log))
    gaps.append(check(stats[("winsorized", "Y0")], cm.frozen_value(clock, "robustness", "no_alpha", "tau2"), f"{clock} K00 winsorized vs results.csv robustness/no_alpha", log))
    gaps.append(check(stats[("raw", "Y1")], cm.frozen_value(clock, "robustness", "raw_outcome", "tau2"), f"{clock} K11 raw vs results.csv robustness/raw_outcome", log))
    meta = {"n_households": int(len(raw)), "n_purchases": int(raw["n_A"].sum() + raw["n_B"].sum()),
            "n_purchases_A": int(raw["n_A"].sum()), "n_purchases_B": int(raw["n_B"].sum()),
            "cutoffs_bp": {"Y1": cm.pooled_cutoffs_record(clock, "ff5mom_h20"), "Y0": cm.pooled_cutoffs_record(clock, "ff5mom_noalpha_h20")},
            "covariances_bp2": {f"{s}_{k}": v for (s, k), v in stats.items()}, "max_abs_gap_vs_frozen": float(max(gaps))}
    return frame, meta


# ---------------------------------------------------------------- export 2

def table3_means(ctx: rs.Context, clock: str, key: pd.Series, log) -> tuple[pd.DataFrame, list[dict], dict]:
    specs = sp.analysis_specs(clock)
    rows, metas, panels = [], [], {}
    for label, (name, analysis, variant, bench, outcome) in TABLE3.items():
        spec = specs[name]
        panel = ctx.panel(spec)
        panels[label] = panel
        table = tau_mod.household_table(panel, spec.weights, residualize_activity=spec.residualize_activity)
        hid = pd.Index(table.household_id).astype(str)
        frame = pd.DataFrame({"clock": clock, "specification": label, "household_index": key.reindex(hid).to_numpy(),
                              "n_A": table.n_a.astype(np.int64), "n_B": table.n_b.astype(np.int64), "m_A_bp": table.ybar_a, "m_B_bp": table.ybar_b,
                              "weight": table.w, "activity_log": np.log1p(table.n_a)})
        kappa = household_covariance(table.ybar_a, table.ybar_b, table.n_a, table.w, spec.residualize_activity)
        gap = check(kappa, cm.frozen_value(clock, analysis, variant, "tau2"), f"{clock} {label} kappa vs results.csv {analysis}/{variant}", log)
        frozen_n = int(cm.frozen_value(clock, analysis, variant, "tau2", column="n_events"))
        frozen_h = int(cm.frozen_value(clock, analysis, variant, "tau2", column="n_households"))
        if frozen_n != int(table.n_a.sum() + table.n_b.sum()) or frozen_h != len(table.n_a):
            raise AssertionError(f"{clock} {label}: counts differ from results.csv")
        rows.append(frame)
        metas.append({"clock": clock, "specification": label, "spec_name": spec.name, "results_key": f"{analysis}/{variant}/tau2",
                      "benchmark": bench, "outcome": spec.outcome, "transformation": "winsorized at the committed pooled p1/p99 by period" if spec.winsorized else "raw",
                      "cutoffs_bp": cm.pooled_cutoffs_record(clock, outcome), "weights": spec.weights,
                      "weight_rule": {"ES": "sqrt(n_A*n_B)", "EH": "1", "PW": "n_A*n_B"}[spec.weights],
                      "activity_adjusted": bool(spec.residualize_activity), "stage_two_activity_control": bool(spec.activity_control),
                      "control_design": "[1, log(1+n_A)] in both periods" if spec.residualize_activity else "[1] (constant only) in both periods",
                      "eligibility": f"rule {spec.rule} on finite outcomes; duplicate treatment {spec.dup}", "n_households": len(table.n_a),
                      "n_purchases": int(table.n_a.sum() + table.n_b.sum()), "kappa_bp2": kappa, "abs_gap_vs_frozen": gap,
                      "bootstrap_seed_name": f"seed_from('{spec.name}', '{clock}', 'household bootstrap')"})
    out = pd.concat(rows, ignore_index=True)
    if out["household_index"].isna().any():
        raise AssertionError("household without an anonymous key")
    return out, metas, panels


# ---------------------------------------------------------------- export 3

def household_tests(ctx: rs.Context, clock: str, key: pd.Series, log) -> tuple[pd.DataFrame, dict]:
    spec = sp.analysis_specs(clock)["best_household"]
    panel = ctx.panel(spec)
    hh = winners.household_stats(panel, min_n=winners.MIN_EVENTS_FULL_SAMPLE)
    out = pd.DataFrame({"clock": clock, "household_index": key.reindex(hh["household_id"].astype(str)).to_numpy(), "n": hh["n"].astype(np.int64),
                        "mean_bp": hh["mean"].to_numpy(float), "sd_bp": hh["sd"].to_numpy(float), "se_bp": hh["se"].to_numpy(float),
                        "t": hh["t"].to_numpy(float), "dof": (hh["n"] - 1).astype(np.int64), "p_two_sided": hh["p"].to_numpy(float)})
    n_sig = int((out["p_two_sided"] < 0.05).sum())
    frozen_n = int(cm.frozen_value(clock, "best_household", "vs_centered_placebo", "n_p05"))
    frozen_h = int(cm.frozen_value(clock, "best_household", "vs_centered_placebo", "n_p05", column="n_households"))
    if n_sig != frozen_n or len(out) != frozen_h:
        raise AssertionError(f"{clock}: household tests {n_sig}/{len(out)} differ from results.csv {frozen_n}/{frozen_h}")
    check(float(out["mean_bp"].max()), cm.frozen_value(clock, "best_household", "vs_centered_placebo", "max_mean"), f"{clock} max household mean", log)
    check(float(out["t"].max()), cm.frozen_value(clock, "best_household", "vs_centered_placebo", "max_t"), f"{clock} max household t", log)
    meta = {"spec_name": spec.name, "outcome": spec.outcome, "transformation": "winsorized at the committed pooled p1/p99 by period",
            "cutoffs_bp": cm.pooled_cutoffs_record(clock, "ff5mom_h20"), "eligibility": "at least 20 purchases in periods A and B together (rule full_n20); duplicate treatment default",
            "sd": "sample SD, ddof=1", "se": "sd/sqrt(n)", "dof": "n-1", "test": "two-sided Student t, reject when p < 0.05 (strict)",
            "n_households": int(len(out)), "n_rejections": n_sig, "code": "estimators.winners.household_stats"}
    return out, meta


# ---------------------------------------------------------------- export 5

def regression_design(ctx: rs.Context, clock: str, panels: dict[str, pd.DataFrame], key: pd.Series, log) -> pd.DataFrame:
    """Period-B purchase rows of the principal covariance regressions (both benchmarks): y_B, x_A, controls, weights,
    20-session entry block, WLS fit and residual. Purchase level: restricted."""
    frames = []
    specs = sp.analysis_specs(clock)
    for label in ("principal", "intercept_omitted"):
        spec = specs[TABLE3[label][0]]
        panel = rs.block_cluster(panels[label], ctx.calendar)
        table = tau_mod.household_table(panel, spec.weights)
        ev = tau_mod.event_design(panel, table, anchor="anchor_blk")
        b_rows = tau_mod.finite_outcome_rows(panel)[0]
        b_rows = b_rows.loc[b_rows["period"] == tau_mod.PERIOD_B, ["household_id", "event_id"]].reset_index(drop=True)
        hh_codes = pd.Index(table.household_id).get_indexer(b_rows["household_id"])
        b_rows = b_rows.loc[hh_codes >= 0].reset_index(drop=True)
        if len(b_rows) != len(ev) or not np.array_equal(b_rows["household_id"].to_numpy(), ev["household_id"].to_numpy()):
            raise AssertionError(f"{clock} {label}: event ids do not align with event_design rows")
        ev["event_id"] = b_rows["event_id"].to_numpy()
        X = tau_mod.design_matrix(ev, True)
        beta = tau_mod.wls_slope(ev["y"].to_numpy(), X, ev["event_weight"].to_numpy())
        fitted = X @ beta
        var_x = tau_mod.var_w(table.x, table.w)
        check(float(beta[1] * var_x), cm.frozen_value(clock, *TABLE3[label][1:3], "tau2"), f"{clock} {label} event-level slope x Var_w(x)", log)
        hhm = pd.DataFrame({"household_id": table.household_id.astype(str), "n_A": table.n_a, "n_B": table.n_b, "w_h": table.w, "m_A_bp": table.ybar_a})
        ev = ev.merge(hhm, on="household_id", how="left")
        frames.append(pd.DataFrame({"clock": clock, "specification": label, "household_index": key.reindex(ev["household_id"].astype(str)).to_numpy(),
                                    "event_id": ev["event_id"].astype(str).to_numpy(), "entry_block": ev["anchor_blk"].astype(np.int64),
                                    "y_B_bp": ev["y"].to_numpy(float), "x_A_bp": ev["x"].to_numpy(float), "activity_log": ev["log_n_a"].to_numpy(float),
                                    "event_weight": ev["event_weight"].to_numpy(float), "w_h": ev["w_h"].to_numpy(float), "n_A": ev["n_A"].astype(np.int64),
                                    "n_B": ev["n_B"].astype(np.int64), "m_A_bp": ev["m_A_bp"].to_numpy(float), "fitted_bp": fitted, "residual_bp": ev["y"].to_numpy(float) - fitted,
                                    "coef_intercept": beta[0], "coef_slope": beta[1], "coef_activity": beta[2], "var_w_x_A": var_x}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- figure input

def figure_input(principal: pd.DataFrame) -> pd.DataFrame:
    g = principal[(principal["clock"] == "T") & (principal["scale"] == "winsorized")].sort_values("household_index")
    rows = []
    for bench, k in (("fitted_intercept", "Y1"), ("intercept_omitted", "Y0")):
        rows.append(pd.DataFrame({"clock": "T", "household_index": g["household_index"].to_numpy(), "benchmark": bench,
                                  "period_A_mean_bp": g[f"{k}_A"].to_numpy(), "period_B_mean_bp": g[f"{k}_B"].to_numpy(),
                                  "activity_log": np.log1p(g["n_A"].to_numpy(float)), "weight": np.sqrt(g["n_A"].to_numpy(float) * g["n_B"].to_numpy(float)),
                                  "outcome_transform": "separately_winsorized"}))
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Round 2 closeout exports 1, 2, 3, 5 and the figure input")
    ap.parse_args(argv)
    log = cm.logger("exports")
    t0 = time.time()
    ctx, _ = cm.load_context(CLOCKS)
    key = cm.household_key_map(ctx.frames)
    log(f"anonymous key over {len(key)} households (crosswalk under the restricted run folder)")
    principal, t3, tests, design = [], [], [], []
    meta = {"run_id": cm.RUN_ID, "environment": cm.environment(), "units": {"means": "bp", "covariances": "bp^2", "counts": "purchases"},
            "household_index": "global anonymous key: 1-based rank of the household id in the sorted union of both clocks' event frames; "
                               "monotone in the original sorted order, so sorting a cohort by it reproduces the order of the original routines",
            "periods": {"A": "2015-2020, return windows closing by 31 December 2020", "B": "2021-2025"},
            "return_window": "20-session buy-and-hold abnormal return after the entry session (bp)",
            "factor_model": "FF5 + momentum, 252-session window ending 21 sessions before the anchor, minimum 60 observations (forward cache)",
            "exports": {}}
    for clock in CLOCKS:
        f1, m1 = principal_means(ctx, clock, key, log)
        principal.append(f1)
        meta["exports"].setdefault("principal_household_means", {})[clock] = m1
        f2, m2, panels = table3_means(ctx, clock, key, log)
        t3.append(f2)
        meta["exports"].setdefault("table3_household_means", {})[clock] = m2
        f3, m3 = household_tests(ctx, clock, key, log)
        tests.append(f3)
        meta["exports"].setdefault("household_test_statistics", {})[clock] = m3
        design.append(regression_design(ctx, clock, panels, key, log))
    principal = pd.concat(principal, ignore_index=True)
    t3 = pd.concat(t3, ignore_index=True)
    tests = pd.concat(tests, ignore_index=True)
    design = pd.concat(design, ignore_index=True)
    # anonymous event keys; the event-id crosswalk stays restricted
    design = design.sort_values(["clock", "specification", "household_index", "entry_block", "event_id"], kind="stable").reset_index(drop=True)
    design.insert(0, "event_index", np.arange(1, len(design) + 1, dtype=np.int64))
    cm.write_csv(design[["event_index", "clock", "specification", "event_id"]], cm.RESTRICTED / "private" / "event_crosswalk_design.csv")
    design = design.drop(columns=["event_id"])
    fig = figure_input(principal)
    paths = {"principal_household_means": cm.write_csv(principal, cm.INPUTS / "principal_household_means.csv"),
             "table3_household_means": cm.write_csv(t3, cm.INPUTS / "table3_household_means.csv"),
             "household_test_statistics": cm.write_csv(tests, cm.INPUTS / "household_test_statistics.csv"),
             "benchmark_figure": cm.write_csv(fig, cm.INPUTS / "benchmark_figure.csv"),
             "covariance_regression_design": cm.write_csv(design, cm.RESTRICTED / "inputs" / "covariance_regression_design.csv")}
    meta["outputs"] = {k: {**cm.file_entry(p, rows=len(pd.read_csv(p))), "restricted": "restricted" in str(p)} for k, p in paths.items()}
    meta["inputs"] = [cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / f"events_{c}.parquet") for c in CLOCKS] + [
        cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "event_duplicate_flags.parquet"),
        cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache.parquet"),
        cm.file_entry(evf.POOLED_CSV), cm.file_entry(cm.FROZEN_RESULTS), cm.file_entry(cm.FROZEN_DECOMP)]
    meta["covariance_regression_design_schema"] = {
        "location": "restricted (purchase level); not part of a transferable package", "columns": list(design.columns),
        "event_weight": "w_h / n_hB", "x_A_bp": "period-A household mean residualized on [1, log(1+n_A)] with weights w_h (repeated on period-B purchases)",
        "design": "WLS of y_B on [1, x_A, log(1+n_A)] with event weights; entry_block = calendar position of the entry session // 20"}
    meta["seconds"] = round(time.time() - t0, 1)
    cm.write_json(meta, cm.INPUTS / "exports_manifest.json")
    log(f"exports written in {meta['seconds']} s")


if __name__ == "__main__":
    main()
