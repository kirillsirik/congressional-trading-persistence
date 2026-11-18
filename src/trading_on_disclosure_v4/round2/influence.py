"""Tasks C2 and C3: signs of the household rejections; slopes, weighted correlations, full leave-one-household-out
re-estimation and additive full-fit contributions for every Table 3 specification and the principal benchmark pair.

python -m trading_on_disclosure_v4.round2.influence
Reads <run>/inputs/{table3_household_means,principal_household_means,household_test_statistics}.csv and writes
<run>/influence/ and <run>/rejections/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.round2 import common as cm
from trading_on_disclosure_v4.round2 import exports as ex

OUT = cm.RUN_DIR / "influence"
OUT_REJ = cm.RUN_DIR / "rejections"
TOL = 1e-9


def residuals(m: np.ndarray, n_a: np.ndarray, w: np.ndarray, activity: bool) -> np.ndarray:
    C = tau_mod.activity_design(n_a) if activity else np.ones((len(n_a), 1))
    return ex.weighted_projection_residual(np.asarray(m, float), C, np.asarray(w, float))


def statistics(m_a, m_b, n_a, w, activity: bool) -> dict:
    """kappa = <a,b>_w, slope = sum w a b / sum w a^2, rho with both periods residualized on the actual controls."""
    a, b = residuals(m_a, n_a, w, activity), residuals(m_b, n_a, w, activity)
    sab, saa, sbb, wp = float(np.sum(w * a * b)), float(np.sum(w * a * a)), float(np.sum(w * b * b)), float(np.sum(w))
    C = tau_mod.activity_design(n_a) if activity else np.ones((len(n_a), 1))
    orth = float(max(np.max(np.abs(C.T @ (w * a))), np.max(np.abs(C.T @ (w * b)))) / max(1.0, np.sqrt(saa * sbb)))
    rank = int(np.linalg.matrix_rank(C * np.sqrt(w)[:, None]))
    return {"kappa": sab / wp, "slope": sab / saa if saa > 0 else np.nan, "corr": sab / np.sqrt(saa * sbb) if saa > 0 and sbb > 0 else np.nan,
            "var_a": saa / wp, "var_b": sbb / wp, "orthogonality": orth, "control_rank": rank, "n_households": int(len(w)),
            "degenerate": bool(saa <= 0 or sbb <= 0 or rank < C.shape[1]), "a": a, "b": b}


def leave_one_out(m_a, m_b, n_a, w, activity: bool, keys) -> pd.DataFrame:
    rows = []
    wshare = np.asarray(w, float) / float(np.sum(w))
    for j in range(len(w)):
        keep = np.arange(len(w)) != j
        s = statistics(np.asarray(m_a)[keep], np.asarray(m_b)[keep], np.asarray(n_a)[keep], np.asarray(w)[keep], activity)
        rows.append({"omitted_household_index": int(keys[j]), "omitted_weight_share": float(wshare[j]), "kappa_minus_h": s["kappa"],
                     "slope_minus_h": s["slope"], "corr_minus_h": s["corr"], "degenerate": s["degenerate"]})
    return pd.DataFrame(rows)


def summarize(label: dict, full: dict, loo: pd.DataFrame, contrib: np.ndarray) -> dict:
    ch = loo["kappa_minus_h"] - full["kappa"]
    i = int(ch.abs().idxmax())
    sign_changes = int((np.sign(loo["kappa_minus_h"]) != np.sign(full["kappa"])).sum())
    return {**label, "kappa_bp2": full["kappa"], "slope": full["slope"], "weighted_corr": full["corr"], "var_a_bp2": full["var_a"], "var_b_bp2": full["var_b"],
            "n_households": full["n_households"], "control_rank": full["control_rank"], "max_orthogonality_residual": full["orthogonality"],
            "loo_kappa_min_bp2": float(loo["kappa_minus_h"].min()), "loo_kappa_max_bp2": float(loo["kappa_minus_h"].max()),
            "loo_slope_min": float(loo["slope_minus_h"].min()), "loo_slope_max": float(loo["slope_minus_h"].max()),
            "loo_corr_min": float(loo["corr_minus_h"].min()), "loo_corr_max": float(loo["corr_minus_h"].max()),
            "loo_n_sign_changes": sign_changes, "loo_max_abs_change_bp2": float(ch.abs().max()),
            "loo_max_abs_change_signed_bp2": float(ch.iloc[i]), "loo_max_abs_change_household_index": int(loo["omitted_household_index"].iloc[i]),
            "loo_max_abs_change_weight_share": float(loo["omitted_weight_share"].iloc[i]),
            "loo_n_degenerate": int(loo["degenerate"].sum()), "contributions_sum_bp2": float(contrib.sum()),
            "contributions_sum_gap_bp2": float(contrib.sum() - full["kappa"]),
            "largest_contribution_bp2": float(contrib[np.argmax(np.abs(contrib))])}


def table3_rows(t3: pd.DataFrame, log) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    summ, loos, contribs = [], [], []
    for (clock, spec), g in t3.groupby(["clock", "specification"], sort=False):
        g = g.sort_values("household_index").reset_index(drop=True)
        activity = spec != "no_activity_adjustment"
        args = (g["m_A_bp"].to_numpy(float), g["m_B_bp"].to_numpy(float), g["n_A"].to_numpy(float), g["weight"].to_numpy(float))
        full = statistics(*args, activity)
        _, analysis, variant = ex.TABLE3[spec][0], ex.TABLE3[spec][1], ex.TABLE3[spec][2]
        ex.check(full["kappa"], cm.frozen_value(clock, analysis, variant, "tau2"), f"{clock} {spec} kappa", log)
        loo = leave_one_out(*args, activity, g["household_index"].to_numpy())
        w = args[3]
        contrib = (w / w.sum()) * full["a"] * full["b"]
        label = {"clock": clock, "specification": spec, "scale": "winsorized", "benchmark": ex.TABLE3[spec][3],
                 "control_design": "[1, log(1+n_A)] both periods" if activity else "[1] both periods (no activity adjustment)"}
        summ.append(summarize(label, full, loo, contrib))
        loos.append(loo.assign(**{k: label[k] for k in ("clock", "specification", "scale")}, change_bp2=loo["kappa_minus_h"] - full["kappa"]))
        contribs.append(pd.DataFrame({"clock": clock, "specification": spec, "scale": "winsorized", "household_index": g["household_index"],
                                      "weight_share": w / w.sum(), "residual_A_bp": full["a"], "residual_B_bp": full["b"], "contribution_bp2": contrib}))
    return summ, loos, contribs


def benchmark_rows(pm: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    """Both principal benchmarks and their difference, raw and winsorized, at both return dates."""
    summ, loos, contribs = [], [], []
    for (clock, scale), g in pm.groupby(["clock", "scale"], sort=False):
        g = g.sort_values("household_index").reset_index(drop=True)
        n_a, n_b = g["n_A"].to_numpy(float), g["n_B"].to_numpy(float)
        w = np.sqrt(n_a * n_b)
        keys = g["household_index"].to_numpy()
        f1 = statistics(g["Y1_A"], g["Y1_B"], n_a, w, True)
        f0 = statistics(g["Y0_A"], g["Y0_B"], n_a, w, True)
        l1 = leave_one_out(g["Y1_A"], g["Y1_B"], n_a, w, True, keys)
        l0 = leave_one_out(g["Y0_A"], g["Y0_B"], n_a, w, True, keys)
        c1 = (w / w.sum()) * f1["a"] * f1["b"]
        c0 = (w / w.sum()) * f0["a"] * f0["b"]
        for bench, full, loo, c in (("fitted_intercept", f1, l1, c1), ("intercept_omitted", f0, l0, c0)):
            label = {"clock": clock, "specification": f"principal_{bench}", "scale": scale, "benchmark": bench, "control_design": "[1, log(1+n_A)] both periods"}
            summ.append(summarize(label, full, loo, c))
            loos.append(loo.assign(clock=clock, specification=f"principal_{bench}", scale=scale, change_bp2=loo["kappa_minus_h"] - full["kappa"]))
            contribs.append(pd.DataFrame({"clock": clock, "specification": f"principal_{bench}", "scale": scale, "household_index": keys,
                                          "weight_share": w / w.sum(), "residual_A_bp": full["a"], "residual_B_bp": full["b"], "contribution_bp2": c}))
        d_full = {"kappa": f1["kappa"] - f0["kappa"], "slope": np.nan, "corr": np.nan, "var_a": np.nan, "var_b": np.nan, "n_households": len(w),
                  "control_rank": f1["control_rank"], "orthogonality": max(f1["orthogonality"], f0["orthogonality"])}
        d_loo = pd.DataFrame({"omitted_household_index": keys, "omitted_weight_share": w / w.sum(),
                              "kappa_minus_h": l1["kappa_minus_h"] - l0["kappa_minus_h"], "slope_minus_h": np.nan, "corr_minus_h": np.nan,
                              "degenerate": l1["degenerate"] | l0["degenerate"]})
        label = {"clock": clock, "specification": "principal_difference", "scale": scale, "benchmark": "fitted_intercept_minus_intercept_omitted",
                 "control_design": "[1, log(1+n_A)] both periods"}
        summ.append(summarize(label, d_full, d_loo, c1 - c0))
        loos.append(d_loo.assign(clock=clock, specification="principal_difference", scale=scale, change_bp2=d_loo["kappa_minus_h"] - d_full["kappa"]))
        contribs.append(pd.DataFrame({"clock": clock, "specification": "principal_difference", "scale": scale, "household_index": keys,
                                      "weight_share": w / w.sum(), "residual_A_bp": np.nan, "residual_B_bp": np.nan, "contribution_bp2": c1 - c0}))
    return summ, loos, contribs


def validate_against_original(ctx, loos: pd.DataFrame, key: pd.Series, log) -> pd.DataFrame:
    """Every household-level deletion against the original event-level estimator (estimate_tau2 on the panel without
    that household, with each specification's own flags), and the principal range against results.csv."""
    inv = pd.Series(key.index.to_numpy(), index=key.to_numpy())
    rows = []
    for clock in ("T", "P"):
        specs = sp.analysis_specs(clock)
        for label, (name, analysis, variant, _, _) in ex.TABLE3.items():
            spec = specs[name]
            panel = ctx.panel(spec)
            mine = loos[(loos["clock"] == clock) & (loos["specification"] == label) & (loos["scale"] == "winsorized")]
            gaps = []
            for r in mine.itertuples(index=False):
                hid = inv[int(r.omitted_household_index)]
                o = tau_mod.estimate_tau2(panel.loc[panel["household_id"] != hid], spec.weights, activity_control=spec.activity_control,
                                          residualize_activity=spec.residualize_activity)
                gaps.append(abs(o.tau2_hat - r.kappa_minus_h))
            rows.append({"clock": clock, "specification": label, "original_function": "estimators.tau.estimate_tau2 (event-level WLS, own flags)",
                         "residualize_activity": spec.residualize_activity, "activity_control": spec.activity_control,
                         "n_deletions": len(gaps), "max_abs_diff_bp2": float(max(gaps))})
            if label == "principal":
                lo, hi = cm.frozen_value(clock, "influence", "leave_one_household_out", "tau2_min"), cm.frozen_value(clock, "influence", "leave_one_household_out", "tau2_max")
                ex.check(float(mine["kappa_minus_h"].min()), lo, f"{clock} principal LOO min vs results.csv", log)
                ex.check(float(mine["kappa_minus_h"].max()), hi, f"{clock} principal LOO max vs results.csv", log)
                orig = tau_mod.leave_one_household_out(panel, "ES")
                rows.append({"clock": clock, "specification": "principal", "original_function": "estimators.tau.leave_one_household_out",
                             "residualize_activity": True, "activity_control": True, "n_deletions": int(len(orig)),
                             "max_abs_diff_bp2": float(np.max(np.abs(np.sort(orig["tau2_hat"].to_numpy()) - np.sort(mine["kappa_minus_h"].to_numpy()))))})
    return pd.DataFrame(rows)


def engine_loo_check(loos: pd.DataFrame) -> pd.DataFrame:
    """The paired engine's leave-one-out output against this module's, for the two benchmarks and the difference."""
    eng = pd.read_csv(cm.RUN_DIR / "paired" / "leave_one_household_out.csv")
    rows = []
    for (clock, scale), e in eng.groupby(["clock", "scale"]):
        for term, spec in (("K11", "principal_fitted_intercept"), ("K00", "principal_intercept_omitted"), ("difference", "principal_difference")):
            m = loos[(loos["clock"] == clock) & (loos["scale"] == scale) & (loos["specification"] == spec)].set_index("omitted_household_index")["kappa_minus_h"]
            x = e.set_index("omitted_household_index")[term]
            rows.append({"clock": clock, "scale": scale, "term": term, "n": int(len(x)), "max_abs_diff_bp2": float((x - m.reindex(x.index)).abs().max())})
    return pd.DataFrame(rows)


def rejection_signs(tests: pd.DataFrame, rank_sets_path=cm.REPO_ROOT / "data" / "v4" / "results_rank_sets.csv") -> pd.DataFrame:
    rows = []
    ranks = pd.read_csv(rank_sets_path)
    for clock, g in tests.groupby("clock", sort=False):
        rej = g["p_two_sided"] < 0.05
        pos, neg = int((rej & (g["t"] > 0)).sum()), int((rej & (g["t"] < 0)).sum())
        imin = int(g["t"].idxmin())
        rr = ranks[ranks["clock"] == clock]
        rows.append({"clock": clock, "n_households": int(len(g)), "n_rejections": int(rej.sum()), "n_positive_rejections": pos,
                     "n_negative_rejections": neg, "rejections_add_up": bool(pos + neg == int(rej.sum())), "min_t": float(g.loc[imin, "t"]),
                     "min_t_household_index": int(g.loc[imin, "household_index"]), "min_t_n": int(g.loc[imin, "n"]), "min_t_mean_bp": float(g.loc[imin, "mean_bp"]),
                     "max_t": float(g["t"].max()), "n_negative_means": int((g["mean_bp"] < 0).sum()), "n_positive_means": int((g["mean_bp"] > 0).sum()),
                     "n_negative_means_rank_sets_file": int((rr["estimate"] < 0).sum()), "n_households_rank_sets_file": int(len(rr)),
                     "test": "two-sided Student t, dof n-1, SE = SD/sqrt(n) with ddof=1, reject when p < 0.05; winsorized outcomes, >= 20 purchases"})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tasks C2 and C3")
    ap.parse_args(argv)
    log = cm.logger("influence")
    t0 = time.time()
    t3 = pd.read_csv(cm.INPUTS / "table3_household_means.csv")
    pm = pd.read_csv(cm.INPUTS / "principal_household_means.csv")
    tests = pd.read_csv(cm.INPUTS / "household_test_statistics.csv")
    s1, l1, c1 = table3_rows(t3, log)
    s2, l2, c2 = benchmark_rows(pm)
    summary = pd.DataFrame(s1 + s2)
    loos = pd.concat(l1 + l2, ignore_index=True)
    contribs = pd.concat(c1 + c2, ignore_index=True)
    if (summary["contributions_sum_gap_bp2"].abs() > TOL * np.maximum(1.0, summary["kappa_bp2"].abs())).any():
        raise AssertionError("full-fit contributions do not sum to the covariance")
    finite_corr = summary["weighted_corr"].dropna()
    if ((finite_corr < -1 - 1e-12) | (finite_corr > 1 + 1e-12)).any():
        raise AssertionError("a weighted correlation lies outside [-1, 1]")
    ctx, _ = cm.load_context()
    key = pd.read_csv(cm.CROSSWALK).set_index("household_id")["household_index"]
    val = validate_against_original(ctx, loos, key, log)
    eng = engine_loo_check(loos)
    rej = rejection_signs(tests)
    for name, df in (("specification_statistics", summary), ("leave_one_out", loos), ("full_fit_contributions", contribs),
                     ("validation_original_estimator", val), ("validation_paired_engine_loo", eng)):
        cm.write_csv(df, OUT / f"{name}.csv")
    cm.write_csv(rej, OUT_REJ / "rejection_signs.csv")
    log(f"max |diff| vs original estimator {val['max_abs_diff_bp2'].max():.2e}; vs engine LOO {eng['max_abs_diff_bp2'].max():.2e}")
    log("rejections: " + "; ".join(f"{r.clock}: {r.n_rejections} = {r.n_positive_rejections} positive + {r.n_negative_rejections} negative, min t {r.min_t:.3f}"
                                   for r in rej.itertuples(index=False)))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
