"""Persistence slopes with their SEs, statistic-specific effective cluster counts for the
recurrence regression, recurrence and gap interval sensitivities, and the conclusion matrix.

python -m trading_on_disclosure_v4.writing_sample.inference_extras
Inputs: the Round 2 closeout's household means, paired selections, paired summary, leave-one-out vectors and
small-cluster file (read only); the restricted purchase panel for the recurrence regression's purchase-level
inference. Component `inference_extras`.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.round2 import small_cluster as scm
from trading_on_disclosure_v4.writing_sample import common as cm

sys.path.insert(0, str(cm.REPO_ROOT / "scripts" / "coherence"))
from aggregate_checks import jackknife, moments, partial_cov, validate_means  # noqa: E402

RUN = cm.CLOSEOUT
OUT = cm.OUT / "inference"
EXPECTED_G = {("T", "principal"): 8.1591283356, ("T", "intercept_omitted"): 4.1004482865, ("P", "principal"): 5.8277981726, ("P", "intercept_omitted"): 5.4702533103}
EXPECTED_GD = {"T": 4.6226214698, "P": 3.5021162788}
EXPECTED_SLOPE_SE = {("T", "principal"): 0.3749212774, ("T", "intercept_omitted"): 0.1949373647, ("P", "principal"): 0.2094914464, ("P", "intercept_omitted"): 0.2303788187}
BENCH = {"principal": ("event_study", "Y1"), "intercept_omitted": ("factor_only", "Y0")}


def residual(m, C, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(C * sw[:, None], m * sw, rcond=None)
    return m - C @ beta


def expanded_design(g: pd.DataFrame, early: np.ndarray):
    """[1, early residual, log(1+n_A)] repeated on n_B identical rows per household; event weight w / n_B."""
    n_a, n_b = g.n_A.to_numpy(float), g.n_B.to_numpy(float)
    w = np.sqrt(n_a * n_b)
    C = np.column_stack([np.ones(len(g)), np.log1p(n_a)])
    x = residual(early, C, w)
    reps = g.n_B.to_numpy(int)
    X = np.repeat(np.column_stack([np.ones(len(g)), x, np.log1p(n_a)]), reps, axis=0)
    v = np.repeat(w / n_b, reps)
    cl = np.repeat(g.household_index.to_numpy(), reps)
    return X, v, cl, x, w


def css(g: pd.DataFrame, early: np.ndarray) -> float:
    X, v, cl, _, _ = expanded_design(g, early)
    return float(scm.css_effective_clusters(X, v, cl, coef_index=1, rho=1.0, working="inverse_weights")["G_star"])


def t_int(est, se, df):
    if not (np.isfinite(se) and np.isfinite(df) and df > 0):
        return np.nan, np.nan
    c = stats.t.ppf(0.975, df)
    return est - c * se, est + c * se


def two_way(scores, h, b, k):
    n = len(scores)
    cl = pd.DataFrame({"h": h, "b": b, "s": scores})
    parts, G = [], []
    for cols in (["h"], ["b"], ["h", "b"]):
        s = cl.groupby(cols).s.sum()
        parts.append(len(s) / (len(s) - 1) * (n - 1) / (n - k) * float(np.sum(s.to_numpy() ** 2)))
        G.append(len(s))
    v = parts[0] + parts[1] - parts[2]
    return v, (float(np.sqrt(v)) if v > 0 else np.nan), min(G[:2]) - 1


def recurrence_regression(clock: str, g: pd.DataFrame, rows: cm.Rows, log) -> dict:
    """Purchase-level defining regression of the winsorized recurrence: late D_i on [1, early residualized D, log(1+n_A)]."""
    pp = pd.read_parquet(cm.RESTRICTED / f"purchase_panel_{clock}.parquet")
    coh = pp[pp["sample"] == "cohort"].reset_index(drop=True)
    per = coh["period"].to_numpy()
    y1, y0 = coh["Y1_bp"].to_numpy(float).copy(), coh["Y0_bp"].to_numpy(float).copy()
    c1, c0 = evf.pooled_cutoffs(clock, "ff5mom_h20"), evf.pooled_cutoffs(clock, "ff5mom_noalpha_h20")
    for p in ("A", "B"):
        m = per == p
        y1[m] = np.clip(y1[m], *c1[p])
        y0[m] = np.clip(y0[m], *c0[p])
    d = y1 - y0
    gg = g.sort_values("household_index").reset_index(drop=True)
    early = (gg.Y1_A - gg.Y0_A).to_numpy(float)
    n_a, n_b = gg.n_A.to_numpy(float), gg.n_B.to_numpy(float)
    w = np.sqrt(n_a * n_b)
    C = np.column_stack([np.ones(len(gg)), np.log1p(n_a)])
    x = residual(early, C, w)
    var_x = float(np.sum(w * x * x) / np.sum(w))
    hh = pd.Series(np.arange(len(gg)), index=gg.household_index.to_numpy())
    b = per == "B"
    code = hh.reindex(coh.loc[b, "household_index"].to_numpy()).to_numpy()
    X = np.column_stack([np.ones(b.sum()), x[code], np.log1p(n_a)[code]])
    v = (w / n_b)[code]
    y = d[b]
    bread = np.linalg.inv(X.T @ (v[:, None] * X))
    beta = bread @ (X.T @ (v * y))
    e = y - X @ beta
    kdd = float(beta[1] * var_x)
    scores = (X @ bread[:, 1]) * v * e * var_x
    var_tw, se_tw, df_tw = two_way(scores, coh.loc[b, "household_index"].to_numpy(), coh.loc[b, "entry_block"].to_numpy(), 3)
    cr2 = scm.cr2_inference(y, X, v, coh.loc[b, "household_index"].to_numpy(), coef_index=1, working="inverse_weights")
    rec = {"KDD_purchase_level": kdd, "slope": float(beta[1]), "var_w_x_D": var_x, "se_two_way": se_tw, "variance_two_way_raw": var_tw, "dof_two_way": df_tw,
           "se_cr2": cr2["se_cr2"] * var_x, "dof_ik": cr2["dof_ik"], "dof_bm": cr2["dof_bm"], "se_cr1_household": cr2["se_cr1_cgm"] * var_x, "n_purchases_B": int(b.sum())}
    return rec


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("inference_extras")
    rows = cm.Rows("data/v4/round2/closeout/closeout_2026-11-10/")
    means = pd.read_csv(RUN / "inputs/principal_household_means.csv")
    sc = pd.read_csv(RUN / "small_cluster/small_cluster_inference.csv")
    sc = sc[sc.working_model == "inverse_weights"]
    paired = pd.read_csv(RUN / "paired/paired_summary.csv")
    frozen = pd.read_csv(cm.REPO_ROOT / "data/v4/results.csv")
    s4 = pd.read_csv(cm.OUT / "robustness" / "table_F5_both_benchmarks_full_precision.csv")
    a5 = pd.read_csv(cm.COMPONENTS / "available.csv")
    gstar, slopes, matrix = [], [], []
    for clock in ("T", "P"):
        g = validate_means(means[(means.clock == clock) & (means.scale == "winsorized")])
        G = len(g)
        mult = pd.read_csv(RUN / f"paired/selection_multiplicities_{clock}.csv")
        if mult.columns[1:].tolist() != [f"hh_{i}" for i in g.household_index]:
            raise AssertionError("selection order")
        M = mult.iloc[:, 1:].to_numpy(float)
        w = np.sqrt(g.n_A.to_numpy(float) * g.n_B.to_numpy(float))
        c = np.log1p(g.n_A.to_numpy(float))
        # S2: slopes, conditional two-way, CR2 and within-draw bootstrap ratio
        for spec, (bench, k) in BENCH.items():
            r = sc[(sc.clock == clock) & (sc.specification == spec)].iloc[0]
            slope = r.covariance_bp2 / r.var_w_x_A_bp2
            se = r.se_cov_original_two_way_bp2 / r.var_w_x_A_bp2
            if not np.isclose(se, EXPECTED_SLOPE_SE[(clock, spec)], atol=5e-11):
                raise AssertionError(f"{clock}/{spec} slope SE {se}")
            df = G - 1
            lo, hi = t_int(slope, se, df)
            a, bb_ = g[f"{k}_A"].to_numpy(float), g[f"{k}_B"].to_numpy(float)
            K = partial_cov(a, bb_, c, M * w)
            V = partial_cov(a, a, c, M * w)
            ratio = K / V
            q = np.quantile(ratio[np.isfinite(ratio)], [0.025, 0.975])
            ci_cr2 = (r.ci_slope_low_ik, r.ci_slope_high_ik)
            rec = {"clock": clock, "benchmark": bench, "slope": slope, "slope_check": float(r.slope), "se_conditional_two_way": se, "dof": df, "ci_low": lo, "ci_high": hi,
                   "p_two_sided": 2 * stats.t.sf(abs(slope / se), df), "var_w_x_A_bp2": r.var_w_x_A_bp2, "covariance_bp2": r.covariance_bp2,
                   "se_cov_two_way_bp2": r.se_cov_original_two_way_bp2, "se_slope_cr2": r.se_slope_cr2, "dof_ik": r.dof_ik, "ci_cr2_ik_low": ci_cr2[0], "ci_cr2_ik_high": ci_cr2[1],
                   "boot_slope_ratio_low": q[0], "boot_slope_ratio_high": q[1], "boot_valid": int(np.isfinite(ratio).sum())}
            slopes.append(rec)
            base = dict(clock=clock, scale="winsorized", sample="persistence_cohort", benchmark=bench, weighting="sqrt(n_A n_B)", controls="log(1+n_A), both periods",
                        source="small_cluster/small_cluster_inference.csv (inverse_weights); paired/selection_multiplicities")
            rows.add("S2_slopes", "slope", slope, "slope", term="persistence_slope", inference="covariance / period-A weighted residual variance", **base)
            rows.add("S2_slopes", "se_conditional_two_way", se, "slope", term="persistence_slope", inference="se_cov_original_two_way_bp2 / var_w_x_A_bp2: household x 20-session block CGM, early residual variance held fixed", **base)
            rows.add("S2_slopes", "ci_low_conditional_two_way", lo, "slope", term="persistence_slope", inference=f"t({df}) conditional two-way", **base)
            rows.add("S2_slopes", "ci_high_conditional_two_way", hi, "slope", term="persistence_slope", inference=f"t({df}) conditional two-way", **base)
            rows.add("S2_slopes", "p_two_sided_conditional_two_way", rec["p_two_sided"], "p", term="persistence_slope", inference=f"t({df})", **base)
            rows.add("S2_slopes", "dof", df, "count", term="persistence_slope", **base)
            rows.add("S2_slopes", "var_w_x_A", r.var_w_x_A_bp2, "bp2", term="persistence_slope", inference="early residual variance (fixed in the conditional SE)", **base)
            rows.add("S2_slopes", "covariance", r.covariance_bp2, "bp2", term="persistence_slope", **base)
            rows.add("S2_slopes", "se_covariance_two_way", r.se_cov_original_two_way_bp2, "bp2", term="persistence_slope", inference="original household x block CGM", **base)
            rows.add("S2_slopes", "se_slope_cr2_household", r.se_slope_cr2, "slope", term="persistence_slope", inference="CR2 (household), small_cluster.cr2_inference", **base)
            rows.add("S2_slopes", "dof_ik", r.dof_ik, "count", term="persistence_slope", **base)
            rows.add("S2_slopes", "ci_low_cr2_ik", ci_cr2[0], "slope", term="persistence_slope", inference="CR2 with Imbens-Kolesar dof (household only)", **base)
            rows.add("S2_slopes", "ci_high_cr2_ik", ci_cr2[1], "slope", term="persistence_slope", inference="CR2 with Imbens-Kolesar dof (household only)", **base)
            rows.add("S2_slopes", "boot_ci_low_slope_within_draw", q[0], "slope", term="persistence_slope", inference="paired household bootstrap: slope = covariance / early variance computed within each of 4,999 draws (activity refit)", **base)
            rows.add("S2_slopes", "boot_ci_high_slope_within_draw", q[1], "slope", term="persistence_slope", inference="paired household bootstrap percentile of the within-draw ratio", **base)
            # S5: CSS counts of the endpoint regressions
            gs = css(g, g[f"{k}_A"].to_numpy(float))
            if not np.isclose(gs, EXPECTED_G[(clock, spec)], atol=5e-10):
                raise AssertionError(f"{clock}/{spec}: G* {gs} does not reproduce {EXPECTED_G[(clock, spec)]}")
            gstar.append({"clock": clock, "statistic": bench, "G_star": gs, "expected": EXPECTED_G[(clock, spec)], "stored": float(r.G_star_rho1)})
            rows.add("S5_effective_counts", "G_star", gs, "clusters", clock=clock, scale="winsorized", term=f"{bench}_slope", sample="persistence_cohort", benchmark=bench,
                     inference="CSS effective clusters, working model inverse_weights, rho = 1, defining regression of the endpoint covariance", source="principal_household_means.csv (expanded design)")
        # S5: recurrence regression
        early_d = (g.Y1_A - g.Y0_A).to_numpy(float)
        gd = css(g, early_d)
        if not np.isclose(gd, EXPECTED_GD[clock], atol=5e-10):
            raise AssertionError(f"{clock}: recurrence G* {gd} vs {EXPECTED_GD[clock]}")
        gr = css(validate_means(means[(means.clock == clock) & (means.scale == "raw")]), (lambda r: (r.Y1_A - r.Y0_A).to_numpy(float))(validate_means(means[(means.clock == clock) & (means.scale == "raw")])))
        gstar.append({"clock": clock, "statistic": "recurrence_KDD", "G_star": gd, "expected": EXPECTED_GD[clock], "stored": np.nan})
        gstar.append({"clock": clock, "statistic": "recurrence_KDD_raw", "G_star": gr, "expected": np.nan, "stored": np.nan})
        for lab, v in (("recurrence_KDD", gd), ("recurrence_KDD_raw", gr)):
            rows.add("S5_effective_counts", "G_star", v, "clusters", clock=clock, scale="raw" if lab.endswith("raw") else "winsorized", term=lab, sample="persistence_cohort", benchmark="recurrence",
                     inference="CSS effective clusters (inverse_weights, rho = 1) of the defining recurrence regression: late D on [1, early activity-residualized D, log(1+n_A)]; D = difference of separately winsorized endpoints",
                     source="principal_household_means.csv (expanded design)")
        reg = recurrence_regression(clock, g, rows, log)
        # recurrence and gap intervals
        pt = moments(g)[0]
        point, deleted, cov, se_j, lo_j, hi_j = jackknife(g)
        a5c = a5[(a5.analysis == "A5") & (a5.clock == clock) & (a5.scale == "winsorized")]
        kdd, se_kdd = point[2], se_j[2]
        gap, se_gap = point[5], se_j[5]
        if not (np.isclose(se_kdd, float(a5c[(a5c.term == "KDD") & (a5c.statistic == "se_jackknife")].value.iloc[0]), rtol=1e-9)):
            raise AssertionError("jackknife SE does not reproduce A5")
        ges = gstar[-5]["G_star"] if False else [x for x in gstar if x["clock"] == clock and x["statistic"] == "event_study"][0]["G_star"]
        gfo = [x for x in gstar if x["clock"] == clock and x["statistic"] == "factor_only"][0]["G_star"]
        ps = paired[(paired.clock == clock) & (paired.scale == "winsorized")].set_index("term")
        f5 = s4[(s4.clock == clock) & (s4.row == "baseline")].iloc[0]
        se_es = float(frozen[(frozen.clock == clock) & (frozen.analysis == "tau") & (frozen.variant == "event_share") & (frozen.statistic == "tau2")].se_block.iloc[0])
        se_fo = float(frozen[(frozen.clock == clock) & (frozen.analysis == "robustness") & (frozen.variant == "no_alpha") & (frozen.statistic == "tau2")].se_block.iloc[0])
        intervals = []
        def add(stat, method, est, lo, hi, df=np.nan, note=""):
            intervals.append({"clock": clock, "statistic": stat, "method": method, "estimate": est, "ci_low": lo, "ci_high": hi, "df": df,
                              "includes_zero": bool(lo <= 0 <= hi) if np.isfinite(lo) and np.isfinite(hi) else np.nan, "note": note})
        add("event_study_covariance", "conditional two-way CGM, t(G-1)", pt[0], *t_int(pt[0], se_es, G - 1), G - 1, "results.csv tau/event_share se_block")
        add("event_study_covariance", "paired household bootstrap percentile", pt[0], ps.loc["K11", "ci_low"], ps.loc["K11", "ci_high"], note="4,999 common selections")
        add("event_study_covariance", "CR2 IK (household)", pt[0], float(sc[(sc.clock == clock) & (sc.specification == "principal")].ci_cov_low_ik_bp2.iloc[0]),
            float(sc[(sc.clock == clock) & (sc.specification == "principal")].ci_cov_high_ik_bp2.iloc[0]), float(sc[(sc.clock == clock) & (sc.specification == "principal")].dof_ik.iloc[0]))
        add("factor_only_covariance", "conditional two-way CGM, t(G-1)", pt[1], *t_int(pt[1], se_fo, G - 1), G - 1, "results.csv robustness/no_alpha se_block")
        add("factor_only_covariance", "paired household bootstrap percentile", pt[1], ps.loc["K00", "ci_low"], ps.loc["K00", "ci_high"], note="4,999 common selections")
        add("factor_only_covariance", "CR2 IK (household)", pt[1], float(sc[(sc.clock == clock) & (sc.specification == "intercept_omitted")].ci_cov_low_ik_bp2.iloc[0]),
            float(sc[(sc.clock == clock) & (sc.specification == "intercept_omitted")].ci_cov_high_ik_bp2.iloc[0]), float(sc[(sc.clock == clock) & (sc.specification == "intercept_omitted")].dof_ik.iloc[0]))
        add("gap", "paired household bootstrap percentile", gap, ps.loc["difference", "ci_low"], ps.loc["difference", "ci_high"], note="gap formed within each common draw")
        add("gap", "conditional two-way CGM on score difference, t(G-1)", gap, *t_int(gap, float(f5.gap_se_two_way), G - 1), G - 1, "robustness module; period-A variances fixed")
        for lab, df in (("jackknife t(G-1)", G - 1), ("jackknife t(G*_event_study - 1) heuristic", ges - 1), ("jackknife t(G*_factor_only - 1) heuristic", gfo - 1),
                        ("jackknife t(min endpoint G* - 1) heuristic", min(ges, gfo) - 1)):
            add("gap", lab, gap, *t_int(gap, se_gap, df), df, "heuristic critical-value sensitivity; not calibrated for the nonlinear gap" if "heuristic" in lab else "")
        add("recurrence_KDD", "paired household bootstrap percentile", kdd, ps.loc["KDD", "ci_low"], ps.loc["KDD", "ci_high"])
        for lab, df in (("jackknife t(G-1)", G - 1), ("jackknife t(G*_recurrence - 1)", gd - 1), ("jackknife t(min endpoint G* - 1) sensitivity", min(ges, gfo) - 1)):
            add("recurrence_KDD", lab, kdd, *t_int(kdd, se_kdd, df), df)
        add("recurrence_KDD", "purchase-level two-way CGM (defining regression), t(G-1)", reg["KDD_purchase_level"], *t_int(reg["KDD_purchase_level"], reg["se_two_way"], reg["dof_two_way"]), reg["dof_two_way"],
            "conditional on the early residualized D; household x 20-session block")
        add("recurrence_KDD", "purchase-level CR2 household, IK dof (defining regression)", reg["KDD_purchase_level"], *t_int(reg["KDD_purchase_level"], reg["se_cr2"], reg["dof_ik"]), reg["dof_ik"],
            "bias-reduced household clustering; ignores common calendar shocks")
        if not np.isclose(reg["KDD_purchase_level"], kdd, rtol=1e-9):
            raise AssertionError(f"{clock}: purchase-level KDD {reg['KDD_purchase_level']} vs household {kdd}")
        for r_ in intervals:
            matrix.append(r_)
            base = dict(clock=clock, scale="winsorized", sample="persistence_cohort", benchmark=r_["statistic"], weighting="sqrt(n_A n_B)", controls="log(1+n_A), both periods",
                        source="round2 closeout paired/, small_cluster/; writing_sample robustness and inference modules")
            t = f"{r_['statistic']}/{r_['method']}"
            rows.add("S5_intervals", "ci_low", r_["ci_low"], "bp2", term=t, inference=r_["method"] + (f"; {r_['note']}" if r_["note"] else ""), **base)
            rows.add("S5_intervals", "ci_high", r_["ci_high"], "bp2", term=t, inference=r_["method"], **base)
            rows.add("S5_intervals", "df", r_["df"], "count", term=t, inference=r_["method"], **base)
            rows.add("S5_intervals", "includes_zero", float(r_["includes_zero"]) if r_["includes_zero"] == r_["includes_zero"] else np.nan, "indicator", term=t, inference=r_["method"], **base)
        for k_, v_ in reg.items():
            rows.add("S5_recurrence_regression", k_, v_, "count" if k_.startswith(("dof", "n_")) else ("slope" if k_ == "slope" else "bp2"), clock=clock, scale="winsorized",
                     term="recurrence_defining_regression", sample="persistence_cohort", benchmark="recurrence", source=f"data/v4/restricted/writing_sample/purchase_panel_{clock}.parquet")
        log(f"{clock}: G* ES {ges:.10f} FO {gfo:.10f} recurrence {gd:.10f}; KDD {kdd:.6f} jackknife SE {se_kdd:.6f}; own-G* interval {t_int(kdd, se_kdd, gd - 1)}")
    cm.write_csv(pd.DataFrame(gstar), OUT / "effective_counts.csv")
    cm.write_csv(pd.DataFrame(slopes), OUT / "slopes_S2.csv")
    cm.write_csv(pd.DataFrame(matrix), OUT / "conclusion_matrix.csv")
    s4c = s4.copy()
    stab = []
    for clock, g in s4c.groupby("clock"):
        core = g[~g.row.isin(["sales", "signed"])]
        stab.append({"clock": clock, "rows": len(core), "event_study_positive": int((core.es_cov > 0).sum()), "factor_only_positive": int((core.fo_cov > 0).sum()),
                     "gap_positive": int((core.gap > 0).sum()), "event_study_boot_excludes_zero": int(((core.es_boot_lo > 0) | (core.es_boot_hi < 0)).sum()),
                     "factor_only_boot_excludes_zero": int(((core.fo_boot_lo > 0) | (core.fo_boot_hi < 0)).sum()), "gap_boot_excludes_zero": int(((core.gap_boot_lo > 0) | (core.gap_boot_hi < 0)).sum())})
    stab = pd.DataFrame(stab)
    cm.write_csv(stab, OUT / "sign_stability_purchase_rows.csv")
    for _, r in stab.iterrows():
        for k_, v_ in r.items():
            if k_ != "clock":
                rows.add("S5_sign_stability", k_, v_, "count", clock=r.clock, scale="winsorized", term="purchase_rows_of_F5",
                         inference="counts over the 13 purchase rows of Table F5 (sales and signed excluded): sign stability of point estimates, separate from significance",
                         source=f"data/v4/writing_sample/{cm.RUN_ID}/robustness/table_F5_both_benchmarks_full_precision.csv")
    rows.write_component("inference_extras")
    log("done")


if __name__ == "__main__":
    main()
