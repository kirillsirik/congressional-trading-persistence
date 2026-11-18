"""Summary tables of the continuation analysis that the paper's exhibits read.

    python -m trading_on_disclosure_v4.writing_sample.tables

Reads the stored results (data/v4/writing_sample/results.csv, data/v4/results.csv, the stored
components under data/v4/writing_sample/computation_2026-11-11/ and the paired bootstrap of the
closeout) and writes data/v4/writing_sample/tables/: Table 3 (persistence under both benchmarks),
Table 4 Panel A (levels of the return components), Table F5 (robustness under both benchmarks),
the six-factor spreads of Table C1, the conclusion matrix, the continuation regressions, the
formation-date grid and the placebo levels.
No restricted input is read.

The percentage of shifted-date placebo estimates below the actual covariance in Table 3 comes from
the stored add-one p-value p of data/v4/results.csv: with 2,000 draws and no ties it is
(2000 - (2001 p - 1)) / 2000.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.writing_sample import common as cm

WS = cm.REPO_ROOT / "data" / "v4" / "writing_sample"
RES = WS / "results.csv"
TABLES = WS / "tables"
PLACEBO_DRAWS = 2000


class Results:
    def __init__(self, path: Path = RES):
        self.d = pd.read_csv(path)
        for c in ("sample", "benchmark", "weighting", "controls", "horizon", "term", "scale", "clock"):
            self.d[c] = self.d[c].fillna("").astype(str)
        self.k = self.d.set_index("result_key")["value"]

    def v(self, key: str) -> float:
        if key not in self.k.index:
            raise KeyError(key)
        return float(self.k[key])

    def q(self, **f) -> pd.DataFrame:
        m = np.ones(len(self.d), bool)
        for c, x in f.items():
            m &= (self.d[c] == x).to_numpy()
        return self.d[m]


def disp(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    s = f"{x:,.{nd}f}"
    return "0" if s in ("-0", "-0.0", "-0.00") and nd == 0 else s.replace("-0.00", "0.00") if s == "-0.00" else s


def interval(lo, hi, nd=0):
    return f"[{disp(lo, nd)}, {disp(hi, nd)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def placebo_shares() -> pd.DataFrame:
    """Share of the 2,000 shifted-date draws strictly below the actual covariance, from the stored p-values."""
    res = pd.read_csv(cm.REPO_ROOT / "data/v4/results.csv")
    rows = []
    for clock in ("T", "P"):
        for analysis, variant in (("tau", "event_share"), ("robustness", "no_alpha"), ("robustness", "market_adjusted")):
            p = float(res[(res.clock == clock) & (res.analysis == analysis) & (res.variant == variant)
                          & (res.statistic == "tau2_placebo_p_timing")].value.iloc[0])
            rows.append({"clock": clock, "spec": f"{analysis}/{variant}",
                         "share_strictly_below": (PLACEBO_DRAWS - round(p * (PLACEBO_DRAWS + 1) - 1)) / PLACEBO_DRAWS})
    return pd.DataFrame(rows)


def table3(r: Results) -> pd.DataFrame:
    f5 = pd.read_csv(cm.OUT / "robustness/table_F5_both_benchmarks_full_precision.csv")
    sl = pd.read_csv(cm.OUT / "inference/slopes_S2.csv")
    pct = placebo_shares()
    paired = pd.read_csv(cm.CLOSEOUT / "paired/paired_summary.csv")
    frozen = pd.read_csv(cm.REPO_ROOT / "data/v4/results.csv")
    rows = []
    for clock in ("T", "P"):
        b = f5[(f5.clock == clock) & (f5.row == "baseline")].iloc[0]
        ps = paired[(paired.clock == clock) & (paired.scale == "winsorized")].set_index("term")
        for bench, cov, se, term, spec in (("event_study", b.es_cov, b.es_se, "K11", "tau/event_share"), ("factor_only", b.fo_cov, b.fo_se, "K00", "robustness/no_alpha")):
            s = sl[(sl.clock == clock) & (sl.benchmark == bench)].iloc[0]
            p = pct[(pct.clock == clock) & (pct.spec == spec)].iloc[0]
            rows.append({"clock": clock, "benchmark": bench, "households": int(b.n_households), "purchases": int(b.n_events), "slope": s.slope, "slope_se_conditional_two_way": s.se_conditional_two_way,
                         "slope_ci_low": s.ci_low, "slope_ci_high": s.ci_high, "slope_p": s.p_two_sided, "slope_boot_low_within_draw": s.boot_slope_ratio_low, "slope_boot_high_within_draw": s.boot_slope_ratio_high,
                         "slope_cr2_ik_low": s.ci_cr2_ik_low, "slope_cr2_ik_high": s.ci_cr2_ik_high, "covariance": cov, "covariance_se_two_way": se, "boot_low": ps.loc[term, "ci_low"], "boot_high": ps.loc[term, "ci_high"],
                         "placebo_share_below_pct": 100 * p.share_strictly_below, "early_residual_variance": s.var_w_x_A_bp2})
        rows.append({"clock": clock, "benchmark": "event_study_minus_factor_only", "households": int(b.n_households), "purchases": int(b.n_events), "covariance": b.gap,
                     "covariance_se_two_way": b.gap_se_two_way, "covariance_se_paired_bootstrap": ps.loc["difference", "bootstrap_se"], "boot_low": ps.loc["difference", "ci_low"], "boot_high": ps.loc["difference", "ci_high"]})
        m = frozen[(frozen.clock == clock) & (frozen.analysis == "robustness") & (frozen.variant == "market_adjusted") & (frozen.statistic == "tau2")].iloc[0]
        p = pct[(pct.clock == clock) & (pct.spec == "robustness/market_adjusted")].iloc[0]
        rows.append({"clock": clock, "benchmark": "market_adjusted", "households": int(m.n_households), "purchases": int(m.n_events), "covariance": m.value, "covariance_se_two_way": m.se_block,
                     "boot_low": m.boot_lo, "boot_high": m.boot_hi, "placebo_share_below_pct": 100 * p.share_strictly_below})
    t = pd.DataFrame(rows)
    t["display_slope"] = t.slope.map(lambda x: disp(x, 2))
    t["display_slope_se"] = t.slope_se_conditional_two_way.map(lambda x: disp(x, 2))
    t["display_slope_ci"] = [interval(a, b, 2) for a, b in zip(t.slope_ci_low, t.slope_ci_high)]
    t["display_covariance"] = t.covariance.map(disp)
    t["display_se"] = [disp(a) for a in t.covariance_se_two_way]
    t["display_boot_interval"] = [interval(a, b) for a, b in zip(t.boot_low, t.boot_high)]
    t["display_placebo_pct"] = t.placebo_share_below_pct.map(lambda x: (f"{x:.2f}".rstrip("0").rstrip(".") if np.isfinite(x) else ""))
    return t


def table4(r: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    names = {"charge_bp": "Total alpha charge (Y0 - Y1)", "alpha20_bp": "Fitted-alpha component (20 x daily alpha)", "compounding_bp": "Compounding component of the charge ((Y0 - Y1) - 20 alpha)",
             "Y0_bp": "Factor-only return (Y0)", "Y1_bp": "Event-study return (Y1)"}
    for o in ("charge_bp", "alpha20_bp", "compounding_bp", "Y0_bp", "Y1_bp"):
        g = lambda s: r.v(f"A4/T/raw/outcome={o}/{s}")
        out.append({"quantity": names[o], "outcome": o, "mean_bp": g("estimate"), "se_two_way_bp": g("se"), "ci_low_bp": g("ci_low"), "ci_high_bp": g("ci_high"), "df": g("df"),
                    "p_zero": g("p_zero"), "variance_raw_bp2": g("variance_raw"), "purchases": g("observations"), "household_clusters": g("household_clusters"), "block_clusters": g("block_clusters"),
                    "display_mean": disp(g("estimate"), 1), "display_se": disp(g("se"), 1), "display_ci": interval(g("ci_low"), g("ci_high"), 1)})
    t = pd.DataFrame(out)
    t["note"] = ["", "", "equals minus the 'compounding remainder in the signed change' Y1 - Y0 + 20 alpha printed as -4.0 in the current Table 4", "", ""]
    cov = r.q(analysis="A4", statistic="joint_covariance").copy()
    cov["row"] = cov.term.str.extract(r"row=([^;]+)")[0]
    cov["col"] = cov.term.str.extract(r"col=([^;]+)")[0]
    return t, cov.pivot(index="row", columns="col", values="value")


def tableF5(r: Results) -> pd.DataFrame:
    f5 = pd.read_csv(cm.OUT / "robustness/table_F5_both_benchmarks_full_precision.csv")
    c14 = r.q(analysis="C14_duplicates", clock="T")
    t = f5.drop(columns=[c for c in f5.columns if c.startswith("_")]).copy()
    for p in ("es", "fo"):
        t[f"display_{p}_cov"] = t[f"{p}_cov"].map(disp)
        t[f"display_{p}_se"] = t[f"{p}_se"].map(disp)
        t[f"display_{p}_interval"] = [interval(a, b) for a, b in zip(t[f"{p}_boot_lo"], t[f"{p}_boot_hi"])]
    t["display_gap"] = t.gap.map(disp)
    t["display_gap_interval"] = [interval(a, b) for a, b in zip(t.gap_boot_lo, t.gap_boot_hi)]
    lo = c14[(c14.statistic == "boot_ci_low_baseline_selections") & c14.term.str.contains("benchmark=event_study;")]
    hi = c14[(c14.statistic == "boot_ci_high_baseline_selections") & c14.term.str.contains("benchmark=event_study;")]
    t["es_interval_shared_selections_C14"] = ""
    if len(lo) == 1:
        m = (t.clock == "T") & (t.row == "vendor_duplicates")
        t.loc[m, "es_interval_shared_selections_C14"] = interval(float(lo.value.iloc[0]), float(hi.value.iloc[0]))
    return t


def tableC1(r: Results) -> pd.DataFrame:
    s = pd.read_csv(cm.OUT / "benchmarks/six_factor_spread_both_benchmarks.csv").iloc[0]
    rows = []
    for lab, p in (("Six-factor event-study returns (current Table C1 'Six-factor-adjusted')", "es"), ("Six-factor factor-only returns (same 39,030 signals)", "fo")):
        rows.append({"specification": lab, "spread_bp": s[f"{p}_estimate"], "se_bp": s[f"{p}_se"], "ci_low_bp": s[f"{p}_ci_low"], "ci_high_bp": s[f"{p}_ci_high"], "p_value": s[f"{p}_p"],
                     "dof": s[f"{p}_dof"], "signals": int(s.n), "households": int(s.households), "blocks": int(s.blocks), "mean_purchases_bp": s[f"{p}_mean_purchases"], "mean_sales_bp": s[f"{p}_mean_sales"],
                     "display": f"{s[f'{p}_estimate']:.2f} & {s[f'{p}_se']:.2f} & [{s[f'{p}_ci_low']:.2f}, {s[f'{p}_ci_high']:.2f}] & {s[f'{p}_p']:.3f} & {int(s.n):,}"})
    rows.append({"specification": "Charge difference: mean (Y0 - Y1) purchases minus sales", "spread_bp": s.charge_difference, "mean_purchases_bp": s.charge_mean_purchases, "mean_sales_bp": s.charge_mean_sales,
                 "signals": int(s.n), "households": int(s.households), "blocks": int(s.blocks), "display": f"{s.charge_difference:.2f} (identity error {s.identity_gap:.1e} bp)"})
    return pd.DataFrame(rows)


def continuation(r: Results) -> pd.DataFrame:
    g = r.q(analysis="A1_purchases").copy()
    p = g.pivot_table(index=["term", "scale"], columns="statistic", values="value").reset_index()
    p["sample"] = p.term.str.extract(r"sample=([^;]+)")[0]
    p["subset"] = p.term.str.extract(r"subset=([^;]+)")[0]
    p["regression"] = p.term.str.extract(r"regression=([^;]+)")[0]
    cols = ["sample", "subset", "regression", "scale", "estimate", "se", "ci_low", "ci_high", "p_zero", "p_one", "df", "observations", "households", "blocks", "purchases_A", "purchases_B",
            "household_clusters", "block_clusters", "household_block_cells", "k_parameters", "factor_household", "factor_block", "factor_household_block", "variance_raw", "variance_valid",
            "fixed_effects", "singleton_blocks", "clipped_x", "clipped_Y0", "clipped_Y1", "excluded_missing_size", "event_study_slope", "remainder_slope", "identity_gap"]
    p = p[[c for c in cols if c in p.columns]].sort_values(["sample", "subset", "scale", "regression"])
    p["display"] = [f"{e:.3f} ({s:.3f}) [{lo:.3f}, {hi:.3f}]" for e, s, lo, hi in zip(p.estimate, p.se, p.ci_low, p.ci_high)]
    return p


def grid(r: Results) -> pd.DataFrame:
    g = r.q(analysis="A1_securities").copy()
    p = g.pivot_table(index=["term", "scale"], columns="statistic", values="value").reset_index()
    p["subset"] = p.term.str.extract(r"subset=([^;]+)")[0]
    p["display"] = [f"{e:.3f} ({s:.3f}) [{lo:.3f}, {hi:.3f}]" for e, s, lo, hi in zip(p.estimate, p.se, p.ci_low, p.ci_high)]
    return p


def placebo(r: Results) -> pd.DataFrame:
    g = r.q(analysis="placebo_levels")
    g = g[g.term.str.contains("_mean;") | g.term.str.contains("_mean$", regex=True)]
    p = g.pivot_table(index=["clock", "sample", "term"], columns="statistic", values="value").reset_index()
    p["quantity"] = p.term.str.split(";").str[0]
    return p


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    r = Results()
    cm.write_csv(table3(r), TABLES / "table3_persistence_both_benchmarks.csv")
    levels, covariance = table4(r)
    cm.write_csv(levels, TABLES / "table4_panelA_levels.csv")
    covariance.to_csv(TABLES / "table4_panelA_joint_covariance_bp2.csv", float_format="%.17g")
    cm.write_csv(pd.read_csv(cm.OUT / "inference/conclusion_matrix.csv"), TABLES / "S5_conclusion_matrix.csv")
    cm.write_csv(tableF5(r), TABLES / "tableF5_both_benchmarks.csv")
    cm.write_csv(tableC1(r), TABLES / "tableC1_six_factor_spread.csv")
    cm.write_csv(continuation(r), TABLES / "S1_continuation_purchases.csv")
    cm.write_csv(grid(r), TABLES / "S1_continuation_formation_grid.csv")
    cm.write_csv(placebo(r), TABLES / "S1_placebo_levels.csv")
    print(f"Wrote the summary tables to {TABLES.relative_to(cm.REPO_ROOT)}")


if __name__ == "__main__":
    main()
