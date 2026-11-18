"""The factor-only six-factor spread on the identical 39,030 signals, the charge identity, the
benchmark behind each detection and spread number, and the Figure G1 thresholds read from its source data.

python -m trading_on_disclosure_v4.writing_sample.spreads
Component `benchmark_labels`; tables under data/v4/writing_sample/computation_2026-11-11/benchmarks/.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.outcomes import panel as panel_mod
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import fits

OUT = cm.OUT / "benchmarks"
CAL = cm.REPO_ROOT / "data" / "v4" / "calibration"
TARGET = {"n": 39030, "households": 324, "blocks": 156, "estimate": -15.402254270544214, "se": 12.991685343425772, "dof": 155,
          "ci_low": -41.06586230866485, "ci_high": 10.261353767576413}
REACHED = {"interpolated", "at_grid_min"}
TEST_NAMES = {"T1_blk20_2s": "pooled-mean test (block-clustered)", "T2_I1cal_ES": "studentized covariance test", "T2_I2b_ES": "household-bootstrap covariance test",
              "T3": "maximum-household test", "T4_z2": "upper-tail test (z > 2)"}


def factor_only_spread(log, rows: cm.Rows) -> dict:
    cal = panel_mod.v2_trading_calendar()
    s = rerun.load_signals()
    d, _ = rerun.dedup_filter(s, rerun.duplicate_signal_uids())
    test = [t for t in rerun.TESTS if t.test_id == "TW_S2_FF5MOM"][0]
    es = rerun.run_test(d, test, "block20", cal)
    data = d.dropna(subset=[test.outcome, test.entry_col]).reset_index(drop=True)
    idx = cal.get_indexer(pd.DatetimeIndex(data[test.entry_col]))
    if (idx < 0).any():
        raise AssertionError("entry date off the calendar")
    ext = fits.load(["permno", "anchor_idx", "alpha_daily", "y1_h20", "y0_h20"])
    m = pd.DataFrame({"permno": data["permno"].astype(np.int64).to_numpy(), "anchor_idx": idx}).merge(ext, on=["permno", "anchor_idx"], how="left")
    y1 = data[test.outcome].to_numpy(float)
    diff = np.abs(y1 - m["y1_h20"].to_numpy(float))
    if not np.isfinite(diff).all() or diff.max() > 1e-12:
        raise AssertionError(f"car_ff5mom_h20 does not equal the forward cache's event-study outcome at the entry cell (max {np.nanmax(diff)})")
    y0 = m["y0_h20"].to_numpy(float)
    buy = (data["direction"] == "purchase").to_numpy(float)
    blk = rerun.block_of(data[test.entry_col], cal)
    hh = data[rerun.HOUSEHOLD].to_numpy()
    fo = rerun.clustered_mean_or_spread(y0, buy, hh, blk, f"{test.entry_col}_blk20")
    es2 = rerun.clustered_mean_or_spread(y1, buy, hh, blk, f"{test.entry_col}_blk20")
    charge = (y0 - y1) * cm.BP
    x = cm.H * m["alpha_daily"].to_numpy(float) * cm.BP
    ch_b, ch_s = float(charge[buy == 1].mean()), float(charge[buy == 0].mean())
    out = {"n": len(data), "households": int(pd.Series(hh).nunique()), "blocks": int(len(np.unique(blk))), "purchases": int(buy.sum()), "sales": int((1 - buy).sum()),
           "es_estimate": es["estimate"] * cm.BP, "es_se": es["se"] * cm.BP, "es_dof": es["dof"], "es_ci_low": es["ci_low"] * cm.BP, "es_ci_high": es["ci_high"] * cm.BP, "es_p": es["p"],
           "fo_estimate": fo["estimate"] * cm.BP, "fo_se": fo["se"] * cm.BP, "fo_dof": fo["dof"], "fo_ci_low": fo["ci_low"] * cm.BP, "fo_ci_high": fo["ci_high"] * cm.BP, "fo_p": fo["p"],
           "fo_psd": fo["psd_corrected"], "es_psd": es["psd_corrected"], "charge_mean_purchases": ch_b, "charge_mean_sales": ch_s, "charge_difference": ch_b - ch_s,
           "alpha20_mean_purchases": float(x[buy == 1].mean()), "alpha20_mean_sales": float(x[buy == 0].mean()),
           "es_mean_purchases": float(y1[buy == 1].mean() * cm.BP), "es_mean_sales": float(y1[buy == 0].mean() * cm.BP),
           "fo_mean_purchases": float(y0[buy == 1].mean() * cm.BP), "fo_mean_sales": float(y0[buy == 0].mean() * cm.BP),
           "max_abs_diff_car_vs_cache_decimal": float(diff.max())}
    out["identity_gap"] = (out["fo_estimate"] - out["es_estimate"]) - out["charge_difference"]
    for k, v in TARGET.items():
        mine = {"n": out["n"], "households": out["households"], "blocks": out["blocks"], "estimate": out["es_estimate"], "se": out["es_se"], "dof": out["es_dof"],
                "ci_low": out["es_ci_low"], "ci_high": out["es_ci_high"]}[k]
        if not np.isclose(mine, v, rtol=1e-10, atol=1e-9):
            raise AssertionError(f"TW_S2_FF5MOM {k}: {mine} vs {v}")
    if abs(out["identity_gap"]) > 1e-9:
        raise AssertionError(f"charge identity fails: {out['identity_gap']}")
    if not np.isclose(es2["estimate"] * cm.BP, out["es_estimate"], rtol=1e-12):
        raise AssertionError("event-study spread differs between run_test and the direct call")
    base = dict(clock="P", scale="raw", sample="TW_S2_FF5MOM_signals", weighting="equal signal", controls="purchase indicator",
                source="outputs/frames/signals.csv (existing.rerun: dedup, car_ff5mom_h20 and follower_entry_date present); factor-only outcome from the forward cache at the entry cell")
    for bench, pfx in (("event_study", "es"), ("factor_only", "fo")):
        for stat, unit in (("estimate", "bp"), ("se", "bp"), ("ci_low", "bp"), ("ci_high", "bp"), ("p", "p"), ("dof", "count")):
            rows.add("S6_spread", {"estimate": "spread"}.get(stat, stat), out[f"{pfx}_{stat}"], unit, benchmark=bench, term="purchase_minus_sale",
                     inference="household x 20-session block CGM (existing.rerun.clustered_mean_or_spread), t(min clusters - 1)", **base)
        rows.add("S6_spread", "mean_purchases", out[f"{pfx}_mean_purchases"], "bp", benchmark=bench, term="purchase_minus_sale", **base)
        rows.add("S6_spread", "mean_sales", out[f"{pfx}_mean_sales"], "bp", benchmark=bench, term="purchase_minus_sale", **base)
    for k in ("n", "households", "blocks", "purchases", "sales"):
        rows.add("S6_spread", k, out[k], "count", term="purchase_minus_sale", **base)
    for k in ("charge_mean_purchases", "charge_mean_sales", "charge_difference", "alpha20_mean_purchases", "alpha20_mean_sales", "identity_gap"):
        rows.add("S6_spread", k, out[k], "bp", term="charge", benchmark="factor_only_minus_event_study",
                 inference="factor-only spread - event-study spread = mean charge (purchases) - mean charge (sales); identity_gap is its numerical error", **base)
    log(f"six-factor spread on {out['n']} signals: event-study {out['es_estimate']:.10f} (SE {out['es_se']:.10f}); factor-only {out['fo_estimate']:.10f} "
        f"(SE {out['fo_se']:.10f}, CI [{out['fo_ci_low']:.4f}, {out['fo_ci_high']:.4f}]); charge difference {out['charge_difference']:.10f}; identity gap {out['identity_gap']:.2e}")
    return out


def figure_g1_thresholds(rows: cm.Rows, log) -> pd.DataFrame:
    """Measured 80%-power crossings per test and the smallest across tests at each measured share, trade date."""
    recs = []
    for panel, share in (("a", "share of households receiving the gain"), ("b", "probability that each purchase receives the gain")):
        t = pd.read_csv(CAL / f"figure2_detection_region_v3_{panel}_tests.csv")
        for frac, g in t.groupby("fraction", sort=True):
            ok = g[g["status"].isin(REACHED)].dropna(subset=["effect_bp_at_target"])
            if len(ok):
                best = ok.loc[ok["effect_bp_at_target"].idxmin()]
                recs.append({"panel": panel, "x_definition": share, "x": float(frac), "households_receiving": float(best["k"]) if pd.notna(best["k"]) else np.nan,
                             "threshold_bp": float(best["effect_bp_at_target"]), "threshold_se_bp": float(best["effect_se_bp"]), "test_id": best["test"],
                             "test": TEST_NAMES.get(best["test"], best["test"]), "status": best["status"], "tests_reaching": len(ok), "variant_field": best["variant"],
                             "analysis_set": best["analysis_set"]})
            else:
                recs.append({"panel": panel, "x_definition": share, "x": float(frac), "households_receiving": np.nan, "threshold_bp": np.nan, "threshold_se_bp": np.nan,
                             "test_id": "", "test": "no test reached 80% power up to 1,600 bp", "status": "not_reached", "tests_reaching": 0, "variant_field": "", "analysis_set": ""})
    out = pd.DataFrame(recs)
    dense = pd.read_csv(CAL / "figure2_detection_region_v3_dense.csv")
    for _, r in out.iterrows():
        rows.add("S7_figure_G1", "threshold_bp", r.threshold_bp, "bp", clock="T", scale="see test", term=f"panel_{r.panel}/x={r.x!r}", benchmark="event_study",
                 inference=f"first 80% crossing, minimum across tests at a measured x; attained by {r.test_id or 'none'}; not a calibrated combined rule; power may fall at larger gains",
                 source=f"data/v4/calibration/figure2_detection_region_v3_{r.panel}_tests.csv", sample="persistence_cohort_trade_date")
    for _, r in dense.iterrows():
        rows.add("S7_figure_G1", "dense_threshold_bp", r.effect_bp_at_target, "bp", clock="T", scale=r.variant, term=f"dense/{r.test}", benchmark="event_study",
                 inference="every household receives the gain (dense alternative)", source="data/v4/calibration/figure2_detection_region_v3_dense.csv", sample="persistence_cohort_trade_date")
    log(f"Figure G1 thresholds: {len(out)} measured shares; at x = 1 the smallest crossing is {out[(out.panel == 'a') & (out.x == 1.0)].threshold_bp.iloc[0]:.6f} bp")
    return out


LABELS = [
    ("Section 7 dispersion limits 87-122 bp; Table G4", "event-study (Y1, fitted intercept extrapolated)", "calibration.bound_validation: OUTCOME = 'headline' (y_h20, committed pooled p1/p99, event-share weights); centered placebo panels N2",
     "data/v4/calibration/bound_validation*.csv; inverted_bound.csv rows outcome=headline", "Table G4 notes already say event-study; Section 7 text should say event-study"),
    ("Figure 4 and Table G2 detection thresholds", "event-study (Y1)", "ext.activity_power and calibration.power_common: run_common.iter_draws(outcome='y_h20') on centered placebo panels; pooled-mean test raw, covariance and maximum-household tests winsorized at the event-study cutoffs",
     "data/v4/ext/activity_power_*.csv", "label thresholds as event-study"),
    ("Table C1 'Six-factor-adjusted returns' -15.4023 bp", "event-study (Y1; car_ff5mom_h20 equals the forward cache's event-study outcome at the entry cell for all 39,030 signals)", "existing.rerun TW_S2_FF5MOM (car_ff5mom_h20), household x 20-session block CGM",
     "data/v4/existing/rerun_tables.csv TW_S2_FF5MOM/dedup/block20", "rename to 'Six-factor event-study returns'; factor-only counterpart in S6_spread"),
    ("Historical-ranking spread 112 bp (SE 70)", "event-study (Y1) at disclosure dates", "ext.forward: score = expanding mean of winsorized y_h20 (historical cutoffs); evaluation = raw y_h20 of next-year purchases; public clock",
     "data/v4/ext/forward_prediction_historical.csv pooled/tercile_diff = 111.699255 (se_block 69.623444)", "label disclosure-date event-study"),
    ("Figure G1 common-household-gain comparison", "event-study (Y1)", "calibration.figure2_v3 from run_power_x/run_power_k outputs on centered placebo panels (y_h20)",
     "data/v4/calibration/figure2_detection_region_v3_{a,b}_tests.csv and _frontier.csv", "label event-study; thresholds in S7_figure_G1"),
]


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("spreads")
    rows = cm.Rows("benchmarks")
    sp_ = factor_only_spread(log, rows)
    cm.write_csv(pd.DataFrame([sp_]), OUT / "six_factor_spread_both_benchmarks.csv")
    g1 = figure_g1_thresholds(rows, log)
    cm.write_csv(g1, OUT / "figure_G1_thresholds.csv")
    lab = pd.DataFrame(LABELS, columns=["exhibit_or_number", "benchmark", "code_path", "source_data", "recommendation"])
    cm.write_csv(lab, OUT / "benchmark_labels.csv")
    fw = pd.read_csv(cm.REPO_ROOT / "data/v4/ext/forward_prediction_historical.csv")
    r = fw[(fw.variant == "pooled") & (fw.statistic == "tercile_diff")].iloc[0]
    rows.add("S7_labels", "historical_ranking_spread", r.value, "bp", clock="P", scale="raw", term="tercile_diff", benchmark="event_study",
             inference="household x 20-session block CGM (ext.forward)", source="data/v4/ext/forward_prediction_historical.csv pooled/tercile_diff")
    rows.add("S7_labels", "historical_ranking_spread_se", r.se_block, "bp", clock="P", scale="raw", term="tercile_diff", benchmark="event_study",
             source="data/v4/ext/forward_prediction_historical.csv pooled/tercile_diff se_block")
    ib = pd.read_csv(CAL / "inverted_bound.csv")
    for _, b in ib[ib.weights == "event_share"].iterrows():
        rows.add("S7_labels", "inverted_bound_gaussian", b.bound_bp, "bp", clock=b.clock, scale="winsorized", term=f"inverted_bound/{b.outcome}",
                 benchmark={"headline": "event_study", "no_alpha": "factor_only", "market_adjusted": "market_adjusted"}[b.outcome],
                 inference="existing Gaussian-shape placebo-injection inversion (calibration.inverted_bound), event-share weights; the factor-only row is the existing no_alpha stream",
                 source="data/v4/calibration/inverted_bound.csv")
    rows.write_component("benchmark_labels")
    log("done")


if __name__ == "__main__":
    main()
