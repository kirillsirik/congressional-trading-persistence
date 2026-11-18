"""Markdown tables of the size and power calibration, generated from the committed CSVs
(`size_null_v2.csv`, `power_surface.csv`, `frontiers_80.csv`) so that no number is transcribed by hand.
Output: data/v4/calibration/report_tables_v2.md.

CLI: python -m trading_on_disclosure_v4.calibration.run_report_tables_v2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration.run_report_tables import _flag, _md, _pct, _pm

SIZE_TESTS_R1 = ("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T4_z2", "T4_z3")
SIZE_TESTS_N20 = ("T1_blk20_2s", "T3", "T4_z2", "T4_z3")
FRONTIER_TESTS = ("T1_blk20_2s", "T1_blk20", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")


def _verdict(r) -> str:
    band = "in 3-7%" if r.in_size_band else ("above 7%" if r.reject_rate > 0.07 else "below 3%")
    if np.isfinite(r.coverage):
        cov = "coverage >= 93%" if r.coverage >= 0.93 else "coverage < 93%"
        ok = bool(r.in_size_band) and r.coverage >= 0.93
        return f"{band}; {cov} -> {'PASS' if ok else 'FAIL'}"
    return f"{band} -> {'PASS' if r.in_size_band else 'FAIL'}"


def size_v2_plan(long: pd.DataFrame) -> str:
    keep = long[long["plan_procedure"].astype(bool)].copy()
    order = {"T1": 0, "T2": 1, "T3": 2, "T4_z2": 3, "T4_z3": 4}
    keep["o"] = keep["test"].map(order)
    keep = keep.sort_values(["clock", "o", "variant", "weights", "inference", "sidedness"])
    rows = []
    for _, r in keep.iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "test": r.test, "weights": r.weights, "inference": r.inference, "sided": r.sidedness,
                     "reject % (MC SE)": _pm(r.reject_rate, r.reject_mc_se), "band": _flag(r.reject_rate),
                     "coverage": _pct(r.coverage) if np.isfinite(r.coverage) else "—", "verdict": _verdict(r), "role": r.role})
    return _md(pd.DataFrame(rows))


def size_v2_reference(long: pd.DataFrame) -> str:
    keep = long[~long["plan_procedure"].astype(bool)].sort_values(["clock", "test", "variant", "weights", "inference", "sidedness"])
    rows = []
    for _, r in keep.iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "test": r.test, "weights": r.weights, "inference": r.inference, "sided": r.sidedness,
                     "reject % (MC SE)": _pm(r.reject_rate, r.reject_mc_se), "band": _flag(r.reject_rate),
                     "coverage": _pct(r.coverage) if np.isfinite(r.coverage) else "—", "role": r.role})
    return _md(pd.DataFrame(rows))


def i1cal_detail(long: pd.DataFrame) -> str:
    d = long[long["inference"] == "I1cal"].sort_values(["clock", "variant", "weights"])
    rows = []
    for _, r in d.iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "weights": r.weights, "c_cal (q95 of null t, draws 1000-1999)": f"{r.crit_value:.3f}",
                     "t(dof) 0.95": f"{st.t.ppf(0.95, r.mean_dof):.3f}", "null t SD (test / reference draws)": f"{r.t_stat_sd_test:.2f} / {r.t_stat_sd_reference:.2f}",
                     "size by c_cal % (SE)": _pm(r.reject_rate, r.reject_mc_se), "size by add-one p over the other 1,999 draws %": _pct(r.reject_rate_p_rule),
                     "mean tau2_hat (MC SD), bp2": f"{r.mean_estimate:.0f} ({r.sd_estimate:.0f})", "mean calibrated upper bound tau2, bp2": f"{r.mean_upper_bound_tau2_cal:.0f}",
                     "psd share": _pct(r.psd_corrected_share)})
    return _md(pd.DataFrame(rows))


def i2b_detail(long: pd.DataFrame) -> str:
    d = long[(long["inference"] == "I2b") & (long["sidedness"] == "one-sided")].sort_values(["clock", "variant", "weights"])
    rows = []
    for _, r in d.iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "weights": r.weights, "1-sided % (SE)": _pm(r.reject_rate, r.reject_mc_se),
                     "coverage of 0": _pct(r.coverage), "mean 95% width, bp2": f"{r.mean_interval_width:.0f}", "boot SD / MC SD": f"{r.boot_sd_over_mc_sd:.2f}",
                     "mean upper bound tau2, bp2": f"{r.mean_upper_bound_tau2:.0f}", "verdict": _verdict(r)})
    return _md(pd.DataFrame(rows))


def t1_detail(long: pd.DataFrame) -> str:
    d = long[(long["test"] == "T1") & (long["sidedness"] == "two-sided")].sort_values(["clock", "inference"])
    rows = []
    for _, r in d.iterrows():
        rows.append({"clock": r.clock, "inference": r.inference, "two-sided % (SE)": _pm(r.reject_rate, r.reject_mc_se), "coverage of 0": _pct(r.coverage) if np.isfinite(r.coverage) else "—",
                     "mean estimate, bp": f"{r.mean_estimate:.2f}", "MC SD, bp": f"{r.sd_estimate:.2f}",
                     "mean SE, bp": f"{r.mean_se:.2f}" if np.isfinite(r.get("mean_se", np.nan)) else "—",
                     "SD / SE": f"{r.sd_over_se:.3f}" if np.isfinite(r.get("sd_over_se", np.nan)) else "—",
                     "mean dof": f"{r.mean_dof:.0f}" if np.isfinite(r.get("mean_dof", np.nan)) else "—",
                     "N1 reference mean (SD), bp": f"{r.null_mean_reference:.2f} ({r.null_sd_reference:.2f})" if np.isfinite(r.get("null_mean_reference", np.nan)) else "—"})
    return _md(pd.DataFrame(rows))


def _cur(surf: pd.DataFrame) -> pd.DataFrame:
    return surf[~surf["superseded"].astype(bool)]


def _plan_rows(d: pd.DataFrame, t: str) -> pd.DataFrame:
    if t.startswith("T1"):
        d = d[d["t1_on_raw_y"].astype("boolean").fillna(False).astype(bool)]
        return d[d["variant"] == d["variant"].min()] if d["variant"].nunique() > 1 else d
    v = {"T3": "wins", "T2_I1cal_ES": "wins", "T2_I2b_ES": "wins", "T2_I1cal_EH": "wins", "T2_I1cal_PW": "wins"}.get(t, "raw")
    if (d["variant"] == v).any():
        return d[d["variant"] == v]
    return d[d["variant"] == "raw"] if (d["variant"] == "raw").any() else d  # T4 is raw by construction inside a wins unit


def dense_v2(surf: pd.DataFrame, tests=("T1_blk20_2s", "T2_I1cal_ES", "T2_I2b_ES", "T2_I1cal_EH", "T2_I1cal_PW", "T3", "T4_z2", "T4_z3")) -> str:
    d = _cur(surf)
    d = d[d["alternative"] == "dense"]
    rows = []
    for clock in ("T", "P"):
        for variant in ("wins", "raw"):
            for t in tests:
                g = d[(d["clock"] == clock) & (d["test"] == t) & (d["variant"] == variant)].sort_values("effect_bp")
                if t.startswith("T1"):
                    g = g[g["t1_on_raw_y"].astype("boolean").fillna(False).astype(bool)]
                if g.empty:
                    continue
                row = {"clock": clock, "Y": ("raw (T1 on raw Y)" if t.startswith("T1") else variant), "test": t, "panel set": g["panel_set"].iloc[0]}
                for _, r in g.iterrows():
                    row[f"tau={r.effect_bp:g}"] = _pct(r.power)
                rows.append(row)
    return _md(pd.DataFrame(rows).fillna("—").drop_duplicates())


def k_power(surf: pd.DataFrame, clock: str, base: str, selection: str, test: str) -> str:
    d = _cur(surf)
    d = d[(d["alternative"] == "sparse_households_k") & (d["clock"] == clock) & (d["base_set"] == base)
          & (d["selection"] == selection) & (d["test"] == test)]
    if d.empty:
        return "(not available)"
    H = d["H_mean"].iloc[0]
    p = d.pivot_table(index="k", columns="effect_bp", values="power")
    p.index = [f"k = {int(k)} ({100 * k / H:.1f}% of H = {H:.0f})" for k in p.index]
    p.columns = [f"{c:g} bp" for c in p.columns]
    p = p.map(_pct).reset_index().rename(columns={"index": "informed households"})
    return _md(p)


def frontier_v2(fr: pd.DataFrame, clock: str, tests=FRONTIER_TESTS) -> str:
    f = fr[(fr["clock"] == clock) & (~fr["superseded"].astype(bool)) & (fr["test"].isin(tests))]
    rows = []
    for (alt, sel, base), g in f.groupby(["alternative", "selection", "base_set"], sort=True):
        for frac, gg in g.groupby("fraction", dropna=False, sort=True):
            H = gg["H_mean"].dropna().iloc[0] if gg["H_mean"].notna().any() else np.nan
            k = gg["k"].dropna().iloc[0] if gg["k"].notna().any() else np.nan
            lab = ("all" if not np.isfinite(frac) else (f"k = {int(k)} of H = {H:.0f} ({100 * frac:.1f}%)" if np.isfinite(k) else f"{100 * frac:g}%"))
            row = {"alternative": alt, "selection": sel, "base": base, "informed": lab}
            for t in tests:
                rr = _plan_rows(gg[gg["test"] == t], t)
                if rr.empty:
                    row[t] = "—"
                    continue
                r = rr.iloc[0]
                e, s, se = r.effect_bp_at_target, r.status, r.effect_se_bp
                row[t] = ("> grid" if s == "not_reached" else (f"<= {e:.0f}" if s == "at_grid_min" else (f"{e:.0f} (±{se:.0f})" if np.isfinite(se) else f"{e:.0f}"))) + f" [{r.variant}]"
            rows.append(row)
    return _md(pd.DataFrame(rows))


def p_old_vs_new(fr: pd.DataFrame, tests=("T1_blk20", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")) -> str:
    """P-clock frontiers: superseded any-line rows vs current all-lines rows, same test and variant (raw)."""
    f = fr[(fr["clock"] == "P") & (fr["alternative"].isin(["dense", "sparse_trades"])) & (fr["test"].isin(tests)) & (fr["variant"] == "raw")]
    rows = []
    for (alt, sel, frac), g in f.groupby(["alternative", "selection", "fraction"], dropna=False, sort=True):
        row = {"alternative": alt, "selection": sel, "fraction": "all" if not np.isfinite(frac) else f"{100 * frac:g}%"}
        for t in tests:
            for sup, lab in ((True, "any-line (superseded)"), (False, "all-lines")):
                r = g[(g["test"] == t) & (g["superseded"].astype(bool) == sup)]
                if r.empty:
                    row[f"{t} {lab}"] = "—"
                    continue
                e, s = r.effect_bp_at_target.iloc[0], r.status.iloc[0]
                row[f"{t} {lab}"] = "> grid" if s == "not_reached" else (f"<= {e:.0f}" if s == "at_grid_min" else f"{e:.0f}")
        rows.append(row)
    return _md(pd.DataFrame(rows))


def expected_informed(surf: pd.DataFrame, size_v2: pd.DataFrame) -> str:
    """Hypergeometric expected number (and P(0)) of informed households inside the R1 and n >= 20 sets when a share
    pi of ALL households is informed, per clock; N and the set sizes are measured."""
    rows = []
    for clock, N, r1 in (("T", 233, 38), ("P", 227, 30)):
        n20 = size_v2[(size_v2["clock"] == clock) & (size_v2["test"] == "T3")]["mean_n_households"].iloc[0]
        for pi in (0.02, 0.05, 0.10, 0.20):
            k_all = int(round(pi * N))
            row = {"clock": clock, "share of all households": f"{100 * pi:g}%", "informed among N": f"{k_all} of {N}"}
            for lab, H in (("R1 set (T2)", r1), ("n >= 20 set (T3)", int(round(n20)))):
                e = k_all * H / N
                p0 = st.hypergeom.pmf(0, N, k_all, H)
                row[f"{lab}: H, E[informed], P(none)"] = f"{H}, {e:.2f}, {100 * p0:.1f}%"
            rows.append(row)
    return _md(pd.DataFrame(rows))


def main() -> None:
    long = pd.read_csv(rc.CALIB_DIR / "size_null_v2.csv")
    surf = pd.read_csv(rc.CALIB_DIR / "power_surface.csv")
    fr = pd.read_csv(rc.CALIB_DIR / "frontiers_80.csv")
    parts = ["### Size v2: plan §11 procedures (1,000 N2 test panels per clock; reference draws 1000-1999)\n", size_v2_plan(long),
             "\n\n### Size v2: reference rows (not plan procedures)\n", size_v2_reference(long),
             "\n\n### I1cal detail\n", i1cal_detail(long), "\n\n### I2b detail (499 draws per panel; the actual run uses 4,999)\n", i2b_detail(long),
             "\n\n### T1 detail\n", t1_detail(long),
             "\n\n### Expected informed households inside each test's set under the wave-2 share grid\n", expected_informed(surf, long),
             "\n\n### Dense power v2 (current panel sets)\n", dense_v2(surf)]
    for clock in ("T", "P"):
        for base, tests in (("R1", SIZE_TESTS_R1), ("n20", SIZE_TESTS_N20)):
            for sel in ("uniform", "activity"):
                for t in tests:
                    parts += [f"\n\n### Power (k grid): {clock} clock / base {base} / {sel} / {t}\n", k_power(surf, clock, base, sel, t)]
        parts += [f"\n\n### 80%-power frontiers v2, {clock} clock (bp per 20 sessions; ± delta-method SE; [variant])\n", frontier_v2(fr, clock)]
    parts += ["\n\n### P clock: any-line (superseded) vs all-lines frontiers, raw Y\n", p_old_vs_new(fr)]
    (rc.CALIB_DIR / "report_tables_v2.md").write_text("\n".join(parts) + "\n")
    print("wrote", rc.CALIB_DIR / "report_tables_v2.md")


if __name__ == "__main__":
    main()
