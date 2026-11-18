"""Markdown tables of the calibration results, generated from the committed CSVs so that no
number is transcribed by hand. Output: data/v4/calibration/report_tables.md.

CLI: python -m trading_on_disclosure_v4.calibration.run_report_tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc


def _pct(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{100 * x:.1f}%"


def _pm(rate: float, se: float) -> str:
    return "—" if not np.isfinite(rate) else f"{100 * rate:.1f} ({100 * se:.1f})"


def _flag(rate: float, lo=0.03, hi=0.07) -> str:
    return "—" if not np.isfinite(rate) else ("in" if lo <= rate <= hi else ("HIGH" if rate > hi else "low"))


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def size_t1_t3_t4(long: pd.DataFrame) -> str:
    rows = []
    for _, r in long[long["test"].isin(["T1", "T3", "T4_z2", "T4_z3"])].iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "test": r.test, "inference": r.inference, "sided": r.sidedness,
                     "reject % (MC SE)": _pm(r.reject_rate, r.reject_mc_se), "band": _flag(r.reject_rate),
                     "coverage": _pct(r.coverage), "mean stat": f"{r.mean_estimate:.3f}", "SD stat": f"{r.sd_estimate:.3f}",
                     "mean SE / crit": f"{r.mean_se:.2f}" if np.isfinite(r.get("mean_se", np.nan)) else (f"{r.critical_value:.3f}" if np.isfinite(r.get("critical_value", np.nan)) else "—")})
    return _md(pd.DataFrame(rows))


def size_t2_wide(long: pd.DataFrame) -> str:
    t2 = long[(long["test"] == "T2")]
    rows = []
    for (clock, variant, w), g in t2.groupby(["clock", "variant", "weights"], sort=False):
        row = {"clock": clock, "Y": variant, "weights": w}
        for inf in ("I1", "I1blk20", "I2L20", "I2b", "I1cal"):
            one = g[(g.inference == inf) & (g.sidedness == "one-sided")]
            two = g[(g.inference == inf) & (g.sidedness == "two-sided")]
            if one.empty:
                row[f"{inf} 1-sided % (SE)"] = "—"
                continue
            row[f"{inf} 1-sided % (SE)"] = _pm(one.reject_rate.iloc[0], one.reject_mc_se.iloc[0]) + f" [{_flag(one.reject_rate.iloc[0])}]"
            if not two.empty:
                row[f"{inf} 2-sided %"] = _pct(two.reject_rate.iloc[0])
                row[f"{inf} coverage"] = _pct(two.coverage.iloc[0]) + (" [ok]" if two.coverage.iloc[0] >= 0.93 else " [low]")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["clock", "Y", "weights"])
    return _md(df.fillna("—"))


def dispersion_t2(long: pd.DataFrame) -> str:
    t2 = long[(long["test"] == "T2") & (long["sidedness"] == "one-sided")]
    rows = []
    for _, r in t2.iterrows():
        rows.append({"clock": r.clock, "Y": r.variant, "weights": r.weights, "inference": r.inference,
                     "mean tau2_hat (bp2)": f"{r.mean_estimate:.0f}", "MC SD tau2_hat": f"{r.sd_estimate:.0f}",
                     "mean 95% width": f"{r.mean_interval_width:.0f}" if np.isfinite(r.get("mean_interval_width", np.nan)) else "—",
                     "boot SD / MC SD": f"{r.boot_sd_over_mc_sd:.2f}" if np.isfinite(r.get("boot_sd_over_mc_sd", np.nan)) else "—",
                     "null t SD": f"{r.t_stat_sd:.2f}" if np.isfinite(r.get("t_stat_sd", np.nan)) else "—",
                     "psd share": _pct(r.psd_corrected_share) if np.isfinite(r.get("psd_corrected_share", np.nan)) else "—"})
    return _md(pd.DataFrame(rows))


def dense_power(surf: pd.DataFrame, tests=("T1", "T2_I1_ES", "T2_I1cal_ES", "T2_I2b_ES", "T2_I2_ES", "T2_I1_EH", "T2_I1_PW", "T3", "T4_z2", "T4_z3")) -> str:
    d = surf[(surf["alternative"] == "dense") & (surf["test"].isin(tests))]
    rows = []
    for (clock, variant), g in d.groupby(["clock", "variant"], sort=True):
        for t in tests:
            gg = g[g.test == t].sort_values("effect_bp")
            if gg.empty:
                continue
            row = {"clock": clock, "Y": variant, "test": t}
            for _, r in gg.iterrows():
                row[f"tau={r.effect_bp:g}"] = _pct(r.power)
            rows.append(row)
    return _md(pd.DataFrame(rows).fillna("—"))


def sparse_power(surf: pd.DataFrame, alternative: str, clock: str, selection: str, test: str) -> str:
    d = surf[(surf["alternative"] == alternative) & (surf["clock"] == clock) & (surf["selection"] == selection)
             & (surf["test"] == test) & (surf["variant"] == "raw")]
    if d.empty:
        return "(not yet available)"
    p = d.pivot_table(index="fraction", columns="effect_bp", values="power")
    p.index = [f"{100 * x:g}%" for x in p.index]
    p.columns = [f"{c:g} bp" for c in p.columns]
    p = p.map(_pct).reset_index().rename(columns={"index": "fraction"})
    return _md(p)


def frontier_table(fr: pd.DataFrame, clock: str, variant: str = "raw", tests=("T1", "T1_blk20", "T2_I1_ES", "T2_I1cal_ES", "T2_I2b_ES", "T3", "T4_z2", "T4_z3")) -> str:
    f = fr[(fr["clock"] == clock) & (fr["variant"] == variant) & (fr["test"].isin(tests))]
    rows = []
    for (alt, sel), g in f.groupby(["alternative", "selection"], sort=True):
        for frac, gg in g.groupby("fraction", dropna=False, sort=True):
            row = {"alternative": alt, "selection": sel, "fraction": "all" if not np.isfinite(frac) else f"{100 * frac:g}%"}
            for t in tests:
                r = gg[gg.test == t]
                if r.empty:
                    row[t] = "—"
                    continue
                e, st = r.effect_bp_at_target.iloc[0], r.status.iloc[0]
                row[t] = ("> grid" if st == "not_reached" else (f"<= {e:.0f}" if st == "at_grid_min" else f"{e:.0f}"))
            rows.append(row)
    return _md(pd.DataFrame(rows))


def main() -> None:
    long = pd.read_csv(rc.CALIB_DIR / "size_null.csv")
    parts = ["### Size: T1, T3, T4 (1,000 N2 panels per clock x Y)\n", size_t1_t3_t4(long),
             "\n\n### Size: T2 (tau2) by inference x weights, side by side\n", size_t2_wide(long),
             "\n\n### T2 dispersion diagnostics\n", dispersion_t2(long)]
    surf_path, fr_path = rc.CALIB_DIR / "power_surface.csv", rc.CALIB_DIR / "frontiers_80.csv"
    if surf_path.exists():
        surf, fr = pd.read_csv(surf_path), pd.read_csv(fr_path)
        parts += ["\n\n### Dense power (raw vs winsorized)\n", dense_power(surf)]
        for clock in ("T", "P"):
            for alt, sels in (("sparse_households", ("uniform", "activity")), ("sparse_trades", ("common", "household"))):
                for sel in sels:
                    for t in ("T1", "T2_I1cal_ES", "T3", "T4_z2"):
                        parts += [f"\n\n### Power: {alt} / {sel} / {clock} clock / {t}\n", sparse_power(surf, alt, clock, sel, t)]
            parts += [f"\n\n### 80%-power frontiers, {clock} clock (effect in bp per 20 sessions)\n", frontier_table(fr, clock)]
    (rc.CALIB_DIR / "report_tables.md").write_text("\n".join(parts) + "\n")
    print("wrote", rc.CALIB_DIR / "report_tables.md")


if __name__ == "__main__":
    main()
