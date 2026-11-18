"""Aggregate the wave-2 and wave-3 power checkpoints into power_surface.csv / frontiers_80.csv (v2).

Sources: grid `power` (wave 2: T cells, current; P cells on the any-line set — dense and sparse-trade
cells superseded by `power_w3`, the share-based sparse-household cells kept as the plan's sensitivity),
`power_w3` (P dense + sparse trades on the all-lines set), `power_k` (k informed households from the
R1 / n >= 20 sets, both clocks). References: T3 critical value, T4 null shares and the I1cal critical
value all come from the `size_v2` grid, N2 draws 1000-1999 of the current set of each clock (disjoint
from the injected draws 0-499). New columns: `x_kind` (share_of_all | k_over_H), `k`, `base_set`,
`H_mean`, `panel_set`, `superseded`, `grid`; frontiers gain `effect_se_bp` (delta method over the
bracketing grid points). The registered T1 sidedness is two-sided (`T1_blk20_2s`); one-sided kept.

CLI: python -m trading_on_disclosure_v4.calibration.run_power_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.calibration import run_power as rp
from trading_on_disclosure_v4.calibration.run_size_v2 import GRID as SIZE_GRID
from trading_on_disclosure_v4.calibration.run_size_v2 import N_TEST

TEST_COLUMNS = dict(rp.TEST_COLUMNS)
TEST_COLUMNS["T1_blk20_2s"] = "T1_I1blk20_rej2"
SIZE_CORRECT_TESTS = ("T1_blk20", "T1_blk20_2s", "T2_I1cal_ES", "T2_I1cal_EH", "T2_I1cal_PW", "T2_I2b_ES", "T2_I2b_EH", "T2_I2b_PW",
                      "T3", "T4_z2", "T4_z3")


def reference_from_size_v2(size_stats: pd.DataFrame, clock: str) -> pd.DataFrame:
    """Reference frame in `rc.apply_reference`'s layout from size_v2 draws >= 1000 (raw + wins rows)."""
    s = size_stats[(size_stats["clock"] == clock) & (size_stats["draw"] >= N_TEST)]
    raw = s[s["variant"] == "raw"].set_index("draw").sort_index()
    wins = s[s["variant"] == "wins"].set_index("draw").sort_index()
    if len(raw) != len(wins) or len(raw) == 0:
        raise ValueError(f"reference draws for {clock}: raw {len(raw)}, wins {len(wins)}")
    return pd.DataFrame({"draw": raw.index, "max_t_raw": raw["T3_max_t"].to_numpy(), "max_t_wins": wins["T3_max_t"].to_numpy(),
                         "share_gt_2": raw["T4_share_gt_2"].to_numpy(), "share_gt_3": raw["T4_share_gt_3"].to_numpy(), "clock": clock})


def _load(grid: str) -> pd.DataFrame:
    try:
        df = rc.load_grid(grid)
    except FileNotFoundError:
        return pd.DataFrame()
    df["grid"] = grid
    return df


def load_all() -> pd.DataFrame:
    w2, w3, k, x = _load("power"), _load("power_w3"), _load("power_k"), _load("power_x")
    if "panel_set" not in w2.columns:
        w2["panel_set"] = w2["clock"].map(rc.WAVE2_SET)
    w2["superseded"] = False
    if len(w3):
        done = set(zip(w3["clock"], w3["variant"], w3["spec_json"], strict=True))
        w2["superseded"] = [(c, v, s) in done for c, v, s in zip(w2["clock"], w2["variant"], w2["spec_json"], strict=True)]
    for f in (w3, k, x):
        if len(f):
            f["superseded"] = False
    return pd.concat([f for f in (w2, w3, k, x) if len(f)], ignore_index=True)


def _family(spec: dict) -> tuple[str, float, float, str, str, float]:
    """(alternative, fraction-or-nan, effect, selection, x_kind, k)."""
    if spec["kind"] == "sparse_k":  # k_frac cells (Figure 2 v3): fraction = k_frac, k = mean realized k
        return ("sparse_households_k", float(spec.get("k_frac", np.nan)), float(spec["mu_bp"]), spec["informed"], "k_over_H",
                float(spec["k"]) if "k" in spec else np.nan)
    alt, frac, eff, sel = rp._family(spec)
    return alt, frac, eff, sel, ("share_of_all" if alt != "dense" else "none"), np.nan


def surface(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["clock", "variant", "spec_json", "grid", "panel_set", "superseded"]
    for (clock, variant, sj, grid, pset, sup), g in stats.groupby(keys, sort=True):
        spec = json.loads(sj)
        alt, frac, eff, sel, x_kind, k = _family(spec)
        base = spec.get("base", "all") if alt != "dense" else "all"
        H = float(g["inj_n_base"].mean()) if "inj_n_base" in g and g["inj_n_base"].notna().any() else np.nan
        if x_kind == "k_over_H":
            if np.isnan(k) and "inj_k_effective" in g:
                k = float(g["inj_k_effective"].mean())
            frac = k / H if np.isnan(frac) else frac
        t1_prefix = "T1raw_" if (variant == "wins" and "T1raw_I1blk20_rej1" in g and g["T1raw_I1blk20_rej1"].notna().any()) else "T1_"
        for test, col in TEST_COLUMNS.items():
            if test.startswith("T1"):
                col = t1_prefix + col[3:]
            if col not in g or g[col].isna().all():
                continue
            r = g[col].to_numpy(dtype=float)
            n = int(np.isfinite(r).sum())
            p = float(np.nanmean(r))
            row = {"clock": clock, "variant": variant, "alternative": alt, "fraction": frac, "effect_bp": eff, "selection": sel,
                   "x_kind": x_kind, "k": k, "base_set": base, "H_mean": H, "params": sj, "test": test, "power": p,
                   "mc_se": rc.mc_se(p, n), "n_panels": n, "truth_tau2": rc.truth_of(spec)["tau2"], "truth_mean": rc.truth_of(spec)["mean"],
                   "panel_set": pset, "superseded": bool(sup), "grid": grid, "size_correct": test in SIZE_CORRECT_TESTS,
                   "t1_on_raw_y": (t1_prefix == "T1raw_" or variant == "raw") if test.startswith("T1") else np.nan}
            if test.startswith("T2_"):
                tag = test[3:].rsplit("_", 1)
                base_tag = {"I2": "I2L20", "I1cal": "I1"}.get(tag[0], tag[0])
                kk = f"T2_{base_tag}_{tag[1]}"
                row.update({"mean_estimate": g[f"{kk}_tau2"].mean(), "coverage": g[f"{kk}_cov"].mean(), "mean_target": g[f"T2_target_{tag[1]}"].mean()})
            elif test.startswith("T1"):
                row.update({"mean_estimate": g[f"{t1_prefix}I1blk20_est"].mean() if f"{t1_prefix}I1blk20_est" in g else np.nan, "mean_target": g["T1_target"].mean()})
            if "inj_k_effective" in g and g["inj_k_effective"].notna().any():
                row["k_effective_mean"] = float(g["inj_k_effective"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def frontier_se(eff: np.ndarray, power: np.ndarray, se: np.ndarray, e: float, status: str, target: float = rp.POWER_TARGET) -> float:
    """Delta-method SE (bp) of the log-linear frontier from the MC SEs of the two bracketing grid points."""
    if status != "interpolated" or not np.isfinite(e):
        return np.nan
    order = np.argsort(eff)
    eff, power, se = np.asarray(eff, float)[order], np.asarray(power, float)[order], np.asarray(se, float)[order]
    i = int(np.flatnonzero(power >= target)[0])
    lo, hi, plo, phi, slo, shi = eff[i - 1], eff[i], power[i - 1], power[i], se[i - 1], se[i]
    if phi <= plo or lo <= 0:
        return np.nan
    f = (target - plo) / (phi - plo)
    var_log = (np.log(hi) - np.log(lo)) ** 2 * ((1 - f) ** 2 * slo ** 2 + f ** 2 * shi ** 2) / (phi - plo) ** 2
    return float(e * np.sqrt(var_log))


def frontiers(surf: pd.DataFrame, target: float = rp.POWER_TARGET) -> pd.DataFrame:
    rows = []
    keys = ["clock", "variant", "alternative", "selection", "x_kind", "base_set", "panel_set", "superseded", "grid", "test", "fraction"]
    for key, g in surf.groupby(keys, sort=True, dropna=False):
        e, status = rp.interpolate_frontier(g["effect_bp"].to_numpy(), g["power"].to_numpy(), target)
        row = dict(zip(keys, key, strict=True))
        gg = g.sort_values("effect_bp")
        row.update({"k": float(gg["k"].iloc[0]), "H_mean": float(gg["H_mean"].iloc[0]), "target_power": target,
                    "effect_bp_at_target": e, "effect_se_bp": frontier_se(gg["effect_bp"].to_numpy(), gg["power"].to_numpy(), gg["mc_se"].to_numpy(), e, status, target),
                    "status": status,
                    "grid_min_effect_reaching_target": float(gg.loc[gg["power"] >= target, "effect_bp"].min()) if (gg["power"] >= target).any() else np.nan,
                    "max_power_on_grid": float(gg["power"].max()), "n_grid_points": int(len(gg)), "n_panels": int(gg["n_panels"].min()),
                    "size_correct": bool(gg["size_correct"].iloc[0]),
                    "t1_on_raw_y": bool(gg["t1_on_raw_y"].iloc[0]) if gg["t1_on_raw_y"].notna().any() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = load_all()
    size_v2 = rc.load_grid(SIZE_GRID)
    cur, old = stats[~stats["superseded"]], stats[stats["superseded"]]
    parts = []
    for clock in ("T", "P"):
        ref = reference_from_size_v2(size_v2, clock)
        for variant in ("raw", "wins"):
            g = cur[(cur["clock"] == clock) & (cur["variant"] == variant)]
            if len(g):
                parts.append(rc.apply_reference(g, ref, variant))
    cur = rc.add_i1cal(pd.concat(parts, ignore_index=True), size_v2, first=N_TEST, n=N_TEST)
    frames = [cur]
    if len(old):  # superseded wave-2 P rows keep their wave-2 references (any-line reference draws; I1cal from size draws 500-999)
        parts = []
        for clock in old["clock"].unique():
            ref = rc.load_reference(clock)
            for variant in ("raw", "wins"):
                g = old[(old["clock"] == clock) & (old["variant"] == variant)]
                if len(g):
                    parts.append(rc.apply_reference(g, ref, variant))
        frames.append(rc.add_i1cal(pd.concat(parts, ignore_index=True), rc.load_grid("size")))
    surf = surface(pd.concat(frames, ignore_index=True))
    fr = frontiers(surf)
    rc.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    surf.to_csv(rc.CALIB_DIR / "power_surface.csv", index=False)
    fr.to_csv(rc.CALIB_DIR / "frontiers_80.csv", index=False)
    rc.log_line(log, f"power_surface.csv {len(surf)} rows ({int(surf['superseded'].sum())} superseded) over "
                     f"{surf['params'].nunique()} cells; frontiers_80.csv {len(fr)} rows")
    return surf, fr


if __name__ == "__main__":
    aggregate(rc.LOG_DIR / "run_power_v2.log")
