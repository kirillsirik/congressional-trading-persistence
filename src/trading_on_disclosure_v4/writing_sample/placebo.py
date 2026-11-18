"""Actual-date versus shifted-date (stored timing-only placebo) return levels and within-draw continuation.

python -m trading_on_disclosure_v4.writing_sample.placebo
Reads the 2,000 stored N1 draws (uncentered, shifted entry dates) of each clock and applies the original
specification transformation (`specs.to_panel`, eligibility recomputed on each draw exactly as the stored
placebo statistics were) for two samples: all purchases (spec `purchase_mean`, rule all, raw) and the
persistence sample (headline spec, rule R1, raw). For each draw it records purchase-weighted means of Y1, Y0,
the charge Y0-Y1, 20*alpha and the compounding remainder, on the SAME rows, and the slope of Y0 on 20*alpha with
entry-block fixed effects (shifted entry-calendar blocks define the effects; original shift blocks as a
sensitivity). Alpha at a shifted entry is the forward cache's own fit at that (security, session).
Writes per-draw summaries and the aggregate rows (component `placebo_levels`).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import inference as einf
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel as panel_mod
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import fits
from trading_on_disclosure_v4.writing_sample import panels as pn

OUT = cm.OUT / "placebo"
STORED_STATS = cm.REPO_ROOT / "data" / "v4" / "restricted" / "e" / "placebo_stats"
SAMPLES = ("all_purchases", "cohort")


class Grid:
    """Dense (security, session) lookup of the extended fit cache."""

    def __init__(self):
        ext = fits.load(["permno", "anchor_idx", "alpha_daily", "y1_h20", "y0_h20"])
        self.permnos = np.sort(ext["permno"].unique())
        self.lo = int(ext["anchor_idx"].min())
        n_a = int(ext["anchor_idx"].max()) - self.lo + 1
        pos = np.searchsorted(self.permnos, ext["permno"].to_numpy())
        flat = pos * n_a + (ext["anchor_idx"].to_numpy() - self.lo)
        self.n_a = n_a
        self.arrays = {}
        for c in ("alpha_daily", "y1_h20", "y0_h20"):
            a = np.full(len(self.permnos) * n_a, np.nan)
            a[flat] = ext[c].to_numpy(float)
            self.arrays[c] = a

    def get(self, col: str, permno: np.ndarray, idx: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.permnos, permno)
        ok = (pos < len(self.permnos)) & (self.permnos[np.minimum(pos, len(self.permnos) - 1)] == permno) & (idx >= self.lo) & (idx < self.lo + self.n_a)
        out = np.full(len(permno), np.nan)
        out[ok] = self.arrays[col][pos[ok] * self.n_a + (idx[ok] - self.lo)]
        return out


def fe_slope(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> float:
    g = pd.factorize(groups)[0]
    cnt = np.bincount(g)
    xd = x - (np.bincount(g, weights=x) / cnt)[g]
    yd = y - (np.bincount(g, weights=y) / cnt)[g]
    sxx = float(xd @ xd)
    return float(xd @ yd / sxx) if sxx > 1e-12 else np.nan


def clip_period(v: np.ndarray, period: np.ndarray, cuts: dict) -> np.ndarray:
    out = v.copy()
    for p, (lo, hi) in cuts.items():
        m = period == p
        out[m] = np.clip(out[m], lo, hi)
    return out


def draw_stats(p1: pd.DataFrame, p0: pd.DataFrame, x: np.ndarray, orig_blk: np.ndarray, shift_blk: np.ndarray, wcuts: dict) -> dict:
    y1 = p1["y_h20"].to_numpy(float)
    y0 = p0["y_h20"].to_numpy(float)
    charge = y0 - y1
    rem = charge - x
    per = p1["period"].to_numpy()
    out = {"purchases": len(p1), "households": int(p1["household_id"].nunique()), "purchases_A": int((per == "A").sum()), "purchases_B": int((per == "B").sum()),
           "shifted_blocks": int(len(np.unique(shift_blk))), "original_blocks": int(len(np.unique(orig_blk))),
           "mean_Y1": float(y1.mean()), "mean_Y0": float(y0.mean()), "mean_charge": float(charge.mean()), "mean_alpha20": float(x.mean()),
           "mean_compounding": float(rem.mean()), "identity_gap": float(y0.mean() - y1.mean() - charge.mean()),
           "slope_raw_shifted_block_fe": fe_slope(y0, x, shift_blk), "slope_raw_original_block_fe": fe_slope(y0, x, orig_blk)}
    xw = clip_period(x, per, wcuts["x"])
    y0w = clip_period(y0, per, wcuts["Y0"])
    out["slope_winsorized_shifted_block_fe"] = fe_slope(y0w, xw, shift_blk)
    return out


def run_clock(clock: str, ctx, grid: Grid, actual: dict, log) -> pd.DataFrame:
    cal = ctx.calendar
    meta = evf.event_metadata(clock)
    orig_idx = pd.Series(cal.get_indexer(pd.DatetimeIndex(meta["anchor_date"])), index=meta["event_id"].astype(str))
    specs = {"all_purchases": pn.all_purchases_spec(clock), "cohort": pn.cohort_spec(clock)}
    rows = []
    checks = {"rows_checked": 0, "y1_max_abs_diff_bp": 0.0, "y0_max_abs_diff_bp": 0.0, "rows_without_alpha": 0}
    t0 = time.time()
    for i, batch in evf.placebo_event_frames(clock, "N1", meta, dup="default", batches=True):
        batch = batch[batch["draw"] < 2000]
        sidx = cal.get_indexer(pd.DatetimeIndex(batch["anchor_date"]))
        perm = batch["permno"].to_numpy(np.int64)
        a = grid.get("alpha_daily", perm, sidx)
        y1c = grid.get("y1_h20", perm, sidx) * cm.BP
        y0c = grid.get("y0_h20", perm, sidx) * cm.BP
        ok = np.isfinite(batch["y_h20"].to_numpy(float))
        d1 = np.abs(batch["y_h20"].to_numpy(float)[ok] - y1c[ok])
        d0 = np.abs(batch["y_noalpha_h20"].to_numpy(float)[ok] - y0c[ok])
        checks["rows_checked"] += int(ok.sum())
        checks["y1_max_abs_diff_bp"] = max(checks["y1_max_abs_diff_bp"], float(np.nanmax(d1)) if len(d1) else 0.0)
        checks["y0_max_abs_diff_bp"] = max(checks["y0_max_abs_diff_bp"], float(np.nanmax(d0)) if len(d0) else 0.0)
        checks["rows_without_alpha"] += int((ok & ~np.isfinite(a)).sum())
        batch = batch.assign(x_alpha20=cm.H * a * cm.BP, shifted_block=sidx // 20,
                             original_block=orig_idx.reindex(batch["event_id"].astype(str)).to_numpy() // 20)
        for draw, g in batch.groupby("draw", sort=True):
            g = g.reset_index(drop=True)
            for sample, spec in specs.items():
                p1 = sp.to_panel(g, spec, None, draw=int(draw))
                p0 = sp.to_panel(g, replace(spec, outcome="ff5mom_noalpha_h20"), None, draw=int(draw))
                if not p1["event_id"].equals(p0["event_id"]):
                    raise AssertionError(f"{clock} draw {draw} {sample}: benchmarks select different rows")
                idx = pd.Index(g["event_id"]).get_indexer(p1["event_id"])
                x = g["x_alpha20"].to_numpy(float)[idx]
                if not np.isfinite(x).all():
                    raise AssertionError(f"{clock} draw {draw}: placebo row without alpha")
                st = draw_stats(p1, p0, x, g["original_block"].to_numpy()[idx], g["shifted_block"].to_numpy()[idx], actual[(clock, sample)]["cuts"])
                rows.append({"clock": clock, "sample": sample, "draw": int(draw), **st})
        log(f"{clock} batch {i}: {len(rows)} draw-samples, {time.time() - t0:.0f} s")
    out = pd.DataFrame(rows)
    out.attrs["checks"] = checks
    log(f"{clock}: stored shifted outcomes vs cache at the shifted cell: {checks}")
    return out


def actual_values(ctx, clock: str, log) -> dict:
    """Actual-date means on both samples, with the original household x 20-session-block CGM SE."""
    res = {}
    cal = ctx.calendar
    for sample in SAMPLES:
        spec = pn.all_purchases_spec(clock) if sample == "all_purchases" else pn.cohort_spec(clock)
        p = pn.attach(ctx.panel(spec), ctx.frames[clock], ["y_noalpha_h20", "x_alpha20", "entry_block"])
        y1, y0, x = (p[c].to_numpy(float) for c in ("y_h20", "y_noalpha_h20", "x_alpha20"))
        clusters = pd.DataFrame({"household": p["household_id"].to_numpy(), "block20": einf.date_clusters(p["anchor_date"], cal)["block20"]})
        rec = {"purchases": len(p), "households": int(p["household_id"].nunique()), "blocks": int(p["entry_block"].nunique())}
        for name, v in (("Y1", y1), ("Y0", y0), ("charge", y0 - y1), ("alpha20", x), ("compounding", y0 - y1 - x)):
            r = einf.clustered(v, np.ones((len(v), 1)), None, clusters, 0)
            rec[f"mean_{name}"] = r["estimate"]
            rec[f"se_{name}"] = r["se"]
            rec[f"dof_{name}"] = r["dof"]
            rec[f"psd_{name}"] = r["psd_corrected"]
        rec["slope_raw"] = fe_slope(y0, x, p["entry_block"].to_numpy())
        per = p["period"].to_numpy()
        cuts = {"x": {q: tuple(np.percentile(x[per == q], [1, 99])) for q in ("A", "B")},
                "Y0": ctx.cutoffs(clock, "ff5mom_noalpha_h20")}
        rec["slope_winsorized"] = fe_slope(clip_period(y0, per, cuts["Y0"]), clip_period(x, per, cuts["x"]), p["entry_block"].to_numpy())
        rec["cuts"] = cuts
        res[(clock, sample)] = rec
        log(f"{clock}/{sample} actual: {rec['purchases']} purchases, {rec['households']} households; mean Y1 {rec['mean_Y1']:.10f} (SE {rec['se_Y1']:.10f})")
    return res


def stored_placebo_means(clock: str) -> np.ndarray:
    """The stored per-draw placebo values of the all-purchase mean (read only)."""
    t = pd.read_parquet(STORED_STATS / f"placebo_{clock}_N1_default.parquet")
    from trading_on_disclosure_v4.estimators import placebo_stats as ps

    key = ps.spec_key(pn.all_purchases_spec(clock))
    v = t[(t["key"] == key) & (t["stat"] == "mean")].sort_values("draw")
    return v["draw"].to_numpy(), v["value"].to_numpy(float)


def summarize(per: pd.DataFrame, actual: dict, rows: cm.Rows, log) -> None:
    stats = {"mean_Y1": ("event_study_mean", "Y1"), "mean_Y0": ("factor_only_mean", "Y0"), "mean_charge": ("alpha_charge_mean", "charge"),
             "mean_alpha20": ("alpha20_mean", "alpha20"), "mean_compounding": ("compounding_remainder_mean", "compounding")}
    for (clock, sample), g in per.groupby(["clock", "sample"]):
        act = actual[(clock, sample)]
        n = len(g)
        src = f"data/v4/restricted/n/panels/{clock}/N1 (stored timing-only draws); data/v4/writing_sample/{cm.RUN_ID}/placebo/placebo_per_draw.csv"
        base = dict(clock=clock, sample=sample, scale="raw", weighting="equal purchase", source=src)
        for col, (name, k) in stats.items():
            v = g[col].to_numpy(float)
            rows.add("placebo_levels", "actual", act[f"mean_{k}"], "bp", term=name, inference="actual-date purchase mean", benchmark=k, **base)
            rows.add("placebo_levels", "actual_se_two_way", act[f"se_{k}"], "bp", term=name, inference="household x 20-session block CGM (ext.inference.clustered), actual dates", benchmark=k, **base)
            rows.add("placebo_levels", "actual_dof", act[f"dof_{k}"], "count", term=name, benchmark=k, **base)
            rows.add("placebo_levels", "placebo_mean_uncentered", float(v.mean()), "bp", term=name, benchmark=k,
                     inference="mean over 2,000 stored timing-only (N1, uncentered) draws of the per-draw purchase mean", **base)
            rows.add("placebo_levels", "placebo_sd_across_draws", float(v.std(ddof=1)), "bp", term=name, benchmark=k,
                     inference="SD across draws (ddof=1); dispersion of the placebo statistic, not a standard error of a population mean", **base)
            for q in (2.5, 50, 97.5):
                rows.add("placebo_levels", f"placebo_p{q:g}", float(np.percentile(v, q)), "bp", term=name, benchmark=k, inference="percentile across draws", **base)
            rows.add("placebo_levels", "share_draws_strictly_below_actual", float(np.mean(v < act[f"mean_{k}"])), "share", term=name, benchmark=k,
                     inference="descriptive placement of the actual value among draws; not a randomization p-value", **base)
        rows.add("placebo_levels", "draws", n, "count", term="draws", **base)
        for col in ("purchases", "households"):
            v = g[col].to_numpy(float)
            rows.add("placebo_levels", f"{col}_actual", act[col], "count", term="sample_size", **base)
            for s, f in (("min", v.min()), ("mean", v.mean()), ("max", v.max())):
                rows.add("placebo_levels", f"{col}_{s}_across_draws", f, "count", term="sample_size", **base)
        rows.add("placebo_levels", "max_abs_identity_gap", float(np.abs(g["identity_gap"]).max()), "bp", term="mean(Y0)-mean(Y1)-mean(charge)", **base)
        for col, lab in (("slope_raw_shifted_block_fe", "raw, shifted entry-calendar block FE (defines the effects)"),
                         ("slope_raw_original_block_fe", "raw, original shift-block FE (sensitivity)"),
                         ("slope_winsorized_shifted_block_fe", "winsorized at the actual sample's x and pooled Y0 cutoffs, shifted-block FE")):
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            b = dict(base, scale="winsorized" if "winsorized" in col else "raw")
            rows.add("placebo_continuation", "draws_with_slope", len(v), "count", term=col, inference=lab, benchmark="Y0_on_alpha20", **{k: x for k, x in b.items()})
            for s, f in (("mean", v.mean()), ("sd", v.std(ddof=1)), ("p2.5", np.percentile(v, 2.5)), ("p50", np.percentile(v, 50)), ("p97.5", np.percentile(v, 97.5))):
                rows.add("placebo_continuation", f"within_draw_slope_{s}", float(f), "slope", term=col, benchmark="Y0_on_alpha20",
                         inference=f"distribution of the within-draw slope across 2,000 draws ({lab}); descriptive, not a sampling interval", **{k: x for k, x in b.items()})
        a_raw = act["slope_raw"]
        v = g["slope_raw_shifted_block_fe"].to_numpy(float)
        rows.add("placebo_continuation", "actual_slope", a_raw, "slope", term="slope_raw_shifted_block_fe", benchmark="Y0_on_alpha20", inference="actual-date slope, entry-block FE", **base)
        rows.add("placebo_continuation", "share_draws_strictly_below_actual", float(np.mean(v < a_raw)), "share", term="slope_raw_shifted_block_fe", benchmark="Y0_on_alpha20",
                 inference="descriptive placement; not a randomization p-value", **base)
        b = dict(base, scale="winsorized")
        rows.add("placebo_continuation", "actual_slope", act["slope_winsorized"], "slope", term="slope_winsorized_shifted_block_fe", benchmark="Y0_on_alpha20", inference="actual-date slope, entry-block FE", **b)
        if sample == "all_purchases":
            d, stored = stored_placebo_means(clock)
            mine = g.sort_values("draw")["mean_Y1"].to_numpy(float)
            gap = float(np.max(np.abs(mine - stored))) if len(stored) == len(mine) else np.nan
            rows.add("placebo_levels", "max_abs_diff_vs_stored_placebo_means", gap, "bp", term="event_study_mean", benchmark="Y1",
                     inference="per-draw all-purchase event-study means vs the stored placebo statistic cache (data/v4/restricted/e/placebo_stats)", **base)
            log(f"{clock}: per-draw all-purchase means vs stored cache: max |diff| {gap:.3e} bp over {len(stored)} draws")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clocks", default="T,P")
    args = ap.parse_args(argv)
    log = cm.logger("placebo")
    t0 = time.time()
    ctx, cal, ext, new = pn.load(log)
    del ext
    grid = Grid()
    log(f"forward cache sha256 {cm.sha256_file(fc.CACHE_PATH)[:12]}")
    clocks = args.clocks.split(",")
    actual = {}
    for c in clocks:
        actual.update(actual_values(ctx, c, log))
    parts = [run_clock(c, ctx, grid, actual, log) for c in clocks]
    checks = {c: dict(x.attrs["checks"]) for c, x in zip(clocks, parts)}
    per = pd.concat(parts, ignore_index=True)
    cm.write_csv(per, OUT / "placebo_per_draw.csv")
    rows = cm.Rows(f"data/v4/writing_sample/{cm.RUN_ID}/placebo/placebo_per_draw.csv")
    summarize(per, actual, rows, log)
    rows.write_component("placebo_levels")
    act = []
    for (c, s), r in actual.items():
        rr = {k: v for k, v in r.items() if k != "cuts"}
        rr.update(clock=c, sample=s, x_cutoffs_bp=str(r["cuts"]["x"]), y0_cutoffs_bp=str(r["cuts"]["Y0"]))
        act.append(rr)
    cm.write_csv(pd.DataFrame(act), OUT / "placebo_actual_values.csv")
    cm.write_json({"seconds": round(time.time() - t0, 1), "environment": cm.environment(), "per_draw_rows": int(len(per)), "stored_outcome_checks": checks,
                   "forward_cache_sha256": cm.sha256_file(fc.CACHE_PATH), "aggregation": "per draw: equal-weight purchase means on the draw's own eligible rows; "
                   "across draws: mean, SD (ddof=1) and percentiles of the per-draw values", "fixed_effects": "shifted entry-calendar 20-session block "
                   "(calendar position of the shifted entry // 20) defines the effects; original shift block (actual entry position // 20) is the sensitivity"},
                  OUT / "placebo_manifest.json")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
