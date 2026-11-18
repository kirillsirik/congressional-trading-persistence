"""Restricted inputs of the continuation consumers (scripts/writing_sample/restricted_consumers.py), built from the
original panels.

python -m trading_on_disclosure_v4.writing_sample.exports
Writes, under data/v4/restricted/writing_sample/: continuation_purchases.csv, continuation_security_grid.csv,
cohort_purchase_means.csv, sales_signed_means.csv, spread_household_counts.csv, purchase_panel_T.parquet,
purchase_panel_P.parquet, private/spread_household_crosswalk.csv; and the export manifest (hashes, counts,
cutoffs, exclusion counts) under data/v4/writing_sample/computation_2026-11-11/.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.outcomes import panel as panel_mod
from trading_on_disclosure_v4.writing_sample import common as cm
from trading_on_disclosure_v4.writing_sample import market_values as mvm
from trading_on_disclosure_v4.writing_sample import panels as pn

MANIFEST = cm.OUT / "exports_manifest.json"
GRID_START = pd.Timestamp("2015-01-01")
GRID_END = pd.Timestamp("2025-12-31")
GRID_STEP = 20
NEW_INDEX_BASE = 1000  # spread-sample households absent from the closeout crosswalk get 1001, 1002, ...


def large_third_flags(mv: pd.DataFrame, cells: pd.DataFrame, calendar: pd.DatetimeIndex) -> np.ndarray:
    """1 if the security's DlyCap on the formation session is at or above the 2/3 quantile (pandas linear) of the
    positive DlyCap values of all universe securities on that session; NaN if its own DlyCap is missing."""
    mv = mv[mv["dlycap"] > 0]
    cut = mv.groupby("date")["dlycap"].quantile(2 / 3)
    own = cells[["permno", "date"]].merge(mv[["permno", "date", "dlycap"]], on=["permno", "date"], how="left")["dlycap"].to_numpy(float)
    c = cut.reindex(pd.DatetimeIndex(cells["date"])).to_numpy(float)
    out = np.where(np.isfinite(own) & np.isfinite(c), (own >= c).astype(float), np.nan)
    return out, own


def purchase_panel(ctx, clock: str, sample: str, key: pd.Series, mv: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase of the sample with both benchmarks at 10/20/60 sessions, alpha, block and period."""
    spec = pn.cohort_spec(clock) if sample == "cohort" else pn.all_purchases_spec(clock)
    p = ctx.panel(spec)
    f = ctx.frames[clock]
    p = pn.attach(p, f, ["x_alpha20", "y_noalpha_h20", "y_h10", "y_h60", "y_noalpha_h10", "y_noalpha_h60", "entry_block", "anchor_idx",
                         "n_est", "alpha_daily", "record_source"])
    p = p.rename(columns={"y_h20": "Y1_bp", "y_noalpha_h20": "Y0_bp", "x_alpha20": "alpha20_bp"})
    cm.check_key_covers(key, p["household_id"])
    p.insert(0, "sample", sample)
    p["household_index"] = key.reindex(p["household_id"].astype(str)).to_numpy()
    flags, own = large_third_flags(mv, pd.DataFrame({"permno": p["permno"].to_numpy(), "date": pd.DatetimeIndex(p["anchor_date"])}), ctx.calendar)
    p["large_stock"] = flags
    p["market_value"] = own
    if not np.isfinite(p[["alpha20_bp", "Y0_bp", "Y1_bp"]].to_numpy(float)).all():
        raise AssertionError(f"{clock}/{sample}: purchase without both outcomes or alpha")
    if not (p["n_est"] >= fm.MIN_ESTIMATION_OBS).all():
        raise AssertionError(f"{clock}/{sample}: fit with fewer than 60 observations")
    return p.reset_index(drop=True)


def exclusion_counts(ctx, clock: str, spec: sp.Spec) -> dict:
    """Purchases of periods A/B in the frame, and where the sample loses the rest (same order as to_panel)."""
    f = ctx.frames[clock]
    buy = f[(f["direction"] == "purchase") & f["period"].isin(["A", "B"])]
    y = pd.to_numeric(buy["y_h20"], errors="coerce")
    out = {"purchases_periods_AB": int(len(buy)), "nonfinite_event_study_outcome": int((~np.isfinite(y)).sum())}
    ok = buy[np.isfinite(y)]
    out["dup_all_lines_dropped"] = int(ok["dup_all_lines"].astype(bool).sum()) if spec.dup == "default" else 0
    ok = ok[~ok["dup_all_lines"].astype(bool)] if spec.dup == "default" else ok
    panel = ctx.panel(spec)
    out["after_finite_and_duplicates"] = int(len(ok))
    out["dropped_by_household_eligibility"] = int(len(ok) - len(panel))
    out["sample_purchases"] = int(len(panel))
    out["sample_households"] = int(panel["household_id"].nunique())
    out["households_before_eligibility"] = int(ok["household_id"].nunique())
    return out


def reason_counts(ctx, clock: str, ext: pd.DataFrame) -> dict:
    f = ctx.frames[clock]
    buy = f[(f["direction"] == "purchase") & f["period"].isin(["A", "B"])][["permno", "anchor_date"]].copy()
    buy["anchor_idx"] = ctx.calendar.get_indexer(pd.DatetimeIndex(buy["anchor_date"]))
    j = buy.merge(ext[["permno", "anchor_idx", "reason_h20"]], on=["permno", "anchor_idx"], how="left")
    return {str(k): int(v) for k, v in j["reason_h20"].astype(str).replace("nan", "not_in_cache").value_counts().items()}


def continuation_purchases(panels: dict[str, pd.DataFrame], clock: str) -> tuple[pd.DataFrame, dict]:
    rows, cuts = [], {}
    for sample, p in panels.items():
        out = p[["sample", "event_id", "household_index", "entry_block", "period", "alpha20_bp", "Y0_bp", "Y1_bp", "large_stock"]].copy()
        y0c, y1c = ctx_cut[(clock, "ff5mom_noalpha_h20")], ctx_cut[(clock, "ff5mom_h20")]
        for col in ("x_low", "x_high", "Y0_low", "Y0_high", "Y1_low", "Y1_high"):
            out[col] = np.nan
        for per in ("A", "B"):
            m = out["period"] == per
            lo, hi = np.percentile(out.loc[m, "alpha20_bp"].to_numpy(float), [1, 99])
            out.loc[m, "x_low"], out.loc[m, "x_high"] = lo, hi
            out.loc[m, "Y0_low"], out.loc[m, "Y0_high"] = y0c[per]
            out.loc[m, "Y1_low"], out.loc[m, "Y1_high"] = y1c[per]
            cuts[f"{sample}/{per}"] = {"x_low_bp": float(lo), "x_high_bp": float(hi), "Y0_low_bp": y0c[per][0], "Y0_high_bp": y0c[per][1],
                                       "Y1_low_bp": y1c[per][0], "Y1_high_bp": y1c[per][1], "purchases": int(m.sum())}
        rows.append(out)
    cols = ["sample", "event_id", "household_index", "entry_block", "period", "alpha20_bp", "Y0_bp", "Y1_bp", "x_low", "x_high", "Y0_low",
            "Y0_high", "Y1_low", "Y1_high", "large_stock"]
    return pd.concat(rows, ignore_index=True)[cols], cuts


ctx_cut: dict = {}


def security_grid(ctx, ext: pd.DataFrame, mv: pd.DataFrame, log) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every 20th session from the first 2015 session through 2025; all universe securities with an ok 20-session cell."""
    cal = ctx.calendar
    p0 = int(cal.searchsorted(GRID_START))
    grid = np.arange(p0, int(cal.searchsorted(GRID_END, side="right")), GRID_STEP)
    if cal[grid[-1]] > GRID_END or cal[grid[0]].year != 2015:
        raise AssertionError("grid outside 2015-2025")
    universe = panel_mod.universe_permnos()
    cells = ext.loc[ext["anchor_idx"].isin(grid), ["permno", "anchor_idx", "alpha_daily", "n_est", "y0_h20", "y1_h20", "reason_h20"]].copy()
    if not np.array_equal(np.sort(cells["permno"].unique()), np.sort(universe)):
        raise AssertionError("grid cells do not cover the universe")
    if len(cells) != len(grid) * len(universe):
        raise AssertionError("missing (security, grid date) cells in the cache")
    cells["reason_h20"] = cells["reason_h20"].astype(str)
    cells["date"] = cal[cells["anchor_idx"].to_numpy()]
    # delisting-return rows inside each window (the existing convention keeps a window only if all 20 returns exist)
    daily = panel_mod.load_local_daily()[["permno", "date", "is_delisting_return_row"]]
    dl = daily[daily["is_delisting_return_row"].astype(bool)].copy()
    dl["pos"] = cal.get_indexer(pd.DatetimeIndex(dl["date"]))
    dl_pos = dl.groupby("permno")["pos"].apply(lambda s: np.sort(s.to_numpy()))
    def window_has_delisting(perm, pos):
        a = dl_pos.get(perm)
        if a is None:
            return False
        k = np.searchsorted(a, pos + 1)
        return bool(k < len(a) and a[k] <= pos + GRID_STEP)
    cells["delisting_in_window"] = [window_has_delisting(int(a), int(b)) for a, b in zip(cells["permno"], cells["anchor_idx"])]
    per_date = cells.groupby("anchor_idx").apply(lambda g: pd.Series({
        "formation_date": str(cal[int(g.name)].date()), "universe_securities": len(g), "ok": int((g["reason_h20"] == fm.REASON_OK).sum()),
        **{f"invalid_{r}": int((g["reason_h20"] == r).sum()) for r in (fm.REASON_SHORT_ESTIMATION, fm.REASON_MISSING_RETURN,
                                                                        fm.REASON_FACTOR_GAP, fm.REASON_TRUNCATED)},
        "delisting_in_window_kept": int(((g["reason_h20"] == fm.REASON_OK) & g["delisting_in_window"]).sum()),
        "delisting_in_window_dropped_incomplete": int(((g["reason_h20"] != fm.REASON_OK) & g["delisting_in_window"]).sum())}), include_groups=False)
    other = set(cells["reason_h20"]) - {fm.REASON_OK, fm.REASON_SHORT_ESTIMATION, fm.REASON_MISSING_RETURN, fm.REASON_FACTOR_GAP, fm.REASON_TRUNCATED}
    if other:
        raise AssertionError(f"unexpected reasons {other}")
    ok = cells[cells["reason_h20"] == fm.REASON_OK].copy()
    if not (ok["n_est"] >= fm.MIN_ESTIMATION_OBS).all() or not np.isfinite(ok[["alpha_daily", "y0_h20", "y1_h20"]].to_numpy(float)).all():
        raise AssertionError("ok grid cell without a valid fit")
    flags, own = large_third_flags(mv, ok[["permno", "date"]], cal)
    out = pd.DataFrame({"sample": "traded_securities", "security_id": ok["permno"].astype(np.int64).to_numpy(),
                        "formation_session": ok["anchor_idx"].astype(np.int64).to_numpy(), "formation_date": [str(d.date()) for d in ok["date"]],
                        "fit_end_session": ok["anchor_idx"].astype(np.int64).to_numpy() - 22, "window_end_session": ok["anchor_idx"].astype(np.int64).to_numpy() + 20,
                        "alpha20_bp": cm.H * ok["alpha_daily"].to_numpy(float) * cm.BP, "Y0_bp": ok["y0_h20"].to_numpy(float) * cm.BP,
                        "Y1_bp": ok["y1_h20"].to_numpy(float) * cm.BP, "market_value": own, "large_stock": flags,
                        "n_est": ok["n_est"].astype(np.int64).to_numpy(), "delisting_in_window": ok["delisting_in_window"].astype(int).to_numpy()})
    out = out.sort_values(["formation_session", "security_id"], ignore_index=True)
    rep = {"grid_sessions": int(len(grid)), "first_formation_date": str(cal[grid[0]].date()), "last_formation_date": str(cal[grid[-1]].date()),
           "first_session_position": int(grid[0]), "step_sessions": GRID_STEP, "universe_securities": int(len(universe)),
           "security_dates_total": int(len(cells)), "security_dates_ok": int(len(out)),
           "security_dates_invalid": int(len(cells) - len(out)), "dates_without_any_ok_cell": int((per_date["ok"] == 0).sum()),
           "ok_cells_missing_market_value": int((~np.isfinite(out["market_value"])).sum()),
           "ok_cells_with_delisting_in_window": int(out["delisting_in_window"].sum()),
           "fit_window": "calendar positions formation-273 .. formation-22 (fit_end_session = formation_session - 22), >= 60 finite rows",
           "return_window": "positions formation+1 .. formation+20 (window_end_session = formation_session + 20)"}
    log(f"grid: {rep['grid_sessions']} dates {rep['first_formation_date']}..{rep['last_formation_date']}; {rep['security_dates_ok']:,} ok of {rep['security_dates_total']:,} security-dates")
    return out, per_date.reset_index().rename(columns={"anchor_idx": "formation_session"}), rep


def sales_signed_means(ctx, key: pd.Series, log) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {}
    for clock in ("T", "P"):
        specs = sp.analysis_specs(clock)
        for design in ("sales", "signed_all"):
            s1 = specs[f"robustness/{design}"]
            pans = {}
            for scale in ("raw", "winsorized"):
                for bench, spec in (("Y1", s1), ("Y0", pn.factor_only(s1))):
                    spec = spec if scale == "winsorized" else replace(spec, winsorized=False)
                    pans[(scale, bench)] = ctx.panel(spec)
            base = pans[("raw", "Y1")][["event_id", "household_id", "period"]]
            for k, p in pans.items():
                if not p[["event_id", "household_id", "period"]].reset_index(drop=True).equals(base.reset_index(drop=True)):
                    raise AssertionError(f"{clock}/{design}/{k}: events differ across benchmarks or scales")
            for scale in ("raw", "winsorized"):
                panel = base.copy()
                panel["Y1"] = pans[(scale, "Y1")]["y_h20"].to_numpy(float)
                panel["Y0"] = pans[(scale, "Y0")]["y_h20"].to_numpy(float)
                hh = dc.household_means(panel, ["Y1", "Y0"])
                cm.check_key_covers(key, hh.index)
                rows.append(pd.DataFrame({"clock": clock, "design": design, "scale": scale, "household_index": key.reindex(hh.index.astype(str)).to_numpy(),
                                          "n_A": hh["n_A"].astype(np.int64).to_numpy(), "n_B": hh["n_B"].astype(np.int64).to_numpy(),
                                          "Y1_A": hh["Y1_A"].to_numpy(), "Y1_B": hh["Y1_B"].to_numpy(), "Y0_A": hh["Y0_A"].to_numpy(), "Y0_B": hh["Y0_B"].to_numpy()}))
            cut1 = ctx.cutoffs_for(s1)
            cut0 = ctx.cutoffs_for(pn.factor_only(s1))
            meta[f"{clock}/{design}"] = {"events": int(len(base)), "households": int(base["household_id"].nunique()),
                                         "event_study_cutoffs_bp": cut1, "factor_only_cutoffs_bp": cut0,
                                         "cutoff_rule": "p1/p99 by period of the direction-applied frame (sales negated for signed trades before the cutoffs), "
                                                        "estimators.results.Context.cutoffs_for; winsorization after the sign"}
            log(f"{clock}/{design}: {len(base)} events, {base['household_id'].nunique()} households, identical across benchmarks and scales")
    return pd.concat(rows, ignore_index=True), meta


def spread_counts(key: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    s = rerun.load_signals()
    d, rep = rerun.dedup_filter(s, rerun.duplicate_signal_uids())
    sp_ = d.dropna(subset=["ar_E_h20", "follower_entry_date"])
    ids = sp_[rerun.HOUSEHOLD].astype(str)
    counts = pd.crosstab(ids, sp_["direction"]).reindex(columns=["purchase", "sale"], fill_value=0)
    known = counts.index.isin(key.index)
    new_ids = sorted(counts.index[~known])
    extra = pd.Series(np.arange(NEW_INDEX_BASE + 1, NEW_INDEX_BASE + 1 + len(new_ids), dtype=np.int64), index=pd.Index(new_ids, name="household_id"))
    full = pd.concat([key, extra])
    out = pd.DataFrame({"household_index": full.reindex(counts.index).to_numpy(np.int64), "purchase_signals": counts["purchase"].astype(np.int64).to_numpy(),
                        "sale_signals": counts["sale"].astype(np.int64).to_numpy()}).sort_values("household_index", ignore_index=True)
    meta = {"dedup_report": rep, "signals": int(len(sp_)), "households": int(len(counts)), "purchase_signals": int(counts["purchase"].sum()),
            "sale_signals": int(counts["sale"].sum()), "households_sales_only": int(((counts["purchase"] == 0) & (counts["sale"] > 0)).sum()),
            "households_purchases_only": int(((counts["sale"] == 0) & (counts["purchase"] > 0)).sum()),
            "households_new_index": len(new_ids), "sample_rule": "existing.rerun: primary non-mixed signals, vendor duplicates (dup_primary_all_lines) "
                                                                 "dropped, ar_E_h20 and follower_entry_date present (the TW_P1_E20 spread sample)"}
    return out, extra.reset_index().rename(columns={0: "household_index"}), meta


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    log = cm.logger("exports")
    t0 = time.time()
    ctx, cal, ext, new = pn.load(log)
    key = cm.household_key()
    mv = mvm.load()
    for clock in ("T", "P"):
        for outcome in ("ff5mom_h20", "ff5mom_noalpha_h20"):
            ctx_cut[(clock, outcome)] = ctx.cutoffs(clock, outcome)
    manifest = {"run_id": cm.RUN_ID, "environment": cm.environment(), "outputs": {}, "exclusions": {}, "reason_counts_h20_purchases": {}}
    panels = {}
    for clock in ("T", "P"):
        for sample in ("cohort", "all_purchases"):
            spec = pn.cohort_spec(clock) if sample == "cohort" else pn.all_purchases_spec(clock)
            panels[(clock, sample)] = purchase_panel(ctx, clock, sample, key, mv)
            manifest["exclusions"][f"{clock}/{sample}"] = exclusion_counts(ctx, clock, spec)
        manifest["reason_counts_h20_purchases"][clock] = reason_counts(ctx, clock, ext)
        allp = pd.concat([panels[(clock, "cohort")], panels[(clock, "all_purchases")]], ignore_index=True)
        allp = allp.drop(columns=["household_id"])
        path = cm.RESTRICTED / f"purchase_panel_{clock}.parquet"
        allp.to_parquet(path, index=False)
        manifest["outputs"][f"purchase_panel_{clock}"] = cm.file_entry(path, rows=len(allp))
    cont, cuts = continuation_purchases({"cohort": panels[("T", "cohort")], "all_purchases": panels[("T", "all_purchases")]}, "T")
    manifest["continuation_cutoffs_T"] = cuts
    manifest["large_stock_missing"] = {f"{c}/{s}": int(panels[(c, s)]["large_stock"].isna().sum()) for (c, s) in panels}
    paths = {"continuation_purchases": cm.write_csv(cont, cm.RESTRICTED / "continuation_purchases.csv")}
    coh = panels[("T", "cohort")]
    paths["cohort_purchase_means"] = cm.write_csv(coh[["event_id", "household_index", "entry_block", "Y0_bp", "Y1_bp", "alpha20_bp"]], cm.RESTRICTED / "cohort_purchase_means.csv")
    grid, per_date, grep = security_grid(ctx, ext, mv, log)
    paths["continuation_security_grid"] = cm.write_csv(grid, cm.RESTRICTED / "continuation_security_grid.csv")
    cm.write_csv(per_date, cm.OUT / "grid_date_counts.csv")
    manifest["grid"] = grep
    ss, ss_meta = sales_signed_means(ctx, key, log)
    paths["sales_signed_means"] = cm.write_csv(ss, cm.RESTRICTED / "sales_signed_means.csv")
    manifest["sales_signed"] = ss_meta
    sc, extra, sc_meta = spread_counts(key)
    paths["spread_household_counts"] = cm.write_csv(sc, cm.RESTRICTED / "spread_household_counts.csv")
    cm.write_csv(extra, cm.RESTRICTED / "private" / "spread_household_crosswalk.csv")
    manifest["spread_counts"] = sc_meta
    for k, p in paths.items():
        manifest["outputs"][k] = cm.file_entry(p, rows=len(pd.read_csv(p)))
    manifest["inputs"] = {"fit_cache_extended": cm.file_entry(cm.RESTRICTED / "fit_cache_extended.parquet"), "market_values": cm.file_entry(mvm.PATH),
                          "household_crosswalk": cm.file_entry(cm.CROSSWALK), "pooled_cutoffs_committed": cm.file_entry(pn.evf.POOLED_CSV),
                          "pooled_cutoffs_new": cm.file_entry(pn.POOLED_NEW)}
    manifest["seconds"] = round(time.time() - t0, 1)
    cm.write_json(manifest, MANIFEST)
    log(f"exports written in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
