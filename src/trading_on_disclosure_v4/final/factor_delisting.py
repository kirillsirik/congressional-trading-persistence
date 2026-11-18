"""Consistent full-horizon factor benchmark for delisted purchase windows of the persistence households (closeout
plan section 3e).

For every purchase of a persistence household (baseline and corrected cohorts, transaction clock) whose 20-session
window holds the security's CRSP delisting-return row and whose factor outcome is missing, the path is verified
(finite security returns on every session through the delisting row, none after it, the six factors and the
risk-free rate finite on all 20 sessions, loadings estimable on the event's own 252/21/60 window) and the
outcome completed against the event's own fitted expected return e_t = rf_t + alpha_hat + beta_hat' f_t:
reinvested, Y = G_s^d * G_e^{d:H} - G_e^H; cash, Y = G_s^d - G_e^H; the partial-window comparison G_s^d - G_e^d is
kept under its own label. The terminal payoff enters once (the delisting row's own return). The persistence
covariance with the completed purchases is reported beside the headline for both cohorts; the earlier
market-adjusted substitution stays a labelled diagnostic. Writes data/v4/final/delisting_factor_completion.csv
(path facts without identities) and delisting_factor_estimates.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.final import cumulative as cu
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

OUT_PATHS = fc.OUT_DIR / "delisting_factor_completion.csv"
OUT_EST = fc.OUT_DIR / "delisting_factor_estimates.csv"
H = fc.H
BP = fc.BP


def factor_path(market: w.Market, permno: int, anchor_date, anchor_idx: int) -> dict:
    """Path facts and the three factor-benchmark outcomes (bp) of one delisted window; NaN outcomes when unverified."""
    g = market.grid(permno)
    lo, hi = anchor_idx + 1, anchor_idx + H
    di = int(market.delist[permno])
    k = di - lo
    seg = g[lo:hi + 1]
    rec = {"window_sessions": int(hi - lo + 1), "delisting_row_position": k + 1, "sessions_before_delisting": int(k),
           "gaps_before_delisting": int((~np.isfinite(seg[:k])).sum()), "delisting_return": float(g[di]) if 0 <= k < H else np.nan,
           "finite_after_delisting": int(np.isfinite(seg[k + 1:]).sum()) if 0 <= k < H else -1,
           "factors_finite_sessions": int(np.all(np.isfinite(market.F[lo:hi + 1]), axis=1).sum() if hi < market.n else 0),
           "rf_finite_sessions": int(np.isfinite(market.rf[lo:hi + 1]).sum() if hi < market.n else 0), "loadings_available": False,
           "verified": False, "y_reinvest_factor_bp": np.nan, "y_cash_factor_bp": np.nan, "y_partial_factor_bp": np.nan,
           "expected_growth_after_delisting": np.nan, "alpha_daily": np.nan}
    beta = market.loadings(permno, anchor_date)
    rec["loadings_available"] = beta is not None
    rec["verified"] = (0 <= k < H and hi < market.n and rec["gaps_before_delisting"] == 0 and np.isfinite(rec["delisting_return"]) and rec["finite_after_delisting"] == 0
                       and rec["factors_finite_sessions"] == H and rec["rf_finite_sessions"] == H and beta is not None)
    if rec["verified"]:
        e = w.expected_return_grid(beta, market.F, market.rf)[lo:hi + 1]
        g_s = float(np.prod(1.0 + seg[:k + 1]))
        g_e_to, g_e_after, g_e_full = float(np.prod(1.0 + e[:k + 1])), float(np.prod(1.0 + e[k + 1:])), float(np.prod(1.0 + e))
        rec.update({"y_reinvest_factor_bp": (g_s * g_e_after - g_e_full) * BP, "y_cash_factor_bp": (g_s - g_e_full) * BP,
                    "y_partial_factor_bp": (g_s - g_e_to) * BP, "expected_growth_after_delisting": g_e_after, "alpha_daily": float(beta[0])})
    return rec


def targets(frame: pd.DataFrame, households: set[str], market: w.Market, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Purchases of `households` in periods A and B without a finite factor outcome whose window holds a delisting row."""
    f = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"]) & ~frame["dup_all_lines"].astype(bool)
                  & frame["household_id"].astype(str).isin(households)].copy()
    f["y"] = pd.to_numeric(f["y_h20"], errors="coerce")
    f = f.loc[~np.isfinite(f["y"])].copy()
    f["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(f["anchor_date"]))
    market.load(f["permno"].unique())
    dl = market.delist.reindex(f["permno"].to_numpy()).to_numpy(dtype=float)
    f["has_delisting"] = w.delisting_in_window(f["anchor_idx"].to_numpy(), dl)
    return f.loc[f["has_delisting"]].reset_index(drop=True)


def completed_tau(frame: pd.DataFrame, panel: pd.DataFrame, paths: pd.DataFrame, col: str, cutoffs, calendar: pd.DatetimeIndex) -> dict:
    """The headline covariance with the verified completed purchases appended to the winsorized panel."""
    add = paths.loc[paths["verified"], ["event_id", "household_id", "period", "anchor_date", col]].rename(columns={col: "y_h20"})
    add = add.assign(draw=0, clock=panel["clock"].iloc[0]) if "clock" in panel.columns else add.assign(draw=0)
    add["household_id"] = add["household_id"].astype(str)
    add = sp.winsorize(add, cutoffs)
    p2 = pd.concat([panel[["draw", "event_id", "household_id", "clock", "period", "y_h20", "anchor_date"]], add[["draw", "event_id", "household_id", "clock", "period", "y_h20", "anchor_date"]]], ignore_index=True)
    return cgm.tau2_i1(rs.block_cluster(p2, calendar), "ES", anchor="anchor_blk")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="consistent factor benchmark for the delisted purchases of the persistence households")
    ap.parse_args(argv)
    log = fc.logger("factor_delisting")
    t0 = time.time()
    calendar = pm.v2_trading_calendar()
    market = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    spec = sp.headline_spec("T")
    cuts = evf.pooled_cutoffs("T", spec.outcome)
    frames = {"baseline": evf.load_event_frame("T", calendar=calendar), "corrected": pd.read_parquet(cu.RESTRICTED / "frame_T_corrected.parquet")}
    path_rows, est = [], []
    for label, frame in frames.items():
        panel = sp.to_panel(frame, spec, cuts)
        hh = set(panel["household_id"].astype(str))
        tg = targets(frame, hh, market, calendar)
        recs = []
        for r in tg.itertuples(index=False):
            rec = factor_path(market, int(r.permno), r.anchor_date, int(r.anchor_idx))
            rec.update({"cohort": label, "event_id": r.event_id, "household_id": str(r.household_id), "period": r.period, "anchor_date": r.anchor_date})
            recs.append(rec)
        paths = pd.DataFrame(recs)
        n_ver = int(paths["verified"].sum()) if len(paths) else 0
        r0 = cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")
        est.append(rs.row("T", "delisting_factor", label, "tau2_observed_only", r0["tau2_hat"], "bp2", se_block=r0["var_wx"] * r0["se_b"],
                          n_events=int(r0["n_events_a"] + r0["n_events_b"]), n_households=int(r0["n_households"]), notes="the panel without the delisted windows"))
        est.append(rs.row("T", "delisting_factor", label, "n_delisted_targets", float(len(paths)), "count", notes="persistence-household purchases without a factor outcome whose window holds a delisting row"))
        est.append(rs.row("T", "delisting_factor", label, "n_verified_paths", float(n_ver), "count"))
        if n_ver:
            for col, tag in (("y_reinvest_factor_bp", "tau2_reinvested_at_fitted_expected_return"), ("y_cash_factor_bp", "tau2_proceeds_in_cash"), ("y_partial_factor_bp", "tau2_partial_window_comparison")):
                r = completed_tau(frame, panel, paths, col, cuts, calendar)
                est.append(rs.row("T", "delisting_factor", label, tag, r["tau2_hat"], "bp2", se_block=r["var_wx"] * r["se_b"], n_events=int(r["n_events_a"] + r["n_events_b"]),
                                  n_households=int(r["n_households"]), notes=f"{n_ver} verified delisted purchases completed against the event's own fitted FF5+MOM expected return (intercept included); winsorized at the committed cutoffs"))
            v = paths.loc[paths["verified"]]
            est.append(rs.row("T", "delisting_factor", label, "mean_y_reinvest_factor", float(v["y_reinvest_factor_bp"].mean()), "bp", n_events=n_ver))
            est.append(rs.row("T", "delisting_factor", label, "mean_y_cash_factor", float(v["y_cash_factor_bp"].mean()), "bp", n_events=n_ver))
            est.append(rs.row("T", "delisting_factor", label, "mean_expected_growth_after_delisting", float(v["expected_growth_after_delisting"].mean()), "ratio", n_events=n_ver,
                              notes="benchmark growth after the delisting row under the fitted expected return (1 would mean no reinvestment effect)"))
            est.append(rs.row("T", "delisting_factor", label, "max_abs_reinvest_minus_partial", float((v["y_reinvest_factor_bp"] - v["y_partial_factor_bp"]).abs().max()), "bp", n_events=n_ver))
        log(f"{label}: {len(paths)} delisted targets, {n_ver} verified; " + (f"tau2 {r0['tau2_hat']:.1f} -> reinvested {est[-5]['value']:.1f}" if n_ver else ""))
        path_rows.append(paths)
    paths_all = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    pub = paths_all.drop(columns=["event_id", "household_id", "anchor_date"], errors="ignore")
    if len(pub):
        pub.insert(0, "window_index", np.arange(1, len(pub) + 1))
    fc.write_csv(pub, OUT_PATHS)
    fc.write_rows(est, OUT_EST)
    fc.write_manifest("factor_delisting", [fc.file_entry(pm.daily_parquet_path()), fc.file_entry(cu.RESTRICTED / "frame_T_corrected.parquet"),
                                           fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"),
                                           fc.file_entry(pm.project_root() / "data" / "raw" / "factors" / "factors_daily.csv")],
                      config={"horizon": H, "estimation": "252/21/60", "benchmark": "rf + alpha + beta'f (FF5+MOM, intercept included)"}, extra={"seconds": round(time.time() - t0, 1)})
    pd.set_option("display.width", 250)
    print(pub.round(2).to_string(index=False) if len(pub) else "no delisted targets")
    print(rs.results_frame(est)[["variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
