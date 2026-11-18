"""Delisting paths verified and the terminal payoff counted once (extension plan round 3): a path
ledger for every window whose security's delisting-return row falls inside its 20 sessions and whose outcome
is missing, the three outcomes on verified paths (proceeds in cash; proceeds reinvested at the benchmark, the
genuine full-horizon outcome; the partial-window comparison under its own label), and the estimates under
each. Writes data/v4/ext/delisting_path_ledger.csv, delisting_outcomes.csv, missing_return_sensitivity_delisting.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

H = 20
BP = 1e4
OUT_LEDGER = cm.OUT_DIR / "delisting_path_ledger.csv"
OUT_OUTCOMES = cm.OUT_DIR / "delisting_outcomes.csv"
OUT_SENS = cm.OUT_DIR / "missing_return_sensitivity_delisting.csv"
RESTRICTED = cm.RESTRICTED_EXT / "delisting"
LOG = cm.LOG_DIR / "delisting.log"
T_CASH = "(ix) delisted proceeds in cash, verified paths"
T_REINVEST = "(x) delisted proceeds reinvested at the benchmark, verified paths"
T_PARTIAL = "(xi) partial-window comparison, verified paths (as treatments ii and viii)"


def path_ledger(frame: pd.DataFrame, market: w.Market, clock: str) -> pd.DataFrame:
    """One line per window with a delisting-return row inside it and no outcome: the path facts and the three
    outcomes on verified paths (NaN otherwise)."""
    rows = []
    for r in frame.itertuples(index=False):
        p = int(r.permno)
        g = market.grid(p)
        lo, hi = int(r.anchor_idx) + 1, int(r.anchor_idx) + H
        di = int(market.delist[p])
        seg, b = g[lo:hi + 1], market.bench[lo:hi + 1]
        k = di - lo  # 0-based position of the delisting row inside the window
        before, after = seg[:k], seg[k + 1:]
        rec = {"clock": clock, "event_id": r.event_id, "direction": r.direction, "period": r.period, "window_sessions": int(hi - lo + 1),
               "delisting_row_position": k + 1, "sessions_before_delisting": int(k), "gaps_before_delisting": int((~np.isfinite(before)).sum()),
               "delisting_return": float(g[di]), "delisting_return_finite": bool(np.isfinite(g[di])), "finite_after_delisting": int(np.isfinite(after).sum()),
               "benchmark_finite_sessions": int(np.isfinite(b).sum()), "verified": False, "y_cash_bp": np.nan, "y_reinvest_bp": np.nan, "y_partial_bp": np.nan,
               "benchmark_growth_after_delisting": np.nan}
        rec["verified"] = (rec["gaps_before_delisting"] == 0 and rec["delisting_return_finite"] and rec["finite_after_delisting"] == 0
                           and rec["benchmark_finite_sessions"] == H and hi - lo + 1 == H)
        if rec["verified"]:
            to_dl = float(np.prod(1.0 + seg[:k + 1]))  # the delisting return enters here, once, as the row's own return
            b_to, b_after, b_full = float(np.prod(1.0 + b[:k + 1])), float(np.prod(1.0 + b[k + 1:])), float(np.prod(1.0 + b))
            rec["y_cash_bp"] = (to_dl - 1.0 - (b_full - 1.0)) * BP
            rec["y_reinvest_bp"] = (to_dl * b_after - b_full) * BP
            rec["y_partial_bp"] = (to_dl - b_to) * BP
            rec["benchmark_growth_after_delisting"] = b_after
        rows.append(rec)
    return pd.DataFrame(rows)


def spread_rows(u: pd.DataFrame, ledger: pd.DataFrame, completed: pd.Series | None) -> list[dict]:
    """The de-duplicated public-clock spread under the three verified-path treatments, with and without the July completion."""
    y0 = u["y_market_adjusted"].to_numpy(float)
    out = []
    base = w.spread_estimate(u, y0)
    out.append(w.sens_row("public_spread_dedup", "P", w.T_OBSERVED, base["estimate"], "bp", base["se"], base["n"], 0, np.nan, np.nan, "the record"))
    led = ledger.loc[ledger["verified"]].set_index("event_id")
    for july in (False, True):
        y_base = y0.copy()
        n_july = 0
        if july and completed is not None:
            f = u["event_id"].map(completed)
            m = ~np.isfinite(y_base) & f.notna().to_numpy()
            y_base[m] = f.to_numpy(dtype=float)[m]
            n_july = int(m.sum())
        for label, col in ((T_CASH, "y_cash_bp"), (T_REINVEST, "y_reinvest_bp"), (T_PARTIAL, "y_partial_bp")):
            y = y_base.copy()
            f = u["event_id"].map(led[col])
            m = ~np.isfinite(y) & f.notna().to_numpy()
            y[m] = f.to_numpy(dtype=float)[m]
            r = w.spread_estimate(u, y)
            out.append(w.sens_row("public_spread_dedup", "P", label + (" + July completion" if july else ""), r["estimate"], "bp", r["se"], r["n"],
                                  int(m.sum() + n_july), np.nan, np.nan,
                                  f"{int(m.sum())} verified delisted windows imputed" + (f"; {n_july} endpoint-censored windows completed on the July 2026 slice" if july else "")))
    return out


def transaction_rows(ctx: rs.Context, calendar: pd.DatetimeIndex, market: w.Market, ledger_t: pd.DataFrame) -> list[dict]:
    """The persistence sample's purchase mean and tau2 with its delisted purchases imputed on verified paths (factor outcome
    stays the headline's; the imputed value is the market-adjusted path outcome, as treatment (ii) did)."""
    spec = sp.headline_spec("T")
    panel = ctx.panel(spec)
    raw_spec = sp.Spec(**{**spec.__dict__, "winsorized": False})
    frame = ctx.frames["T"]
    hh = set(panel["household_id"])
    targets = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"]) & frame["household_id"].astype(str).isin(hh)
                        & ~np.isfinite(pd.to_numeric(frame["y_h20"], errors="coerce"))]
    led = ledger_t.loc[ledger_t["verified"]].set_index("event_id")
    hit = targets["event_id"].isin(led.index)
    out = []
    n_hit = int(hit.sum())
    raw_panel = ctx.panel(raw_spec)
    base_mean = rs.block_cluster(raw_panel, calendar)
    from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt

    r0 = tt.t1_clustered_mean(base_mean, anchor="anchor_blk")
    out.append(rs.row("T", "delisting", "purchase_mean_persistence", "observed_only", r0["estimate"], "bp", se_block=r0["se"], n_events=r0["n_events"],
                      n_households=r0["n_households"], notes="raw outcome, the persistence households' purchases"))
    for label, col in ((T_CASH, "y_cash_bp"), (T_REINVEST, "y_reinvest_bp"), (T_PARTIAL, "y_partial_bp")):
        add = targets.loc[hit, ["event_id", "household_id", "period", "anchor_date"]].copy()
        add["y_h20"] = add["event_id"].map(led[col]).to_numpy(dtype=float)
        add["household_id"] = add["household_id"].astype(str)
        pr = pd.concat([raw_panel[["event_id", "household_id", "period", "anchor_date", "y_h20"]], add], ignore_index=True)
        r = tt.t1_clustered_mean(rs.block_cluster(pr, calendar), anchor="anchor_blk")
        out.append(rs.row("T", "delisting", "purchase_mean_persistence", label, r["estimate"], "bp", se_block=r["se"], n_events=r["n_events"],
                          n_households=r["n_households"], notes=f"{n_hit} delisted purchases of the persistence households imputed on verified paths (market-adjusted path outcome)"))
        pw = sp.winsorize(pr, ctx.cutoffs_for(spec))
        t = cgm.tau2_i1(rs.block_cluster(pw, calendar), "ES", anchor="anchor_blk")
        out.append(rs.row("T", "delisting", "tau2_headline", label, t["tau2_hat"], "bp2", se_block=t["var_wx"] * t["se_b"], n_events=int(t["n_events_a"] + t["n_events_b"]),
                          n_households=int(t["n_households"]), notes=f"{n_hit} delisted purchases imputed; winsorized at the committed cutoffs; no placebo p (the stored placebo datasets hold no imputed events)"))
    t0 = cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")
    out.append(rs.row("T", "delisting", "tau2_headline", "observed_only", t0["tau2_hat"], "bp2", se_block=t0["var_wx"] * t0["se_b"], n_events=int(t0["n_events_a"] + t0["n_events_b"]),
                      n_households=int(t0["n_households"]), notes="the headline"))
    out.append(rs.row("T", "delisting", "purchase_mean_persistence", "n_delisted_targets", n_hit, "count", notes="persistence-household purchases without an outcome whose window holds a verified delisting path"))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="delisting paths verified; terminal payoff counted once")
    ap.parse_args(argv)
    log = lambda m: cm.log_line(LOG, m)  # noqa: E731
    t0 = time.time()
    calendar = pm.v2_trading_calendar()
    market = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    u, _ = w.public_universe(calendar)
    u["blk"] = u["anchor_idx"] // 20
    market.load(u["permno"].unique())
    yv = u["y_market_adjusted"].to_numpy(float)
    dl = market.delist.reindex(u["permno"].to_numpy()).to_numpy(float)
    has = w.delisting_in_window(u["anchor_idx"].to_numpy(), dl)
    led_p = path_ledger(u.loc[has & ~np.isfinite(yv)], market, "P")
    tu = w.transaction_universe(calendar)
    tu = tu.loc[tu["direction"].isin(["purchase", "sale"])]
    market.load(tu["permno"].unique())
    dlt = market.delist.reindex(tu["permno"].to_numpy()).to_numpy(float)
    hast = w.delisting_in_window(tu["anchor_idx"].to_numpy(), dlt)
    yt = tu["y_market_adjusted"].to_numpy(float)
    led_t = path_ledger(tu.loc[hast & ~np.isfinite(yt)], market, "T")
    ledger = pd.concat([led_p, led_t], ignore_index=True)
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(RESTRICTED / "path_ledger_rows.parquet", index=False)
    pub = ledger.drop(columns=["event_id"])
    pub.insert(1, "window_index", np.arange(1, len(pub) + 1))
    pub.to_csv(OUT_LEDGER, index=False, lineterminator="\n")
    # outcome summary by clock and direction
    summ = []
    for (clock, dirn), g in ledger.groupby(["clock", "direction"]):
        v = g.loc[g["verified"]]
        summ.append({"clock": clock, "direction": dirn, "windows": int(len(g)), "verified": int(len(v)), "unverified": int((~g["verified"]).sum()),
                     "delisting_return_mean": float(g["delisting_return"].mean()), "delisting_return_median": float(g["delisting_return"].median()),
                     "sessions_before_delisting_mean": float(g["sessions_before_delisting"].mean()),
                     "y_cash_mean_bp": float(v["y_cash_bp"].mean()) if len(v) else np.nan, "y_reinvest_mean_bp": float(v["y_reinvest_bp"].mean()) if len(v) else np.nan,
                     "y_partial_mean_bp": float(v["y_partial_bp"].mean()) if len(v) else np.nan,
                     "reinvest_minus_partial_mean_bp": float((v["y_reinvest_bp"] - v["y_partial_bp"]).mean()) if len(v) else np.nan,
                     "reinvest_minus_partial_max_abs_bp": float((v["y_reinvest_bp"] - v["y_partial_bp"]).abs().max()) if len(v) else np.nan,
                     "benchmark_growth_after_delisting_mean": float(v["benchmark_growth_after_delisting"].mean()) if len(v) else np.nan})
    for clock in ("P", "T"):
        g = ledger[ledger["clock"] == clock]
        v = g.loc[g["verified"]]
        summ.append({"clock": clock, "direction": "all", "windows": int(len(g)), "verified": int(len(v)), "unverified": int((~g["verified"]).sum()),
                     "delisting_return_mean": float(g["delisting_return"].mean()), "delisting_return_median": float(g["delisting_return"].median()),
                     "sessions_before_delisting_mean": float(g["sessions_before_delisting"].mean()),
                     "y_cash_mean_bp": float(v["y_cash_bp"].mean()) if len(v) else np.nan, "y_reinvest_mean_bp": float(v["y_reinvest_bp"].mean()) if len(v) else np.nan,
                     "y_partial_mean_bp": float(v["y_partial_bp"].mean()) if len(v) else np.nan,
                     "reinvest_minus_partial_mean_bp": float((v["y_reinvest_bp"] - v["y_partial_bp"]).mean()) if len(v) else np.nan,
                     "reinvest_minus_partial_max_abs_bp": float((v["y_reinvest_bp"] - v["y_partial_bp"]).abs().max()) if len(v) else np.nan,
                     "benchmark_growth_after_delisting_mean": float(v["benchmark_growth_after_delisting"].mean()) if len(v) else np.nan})
    pd.DataFrame(summ).to_csv(OUT_OUTCOMES, index=False, lineterminator="\n")
    # the spread, with and without the July completion
    completed = None
    cpath = cm.RESTRICTED_EXT / "endpoint" / "completed_windows.parquet"
    if cpath.exists():
        c = pd.read_parquet(cpath)
        completed = c.loc[c["completed"]].set_index("event_id")["y_completed_bp"]
    sens = spread_rows(u, led_p, completed)
    # the transaction clock's persistence sample
    ctx, _ = cm.load_headline_context(("T",), with_placebo=False)
    trows = transaction_rows(ctx, calendar, market, led_t)
    sens_df = pd.DataFrame(sens)[w.SENS_COLUMNS]
    t_df = rs.results_frame(trows)
    sens_df.to_csv(OUT_SENS, index=False, lineterminator="\n")
    t_df.to_csv(OUT_SENS.with_name("delisting_transaction_clock.csv"), index=False, lineterminator="\n")
    cm.write_manifest("delisting", [cm.file_entry(pm.daily_parquet_path()), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv"),
                                    cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet")], {"seconds": round(time.time() - t0, 1)})
    log(f"ledger {len(ledger)} windows ({int(ledger['verified'].sum())} verified); written in {time.time() - t0:.0f} s")
    print(pd.DataFrame(summ).round(2).to_string(index=False))
    print(sens_df[["treatment", "estimate", "se_block", "n_used", "n_imputed"]].round(3).to_string(index=False))
    print(t_df[["analysis", "variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
