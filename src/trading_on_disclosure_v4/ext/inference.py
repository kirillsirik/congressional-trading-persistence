"""Inference ladder against overlapping windows and common calendar or security shocks (extension plan
2026-10-19, section 8): the de-duplicated public-clock spread, the transaction-clock purchase mean and the
headline tau2 under household x {day, 20-, 40-, 60-day block, quarter} and household x block x security
clusters; the calendar-time portfolio alternative for the spread; the common-shock-removed outcome; and,
beside each row, the placebo rejection rate already measured for that procedure where one exists.
Writes data/v4/ext/inference_ladder.csv.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v2.inference.cgm3 import cgm_cluster_cov
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.outcomes import panel as panel_mod

ANALYSIS = "inference_ladder"
BP = 1e4
H = 20
BLOCKS = (20, 40, 60)
OUT_CSV = cm.OUT_DIR / "inference_ladder.csv"
LOG = cm.LOG_DIR / "inference.log"
SIZE_V2 = cm.REPO_ROOT / "data" / "v4" / "calibration" / "size_null_v2.csv"


# ---------------------------------------------------------------- cluster ladders

def date_clusters(dates: pd.Series, calendar: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Anchor day, 20/40/60-session blocks of the calendar position, and calendar quarter."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    pos = calendar.get_indexer(d)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} anchors are not calendar sessions")
    out = {"day": d.to_numpy(dtype="datetime64[ns]").astype(np.int64)}
    for L in BLOCKS:
        out[f"block{L}"] = pos // L
    out["quarter"] = (d.year * 4 + (d.month - 1) // 3).to_numpy()
    return out


def clustered(y: np.ndarray, X: np.ndarray, w: np.ndarray | None, clusters: pd.DataFrame, j: int) -> dict:
    """One coefficient with CGM multi-way clusters (WLS through the sqrt-weight transform when w is given)."""
    if w is not None:
        sw = np.sqrt(np.asarray(w, dtype=float))
        y, X = y * sw, X * sw[:, None]
    res = cgm_cluster_cov(y, X, clusters.reset_index(drop=True))
    est, se = float(res.beta[j]), float(np.sqrt(res.cov[j, j]))
    dof = max(int(res.dof), 1)
    t = est / se if se > 0 else np.nan
    return {"estimate": est, "se": se, "t": t, "p_two_sided": float(2 * stats.t.sf(abs(t), df=dof)) if np.isfinite(t) else np.nan, "dof": dof,
            "cluster_counts": dict(res.cluster_counts), "psd_corrected": bool(res.psd_corrected)}


def ladder(y, X, w, household, dates, permno, calendar, j) -> list[tuple[str, dict]]:
    """household x {day, block20, block40, block60, quarter}; household x block20 x security."""
    dc = date_clusters(dates, calendar)
    out = []
    for name, col in dc.items():
        out.append((f"household_x_{name}", clustered(y, X, w, pd.DataFrame({"household": household, name: col}), j)))
    out.append(("household_x_block20_x_security", clustered(y, X, w, pd.DataFrame({"household": household, "block20": dc["block20"], "security": permno}), j)))
    out.append(("household_x_block60_x_security", clustered(y, X, w, pd.DataFrame({"household": household, "block60": dc["block60"], "security": permno}), j)))
    return out


def _row(estimand: str, clock: str, procedure: str, r: dict, unit: str, n: int, n_hh: int, notes: str = "", **kw) -> dict:
    cc = "; ".join(f"{k} {v}" for k, v in r["cluster_counts"].items() if "×" not in k)
    return rs.row(clock, ANALYSIS, estimand, procedure, r["estimate"], unit, se_block=r["se"], n_events=n, n_households=n_hh,
                  notes=f"t {r['t']:.3f}, two-sided p {r['p_two_sided']:.3f}, dof {r['dof']}; clusters {cc}; psd_corrected {r['psd_corrected']}" + (f"; {notes}" if notes else ""), **kw)


# ---------------------------------------------------------------- the spread

def spread_rows(calendar: pd.DatetimeIndex, log) -> list[dict]:
    signals = rerun.load_signals()
    dedup, _ = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    d = dedup.dropna(subset=["ar_E_h20", "follower_entry_date"]).reset_index(drop=True)
    y = d["ar_E_h20"].to_numpy(float) * BP
    purchase = (d["direction"] == "purchase").to_numpy(float)
    X = np.column_stack([np.ones(len(y)), purchase])
    n, n_hh = int(len(d)), int(d["filer_uid"].nunique())
    rows = []
    for proc, r in ladder(y, X, None, d["filer_uid"].to_numpy(), d["follower_entry_date"], d["permno"].to_numpy(), calendar, 1):
        rows.append(_row("public_spread_dedup", "P", proc, r, "bp", n, n_hh, "purchase-minus-sale spread at 20 days, market-adjusted, de-duplicated sample"))
    # common-shock-removed: outcome minus the cross-sectional mean of all de-duplicated signals (both directions) in the same 20-day block
    blk = date_clusters(d["follower_entry_date"], calendar)["block20"]
    y_dm = y - pd.Series(y).groupby(blk).transform("mean").to_numpy()
    for proc, r in ladder(y_dm, X, None, d["filer_uid"].to_numpy(), d["follower_entry_date"], d["permno"].to_numpy(), calendar, 1):
        if proc in ("household_x_block20", "household_x_block20_x_security"):
            rows.append(_row("public_spread_dedup_block_demeaned", "P", proc, r, "bp", n, n_hh, "outcome minus the mean over all de-duplicated signals (both directions) anchored in the same 20-day block"))
    rows += calendar_time_rows(d, calendar, log)
    return rows


def calendar_time_rows(d: pd.DataFrame, calendar: pd.DatetimeIndex, log) -> list[dict]:
    """Daily long-purchase / short-sale portfolio held 20 sessions after entry, equal-weighted over positions within
    each leg, market-adjusted daily returns; mean daily long-minus-short with Newey-West (lag 20) standard errors."""
    daily = panel_mod.load_local_daily()
    permnos = np.sort(d["permno"].unique().astype(int))
    daily = daily[daily["permno"].isin(permnos)]
    bench = panel_mod.load_benchmark().reindex(calendar).to_numpy(float)
    pidx = {int(p): i for i, p in enumerate(permnos)}
    R = np.full((len(permnos), len(calendar)), np.nan)
    ci = calendar.get_indexer(pd.DatetimeIndex(daily["date"]))
    ok = ci >= 0
    rows_idx = pd.Series(pidx).reindex(daily["permno"].to_numpy()[ok].astype(int)).to_numpy(dtype=int)
    R[rows_idx, ci[ok]] = daily["total_ret"].to_numpy(float)[ok]
    AR = R - bench[None, :]
    count = {"purchase": np.zeros_like(R, dtype=np.int32), "sale": np.zeros_like(R, dtype=np.int32)}
    pos = calendar.get_indexer(pd.DatetimeIndex(d["follower_entry_date"]))
    for p, e, dirn in zip(d["permno"].to_numpy().astype(int), pos, d["direction"].to_numpy(), strict=True):
        lo, hi = e + 1, min(e + H, len(calendar) - 1)
        if lo <= hi:
            count[dirn][pidx[p], lo:hi + 1] += 1
    legs = {}
    for dirn, c in count.items():
        c = c.astype(float)
        c[~np.isfinite(AR)] = 0.0
        num = np.nansum(np.where(c > 0, AR * c, 0.0), axis=0)
        den = c.sum(axis=0)
        legs[dirn] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        legs[dirn + "_n"] = den
    both = np.isfinite(legs["purchase"]) & np.isfinite(legs["sale"])
    ls = (legs["purchase"] - legs["sale"])[both] * BP
    t_days = int(both.sum())
    mean, se = nw_mean(ls, H)
    rows = [rs.row("P", ANALYSIS, "public_spread_dedup_calendar_time", "daily_long_minus_short_nw20", mean, "bp_per_day", se_block=se, n_events=t_days, n_households=np.nan,
                   notes=f"mean daily return of the long-purchase / short-sale 20-session portfolio over {t_days} sessions with both legs; Newey-West lag {H}; "
                         f"mean positions per day: long {legs['purchase_n'][both].mean():.1f}, short {legs['sale_n'][both].mean():.1f}; a different estimand from the signal-level spread"),
            rs.row("P", ANALYSIS, "public_spread_dedup_calendar_time", "daily_long_minus_short_x20", mean * H, "bp", se_block=se * H, n_events=t_days, n_households=np.nan,
                   notes="the daily mean and its standard error multiplied by 20 (the holding length), for scale only")]
    for dirn in ("purchase", "sale"):
        v = legs[dirn][np.isfinite(legs[dirn])] * BP
        m, s = nw_mean(v, H)
        rows.append(rs.row("P", ANALYSIS, "public_spread_dedup_calendar_time", f"daily_{dirn}_leg_nw20", m, "bp_per_day", se_block=s, n_events=int(len(v)), n_households=np.nan,
                           notes="mean daily market-adjusted return of the leg; Newey-West lag 20"))
    log(f"calendar-time portfolio: {t_days} sessions with both legs")
    return rows


def nw_mean(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean with a Newey-West (Bartlett, fixed lag) standard error."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, lag + 1):
        wgt = 1.0 - k / (lag + 1.0)
        s += 2.0 * wgt * float(e[k:] @ e[:-k]) / n
    return float(x.mean()), float(np.sqrt(max(s, 0.0) / n))


# ---------------------------------------------------------------- the transaction-clock purchase mean and tau2

def mean_rows(ctx: rs.Context, clock: str, spec: sp.Spec, estimand: str, note: str) -> list[dict]:
    panel = ctx.panel(spec)
    y = panel["y_h20"].to_numpy(float)
    X = np.ones((len(y), 1))
    rows = []
    for proc, r in ladder(y, X, None, panel["household_id"].to_numpy(), panel["anchor_date"], panel["permno"].to_numpy(), ctx.calendar, 0):
        rows.append(_row(estimand, clock, proc, r, "bp", int(len(panel)), int(panel["household_id"].nunique()), note))
    return rows


def tau2_with_clusters(panel: pd.DataFrame, calendar: pd.DatetimeIndex, weights: str = "ES") -> list[tuple[str, dict, float]]:
    """The headline WLS slope with each clustering; tau2 = slope x Var_w(x); returns (procedure, result, var_wx)."""
    table = tau_mod.household_table(panel, weights)
    hh = pd.DataFrame({"household_id": table.household_id, "x": table.x, "log_n_a": table.log_n_a, "event_weight": table.w / table.n_b})
    b = panel.loc[panel["period"] == "B", ["household_id", "y_h20", "anchor_date", "permno"]].merge(hh, on="household_id", how="inner")
    X = np.column_stack([np.ones(len(b)), b["x"].to_numpy(float), b["log_n_a"].to_numpy(float)])
    v = tau_mod.var_w(table.x, table.w)
    out = []
    for proc, r in ladder(b["y_h20"].to_numpy(float), X, b["event_weight"].to_numpy(float), b["household_id"].to_numpy(), b["anchor_date"], b["permno"].to_numpy(), calendar, 1):
        out.append((proc, r, v))
    return out


def tau2_rows(ctx: rs.Context, clock: str) -> list[dict]:
    spec = sp.headline_spec(clock)
    panel = ctx.panel(spec)
    n, n_hh = int(len(panel)), int(panel["household_id"].nunique())
    rows = []
    for proc, r, v in tau2_with_clusters(panel, ctx.calendar):
        rr = {**r, "estimate": r["estimate"] * v, "se": r["se"] * v}
        rows.append(_row("tau2_headline", clock, proc, rr, "bp2", n, n_hh, "slope x Var_w(x); event-share weights, winsorized outcome; one-sided p = t p / 2 when t > 0"))
    return rows


def demeaned_tau2_rows(ctx: rs.Context, clock: str, log, n_placebo: int) -> list[dict]:
    """tau2 on the outcome minus the mean of all purchases (all households, periods A and B) anchored in the same
    20-day block, winsorized at the demeaned outcome's own p1/p99; block clusters; centered-placebo p with the same
    demeaning on every draw's shifted anchors."""
    frame = ctx.frames[clock]
    base = sp.Spec(f"{ANALYSIS}/demeaned", clock, "tau", winsorized=False)
    raw = sp.to_panel(frame, base, None)  # purchases, dedup, periods A/B, headline eligibility, raw outcome
    allp = frame.loc[(frame["direction"] == "purchase") & frame["period"].isin(["A", "B"]) & ~frame["dup_all_lines"].astype(bool)]
    allp = allp.loc[np.isfinite(pd.to_numeric(allp["y_h20"], errors="coerce"))]
    blk_all = date_clusters(allp["anchor_date"], ctx.calendar)["block20"]
    block_mean = pd.Series(pd.to_numeric(allp["y_h20"]).to_numpy(float)).groupby(blk_all).mean()
    blk = date_clusters(raw["anchor_date"], ctx.calendar)["block20"]
    raw = raw.assign(y_h20=raw["y_h20"].to_numpy(float) - block_mean.reindex(blk).to_numpy())
    cuts = evf.sample_cutoffs(raw["y_h20"].to_numpy(), raw["period"].to_numpy())
    panel = sp.winsorize(raw, cuts)
    pb = rs.block_cluster(panel, ctx.calendar)
    from trading_on_disclosure_v4.inference import cgm

    r = cgm.tau2_i1(pb, "ES", anchor="anchor_blk")
    actual = r["tau2_hat"]
    # placebo: the same construction on each centered draw
    vals = []
    meta = evf.event_metadata(clock)
    t0 = time.time()
    for _, batch in evf.placebo_event_frames(clock, "N2", meta, dup="default", batches=True):
        for draw, g in batch.groupby("draw", sort=True):
            if draw >= n_placebo:
                continue
            g = g.reset_index(drop=True)
            gp = g.loc[np.isfinite(pd.to_numeric(g["y_h20"], errors="coerce")) & ~g["dup_all_lines"].astype(bool) & g["period"].isin(["A", "B"])]
            bm = pd.Series(gp["y_h20"].to_numpy(float)).groupby(date_clusters(gp["anchor_date"], ctx.calendar)["block20"]).mean()
            p = sp.to_panel(g, base, None, draw=int(draw))
            if p.empty:
                vals.append(np.nan)
                continue
            p = p.assign(y_h20=p["y_h20"].to_numpy(float) - bm.reindex(date_clusters(p["anchor_date"], ctx.calendar)["block20"]).to_numpy())
            p = sp.winsorize(p, evf.sample_cutoffs(p["y_h20"].to_numpy(), p["period"].to_numpy()))
            try:
                vals.append(tau_mod.tau2_household_covariance(tau_mod.household_table(p, "ES")))
            except ValueError:
                vals.append(np.nan)
        log(f"demeaned tau2 placebo {clock}: {len(vals)} draws, {time.time() - t0:.0f} s")
    pl = placebo_p_value(actual, np.asarray(vals), "greater")
    n, n_hh = int(len(panel)), int(panel["household_id"].nunique())
    return [rs.row(clock, ANALYSIS, "tau2_headline_block_demeaned", "household_x_block20", actual, "bp2", placebo_p=pl["p"], se_block=r["var_wx"] * r["se_b"],
                   n_events=n, n_households=n_hh,
                   notes=f"outcome minus the mean of all purchases anchored in the same 20-day block, winsorized at its own p1/p99; t {r['t_stat']:.3f}; "
                         f"centered-placebo p over {pl['n_null']} draws with the same demeaning; placebo mean {pl['null_mean']:.1f}, 95th percentile {pl['null_q95']:.1f}"),
            rs.row(clock, ANALYSIS, "tau2_headline_block_demeaned", "tau", tau_mod.signed_root(actual), "bp", n_events=n, n_households=n_hh, notes="signed root")]


# ---------------------------------------------------------------- measured placebo rejection rates

def size_rows() -> list[dict]:
    """Restate the measured rejection rates of the procedures the ladder uses (size_null_v2.csv), so the reader sees
    which standard errors were checked on placebo data."""
    s = pd.read_csv(SIZE_V2)
    keep = s[s["test"].isin(["T1", "T2"]) & s["inference"].isin(["I1", "I1blk20", "I1blk60", "I1cal", "I2b"]) & s["weights"].isin(["none", "ES"])]
    rows = []
    for r in keep.itertuples(index=False):
        rows.append(rs.row(r.clock, ANALYSIS, f"placebo_size_{r.test}_{r.variant}", f"{r.inference}_{r.sidedness.replace('-', '_')}", float(r.reject_rate), "share",
                           boot_lo=float(r.coverage) if pd.notna(r.coverage) else np.nan, n_events=int(r.n_panels),
                           notes=f"measured rejection rate at nominal 5% on {int(r.n_panels)} centered placebo datasets (size_null_v2.csv); coverage of zero in boot_lo; plan procedure {bool(r.plan_procedure)}"))
    return rows


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="inference ladder")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-placebo", type=int, default=2000)
    ap.add_argument("--skip-calendar-time", action="store_true")
    a = ap.parse_args(argv)
    clocks = tuple(a.clocks.split(","))
    log = lambda m: cm.log_line(LOG, m)  # noqa: E731
    t0 = time.time()
    ctx, calendar = cm.load_headline_context(clocks, with_placebo=False)
    rows: list[dict] = []
    if "P" in clocks:
        if a.skip_calendar_time:
            global calendar_time_rows
            calendar_time_rows = lambda d, c, lg: []  # noqa: E731
        rows += spread_rows(calendar, log)
    for c in clocks:
        specs = sp.analysis_specs(c)
        rows += mean_rows(ctx, c, specs["purchase_mean"], "purchase_mean_all_households", "raw outcome, all households, periods A + B (the paper's Section 4.3 mean)")
        pers = replace(sp.headline_spec(c), name="purchase_mean_persistence", kind="mean", winsorized=False)
        rows += mean_rows(ctx, c, pers, "purchase_mean_persistence_sample", "raw outcome, the persistence households' purchases, periods A + B")
        rows += tau2_rows(ctx, c)
        rows += demeaned_tau2_rows(ctx, c, log, a.n_placebo)
    rows += size_rows()
    out = cm.write_rows(rows, OUT_CSV)
    cm.write_manifest("inference", [cm.file_entry(SIZE_V2), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv"),
                                    cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache_manifest.json")],
                      {"seconds": round(time.time() - t0, 1)})
    log(f"written {OUT_CSV} ({len(out)} rows) in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
