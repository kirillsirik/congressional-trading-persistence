"""Benchmark decomposition of the persistence sign reversal (extension plan 2026-10-19, section 5).

On the headline persistence panel the estimator tau2(a, b) = Cov_w(residualized period-A mean of a,
period-B mean of b) is bilinear, so with Y1 = Yx + D (D = Y1 - Yx) the headline splits exactly into
base + adjustment + two cross-terms on raw outcomes. Alternatives: the cache's no-intercept outcome
(loadings kept, alpha omitted), loadings re-fitted without an intercept (computed here per event),
and the market-adjusted outcome. Writes data/v4/ext/decomposition.csv and decomposition_households.csv.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.inference.placebo import placebo_p_value
from trading_on_disclosure_v4.outcomes import panel as panel_mod

ANALYSIS = "decomposition"
BP = 1e4
H = 20
COLS = {"Y1": "y_h20", "Y0": "y_noalpha_h20", "Ym": "y_mkt_h20"}
CUTOFF_OUTCOME = {"Y1": "ff5mom_h20", "Y0": "ff5mom_noalpha_h20", "Ym": "mkt_h20"}
ALTERNATIVES = {"no_alpha_omit": "Y0", "no_alpha_refit": "Y0r", "market_adjusted": "Ym"}
TERMS = ("total", "base", "adj", "cross_base_adj", "cross_adj_base")
OUT_CSV = cm.OUT_DIR / "decomposition.csv"
HH_CSV = cm.OUT_DIR / "decomposition_households.csv"
LOG = cm.LOG_DIR / "decompose.log"


# ---------------------------------------------------------------- the bilinear estimator

def household_means(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per household: n_A, n_B and the period means of every column (households present in both periods)."""
    g = panel.groupby(["household_id", "period"])
    n = g.size().unstack("period")
    means = g[cols].mean().unstack("period")
    keep = n.index[(n.get("A", 0) > 0) & (n.get("B", 0) > 0)]
    out = pd.DataFrame({"n_A": n.loc[keep, "A"].astype(float), "n_B": n.loc[keep, "B"].astype(float)}, index=keep)
    for c in cols:
        out[f"{c}_A"] = means.loc[keep, (c, "A")].to_numpy(dtype=float)
        out[f"{c}_B"] = means.loc[keep, (c, "B")].to_numpy(dtype=float)
    return out


def residualize(ybar_a: np.ndarray, n_a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """w-weighted residual of the period-A mean on [1, log(1 + n_A)] (the headline's activity control)."""
    X = tau_mod.activity_design(n_a)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], ybar_a * sw, rcond=None)
    return ybar_a - X @ beta


def tau2_bilinear(x_a: np.ndarray, ybar_b: np.ndarray, w: np.ndarray) -> float:
    """Cov_w(x_a, Ybar_b) = sum w x Ybar_B / sum w: the headline estimand as a bilinear form."""
    return float(np.sum(w * x_a * ybar_b) / np.sum(w))


def terms(hh: pd.DataFrame, y1: str, yx: str, w: np.ndarray) -> dict[str, float]:
    """total = base + adj + cross_base_adj + cross_adj_base for Y1 = Yx + D on household rows."""
    n_a = hh["n_A"].to_numpy()
    d_a = hh[f"{y1}_A"].to_numpy() - hh[f"{yx}_A"].to_numpy()
    d_b = hh[f"{y1}_B"].to_numpy() - hh[f"{yx}_B"].to_numpy()
    x1 = residualize(hh[f"{y1}_A"].to_numpy(), n_a, w)
    xx = residualize(hh[f"{yx}_A"].to_numpy(), n_a, w)
    xd = residualize(d_a, n_a, w)
    out = {"total": tau2_bilinear(x1, hh[f"{y1}_B"].to_numpy(), w), "base": tau2_bilinear(xx, hh[f"{yx}_B"].to_numpy(), w),
           "adj": tau2_bilinear(xd, d_b, w), "cross_base_adj": tau2_bilinear(xx, d_b, w),
           "cross_adj_base": tau2_bilinear(xd, hh[f"{yx}_B"].to_numpy(), w)}
    out["identity_gap"] = out["total"] - (out["base"] + out["adj"] + out["cross_base_adj"] + out["cross_adj_base"])
    out["corr_D_AB"] = tau_mod.corr_w(d_a, d_b, w)
    out["corr_xD_x1"] = tau_mod.corr_w(xd, x1, w)
    out["mean_D_A"] = float(np.sum(w * d_a) / np.sum(w))
    out["mean_D_B"] = float(np.sum(w * d_b) / np.sum(w))
    return out


# ---------------------------------------------------------------- outcomes on the headline events

def multi_outcome_panel(ctx: rs.Context, clock: str) -> pd.DataFrame:
    """The headline panel's events and households (eligibility on the finite headline outcome) with the raw
    cache outcomes Y1, Y0, Ym in bp attached by event id."""
    spec = sp.headline_spec(clock)
    raw = sp.to_panel(ctx.frames[clock], replace(spec, winsorized=False), None)
    frame = ctx.frames[clock].set_index("event_id")
    out = raw[["event_id", "household_id", "period", "anchor_date", "permno"]].copy()
    for k, c in COLS.items():
        out[k] = pd.to_numeric(frame.loc[out["event_id"], c], errors="coerce").to_numpy(dtype=float)
    out.attrs["n_households_eligible"] = raw.attrs["n_households_eligible"]
    return out


def refit_outcomes(events: pd.DataFrame, calendar: pd.DatetimeIndex, log) -> pd.DataFrame:
    """Per event: the with-intercept outcome (to verify the cache), the intercept-omitted outcome, the outcome
    with loadings re-fitted without an intercept, the daily alpha and the three expected-return legs (bp)."""
    daily = panel_mod.load_local_daily()
    daily = daily[daily["permno"].isin(events["permno"].unique())]
    factors = panel_mod.load_factors()
    F = factors[list(panel_mod.FACTOR_COLS)].reindex(calendar).to_numpy(float)
    rf = factors["rf"].reindex(calendar).to_numpy(float)
    pos_all = calendar.get_indexer(pd.DatetimeIndex(events["anchor_date"]))
    if (pos_all < 0).any():
        raise ValueError("anchor not on the calendar")
    stocks = {int(p): panel_mod.align_to_calendar(g.set_index("date")["total_ret"], calendar) for p, g in daily.groupby("permno")}
    rows = []
    t0 = time.time()
    for i, (permno, pos) in enumerate(zip(events["permno"].to_numpy(), pos_all, strict=True)):
        r = stocks.get(int(permno))
        rec = {"event_id": events["event_id"].iat[i], "Y1_refit": np.nan, "Y0_refit": np.nan, "Y0r": np.nan, "alpha_daily": np.nan,
               "E1": np.nan, "E0": np.nan, "E0r": np.nan, "R": np.nan, "n_est": 0}
        if r is not None:
            est_end, est_start = pos - fm.ESTIMATION_GAP, max(0, pos - fm.ESTIMATION_GAP - fm.ESTIMATION_WINDOW)
            excess = r[est_start:est_end] - rf[est_start:est_end]
            Fw = F[est_start:est_end]
            good = np.isfinite(excess) & np.all(np.isfinite(Fw), axis=1)
            win = slice(pos + 1, pos + H + 1)
            path_ok = pos + H < len(calendar) and np.all(np.isfinite(r[win])) and np.all(np.isfinite(F[win])) and np.all(np.isfinite(rf[win]))
            if int(good.sum()) >= fm.MIN_ESTIMATION_OBS and path_ok:
                Xg = np.column_stack([np.ones(int(good.sum())), Fw[good]])
                beta, *_ = np.linalg.lstsq(Xg, excess[good], rcond=None)
                beta_r, *_ = np.linalg.lstsq(Fw[good], excess[good], rcond=None)
                R = float(np.prod(1.0 + r[win]) - 1.0)
                E1 = float(np.prod(1.0 + rf[win] + beta[0] + F[win] @ beta[1:]) - 1.0)
                E0 = float(np.prod(1.0 + rf[win] + F[win] @ beta[1:]) - 1.0)
                E0r = float(np.prod(1.0 + rf[win] + F[win] @ beta_r) - 1.0)
                rec.update({"Y1_refit": (R - E1) * BP, "Y0_refit": (R - E0) * BP, "Y0r": (R - E0r) * BP, "alpha_daily": float(beta[0]),
                            "E1": E1 * BP, "E0": E0 * BP, "E0r": E0r * BP, "R": R * BP, "n_est": int(good.sum())})
        rows.append(rec)
        if (i + 1) % 2000 == 0:
            log(f"refit {i + 1}/{len(events)} events, {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


def winsor_cutoffs(panel: pd.DataFrame, col: str, clock: str) -> dict[str, tuple[float, float]]:
    """The committed pooled cutoffs for cache outcomes; the panel's own p1/p99 per period for derived ones."""
    if col in CUTOFF_OUTCOME:
        return evf.pooled_cutoffs(clock, CUTOFF_OUTCOME[col])
    return evf.sample_cutoffs(panel[col].to_numpy(dtype=float), panel["period"].to_numpy())


def winsorized_column(panel: pd.DataFrame, col: str, clock: str) -> np.ndarray:
    cuts = winsor_cutoffs(panel, col, clock)
    v = panel[col].to_numpy(dtype=float).copy()
    per = panel["period"].to_numpy()
    for p, (lo, hi) in cuts.items():
        m = per == p
        v[m] = np.clip(v[m], lo, hi)
    return v


# ---------------------------------------------------------------- rows

def _row(clock: str, variant: str, stat: str, value, unit: str, n_ev: int, n_hh: int, notes: str = "", **kw) -> dict:
    return rs.row(clock, ANALYSIS, variant, stat, value, unit, n_events=n_ev, n_households=n_hh, notes=notes, **kw)


def decompose_clock(ctx: rs.Context, clock: str, log, n_placebo: int = 2000) -> tuple[list[dict], list[dict]]:
    base = multi_outcome_panel(ctx, clock)
    ref = refit_outcomes(base, ctx.calendar, log)
    base = base.merge(ref, on="event_id", how="left")
    n_hh_head = int(base["household_id"].nunique())
    rows: list[dict] = []
    hh_rows: list[dict] = []
    # verification: the per-event recomputation reproduces the cache (bp)
    d1 = np.nanmax(np.abs(base["Y1_refit"] - base["Y1"]))
    d0 = np.nanmax(np.abs(base["Y0_refit"] - base["Y0"]))
    rows.append(_row(clock, "verification", "max_abs_diff_Y1_recomputed_vs_cache", d1, "bp", len(base), n_hh_head,
                     "per-event FF5+MOM recomputation (252-session window ending 21 sessions before the anchor, intercept kept) against the forward cache"))
    rows.append(_row(clock, "verification", "max_abs_diff_Y0_recomputed_vs_cache", d0, "bp", len(base), n_hh_head,
                     "same loadings, intercept omitted from the expected return, against the cache's no-alpha column"))
    rows.append(_row(clock, "verification", "n_events_refit_missing", int(base["Y0r"].isna().sum()), "count", len(base), n_hh_head,
                     "headline events whose re-fit could not be computed (should be 0)"))
    # headline on the full panel, winsorized at the committed cutoffs, against results.csv
    pw = base[["event_id", "household_id", "period", "anchor_date", "Y1"]].rename(columns={"Y1": "y_h20"})
    pw = sp.winsorize(pw, evf.pooled_cutoffs(clock, "ff5mom_h20"))
    r_head = cgm.tau2_i1(rs.block_cluster(pw, ctx.calendar), "ES", anchor="anchor_blk")
    rows.append(_row(clock, "verification", "tau2_headline_winsorized_full_panel", r_head["tau2_hat"], "bp2", len(base), n_hh_head,
                     "cgm.tau2_i1 on the winsorized headline panel rebuilt here; equals the tau2 row of results.csv"))
    for variant, yx in ALTERNATIVES.items():
        need = ["Y1", yx]
        ok = np.isfinite(base[need].to_numpy(dtype=float)).all(axis=1)
        panel = base.loc[ok].reset_index(drop=True)
        n_hh = int(panel["household_id"].nunique())
        n_ev = int(len(panel))
        n_drop = int((~ok).sum())
        note_s = f"headline households; events with both outcomes finite ({n_drop} headline events dropped for this comparison)"
        # raw
        hh = household_means(panel, ["Y1", yx])
        w = tau_mod.household_weights(hh["n_A"].to_numpy(), hh["n_B"].to_numpy(), "ES")
        t_raw = terms(hh, "Y1", yx, w)
        for k in TERMS:
            rows.append(_row(clock, variant, f"tau2_{k}_raw", t_raw[k], "bp2", n_ev, n_hh, f"raw outcomes; {note_s}"))
        rows.append(_row(clock, variant, "identity_gap_raw", t_raw["identity_gap"], "bp2", n_ev, n_hh, "total minus the sum of the four terms (exact bilinearity: should be ~0)"))
        rows.append(_row(clock, variant, "identity_rel_gap_raw", abs(t_raw["identity_gap"]) / max(1.0, abs(t_raw["total"])), "ratio", n_ev, n_hh))
        # verification against the event-level WLS estimator on the same raw panel
        p_raw = panel[["event_id", "household_id", "period", "anchor_date"]].assign(y_h20=panel["Y1"].to_numpy())
        r_ev = tau_mod.estimate_tau2(p_raw, "ES")
        rows.append(_row(clock, variant, "tau2_total_raw_event_wls", r_ev.tau2_hat, "bp2", n_ev, n_hh, "estimate_tau2 (event-level WLS with the activity control) on the same raw panel"))
        # winsorized: each outcome at its own cutoffs (the plan's rule), D^w := Y1^w - Yx^w (identity exact), and D winsorized on its own
        pw_ = panel.copy()
        pw_["Y1w"] = winsorized_column(panel, "Y1", clock)
        pw_[f"{yx}w"] = winsorized_column(panel, yx, clock)
        pw_["Dw_own"] = winsorized_column(panel.assign(D=panel["Y1"] - panel[yx]), "D", clock)
        hhw = household_means(pw_, ["Y1w", f"{yx}w", "Dw_own"])
        ww = tau_mod.household_weights(hhw["n_A"].to_numpy(), hhw["n_B"].to_numpy(), "ES")
        t_w = terms(hhw, "Y1w", f"{yx}w", ww)
        for k in TERMS:
            rows.append(_row(clock, variant, f"tau2_{k}_wins", t_w[k], "bp2", n_ev, n_hh,
                             "each outcome winsorized at its own pooled p1/p99 by period; D = Y1w - Yxw"))
        rows.append(_row(clock, variant, "identity_gap_wins", t_w["identity_gap"], "bp2", n_ev, n_hh, "exact by construction when D is defined on the winsorized outcomes"))
        xd_own = residualize(hhw["Dw_own_A"].to_numpy(), hhw["n_A"].to_numpy(), ww)
        adj_own = tau2_bilinear(xd_own, hhw["Dw_own_B"].to_numpy(), ww)
        xx_w = residualize(hhw[f"{yx}w_A"].to_numpy(), hhw["n_A"].to_numpy(), ww)
        cross1_own = tau2_bilinear(xx_w, hhw["Dw_own_B"].to_numpy(), ww)
        cross2_own = tau2_bilinear(xd_own, hhw[f"{yx}w_B"].to_numpy(), ww)
        gap_own = t_w["total"] - (t_w["base"] + adj_own + cross1_own + cross2_own)
        rows.append(_row(clock, variant, "tau2_adj_wins_own_cutoffs", adj_own, "bp2", n_ev, n_hh, "D winsorized on its own p1/p99 by period"))
        rows.append(_row(clock, variant, "tau2_cross_base_adj_wins_own_cutoffs", cross1_own, "bp2", n_ev, n_hh))
        rows.append(_row(clock, variant, "tau2_cross_adj_base_wins_own_cutoffs", cross2_own, "bp2", n_ev, n_hh))
        rows.append(_row(clock, variant, "winsorization_nonlinearity_gap", gap_own, "bp2", n_ev, n_hh,
                         "total(winsorized Y1) minus base(winsorized Yx) + the three terms with D winsorized on its own: the non-linearity of winsorization"))
        rows.append(_row(clock, variant, "winsorization_effect_total", t_w["total"] - t_raw["total"], "bp2", n_ev, n_hh, "winsorized minus raw headline on this sample"))
        rows.append(_row(clock, variant, "winsorization_effect_base", t_w["base"] - t_raw["base"], "bp2", n_ev, n_hh, "winsorized minus raw alternative on this sample"))
        # what the adjustment term is
        for k, unit in (("corr_D_AB", "correlation"), ("corr_xD_x1", "correlation"), ("mean_D_A", "bp"), ("mean_D_B", "bp")):
            rows.append(_row(clock, variant, f"{k}_raw", t_raw[k], unit, n_ev, n_hh,
                             {"corr_D_AB": "w-weighted correlation across households of the period-A and period-B means of D",
                              "corr_xD_x1": "w-weighted correlation of the residualized period-A means of D and of Y1",
                              "mean_D_A": "w-weighted mean of household period-A means of D", "mean_D_B": "w-weighted mean of household period-B means of D"}[k]))
        d_ev = panel["Y1"] - panel[yx]
        rows.append(_row(clock, variant, "corr_D_AB_wins", t_w["corr_D_AB"], "correlation", n_ev, n_hh, "the same correlation on the winsorized outcomes"))
        rob = ctx.panel(replace(sp.headline_spec(clock), outcome=CUTOFF_OUTCOME[yx])) if yx in CUTOFF_OUTCOME else None
        if rob is not None:
            r_rob = cgm.tau2_i1(rs.block_cluster(rob, ctx.calendar), "ES", anchor="anchor_blk")
            rows.append(_row(clock, variant, "tau2_alternative_own_sample_wins", r_rob["tau2_hat"], "bp2", int(len(rob)), int(rob["household_id"].nunique()),
                             "the Table 4 robustness row: the alternative outcome on its own finite sample, eligibility re-applied (differs from tau2_base_wins when the sample differs)"))
        rows.append(_row(clock, variant, "sd_D_event", float(np.std(d_ev, ddof=1)), "bp", n_ev, n_hh, "event-level SD of D"))
        rows.append(_row(clock, variant, "mean_D_event", float(np.mean(d_ev)), "bp", n_ev, n_hh, "event-level mean of D"))
        rows.append(_row(clock, variant, "n_events_common", n_ev, "count", n_ev, n_hh, note_s))
        rows.append(_row(clock, variant, "n_events_dropped_for_common_sample", n_drop, "count", n_ev, n_hh))
        # household rows (anonymized)
        x1 = residualize(hh["Y1_A"].to_numpy(), hh["n_A"].to_numpy(), w)
        xd = residualize((hh["Y1_A"] - hh[f"{yx}_A"]).to_numpy(), hh["n_A"].to_numpy(), w)
        order = np.argsort(-x1, kind="stable")
        for i, k in enumerate(order, start=1):
            hh_rows.append({"clock": clock, "variant": variant, "household_index": i, "x_Y1_A": float(x1[k]), "Y1_B": float(hh["Y1_B"].iat[k]),
                            "x_D_A": float(xd[k]), "D_B": float((hh["Y1_B"] - hh[f"{yx}_B"]).iat[k]), "w": float(w[k])})
        # compounding (intercept omitted): D0 against 20 x alpha
        if variant == "no_alpha_omit":
            lin = -panel["alpha_daily"].to_numpy(dtype=float) * H * BP  # D0 = E0 - E1 is about -20 alpha
            resid = d_ev.to_numpy() - lin
            rows.append(_row(clock, variant, "mean_compounding_residual", float(np.nanmean(resid)), "bp", n_ev, n_hh, "mean of D0 - (-20 x alpha-hat) (bp)"))
            rows.append(_row(clock, variant, "sd_compounding_residual", float(np.nanstd(resid, ddof=1)), "bp", n_ev, n_hh))
            rows.append(_row(clock, variant, "mean_20alpha", float(np.nanmean(-lin)), "bp", n_ev, n_hh, "mean of 20 x daily alpha-hat (bp)"))
            pl = panel.assign(Ylin=panel["Y1"] - lin)
            hhl = household_means(pl, ["Y1", "Ylin"])
            wl = tau_mod.household_weights(hhl["n_A"].to_numpy(), hhl["n_B"].to_numpy(), "ES")
            t_lin = terms(hhl, "Y1", "Ylin", wl)
            for k in TERMS:
                rows.append(_row(clock, variant, f"tau2_{k}_linear_alpha", t_lin[k], "bp2", n_ev, n_hh, "D replaced by 20 x alpha-hat (raw outcomes)"))
            rows.append(_row(clock, variant, "identity_gap_linear_alpha", t_lin["identity_gap"], "bp2", n_ev, n_hh))
        if variant == "no_alpha_refit":
            de = panel["E0r"] - panel["E0"]
            rows.append(_row(clock, variant, "mean_E0r_minus_E0", float(np.nanmean(de)), "bp", n_ev, n_hh, "expected-return leg: re-fitted without intercept minus intercept omitted (bp)"))
            rows.append(_row(clock, variant, "sd_E0r_minus_E0", float(np.nanstd(de, ddof=1)), "bp", n_ev, n_hh))
            rows.append(_row(clock, variant, "mean_abs_E0r_minus_E0", float(np.nanmean(np.abs(de))), "bp", n_ev, n_hh))
    # the no-alpha and market alternatives on the timing-only placebo draws (Y0r is not stored on the draws)
    rows += placebo_terms(ctx, clock, log, n_placebo, {v: a for v, a in ALTERNATIVES.items() if a != "Y0r"},
                          {r["variant"] + "|" + r["statistic"]: r["value"] for r in rows})
    return rows, hh_rows


def placebo_terms(ctx: rs.Context, clock: str, log, n_placebo: int, alternatives: dict[str, str], actual: dict) -> list[dict]:
    """The four raw and winsorized terms on every timing-only placebo draw: mean, 5th and 95th percentiles,
    and the placebo p of the actual adjustment and total terms."""
    spec_raw = replace(sp.headline_spec(clock), winsorized=False)
    meta = evf.event_metadata(clock)
    cuts = {k: evf.pooled_cutoffs(clock, CUTOFF_OUTCOME[k]) for k in COLS}
    rec: dict[str, list[float]] = {}
    n_seen = 0
    t0 = time.time()
    for _, batch in evf.placebo_event_frames(clock, "N1", meta, dup="default", batches=True):
        for draw, g in batch.groupby("draw", sort=True):
            if draw >= n_placebo:
                continue
            g = g.reset_index(drop=True)
            p = sp.to_panel(g, spec_raw, None, draw=int(draw))
            gi = g.set_index("event_id")
            for k, c in COLS.items():
                p[k] = pd.to_numeric(gi.loc[p["event_id"], c], errors="coerce").to_numpy(dtype=float)
                v = p[k].to_numpy().copy()
                per = p["period"].to_numpy()
                for pp, (lo, hi) in cuts[k].items():
                    m = per == pp
                    v[m] = np.clip(v[m], lo, hi)
                p[f"{k}w"] = v
            for variant, yx in alternatives.items():
                ok = np.isfinite(p[["Y1", yx]].to_numpy(dtype=float)).all(axis=1)
                q = p.loc[ok]
                for suffix in ("", "w"):
                    hh = household_means(q, [f"Y1{suffix}", f"{yx}{suffix}"])
                    w = tau_mod.household_weights(hh["n_A"].to_numpy(), hh["n_B"].to_numpy(), "ES")
                    t = terms(hh, f"Y1{suffix}", f"{yx}{suffix}", w)
                    for k in TERMS + ("corr_D_AB",):
                        rec.setdefault(f"{variant}|{k}|{'raw' if suffix == '' else 'wins'}", []).append(t[k])
            n_seen += 1
        log(f"placebo terms {clock}: {n_seen} draws, {time.time() - t0:.0f} s")
    rows = []
    for key, vals in rec.items():
        variant, k, kind = key.split("|")
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        unit = "correlation" if k == "corr_D_AB" else "bp2"
        stat_actual = f"{k}_{kind}" if k == "corr_D_AB" and kind == "raw" else (f"tau2_{k}_{kind}" if k != "corr_D_AB" else None)
        a = actual.get(f"{variant}|{stat_actual}") if stat_actual else None
        p_val = placebo_p_value(a, v, "greater")["p"] if a is not None and np.isfinite(a) and len(v) else np.nan
        note = f"timing-only placebo, {len(v)} draws; boot_lo / boot_hi = 5th / 95th percentiles; placebo_p = share of draws at or above the actual value"
        rows.append(rs.row(clock, ANALYSIS, variant, f"placebo_{k}_{kind}", float(v.mean()), unit, placebo_p=p_val,
                           boot_lo=float(np.quantile(v, 0.05)), boot_hi=float(np.quantile(v, 0.95)), notes=note))
    return rows


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="benchmark decomposition of the persistence sign reversal")
    ap.add_argument("--clocks", default="T,P")
    ap.add_argument("--n-placebo", type=int, default=2000)
    a = ap.parse_args(argv)
    clocks = tuple(a.clocks.split(","))
    log = lambda m: cm.log_line(LOG, m)  # noqa: E731
    t0 = time.time()
    ctx, _ = cm.load_headline_context(clocks, with_placebo=False)
    rows, hh_rows = [], []
    for c in clocks:
        r, h = decompose_clock(ctx, c, log, a.n_placebo)
        rows += r
        hh_rows += h
        log(f"{c}: {len(r)} rows")
    out = cm.write_rows(rows, OUT_CSV)
    pd.DataFrame(hh_rows).to_csv(HH_CSV, index=False, lineterminator="\n")
    inputs = [cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "n" / "forward_cache_manifest.json"),
              cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"),
              cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_P.parquet"),
              cm.file_entry(cm.REPO_ROOT / "data" / "v4" / "pooled" / "pooled_outcome_distribution.csv")]
    cm.write_manifest("decompose", inputs, {"seconds": round(time.time() - t0, 1), "placebo_sets": {c: evf.PLACEBO_SETS[(c, "default")] for c in clocks}})
    log(f"written {OUT_CSV} ({len(out)} rows) and {HH_CSV} ({len(hh_rows)} rows) in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
