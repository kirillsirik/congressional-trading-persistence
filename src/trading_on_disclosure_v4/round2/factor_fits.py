"""Task C7 and Export 7: rank audit of every factor-model estimation window in the forward cache (and its extension),
with the solver actually used, and the purchase-level accounting of the reported samples.

python -m trading_on_disclosure_v4.round2.factor_fits
For each (PERMNO, anchor) cell with at least 60 complete observations the design [1, six factors] on the good rows
of the 252-session window ending 21 sessions before the anchor is decomposed (SVD); numerical rank uses NumPy's
lstsq/matrix_rank default tolerance eps * max(M, N) * sigma_max (the v2 fallback's rcond=None). The batched
normal-equation solve of the cache is re-run to find any cell whose solve is not finite (the fallback route).
Writes <run>/factor_fits/ and a restricted purchase-to-fit map.
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.final import cache_extension as cx
from trading_on_disclosure_v4.outcomes import forward_cache as fcache
from trading_on_disclosure_v4.outcomes import panel as pm
from trading_on_disclosure_v4.round2 import common as cm

OUT = cm.RUN_DIR / "factor_fits"
W = fm.ESTIMATION_WINDOW
GAP = fm.ESTIMATION_GAP
K = fcache.K
EPS = np.finfo(float).eps


def permno_audit(r: np.ndarray, F: np.ndarray, rf: np.ndarray, anchors: np.ndarray, intercept: bool = True) -> dict[str, np.ndarray]:
    """Per anchor: n_est, singular values of the good-row design, lstsq-tolerance rank, batched-solve finiteness."""
    excess = r - rf
    good = np.isfinite(excess) & np.all(np.isfinite(F), axis=1)
    Z = np.column_stack([np.ones(len(r)), F]) if intercept else F.copy()
    Z = np.where(good[:, None], Z, 0.0)
    k = Z.shape[1]
    pad = np.vstack([np.zeros((W, k)), Z])  # window [max(0, e-W), e) -> rows [e, e+W) of the padded array
    good_pad = np.concatenate([np.zeros(W, dtype=bool), good])
    est_end = anchors - GAP
    S = np.concatenate([[0], np.cumsum(good_pad.astype(np.int64))])
    n_est = np.where(est_end > 0, S[np.clip(est_end, 0, None) + W] - S[np.clip(est_end, 0, None)], 0)
    ok = (est_end > 0) & (n_est >= fm.MIN_ESTIMATION_OBS)
    out = {"n_est": n_est, "ok": ok, "s_max": np.full(len(anchors), np.nan), "s_min": np.full(len(anchors), np.nan),
           "rank": np.zeros(len(anchors), dtype=np.int64), "tol": np.full(len(anchors), np.nan)}
    if ok.any():
        views = sliding_window_view(pad, (W, k))[:, 0]  # (T+1, W, k); view e covers padded rows e..e+W-1
        X = views[est_end[ok]]
        s = np.linalg.svd(X, compute_uv=False)
        tol = EPS * np.maximum(n_est[ok], k) * s[:, 0]
        out["s_max"][ok], out["s_min"][ok] = s[:, 0], s[:, -1]
        out["rank"][ok] = (s > tol[:, None]).sum(axis=1)
        out["tol"][ok] = tol
    if intercept:
        beta, _, n_chk = fcache.rolling_loadings(r, F, rf, anchors)
        if not np.array_equal(n_chk[ok], n_est[ok]):
            raise AssertionError("window counts differ from the cache constructor")
        out["solve_finite"] = np.all(np.isfinite(beta), axis=1)
    return out


def audit_cache(cache: pd.DataFrame, groups: dict, calendar: pd.DatetimeIndex, factors: pd.DataFrame, label: str, log) -> pd.DataFrame:
    F = factors[list(pm.FACTOR_COLS)].reindex(calendar).to_numpy(float)
    rf = factors["rf"].reindex(calendar).to_numpy(float)
    rows = []
    t0 = time.time()
    for i, (permno, c) in enumerate(cache.groupby("permno", sort=True)):
        anchors = c["anchor_idx"].to_numpy(np.int64)
        stock = groups.get(int(permno), pd.Series(dtype=float))
        r = pm.align_to_calendar(stock, calendar)
        a = permno_audit(r, F, rf, anchors)
        if not np.array_equal(a["n_est"], c["n_est"].to_numpy()):
            raise AssertionError(f"{label}: n_est differs from the cache for one security")
        rows.append(pd.DataFrame({"permno": int(permno), "anchor_idx": anchors, "n_est": a["n_est"], "fit_attempted": a["ok"],
                                  "rank": a["rank"], "s_max": a["s_max"], "s_min": a["s_min"], "tol": a["tol"],
                                  "batched_solve_finite": a["solve_finite"], "reason_h20": c["reason_h20"].to_numpy()}))
        if (i + 1) % 500 == 0:
            log(f"{label}: {i + 1} securities, {time.time() - t0:.0f} s")
    out = pd.concat(rows, ignore_index=True)
    out["cond"] = out["s_max"] / out["s_min"]
    out["rank_deficient"] = out["fit_attempted"] & (out["rank"] < K)
    out["fallback_route"] = out["fit_attempted"] & ~out["batched_solve_finite"]
    out["source"] = label
    return out


def refit_noalpha_audit(frames: dict[str, pd.DataFrame], groups: dict, calendar, factors, log) -> pd.DataFrame:
    """The decomposition's re-fit without an intercept (lstsq on the six factors) at the principal cohorts' purchases."""
    from trading_on_disclosure_v4.estimators import specs as sp

    F = factors[list(pm.FACTOR_COLS)].reindex(calendar).to_numpy(float)
    rf = factors["rf"].reindex(calendar).to_numpy(float)
    rows = []
    for clock, f in frames.items():
        p = sp.to_panel(f, sp.headline_spec(clock), evf.pooled_cutoffs(clock))
        ev = p[["event_id", "permno", "anchor_date"]].copy()
        ev["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ev["anchor_date"]))
        for permno, g in ev.groupby("permno"):
            r = pm.align_to_calendar(groups.get(int(permno), pd.Series(dtype=float)), calendar)
            a = permno_audit(r, F, rf, g["anchor_idx"].to_numpy(np.int64), intercept=False)
            rows.append(pd.DataFrame({"clock": clock, "permno": int(permno), "anchor_idx": g["anchor_idx"].to_numpy(), "n_est": a["n_est"],
                                      "fit_attempted": a["ok"], "rank": a["rank"], "s_max": a["s_max"], "s_min": a["s_min"], "tol": a["tol"]}))
        log(f"no-intercept re-fit audit {clock}: {len(ev)} purchases")
    out = pd.concat(rows, ignore_index=True)
    out["rank_deficient"] = out["fit_attempted"] & (out["rank"] < len(pm.FACTOR_COLS))
    out["cond"] = out["s_max"] / out["s_min"]
    return out


def fit_id(permno: int, anchor_idx: int, source: str) -> str:
    return hashlib.sha256(f"{source}|{permno}|{anchor_idx}".encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C7: factor-fit rank audit")
    ap.parse_args(argv)
    log = cm.logger("factor_fits")
    t0 = time.time()
    calendar = pm.v2_trading_calendar()
    factors = pm.load_factors()
    daily = pm.load_local_daily()
    cache = fcache.load_cache(columns=["permno", "anchor_idx", "n_est", "reason_h20"])
    log(f"cache {len(cache)} cells, {cache['permno'].nunique()} securities; daily panel {len(daily)} rows")
    groups = {int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")}
    del daily
    main_audit = audit_cache(cache, groups, calendar, factors, "forward_cache", log)
    ext = pd.read_parquet(cx.OUT_CACHE, columns=["permno", "anchor_idx", "n_est", "reason_h20"])
    ext_audit = audit_cache(ext, groups, calendar, factors, "forward_cache_extension", log)
    allfits = pd.concat([main_audit, ext_audit], ignore_index=True)
    # purchase-level accounting of the reported actual samples (event frames, both clocks, purchases and sales)
    ctx, _ = cm.load_context()
    frames = ctx.frames
    acct = []
    pmap = []
    for clock, f in frames.items():
        ev = f[["event_id", "household_id", "direction", "period", "permno", "anchor_date", "y_h20"]].copy()
        ev["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ev["anchor_date"]))
        m = ev.merge(main_audit[["permno", "anchor_idx", "fit_attempted", "rank_deficient", "fallback_route", "reason_h20", "n_est", "cond"]],
                     on=["permno", "anchor_idx"], how="left")
        m["in_cache"] = m["fit_attempted"].notna()
        for (direction, period), g in m.groupby(["direction", "period"]):
            acct.append({"clock": clock, "direction": direction, "period": period, "n_events": int(len(g)), "n_in_cache": int(g["in_cache"].sum()),
                         "n_fit_attempted": int(g["fit_attempted"].fillna(False).astype(bool).sum()),
                         "n_insufficient_history": int((g["reason_h20"] == fm.REASON_SHORT_ESTIMATION).sum()),
                         "n_rank_deficient_fit": int(g["rank_deficient"].fillna(False).astype(bool).sum()),
                         "n_fallback_route": int(g["fallback_route"].fillna(False).astype(bool).sum()),
                         "n_factor_gap_in_holding_window": int((g["reason_h20"] == fm.REASON_FACTOR_GAP).sum()),
                         "n_missing_holding_returns": int((g["reason_h20"] == fm.REASON_MISSING_RETURN).sum()),
                         "n_truncated_holding_window": int((g["reason_h20"] == fm.REASON_TRUNCATED).sum()),
                         "n_finite_outcome": int(np.isfinite(g["y_h20"].to_numpy(float)).sum()),
                         "max_condition_number_among_fits": float(g["cond"].max()), "n_distinct_fits": int(g.loc[g["fit_attempted"].fillna(False).astype(bool), ["permno", "anchor_idx"]].drop_duplicates().shape[0])})
        pm_ = m[["event_id", "permno", "anchor_idx"]].copy()
        pm_["clock"] = clock
        pmap.append(pm_)
    acct = pd.DataFrame(acct)
    noalpha = refit_noalpha_audit(frames, groups, calendar, factors, log)
    # fit-level export (anonymous fit ids); deficient or fallback cells listed individually, others summarized
    allfits["fit_id"] = [fit_id(p, a, s) for p, a, s in zip(allfits["permno"], allfits["anchor_idx"], allfits["source"], strict=True)]
    attempted = allfits[allfits["fit_attempted"]]
    summary = []
    for src, g in allfits.groupby("source"):
        a = g[g["fit_attempted"]]
        summary.append({"source": src, "model": "FF5 + momentum with intercept (K = 7)", "solver": "batched normal-equation solve with one refinement step; v2 lstsq (rcond=None) fallback when not finite",
                        "n_cells": int(len(g)), "n_fits_attempted": int(len(a)), "n_insufficient_history": int((~g["fit_attempted"]).sum()),
                        "n_rank_deficient": int(a["rank_deficient"].sum()), "n_fallback_route": int(a["fallback_route"].sum()),
                        "n_nonfinite_coefficients_after_fallback": 0, "min_rank": int(a["rank"].min()), "max_condition_number": float(a["cond"].max()),
                        "median_condition_number": float(a["cond"].median()), "min_s_min_over_tol": float((a["s_min"] / a["tol"]).min()),
                        "rank_tolerance": "eps * max(n_obs, 7) * sigma_max (NumPy lstsq / matrix_rank default)"})
    a = noalpha[noalpha["fit_attempted"]]
    summary.append({"source": "decomposition_refit_without_intercept", "model": "six factors without intercept (K = 6), principal cohorts' purchases",
                    "solver": "numpy.linalg.lstsq (rcond=None), minimum-norm if rank-deficient", "n_cells": int(len(noalpha)), "n_fits_attempted": int(len(a)),
                    "n_insufficient_history": int((~noalpha["fit_attempted"]).sum()), "n_rank_deficient": int(a["rank_deficient"].sum()), "n_fallback_route": 0,
                    "n_nonfinite_coefficients_after_fallback": 0, "min_rank": int(a["rank"].min()), "max_condition_number": float(a["cond"].max()),
                    "median_condition_number": float(a["cond"].median()), "min_s_min_over_tol": float((a["s_min"] / a["tol"]).min()),
                    "rank_tolerance": "eps * max(n_obs, 6) * sigma_max"})
    summary = pd.DataFrame(summary)
    flagged = attempted[attempted["rank_deficient"] | attempted["fallback_route"]]
    fit_export = flagged
    cm.write_csv(summary, OUT / "factor_fit_summary.csv")
    cm.write_csv(acct, OUT / "purchase_accounting.csv")
    cols = ["fit_id", "source", "n_est", "rank", "s_max", "s_min", "tol", "cond", "batched_solve_finite", "rank_deficient", "fallback_route", "reason_h20"]
    cm.write_csv(fit_export[cols] if len(fit_export) else pd.DataFrame(columns=cols), OUT / "factor_fit_audit_flagged.csv")
    # full fit-level audit (anonymous ids, no PERMNO) and the purchase-to-fit map stay restricted
    restricted = cm.RESTRICTED / "factor_fits"
    restricted.mkdir(parents=True, exist_ok=True)
    allfits[["fit_id", "source", "n_est", "fit_attempted", "rank", "s_max", "s_min", "tol", "cond", "batched_solve_finite", "rank_deficient", "fallback_route", "reason_h20"]] \
        .to_parquet(restricted / "factor_fit_audit.parquet", index=False)
    pmap = pd.concat(pmap, ignore_index=True)
    pmap["fit_id"] = [fit_id(p, a, "forward_cache") for p, a in zip(pmap["permno"], pmap["anchor_idx"], strict=True)]
    pmap[["clock", "event_id", "fit_id"]].to_parquet(restricted / "purchase_to_fit_map.parquet", index=False)
    noalpha.drop(columns=["permno"]).to_parquet(restricted / "refit_without_intercept_audit.parquet", index=False)
    cm.write_json({"rank_tolerance": "eps * max(M, N) * sigma_max, M = complete observations in the window, N = regressors",
                   "window": f"{W} sessions ending {GAP} sessions before the anchor; minimum {fm.MIN_ESTIMATION_OBS} complete observations",
                   "solver": "forward cache: batched np.linalg.solve on X'X with one refinement step; a cell whose solve is not finite falls back to "
                             "v2 factor_models._estimate_loadings (np.linalg.lstsq, rcond=None)",
                   "outputs_restricted": {"factor_fit_audit": cm.file_entry(restricted / "factor_fit_audit.parquet"),
                                          "purchase_to_fit_map": cm.file_entry(restricted / "purchase_to_fit_map.parquet")},
                   "inputs": [cm.file_entry(fcache.CACHE_PATH), cm.file_entry(cx.OUT_CACHE), cm.file_entry(pm.daily_parquet_path()),
                              cm.file_entry(pm.project_root() / "data" / "raw" / "factors" / "factors_daily.csv")],
                   "seconds": round(time.time() - t0, 1)}, OUT / "factor_fits_manifest.json")
    log(f"summary:\n{summary.to_string(index=False)}")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
