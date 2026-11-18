"""Extended factor-fit cache: the forward cache's own fits with the daily intercept and loadings kept.

python -m trading_on_disclosure_v4.writing_sample.fits
For every (PERMNO, anchor) cell of the forward cache it re-runs the cache's batched FF5+MOM fit (sessions
anchor-273..anchor-22, at least 60 observations, the same fallback routine) and stores the daily alpha, the six
loadings, and both outcomes at 10, 20 and 60 sessions: Y1 with the fitted intercept in the expected return and Y0
without it, from the SAME loadings. Every stored cache column is reproduced and compared before the file is kept.
Writes data/v4/restricted/writing_sample/fit_cache_extended.parquet (restricted: security-date level).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel
from trading_on_disclosure_v4.writing_sample import common as cm

PATH = cm.RESTRICTED / "fit_cache_extended.parquet"
MANIFEST = cm.OUT / "fit_cache_extended_manifest.json"
BETA_COLS = [f"beta_{f}" for f in panel.FACTOR_COLS]
HORIZONS = fc.HORIZONS


def compute_permno(permno: int, stock: pd.Series, factors: pd.DataFrame, F: np.ndarray, rf: np.ndarray,
                   calendar: pd.DatetimeIndex, anchors: np.ndarray) -> pd.DataFrame:
    """forward_cache.compute_permno with the fitted coefficients and the factor-only outcome at every horizon."""
    r = panel.align_to_calendar(stock, calendar)
    beta, resid_sd, n_est = fc.rolling_loadings(r, F, rf, anchors)
    beta, resid_sd = fc._fallback_loadings(beta, resid_sd, n_est, stock, factors, calendar, anchors)
    short = ~np.all(np.isfinite(beta), axis=1)
    out = {"permno": np.full(len(anchors), int(permno), dtype=np.int64), "anchor_idx": anchors.astype(np.int32),
           "alpha_daily": beta[:, 0]}
    for j, c in enumerate(BETA_COLS, start=1):
        out[c] = beta[:, j]
    out["n_est"] = n_est
    out["resid_sd"] = resid_sd
    for h in HORIZONS:
        prod_s, n_nan, trunc, idx0 = fc._window_products(r, anchors, h)
        prod_e, n_gap = fc._expected_products(beta, F, rf, idx0, h)
        prod_n, _ = fc._expected_products(beta, F, rf, idx0, h, include_alpha=False)
        reason = np.select([trunc, n_nan > 0, n_gap > 0, short],
                           [fm.REASON_TRUNCATED, fm.REASON_MISSING_RETURN, fm.REASON_FACTOR_GAP, fm.REASON_SHORT_ESTIMATION], fm.REASON_OK)
        ok = reason == fm.REASON_OK
        out[f"y1_h{h}"] = np.where(ok, (prod_s - 1.0) - (prod_e - 1.0), np.nan)
        out[f"y0_h{h}"] = np.where(ok, (prod_s - 1.0) - (prod_n - 1.0), np.nan)
        out[f"reason_h{h}"] = reason
    return pd.DataFrame(out)


def build(log) -> pd.DataFrame:
    inputs = fc.CacheInputs()
    calendar, factors, anchors = inputs.calendar, inputs.factors, inputs.anchors
    F = factors[list(panel.FACTOR_COLS)].reindex(calendar).to_numpy(float)
    rf = factors["rf"].reindex(calendar).to_numpy(float)
    universe = panel.universe_permnos()
    daily = inputs.daily[inputs.daily["permno"].isin(universe)]
    groups = {int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")}
    frames = []
    t0 = time.time()
    for i, p in enumerate(universe):
        frames.append(compute_permno(int(p), groups.get(int(p), pd.Series(dtype=float)), factors, F, rf, calendar, anchors))
        if (i + 1) % 500 == 0:
            log(f"fitted {i + 1}/{len(universe)} securities, {time.time() - t0:.0f} s")
    out = pd.concat(frames, ignore_index=True)
    for h in HORIZONS:
        out[f"reason_h{h}"] = out[f"reason_h{h}"].astype("category")
    log(f"fit cache: {len(out):,} cells over {len(universe):,} securities in {time.time() - t0:.0f} s")
    return out


def load(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PATH, columns=columns)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    log = cm.logger("fits")
    t0 = time.time()
    ext = build(log)
    rep = verify_full(ext, log)
    cm.RESTRICTED.mkdir(parents=True, exist_ok=True)
    ext.to_parquet(PATH, index=False)
    man = {"path": cm.rel(PATH), "sha256": cm.sha256_file(PATH), "rows": int(len(ext)), "columns": list(ext.columns),
           "definition": {"alpha_daily": "fitted daily intercept (decimal) of the OLS of r - rf on [1, mkt_rf, smb, hml, rmw, cma, umd] over "
                                          "calendar positions anchor-273..anchor-22 (>= 60 finite rows), the forward cache's own fit",
                          "y1_hH": "prod(1+r) - prod(1+rf+alpha+beta'f) over sessions anchor+1..anchor+H (decimal); NaN unless reason ok",
                          "y0_hH": "prod(1+r) - prod(1+rf+beta'f), same loadings and window (decimal); NaN unless reason ok"},
           "verification_against_forward_cache": rep, "forward_cache": cm.file_entry(fc.CACHE_PATH),
           "daily_panel": cm.file_entry(panel.daily_parquet_path()), "inputs_digests": fc.input_digests(),
           "seconds": round(time.time() - t0, 1), "environment": cm.environment()}
    cm.write_json(man, MANIFEST)
    log(f"wrote {cm.rel(PATH)} ({len(ext):,} rows) in {time.time() - t0:.0f} s")


def verify_full(ext: pd.DataFrame, log) -> dict:
    cache = fc.load_cache()
    e = ext.rename(columns={f"reason_h{h}": f"reason_h{h}_y" for h in HORIZONS} | {"n_est": "n_est_y", "resid_sd": "resid_sd_y"})
    c = cache.rename(columns={"n_est": "n_est_x", "resid_sd": "resid_sd"})
    m = c.merge(e, on=["permno", "anchor_idx"], how="outer", indicator=True, validate="one_to_one")
    rep = {"cells_cache": int(len(cache)), "cells_extended": int(len(ext)), "cells_only_one_side": int((m["_merge"] != "both").sum())}
    if rep["cells_only_one_side"]:
        raise AssertionError(f"{rep['cells_only_one_side']} cells are not in both caches")
    pairs = {"bhar_ff5mom_h10": "y1_h10", "bhar_ff5mom_h20": "y1_h20", "bhar_ff5mom_h60": "y1_h60", "bhar_ff5mom_noalpha_h20": "y0_h20",
             "resid_sd": "resid_sd_y"}
    for a, b in pairs.items():
        x, y = m[a].to_numpy(float), m[b].to_numpy(float)
        nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
        both = np.isfinite(x) & np.isfinite(y)
        diff = np.abs(x[both] - y[both])
        rep[a] = {"nan_pattern_mismatches": nan_mismatch, "finite_cells": int(both.sum()), "exactly_equal": int((diff == 0).sum()),
                  "max_abs_diff_decimal": float(diff.max()) if len(diff) else 0.0}
        if nan_mismatch or (len(diff) and diff.max() > 1e-12):
            raise AssertionError(f"{a}: reproduction differs from the stored cache: {rep[a]}")
    for h in HORIZONS:
        rep[f"reason_h{h}_mismatches"] = int((m[f"reason_h{h}"].astype(str).to_numpy() != m[f"reason_h{h}_y"].astype(str).to_numpy()).sum())
        if rep[f"reason_h{h}_mismatches"]:
            raise AssertionError(f"reason_h{h} differs from the stored cache")
    rep["n_est_mismatches"] = int((m["n_est_x"].to_numpy() != m["n_est_y"].to_numpy()).sum())
    if rep["n_est_mismatches"]:
        raise AssertionError("n_est differs from the stored cache")
    ok20 = m["reason_h20"].astype(str).to_numpy() == fm.REASON_OK
    rep["alpha_finite_where_h20_ok"] = int(np.isfinite(m.loc[ok20, "alpha_daily"]).sum())
    rep["cells_h20_ok"] = int(ok20.sum())
    for h in HORIZONS:
        okh = m[f"reason_h{h}"].astype(str).to_numpy() == fm.REASON_OK
        rep[f"y0_h{h}_finite_equals_ok"] = bool(np.array_equal(np.isfinite(m[f"y0_h{h}"].to_numpy(float)), okh))
    log("verification: " + json.dumps(rep, default=str))
    return rep


if __name__ == "__main__":
    main()
