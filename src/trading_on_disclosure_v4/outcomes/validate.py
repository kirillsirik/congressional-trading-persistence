"""The forward cache equals the v2 constructors called directly.

Checks >= 2,000 random (PERMNO, anchor) cells plus every anchor of 20 random
PERMNOs against ``compute_factor_model_cars`` (h = 10, 20, 60) and
``compute_signal_outcomes`` (market-adjusted h = 20); the no-alpha column
(no v2 counterpart) against a hand-computed lstsq reference on a random
subsample of ok cells. Writes a count-only report (counts, max abs diff) to
data/v4/pooled/forward_cache_validation.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v2.outcomes.factor_models import compute_factor_model_cars
from trading_on_disclosure_v2.outcomes.returns import compute_signal_outcomes
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel
from trading_on_disclosure_v4.outcomes.seeds import seed_from

TOL = 1e-10
REPORT_PATH = panel.REPO_ROOT / "data" / "v4" / "pooled" / "forward_cache_validation.json"


def _compare(cache_val: pd.Series, cache_reason: pd.Series, v2_val: pd.Series, v2_reason: pd.Series) -> dict:
    both_nan = cache_val.isna().to_numpy() & v2_val.isna().to_numpy()
    both_num = cache_val.notna().to_numpy() & v2_val.notna().to_numpy()
    diff = np.abs(cache_val.to_numpy(float)[both_num] - v2_val.to_numpy(float)[both_num])
    max_diff = float(diff.max()) if diff.size else 0.0
    nan_mismatch = int((~both_nan & ~both_num).sum())
    reason_mismatch = int((cache_reason.to_numpy() != v2_reason.to_numpy()).sum())
    return {
        "n_rows": int(len(cache_val)), "n_both_nan": int(both_nan.sum()),
        "n_compared": int(both_num.sum()), "max_abs_diff": max_diff,
        "n_nan_pattern_mismatch": nan_mismatch, "n_reason_mismatch": reason_mismatch,
        "pass": bool(max_diff <= TOL and nan_mismatch == 0 and reason_mismatch == 0),
    }


def select_cells(cache: pd.DataFrame, n_pairs: int, n_permnos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random cells plus every anchor of ``n_permnos`` random PERMNOs."""
    pairs = cache.iloc[rng.choice(len(cache), size=n_pairs, replace=False)]
    permnos = rng.choice(cache["permno"].unique(), size=n_permnos, replace=False)
    full = cache[cache["permno"].isin(permnos)]
    cells = pd.concat([pairs, full]).drop_duplicates(["permno", "anchor_idx"])
    return cells.sort_values(["permno", "anchor_idx"], ignore_index=True)


def _noalpha_reference(cell, stock: pd.Series, factors: pd.DataFrame, calendar: pd.DatetimeIndex) -> float:
    """BHAR with expected return rf + beta'f (no intercept), beta by lstsq on the v2 window."""
    pos = int(cell.anchor_idx)
    window = calendar[max(0, pos - fm.ESTIMATION_GAP - fm.ESTIMATION_WINDOW): pos - fm.ESTIMATION_GAP]
    f = factors.reindex(window)
    excess = (stock.reindex(window) - f["rf"]).to_numpy(float)
    X = np.column_stack([np.ones(len(window)), f[list(panel.FACTOR_COLS)].to_numpy(float)])
    good = np.isfinite(excess) & np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[good], excess[good], rcond=None)
    fw = calendar[pos + 1: pos + fc.PRIMARY_HORIZON + 1]
    ff = factors.reindex(fw)
    expected = ff["rf"].to_numpy(float) + ff[list(panel.FACTOR_COLS)].to_numpy(float) @ beta[1:]
    return float(np.prod(1 + stock.reindex(fw).to_numpy(float)) - 1 - (np.prod(1 + expected) - 1))


def check_noalpha(cells: pd.DataFrame, daily: pd.DataFrame, factors: pd.DataFrame,
                  calendar: pd.DatetimeIndex, n_max: int, rng: np.random.Generator) -> dict:
    ok = cells[cells["reason_h20"] == fm.REASON_OK]
    sub = ok.iloc[np.sort(rng.choice(len(ok), size=min(n_max, len(ok)), replace=False))]
    by_permno = {int(p): g.set_index("date")["total_ret"] for p, g in daily.groupby("permno")}
    ref = np.array([_noalpha_reference(c, by_permno[int(c.permno)], factors, calendar)
                    for c in sub.itertuples(index=False)])
    diff = np.abs(sub["bhar_ff5mom_noalpha_h20"].to_numpy(float) - ref)
    nan_mismatch = int((cells["bhar_ff5mom_noalpha_h20"].isna() != (cells["reason_h20"] != fm.REASON_OK)).sum())
    return {"n_rows": int(len(cells)), "n_compared": int(len(sub)),
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "n_nan_pattern_mismatch": nan_mismatch, "n_reason_mismatch": 0,
            "reference": "lstsq on the v2 window; expected = rf + beta'f (no intercept)",
            "pass": bool((diff.max() if diff.size else 0.0) <= TOL and nan_mismatch == 0)}


def run_validation(
    cache_path: Path = fc.CACHE_PATH, n_pairs: int = 2000, n_permnos: int = 20,
    report_path: Path = REPORT_PATH, test_name: str = "forward_cache_validation",
    n_noalpha: int = 2000,
) -> dict:
    t0 = time.time()
    cache = fc.load_cache(cache_path)
    rng = np.random.default_rng(seed_from(test_name))
    cells = select_cells(cache, n_pairs, n_permnos, rng)
    inputs = fc.CacheInputs()
    daily = inputs.daily[inputs.daily["permno"].isin(cells["permno"].unique())]
    sig = cells[["permno", "anchor_date"]].copy()
    report = {
        "test_name": test_name, "seed": seed_from(test_name), "tolerance": TOL,
        "n_random_pairs": n_pairs, "n_full_permnos": n_permnos,
        "n_cells_checked": int(len(cells)), "cache_sha256": fc.sha256_file(cache_path),
        "cache_version": fc.CACHE_VERSION,
        "checks": {},
    }
    for h in fc.HORIZONS:
        v2 = compute_factor_model_cars(sig, daily, inputs.factors, inputs.calendar,
                                       "anchor_date", h, ("ff5mom",))
        report["checks"][f"bhar_ff5mom_h{h}"] = _compare(
            cells[f"bhar_ff5mom_h{h}"], cells[f"reason_h{h}"],
            v2[f"car_ff5mom_h{h}"], v2[f"car_ff5mom_h{h}_reason"])
    m = compute_signal_outcomes(sig, daily, panel.load_benchmark(), inputs.calendar,
                                anchors=(("M", "anchor_date", (fc.MKT_HORIZON,)),))
    report["checks"]["bhar_mkt_h20"] = _compare(
        cells["bhar_mkt_h20"], cells["reason_mkt_h20"], m["ar_M_h20"], m["ar_M_h20_reason"])
    report["checks"]["bhar_ff5mom_noalpha_h20"] = check_noalpha(
        cells, daily, inputs.factors, inputs.calendar, n_noalpha, rng)
    report["all_pass"] = all(c["pass"] for c in report["checks"].values())
    report["seconds"] = round(time.time() - t0, 1)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run_validation(), indent=2))
