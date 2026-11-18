"""Forward buy-and-hold abnormal returns for every (PERMNO, session) cell.

Vectorized re-implementation of v2 ``compute_factor_model_cars`` (FF5+MOM,
252-session estimation window ending 21 sessions before the anchor, minimum 60
finite observations, alpha included in the expected return, missing returns
never zero-filled) and of v2 ``window_abnormal_return`` (market-adjusted).
Per PERMNO, X'X and X'y are accumulated with cumulative sums over the calendar
grid so every anchor's loadings come from one batched 7x7 solve; any anchor
whose solve is not finite falls back to the v2 routine itself (loadings AND
residual SD recomputed on the window). Reason codes are the v2 strings with the
v2 precedence. ``bhar_ff5mom_noalpha_h20`` (plan §12 row 3) uses the same
loadings and window with expected return rf + beta'f (no intercept).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v2.outcomes import returns as rt
from trading_on_disclosure_v4.outcomes import panel

HORIZONS = (10, 20, 60)
MKT_HORIZON = 20
PRIMARY_HORIZON = 20
ANCHOR_START = pd.Timestamp("2014-01-02")
K = 1 + len(panel.FACTOR_COLS)  # intercept + 6 factors

CACHE_COLUMNS = [
    "permno", "anchor_date", "bhar_ff5mom_h10", "bhar_ff5mom_h20",
    "bhar_ff5mom_h60", "bhar_ff5mom_noalpha_h20", "bhar_mkt_h20", "resid_sd",
    "n_est", "reason_h20", "reason_h10", "reason_h60", "reason_mkt_h20", "anchor_idx",
]
CACHE_VERSION = 2  # 1 = wave-1 columns; 2 = + bhar_ff5mom_noalpha_h20, fallback resid_sd


def anchor_index_range(calendar: pd.DatetimeIndex, horizon: int = PRIMARY_HORIZON) -> tuple[int, int]:
    """First anchor index (>= 2014-01-02) and last with a complete ``horizon`` window."""
    lo = int(calendar.searchsorted(ANCHOR_START, side="left"))
    hi = len(calendar) - horizon - 1
    return lo, hi


def _cumulative_moments(
    y: np.ndarray, F: np.ndarray, rf: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prefix sums (length T+1) of z z', z y, y^2 and the count over 'good' rows."""
    excess = y - rf
    good = np.isfinite(excess) & np.all(np.isfinite(F), axis=1)
    Z = np.column_stack([np.ones(len(y)), F])
    Z = np.where(good[:, None], Z, 0.0)
    ye = np.where(good, excess, 0.0)
    zz = np.einsum("ti,tj->tij", Z, Z)
    S_zz = np.concatenate([np.zeros((1, K, K)), np.cumsum(zz, axis=0)])
    S_zy = np.concatenate([np.zeros((1, K)), np.cumsum(Z * ye[:, None], axis=0)])
    S_yy = np.concatenate([[0.0], np.cumsum(ye * ye)])
    S_n = np.concatenate([[0], np.cumsum(good.astype(np.int64))])
    return S_zz, S_zy, S_yy, S_n


def _solve_refined(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stacked solve with one step of iterative refinement (raises on a singular matrix)."""
    sol = np.linalg.solve(A, b[..., None])[..., 0]
    resid = b - np.einsum("...ij,...j->...i", A, sol)
    return sol + np.linalg.solve(A, resid[..., None])[..., 0]


def _batched_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve every 7x7 system; only the rows whose own matrix is singular become NaN."""
    with np.errstate(all="ignore"):
        try:
            return _solve_refined(A, b)
        except np.linalg.LinAlgError:
            sol = np.full(b.shape, np.nan)
            for i in range(len(A)):
                try:
                    sol[i] = _solve_refined(A[i], b[i])
                except np.linalg.LinAlgError:
                    pass
            return sol


def rolling_loadings(
    y: np.ndarray, F: np.ndarray, rf: np.ndarray, anchors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor OLS of excess return on [1, factors] over the v2 window.

    Returns ``beta`` (n, K) with NaN rows where the v2 rule gives None,
    ``resid_sd`` (n,) = sqrt(SSR / (n_est - K)) and ``n_est`` (n,).
    """
    S_zz, S_zy, S_yy, S_n = _cumulative_moments(y, F, rf)
    est_end = anchors - fm.ESTIMATION_GAP
    est_start = np.maximum(0, est_end - fm.ESTIMATION_WINDOW)
    valid = est_end > est_start
    e, s = np.where(valid, est_end, 0), np.where(valid, est_start, 0)
    XtX = S_zz[e] - S_zz[s]
    Xty = S_zy[e] - S_zy[s]
    yty = S_yy[e] - S_yy[s]
    n_est = np.where(valid, S_n[e] - S_n[s], 0)
    ok = valid & (n_est >= fm.MIN_ESTIMATION_OBS)
    beta = np.full((len(anchors), K), np.nan)
    if ok.any():
        beta[ok] = _batched_solve(XtX[ok], Xty[ok])
    ssr = yty - np.einsum("ni,ni->n", beta, Xty)
    with np.errstate(all="ignore"):
        resid_sd = np.sqrt(np.maximum(ssr, 0.0) / (n_est - K))
    resid_sd = np.where(ok, resid_sd, np.nan)
    return beta, resid_sd, n_est.astype(np.int32)


def _window_products(
    r: np.ndarray, anchors: np.ndarray, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """prod(1+r) over sessions anchor+1..anchor+h; NaN counts; truncation flag."""
    T = len(r)
    truncated = anchors + h > T - 1
    idx0 = np.where(truncated, 0, anchors)
    filled = np.where(np.isfinite(r), r, 0.0)
    nan_flag = (~np.isfinite(r)).astype(np.int64)
    prod = np.ones(len(anchors))
    n_nan = np.zeros(len(anchors), dtype=np.int64)
    for k in range(1, h + 1):
        prod = prod * (1.0 + filled[idx0 + k])
        n_nan = n_nan + nan_flag[idx0 + k]
    return prod, n_nan, truncated, idx0


def _expected_products(
    beta: np.ndarray, F: np.ndarray, rf: np.ndarray, idx0: np.ndarray, h: int,
    include_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """prod(1 + rf [+ alpha] + beta'f) over the window and the factor-gap count."""
    f_bad = (~(np.all(np.isfinite(F), axis=1) & np.isfinite(rf))).astype(np.int64)
    F0, rf0 = np.where(np.isfinite(F), F, 0.0), np.where(np.isfinite(rf), rf, 0.0)
    alpha = beta[:, 0] if include_alpha else 0.0
    prod = np.ones(len(idx0))
    n_gap = np.zeros(len(idx0), dtype=np.int64)
    for k in range(1, h + 1):
        e = rf0[idx0 + k] + alpha + np.einsum("nj,nj->n", F0[idx0 + k], beta[:, 1:])
        prod = prod * (1.0 + e)
        n_gap = n_gap + f_bad[idx0 + k]
    return prod, n_gap


def _direct_resid_sd(
    beta: np.ndarray, stock: pd.Series, factors: pd.DataFrame, calendar: pd.DatetimeIndex, pos: int,
) -> float:
    """sqrt(SSR / (n_est - K)) of ``beta`` on the v2 estimation window of anchor ``pos``."""
    window = calendar[max(0, pos - fm.ESTIMATION_GAP - fm.ESTIMATION_WINDOW): pos - fm.ESTIMATION_GAP]
    f = factors.reindex(window)
    excess = (stock.reindex(window) - f["rf"]).to_numpy(float)
    X = np.column_stack([np.ones(len(window)), f[list(panel.FACTOR_COLS)].to_numpy(float)])
    good = np.isfinite(excess) & np.all(np.isfinite(X), axis=1)
    resid = excess[good] - X[good] @ beta
    return float(np.sqrt(resid @ resid / (good.sum() - K)))


def _fallback_loadings(
    beta: np.ndarray, resid_sd: np.ndarray, n_est: np.ndarray, stock: pd.Series,
    factors: pd.DataFrame, calendar: pd.DatetimeIndex, anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute loadings AND resid_sd with the v2 routine where the batched solve was not finite."""
    bad = (n_est >= fm.MIN_ESTIMATION_OBS) & ~np.all(np.isfinite(beta), axis=1)
    for i in np.flatnonzero(bad):
        b = fm._estimate_loadings(stock, factors, calendar[anchors[i]], calendar, panel.FACTOR_COLS)
        if b is None:
            beta[i], resid_sd[i] = np.nan, np.nan
        else:
            beta[i] = b
            resid_sd[i] = _direct_resid_sd(b, stock, factors, calendar, int(anchors[i]))
    return beta, resid_sd


def compute_permno(
    permno: int, stock: pd.Series, factors: pd.DataFrame, bench: np.ndarray,
    calendar: pd.DatetimeIndex, anchors: np.ndarray,
) -> pd.DataFrame:
    """Cache rows for one PERMNO over ``anchors`` (calendar positions)."""
    r = panel.align_to_calendar(stock, calendar)
    F = factors[list(panel.FACTOR_COLS)].reindex(calendar).to_numpy(float)
    rf = factors["rf"].reindex(calendar).to_numpy(float)
    beta, resid_sd, n_est = rolling_loadings(r, F, rf, anchors)
    beta, resid_sd = _fallback_loadings(beta, resid_sd, n_est, stock, factors, calendar, anchors)
    short = ~np.all(np.isfinite(beta), axis=1)
    out = {"permno": np.full(len(anchors), int(permno), dtype=np.int64),
           "anchor_date": calendar[anchors], "anchor_idx": anchors.astype(np.int32)}
    for h in HORIZONS:
        prod_s, n_nan, trunc, idx0 = _window_products(r, anchors, h)
        prod_e, n_gap = _expected_products(beta, F, rf, idx0, h)
        value = (prod_s - 1.0) - (prod_e - 1.0)
        reason = np.select(
            [trunc, n_nan > 0, n_gap > 0, short],
            [fm.REASON_TRUNCATED, fm.REASON_MISSING_RETURN, fm.REASON_FACTOR_GAP,
             fm.REASON_SHORT_ESTIMATION], fm.REASON_OK)
        out[f"bhar_ff5mom_h{h}"] = np.where(reason == fm.REASON_OK, value, np.nan)
        out[f"reason_h{h}"] = reason
        if h == PRIMARY_HORIZON:  # plan §12 row 3: same loadings, expected return without alpha
            prod_na, _ = _expected_products(beta, F, rf, idx0, h, include_alpha=False)
            out["bhar_ff5mom_noalpha_h20"] = np.where(
                reason == fm.REASON_OK, (prod_s - 1.0) - (prod_na - 1.0), np.nan)
    prod_s, n_nan, trunc, idx0 = _window_products(r, anchors, MKT_HORIZON)
    prod_b, n_bnan, _, _ = _window_products(bench, anchors, MKT_HORIZON)
    reason = np.select(
        [trunc, n_nan >= MKT_HORIZON, n_nan > 0, n_bnan > 0],
        [rt.REASON_TRUNCATED, rt.REASON_NO_SECURITY, rt.REASON_MISSING_RETURN,
         rt.REASON_BENCHMARK_GAP], rt.REASON_OK)
    out["bhar_mkt_h20"] = np.where(reason == rt.REASON_OK, (prod_s - 1.0) - (prod_b - 1.0), np.nan)
    out["reason_mkt_h20"] = reason
    out["resid_sd"], out["n_est"] = resid_sd, n_est
    return pd.DataFrame(out)[CACHE_COLUMNS]


CACHE_PATH = panel.RESTRICTED_N / "forward_cache.parquet"
PARTS_DIR = panel.RESTRICTED_N / "forward_cache_parts"
MANIFEST_PATH = panel.RESTRICTED_N / "forward_cache_manifest.json"


class CacheInputs:
    """Shared market inputs loaded once per build."""

    def __init__(self, root: Path | None = None, daily: pd.DataFrame | None = None):
        self.calendar = panel.v2_trading_calendar(root)
        self.factors = panel.load_factors(root)
        self.bench = panel.align_to_calendar(panel.load_benchmark(root), self.calendar)
        lo, hi = anchor_index_range(self.calendar)
        self.anchors = np.arange(lo, hi + 1)
        self.daily = panel.load_local_daily() if daily is None else daily

    def stock(self, permno: int) -> pd.Series:
        sub = self.daily[self.daily["permno"] == permno]
        return sub.set_index("date")["total_ret"]

    def compute(self, permno: int) -> pd.DataFrame:
        return compute_permno(permno, self.stock(permno), self.factors, self.bench,
                              self.calendar, self.anchors)


def _log(log_path: Path | None, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    if log_path is None:
        print(line, flush=True)
    else:
        with open(log_path, "a") as fh:
            fh.write(line + "\n")


def input_digests(root: Path | None = None) -> dict[str, str]:
    """sha256 of the three market inputs (Dropbox) and of the local daily copy when present."""
    proc = panel.processed_dir(root)
    files = {
        "crsp_daily_2014plus.parquet": proc / "crsp_daily_2014plus.parquet",
        "benchmark_daily_2014plus.parquet": proc / "benchmark_daily_2014plus.parquet",
        "factors_daily.csv": (root or panel.project_root()) / "data" / "raw" / "factors" / "factors_daily.csv",
    }
    if root is None and panel.daily_parquet_path().exists():
        files["local_daily.parquet"] = panel.daily_parquet_path()
    return {k: sha256_file(v) for k, v in files.items() if v.exists()}


def build_key(permnos: np.ndarray, chunk_size: int, digests: dict[str, str]) -> dict:
    """What a resumable part set must agree on; any change invalidates every existing part."""
    return {"cache_version": CACHE_VERSION, "chunk_size": int(chunk_size),
            "n_permnos": int(len(permnos)),
            "universe_sha256": hashlib.sha256(np.asarray(permnos, dtype=np.int64).tobytes()).hexdigest(),
            "input_digests": dict(sorted(digests.items()))}


def _part_is_reusable(part: Path, expected: np.ndarray) -> bool:
    if not part.exists():
        return False
    have = np.sort(pd.read_parquet(part, columns=["permno"])["permno"].unique())
    return bool(np.array_equal(have, np.sort(np.asarray(expected, dtype=np.int64))))


def build_parts(
    permnos: np.ndarray, inputs: CacheInputs, parts_dir: Path = PARTS_DIR,
    chunk_size: int = 50, log_path: Path | None = None, digests: dict[str, str] | None = None,
) -> list[Path]:
    """Write one parquet part per chunk of PERMNOs; resume only parts built under the same key."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    key = build_key(permnos, chunk_size, digests if digests is not None else {})
    key_path = parts_dir / "build_key.json"
    if not key_path.exists() or json.loads(key_path.read_text()) != key:
        stale = sorted(parts_dir.glob("part_*.parquet"))
        for old in stale:
            old.unlink()
        _log(log_path, f"build key changed or absent: removed {len(stale)} stale parts")
        key_path.write_text(json.dumps(key, indent=2))
    inputs.daily = inputs.daily[inputs.daily["permno"].isin(permnos)]
    groups = {int(p): g.set_index("date")["total_ret"] for p, g in inputs.daily.groupby("permno")}
    written: list[Path] = []
    t0 = time.time()
    for i in range(0, len(permnos), chunk_size):
        part = parts_dir / f"part_{i // chunk_size:04d}.parquet"
        chunk = permnos[i:i + chunk_size]
        if _part_is_reusable(part, chunk):
            written.append(part)
            continue
        frames = [compute_permno(int(p), groups.get(int(p), pd.Series(dtype=float)),
                                 inputs.factors, inputs.bench, inputs.calendar, inputs.anchors)
                  for p in chunk]
        pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
        written.append(part)
        _log(log_path, f"chunk {i // chunk_size} permnos {i}-{min(i + chunk_size, len(permnos))} "
                       f"of {len(permnos)} elapsed {time.time() - t0:.1f}s")
    return written


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assemble_cache(
    parts: list[Path], out: Path = CACHE_PATH, manifest_path: Path = MANIFEST_PATH,
    extra: dict | None = None,
) -> dict:
    """Concatenate parts into one parquet and write the count-only manifest."""
    cache = pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True)
    cache = cache.sort_values(["permno", "anchor_idx"], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out, index=False)
    manifest = {
        "path": str(out.relative_to(panel.REPO_ROOT)),
        "sha256": sha256_file(out),
        "n_rows": int(len(cache)),
        "n_permnos": int(cache["permno"].nunique()),
        "n_anchors": int(cache["anchor_idx"].nunique()),
        "anchor_date_min": str(cache["anchor_date"].min().date()),
        "anchor_date_max": str(cache["anchor_date"].max().date()),
        "columns": list(cache.columns),
        "reason_counts": {
            c: {str(k): int(v) for k, v in cache[c].value_counts().items()}
            for c in ("reason_h10", "reason_h20", "reason_h60", "reason_mkt_h20")
        },
        "n_est_ok_min": int(cache.loc[cache["reason_h20"] == fm.REASON_OK, "n_est"].min()),
        "n_ok_h20_with_nan_resid_sd": int(
            ((cache["reason_h20"] == fm.REASON_OK) & cache["resid_sd"].isna()).sum()),
        "n_ok_h20_with_nan_noalpha": int(
            ((cache["reason_h20"] == fm.REASON_OK) & cache["bhar_ff5mom_noalpha_h20"].isna()).sum()),
        "cache_version": CACHE_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.update(extra or {})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_cache(path: Path = CACHE_PATH, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def main(argv: list[str] | None = None) -> None:
    """CLI: full build (resumable parts) + manifest. Log under data/v4/restricted/n/logs/."""
    import argparse

    ap = argparse.ArgumentParser(description="build the v4 forward-return cache")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--log", type=Path, default=panel.RESTRICTED_N / "logs" / "build_forward_cache.log")
    args = ap.parse_args(argv)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(args.log, f"start full build (cache_version {CACHE_VERSION})")
    inputs = CacheInputs()
    universe = panel.universe_permnos()
    digests = input_digests()
    in_panel = np.isin(universe, inputs.daily["permno"].unique())
    parts = build_parts(universe, inputs, chunk_size=args.chunk_size, log_path=args.log, digests=digests)
    _log(args.log, f"parts done {len(parts)} in {time.time() - t0:.1f}s; assembling")
    man = assemble_cache(parts, extra={
        "build_seconds": round(time.time() - t0, 1),
        "universe_source": "outputs/frames/signals.csv + matched_transactions.csv",
        "universe_coverage": {"n_universe_permnos": int(len(universe)),
                              "n_universe_permnos_in_panel": int(in_panel.sum()),
                              "n_universe_permnos_missing_from_panel": int((~in_panel).sum())},
        "input_digests": digests,
        "build_key": build_key(universe, args.chunk_size, digests),
    })
    _log(args.log, f"done rows={man['n_rows']} sha256={man['sha256'][:12]} total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
