"""Forward-return cache extended to the securities of the rule-repaired lines that the released cache does not
hold (closeout plan section 3d): the same vectorized constructor as the released cache, on the same market
inputs, for every anchor of the cache's grid, so the corrected event set can be shifted by the placebo
generator. Validated against the released per-event constructor at the repaired events' actual anchors.
Writes data/v4/restricted/final/forward_cache_extension.parquet and data/v4/final/cache_extension_validation.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.outcomes import forward_cache as fcache
from trading_on_disclosure_v4.outcomes import panel as pm

REPAIR_DIR = fc.REPO_ROOT / "data" / "v4" / "restricted" / "ext" / "repair"
NEW_EPISODES = REPAIR_DIR / "new_episodes.parquet"
NEW_SIGNALS = REPAIR_DIR / "new_signals_outcomes.parquet"
OUT_CACHE = fc.RESTRICTED / "forward_cache_extension.parquet"
OUT_VALID = fc.OUT_DIR / "cache_extension_validation.csv"


def missing_permnos() -> np.ndarray:
    """PERMNOs of the repaired episodes and signals that the released cache does not cover."""
    have = set(fcache.load_cache(columns=["permno"])["permno"].unique())
    ne = pd.read_parquet(NEW_EPISODES, columns=["permno"])
    ns = pd.read_parquet(NEW_SIGNALS, columns=["permno"])
    want = set(int(p) for p in pd.concat([ne["permno"], ns["permno"]]).dropna().unique())
    return np.array(sorted(want - have), dtype=np.int64)


def build(permnos: np.ndarray, log) -> pd.DataFrame:
    inputs = fcache.CacheInputs()
    in_panel = np.isin(permnos, inputs.daily["permno"].unique())
    log(f"{len(permnos)} securities outside the released cache; {int(in_panel.sum())} in the local daily panel")
    frames = [inputs.compute(int(p)) for p in permnos]
    out = pd.concat(frames, ignore_index=True).sort_values(["permno", "anchor_idx"], ignore_index=True)
    return out


def validate(ext: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """The released constructor's values stored with the repaired events against the extension at the same cells."""
    rows = []
    ne = pd.read_parquet(NEW_EPISODES)
    ne = ne.loc[ne["permno"].isin(ext["permno"].unique()) & ne["bhar_ff5mom_h20"].notna()]
    m = ne.merge(ext[["permno", "anchor_idx", "bhar_ff5mom_h20"]].rename(columns={"bhar_ff5mom_h20": "ext"}), on=["permno", "anchor_idx"], how="left")
    d = (m["bhar_ff5mom_h20"] - m["ext"]).abs() * fc.BP
    rows.append({"check": "transaction_clock_factor_outcome_vs_released_constructor", "n": int(len(m)), "n_matched": int(m["ext"].notna().sum()),
                 "max_abs_diff_bp": float(d.max()) if len(d) else np.nan, "mean_abs_diff_bp": float(d.mean()) if len(d) else np.nan})
    ns = pd.read_parquet(NEW_SIGNALS)
    ns = ns.loc[ns["permno"].isin(ext["permno"].unique()) & ns["car_ff5mom_h20"].notna()].copy()
    ns["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(pd.to_datetime(ns["follower_entry_date"])))
    m2 = ns.merge(ext[["permno", "anchor_idx", "bhar_ff5mom_h20", "bhar_mkt_h20"]].rename(columns={"bhar_ff5mom_h20": "ext_f", "bhar_mkt_h20": "ext_m"}), on=["permno", "anchor_idx"], how="left")
    d2 = (m2["car_ff5mom_h20"] - m2["ext_f"]).abs() * fc.BP
    d3 = (m2["ar_E_h20"] - m2["ext_m"]).abs() * fc.BP
    rows.append({"check": "public_clock_factor_outcome_vs_released_constructor", "n": int(len(m2)), "n_matched": int(m2["ext_f"].notna().sum()),
                 "max_abs_diff_bp": float(d2.max()) if len(d2) else np.nan, "mean_abs_diff_bp": float(d2.mean()) if len(d2) else np.nan})
    rows.append({"check": "public_clock_market_adjusted_outcome_vs_released_constructor", "n": int(len(m2)), "n_matched": int(m2["ext_m"].notna().sum()),
                 "max_abs_diff_bp": float(d3.max()) if len(d3) else np.nan, "mean_abs_diff_bp": float(d3.mean()) if len(d3) else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="forward cache extension for the repaired securities")
    ap.parse_args(argv)
    log = fc.logger("cache_extension")
    t0 = time.time()
    permnos = missing_permnos()
    digests = fcache.input_digests()
    entries = [fc.file_entry(NEW_EPISODES), fc.file_entry(NEW_SIGNALS), fc.file_entry(fcache.MANIFEST_PATH)]
    config = {"cache_version": fcache.CACHE_VERSION, "permnos": [int(p) for p in permnos], "input_digests": digests}
    key = fc.config_key(config, entries)
    if fc.reuse_ok(OUT_CACHE, key):
        ext = pd.read_parquet(OUT_CACHE)
        log(f"reused {OUT_CACHE.name} (key matches)")
    else:
        ext = build(permnos, log)
        OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ext.to_parquet(OUT_CACHE, index=False)
        fc.write_sidecar(OUT_CACHE, key, {"n_rows": int(len(ext)), "n_permnos": int(ext["permno"].nunique())})
    calendar = pm.v2_trading_calendar()
    val = validate(ext, calendar)
    fc.write_csv(val, OUT_VALID)
    fc.write_manifest("cache_extension", entries, config=config,
                      extra={"seconds": round(time.time() - t0, 1), "n_rows": int(len(ext)), "n_permnos": int(ext["permno"].nunique()),
                             "reason_counts_h20": {str(k): int(v) for k, v in ext["reason_h20"].value_counts().items()}})
    print(val.to_string(index=False))
    log(f"extension {len(ext)} rows for {ext['permno'].nunique()} securities; validation max |diff| {val['max_abs_diff_bp'].max():.2e} bp; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
