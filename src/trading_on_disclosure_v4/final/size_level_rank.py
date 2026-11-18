"""Size of the level-rank reference rule (closeout plan section 2, uncertainty).

The paper's reference probability for a household covariance is the add-one upper-tail rank of the observed
covariance LEVEL among the 2,000 shifted-date values of the same statistic. The calibration report's
"placebo test" sizes rank the clustered t-statistic, not the level. This module measures the level-rank
rule on the stored zero-effect draws: for every specification with a stored placebo distribution, test
draws 0-999 are ranked against reference draws 1,000-1,999, then the halves are swapped, and the share rejected at
nominal 5% is reported with its Monte Carlo standard error. (A leave-one-out rank over all draws rejects 5% by
construction and says nothing, so it is not reported.)

Verification pass of 7 November 2026: the same split-half readings are added for the per-draw covariance levels of
the reference datasets regenerated on the corrected samples (`final.reference`, placebo_set "corrected_final"), and
the Monte Carlo spread of each reading is simulated under exchangeable draws rather than taken from the binomial
formula: ranking one half against the other also carries the sampling error of the reference half's critical value,
and the two readings share the same 2,000 datasets, so the per-direction spread is wider than binomial and the pooled
spread much narrower (the pooled rate is close to the nominal level whenever the halves are exchangeable).
Writes data/v4/final/size_level_rank.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.final import common as fc

CACHE = fc.REPO_ROOT / "data" / "v4" / "restricted" / "e" / "placebo_stats"
CORRECTED = fc.RESTRICTED  # reference_draw_statistics_<clock>.parquet written by final.reference
OUT = fc.OUT_DIR / "size_level_rank.csv"
OUT_HALF = fc.OUT_DIR / "reference_half_sets.csv"
OUT_MECH = fc.OUT_DIR / "reference_half_mechanics.csv"
PANELS = fc.RESTRICTED / "panels"
SET_NAMES = {"T": "A0_T_purchase_AB_L20_both_keepdups_final", "P": "A0_P_purchase_AB_L20_both_alllines_final"}
RESULTS = fc.REPO_ROOT / "data" / "v4" / "results.csv"
CORRECTED_REFERENCE = fc.OUT_DIR / "persistence_corrected_reference.csv"
N_TEST = 1000
ALPHA = 0.05
EXCH_REPS = 20000
EXCH_SEED = 20261107
CORRECTED_SPECS = {"headline": ("ff5mom_h20", "wins", "ES"), "equal_household": ("ff5mom_h20", "wins", "EH"), "pair": ("ff5mom_h20", "wins", "PW"),
                   "no_alpha": ("ff5mom_noalpha_h20", "wins", "ES"), "market_adjusted": ("mkt_h20", "wins", "ES"), "raw_outcome": ("ff5mom_h20", "raw", "ES")}
TWO_SIDED_STATS = ("mean",)
UPPER_STATS = ("tau2", "max_mean", "max_t", "top_decile_mean", "n_p05", "share_gt_2", "share_gt_3")


def rank_p(value: float, null: np.ndarray, two_sided: bool) -> float:
    if two_sided:
        c = float(np.median(null))
        return (1.0 + float(np.sum(np.abs(null - c) >= abs(value - c)))) / (len(null) + 1.0)
    return (1.0 + float(np.sum(null >= value))) / (len(null) + 1.0)


def size_rows(table: pd.DataFrame, clock: str, flavour: str, dup: str) -> list[dict]:
    rows = []
    for (key, stat), g in table.groupby(["key", "stat"], sort=True):
        if stat not in UPPER_STATS + TWO_SIDED_STATS:
            continue
        g = g.sort_values("draw")
        v = g["value"].to_numpy(dtype=float)
        d = g["draw"].to_numpy()
        ok = np.isfinite(v)
        v, d = v[ok], d[ok]
        test, ref = v[d < N_TEST], v[d >= N_TEST]
        if len(test) < 100 or len(ref) < 100:
            continue
        two = stat in TWO_SIDED_STATS
        p_ref = np.array([rank_p(x, ref, two) for x in test])  # first half ranked against the second
        p_rev = np.array([rank_p(x, test, two) for x in ref])  # second half ranked against the first
        parts = key.split("|")
        name = parts[0]
        rows.append({"clock": clock, "flavour": flavour, "placebo_set": dup, "analysis": name, "statistic": stat, "outcome": parts[2], "winsorized": parts[3],
                     "rule": parts[4], "split": parts[5], "duplicates": parts[6], "direction": parts[8], "weights": parts[9] if len(parts) > 9 else "",
                     "sidedness": "two-sided" if two else "upper", "n_test_draws": int(len(test)), "n_reference_draws": int(len(ref)),
                     "reject_rate_reference": float(np.mean(p_ref < ALPHA)), "mc_se_reference": float(np.sqrt(ALPHA * (1 - ALPHA) / len(test))),
                     "reject_rate_reverse_halves": float(np.mean(p_rev < ALPHA)), "reject_rate_both_halves": float((np.sum(p_ref < ALPHA) + np.sum(p_rev < ALPHA)) / (len(test) + len(ref))),
                     "n_all_draws": int(len(v)),
                     "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)), "null_q95": float(np.quantile(v, 0.95)), "key": key})
    return rows


def exchangeable_mc(n_test: int = N_TEST, n_ref: int = N_TEST, alpha: float = ALPHA, reps: int = EXCH_REPS, seed: int = EXCH_SEED) -> dict:
    """Monte Carlo spread of the split-half rejection rates when all draws are exchangeable. The rule uses ranks only,
    so the result is distribution-free; standard normal draws stand in for any continuous law."""
    rng = np.random.default_rng(seed)
    ref_rates, rev_rates = np.empty(reps), np.empty(reps)
    for i in range(reps):
        a, b = rng.standard_normal(n_test), rng.standard_normal(n_ref)
        sa, sb = np.sort(a), np.sort(b)
        p_ref = (1.0 + (n_ref - np.searchsorted(sb, a, side="left"))) / (n_ref + 1.0)  # add-one rank of each a against b
        p_rev = (1.0 + (n_test - np.searchsorted(sa, b, side="left"))) / (n_test + 1.0)
        ref_rates[i], rev_rates[i] = np.mean(p_ref < alpha), np.mean(p_rev < alpha)
    pooled = (ref_rates * n_test + rev_rates * n_ref) / (n_test + n_ref)
    return {"mc_mean_direction_exch": float(ref_rates.mean()), "mc_sd_direction_exch": float(ref_rates.std(ddof=1)),
            "mc_mean_pooled_exch": float(pooled.mean()), "mc_sd_pooled_exch": float(pooled.std(ddof=1)),
            "mc_corr_directions_exch": float(np.corrcoef(ref_rates, rev_rates)[0, 1]), "mc_reps_exch": int(reps)}


def corrected_table(clock: str) -> pd.DataFrame | None:
    """The regenerated reference draws of the corrected samples in the (key, stat, draw, value) layout of the stored cache."""
    p = CORRECTED / f"reference_draw_statistics_{clock}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    rows = []
    for (flavour, spec, cohort), g in d.groupby(["flavour", "spec", "cohort"], sort=True):
        outcome, wins, weights = CORRECTED_SPECS[spec]
        key = f"{spec}/{cohort}|{clock}|{outcome}|{wins}|R1|AB|default|nohsm|purchase|{weights}|ctrl"
        rows.append(pd.DataFrame({"flavour": flavour, "key": key, "stat": "tau2", "draw": g["draw"].to_numpy(), "value": g["tau2"].to_numpy(dtype=float)}))
    return pd.concat(rows, ignore_index=True)


def half_set_rows() -> list[dict]:
    """The headline reference probability of each observed covariance on all 2,000 centered datasets and on each half
    separately (the spread between the halves is the Monte Carlo uncertainty of the reported probability)."""
    rows = []
    weights = {"event_share": "ES", "equal_household": "EH", "pair": "PW"}
    res = pd.read_csv(RESULTS) if RESULTS.exists() else None
    ref = pd.read_csv(CORRECTED_REFERENCE) if CORRECTED_REFERENCE.exists() else None
    for clock in ("T", "P"):
        p = CACHE / f"placebo_{clock}_N2_default.parquet"
        if res is not None and p.exists():
            t = pd.read_parquet(p)
            for variant, w in weights.items():
                obs = res.loc[(res["clock"] == clock) & (res["analysis"] == "tau") & (res["variant"] == variant) & (res["statistic"] == "tau2"), "value"]
                g = t.loc[(t["stat"] == "tau2") & t["key"].str.startswith(f"tau/{variant}|")].sort_values("draw")
                if obs.empty or g.empty:
                    continue
                v = g["value"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                rows.append(_half_row(clock, "record", "baseline", w, float(obs.iloc[0]), v))
        q = CORRECTED / f"reference_draw_statistics_{clock}.parquet"
        if ref is not None and q.exists():
            d = pd.read_parquet(q)
            for spec, w in (("headline", "ES"), ("equal_household", "EH"), ("pair", "PW")):
                for cohort in ("corrected", "common_cohort"):
                    obs = ref.loc[(ref["clock"] == clock) & (ref["variant"] == f"{cohort}|{spec}") & (ref["statistic"] == "tau2_observed"), "value"]
                    g = d.loc[(d["flavour"] == "N2") & (d["spec"] == spec) & (d["cohort"] == cohort)].sort_values("draw")
                    if obs.empty or g.empty:
                        continue
                    rows.append(_half_row(clock, "corrected_final", cohort, w, float(obs.iloc[0]), g["tau2"].to_numpy(dtype=float)))
    return rows


def half_mechanics_rows() -> list[dict]:
    """Per-draw mechanics of the regenerated sets (rows kept, mean absolute shift in calendar days, share of positive
    shifts, share of windows overlapping the actual window) averaged over draws 0-999 and 1,000-1,999, with the Welch
    t-statistic of the difference: whether the two halves were generated alike."""
    rows = []
    for clock, name in SET_NAMES.items():
        d = PANELS / clock / "N1" / name
        frame = fc.RESTRICTED / "cumulative" / f"frame_{clock}_corrected.parquet"
        if not d.exists() or not frame.exists():
            continue
        anchor = pd.read_parquet(frame, columns=["event_id", "anchor_date"]).drop_duplicates("event_id").set_index("event_id")["anchor_date"]
        parts = []
        for f in sorted(d.glob("n1_batch_*.parquet")):
            b = pd.read_parquet(f, columns=["draw", "event_id", "shifted_anchor_date", "overlaps_actual"])
            days = (pd.to_datetime(b["shifted_anchor_date"]) - pd.to_datetime(b["event_id"].map(anchor))).dt.days
            b = b.assign(abs_shift=days.abs(), positive=(days > 0).astype(float), overlap=b["overlaps_actual"].astype(float))
            parts.append(b.groupby("draw").agg(rows=("draw", "size"), abs_shift=("abs_shift", "mean"), positive=("positive", "mean"), overlap=("overlap", "mean")))
        if not parts:
            continue
        s = pd.concat(parts).sort_index()
        h1, h2 = s.loc[s.index < N_TEST], s.loc[s.index >= N_TEST]
        for col in ("rows", "abs_shift", "positive", "overlap"):
            m1, m2 = float(h1[col].mean()), float(h2[col].mean())
            se = float(np.sqrt(h1[col].var(ddof=1) / len(h1) + h2[col].var(ddof=1) / len(h2)))
            rows.append({"clock": clock, "set_name": name, "quantity": col, "half_1_mean": m1, "half_2_mean": m2, "difference": m1 - m2, "welch_t": (m1 - m2) / se if se > 0 else 0.0,
                         "n_half_1": int(len(h1)), "n_half_2": int(len(h2))})
    return rows


def _half_row(clock: str, placebo_set: str, cohort: str, weights: str, obs: float, v: np.ndarray) -> dict:
    a, b = v[:N_TEST], v[N_TEST:]
    return {"clock": clock, "placebo_set": placebo_set, "cohort": cohort, "weights": weights, "observed": obs, "n_all": int(len(v)),
            "p_all": rank_p(obs, v, False), "p_half_1": rank_p(obs, a, False), "p_half_2": rank_p(obs, b, False),
            "half_1_mean": float(a.mean()), "half_2_mean": float(b.mean()), "half_1_q95": float(np.quantile(a, 0.95)), "half_2_q95": float(np.quantile(b, 0.95)),
            "sd_all": float(v.std(ddof=1))}


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="size of the level-rank reference rule on the stored zero-effect draws")
    ap.parse_args(argv)
    log = fc.logger("size_level_rank")
    t0 = time.time()
    rows, entries = [], []
    for clock in ("T", "P"):
        for flavour in ("N2", "N1"):
            for dup in ("default", "keepdups"):
                p = CACHE / f"placebo_{clock}_{flavour}_{dup}.parquet"
                if not p.exists():
                    continue
                t = pd.read_parquet(p)
                rows += size_rows(t, clock, flavour, dup)
                entries.append(fc.file_entry(p, rows=len(t)))
                log(f"{p.name}: {t['key'].nunique()} specs, {len(t)} rows")
        ct = corrected_table(clock)
        if ct is not None:
            for flavour, t in ct.groupby("flavour", sort=True):
                rows += size_rows(t, clock, str(flavour), "corrected_final")
            p = CORRECTED / f"reference_draw_statistics_{clock}.parquet"
            entries.append(fc.file_entry(p, rows=len(ct)))
            log(f"{p.name}: {ct['key'].nunique()} specs, {len(ct)} rows")
    out = pd.DataFrame(rows)
    exch = exchangeable_mc()
    for k, v in exch.items():
        out[k] = v
    log("exchangeable draws: per-direction rate mean {mc_mean_direction_exch:.4f} sd {mc_sd_direction_exch:.4f}; pooled mean {mc_mean_pooled_exch:.4f} sd {mc_sd_pooled_exch:.4f}; "
        "correlation of the two directions {mc_corr_directions_exch:.3f} ({mc_reps_exch} replications)".format(**exch))
    fc.write_csv(out, OUT)
    half = pd.DataFrame(half_set_rows())
    fc.write_csv(half, OUT_HALF)
    mech = pd.DataFrame(half_mechanics_rows())
    fc.write_csv(mech, OUT_MECH)
    if len(mech):
        print(mech.round(4).to_string(index=False))
    for e in (RESULTS, CORRECTED_REFERENCE):
        if e.exists():
            entries.append(fc.file_entry(e))
    fc.write_manifest("size_level_rank", entries, config={"alpha": ALPHA, "n_test": N_TEST, "rule": "add-one rank of the level, test draws 0-999 against reference draws 1000-1999, and the halves swapped",
                                                          "exchangeable_mc": {"reps": EXCH_REPS, "seed": EXCH_SEED}, "placebo_sets": ["default", "keepdups", "corrected_final"]},
                      extra={"seconds": round(time.time() - t0, 1), "n_rows": int(len(out)), **exch})
    pd.set_option("display.width", 250)
    head = out[(out["statistic"] == "tau2") & (out["flavour"] == "N2") & (out["analysis"].str.startswith("tau/") | (out["placebo_set"] == "corrected_final"))]
    print(head[["clock", "placebo_set", "analysis", "winsorized", "weights", "reject_rate_reference", "reject_rate_reverse_halves", "reject_rate_both_halves", "mc_sd_direction_exch", "mc_sd_pooled_exch"]].round(4).to_string(index=False))
    print(half.round(4).to_string(index=False))
    log(f"written {OUT} ({len(out)} rows) and {OUT_HALF} ({len(half)} rows) in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
