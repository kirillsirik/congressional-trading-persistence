"""Reproduce the frozen baseline at full precision before any addition.

python -m trading_on_disclosure_v4.round2.baseline
Writes <run>/baseline_reconciliation.csv and <run>/input_inventory.csv. The spreads are recomputed with the original
functions (ext.reproduce.spread_rows for the initial match; final.cumulative's chain, in memory, for the extension).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.ext import reproduce as erp
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.final import cumulative as cu
from trading_on_disclosure_v4.outcomes import forward_cache as fcache
from trading_on_disclosure_v4.outcomes import panel as pm
from trading_on_disclosure_v4.round2 import common as cm
from trading_on_disclosure_v4.round2 import exports as ex

OUT = cm.RUN_DIR / "baseline_reconciliation.csv"
FP_TOL = 1e-9  # relative tolerance for recomputed floating-point values (different summation order only)


def row(key: str, archived: float, reproduced: float, units: str, source: str, tol: float = FP_TOL, inputs: str = "") -> dict:
    archived, reproduced = float(archived), float(reproduced)
    diff = abs(reproduced - archived)
    rel = diff / max(1.0, abs(archived))
    exact = units == "count"
    ok = diff == 0 if exact else rel <= tol
    return {"specification_key": key, "archived_value": archived, "reproduced_value": reproduced, "units": units, "abs_diff": diff, "rel_diff": rel,
            "tolerance": "exact" if exact else f"relative {tol:g}", "archived_source": source, "inputs": inputs, "verdict": "PASS" if ok else "FAIL"}


def persistence_rows(ctx, key, log) -> list[dict]:
    rows = []
    inputs = "events_{T,P}.parquet; forward_cache.parquet; pooled_outcome_distribution.csv"
    for clock in ("T", "P"):
        f1, m1 = ex.principal_means(ctx, clock, key, log)
        cov = m1["covariances_bp2"]
        lab = {"T": "trade date", "P": "disclosure date"}[clock]
        rows.append(row(f"{clock} initial persistence cohort: households", cm.frozen_value(clock, "tau", "event_share", "tau2", "n_households"), m1["n_households"], "count", "results.csv tau/event_share/tau2 n_households", inputs=inputs))
        rows.append(row(f"{clock} initial persistence cohort: purchases", cm.frozen_value(clock, "tau", "event_share", "tau2", "n_events"), m1["n_purchases"], "count", "results.csv tau/event_share/tau2 n_events", inputs=inputs))
        rows.append(row(f"{clock} fitted-intercept covariance, winsorized ({lab})", cm.frozen_value(clock, "tau", "event_share", "tau2"), cov["winsorized_Y1"], "bp2", "results.csv tau/event_share/tau2", inputs=inputs))
        rows.append(row(f"{clock} intercept-omitted covariance, fixed loadings, winsorized", cm.frozen_value(clock, "robustness", "no_alpha", "tau2"), cov["winsorized_Y0"], "bp2", "results.csv robustness/no_alpha/tau2", inputs=inputs))
        rows.append(row(f"{clock} fitted-intercept covariance, unwinsorized", cm.frozen_value(clock, "decomposition", "no_alpha_omit", "tau2_total_raw", path=cm.FROZEN_DECOMP), cov["raw_Y1"], "bp2", "ext/decomposition.csv no_alpha_omit/tau2_total_raw", inputs=inputs))
        rows.append(row(f"{clock} intercept-omitted covariance, unwinsorized", cm.frozen_value(clock, "decomposition", "no_alpha_omit", "tau2_base_raw", path=cm.FROZEN_DECOMP), cov["raw_Y0"], "bp2", "ext/decomposition.csv no_alpha_omit/tau2_base_raw", inputs=inputs))
        _, m2, _ = ex.table3_means(ctx, clock, key, log)
        for m in m2:
            a, v = m["results_key"].split("/")[:2]
            rows.append(row(f"{clock} Table 3 {m['specification']} covariance", cm.frozen_value(clock, a, v, "tau2"), m["kappa_bp2"], "bp2", f"results.csv {m['results_key']}", inputs=inputs))
        _, m3 = ex.household_tests(ctx, clock, key, log)
        rows.append(row(f"{clock} broad household tests: households (>= 20 purchases)", cm.frozen_value(clock, "best_household", "vs_centered_placebo", "n_p05", "n_households"), m3["n_households"], "count", "results.csv best_household/vs_centered_placebo/n_p05 n_households", inputs=inputs))
        rows.append(row(f"{clock} broad household tests: two-sided 5% rejections", cm.frozen_value(clock, "best_household", "vs_centered_placebo", "n_p05"), m3["n_rejections"], "count", "results.csv best_household/vs_centered_placebo/n_p05", inputs=inputs))
    return rows


def spread_rows(log) -> list[dict]:
    rows = []
    for r in erp.spread_rows():
        if r["sample"] == "dedup" and r["clusters"] == "block20" and r["statistic"] in ("estimate_bp", "se_bp", "n_signals", "n_households", "n_dates_or_blocks"):
            units = {"estimate_bp": "bp", "se_bp": "bp", "n_signals": "count", "n_households": "count", "n_dates_or_blocks": "count"}[r["statistic"]]
            rows.append(row(f"initial post-disclosure purchase-minus-sale spread (deduplicated, household x 20-session block): {r['statistic']}",
                            r["on_file"], r["recomputed"], units, "existing/rerun_tables.csv TW_P1_E20 dedup block20",
                            inputs="outputs/frames/signals.csv; event_duplicate_flags.parquet"))
    lg = cm.logger("baseline_cumulative")
    u = cu.build_union(lg)
    uni = cu.public_universe_corrected(u, lg)
    base_uni = cu.baseline_universe(u)
    market = w.Market(u["calendar"], pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    market.load(uni["permno"].unique())
    comp, led = cu.completions(uni, market, lg)
    spread, vectors = cu.spread_rows(uni, comp, led, base_uni, u["calendar"], lg)
    on_file = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "final" / "spread_corrected.csv")
    for s in spread:
        if s["statistic"] != "spread":
            continue
        f = on_file[(on_file["variant"] == s["variant"]) & (on_file["statistic"] == "spread")].iloc[0]
        for stat, a, b, units in (("estimate", f["value"], s["value"], "bp"), ("se_block", f["se_block"], s["se_block"], "bp"),
                                  ("n_signals", f["n_events"], s["n_events"], "count"), ("n_households", f["n_households"], s["n_households"], "count")):
            rows.append(row(f"extended spread {s['variant']}: {stat}", a, b, units, f"final/spread_corrected.csv {s['variant']}/spread",
                            inputs="timed/matched/signals frames; repair tiers; forward cache + extension; July 2026 CRSP slice"))
    log(f"extended spread chain rebuilt in memory: {len(uni)} signals in the corrected universe")
    cm.write_csv(uni[["event_id", "household", "direction", "period", "blk"]].assign(y_primary=vectors[cu.PRIMARY_SPREAD]),
                 cm.RESTRICTED / "inputs" / "extended_spread_design.csv")
    return rows


def macro_rows() -> list[dict]:
    """The Round 2 macros already in the manuscript, against this run's household-level recomputation."""
    import json

    src = json.loads((cm.REPO_ROOT / "data" / "v4" / "round2" / "macro_sources.json").read_text())
    spec = pd.read_csv(cm.RUN_DIR / "influence" / "specification_statistics.csv")
    t3 = pd.read_csv(cm.INPUTS / "table3_household_means.csv")
    tests = pd.read_csv(cm.INPUTS / "household_test_statistics.csv")
    rows = []
    for clock, name in (("T", "Trade"), ("P", "Disclosure")):
        s = spec[(spec["clock"] == clock) & (spec["specification"] == "principal")].iloc[0]
        g = t3[(t3["clock"] == clock) & (t3["specification"] == "principal")]
        wv = g["weight"].to_numpy(float)
        a = wv / wv.sum()
        tt_ = tests[tests["clock"] == clock]
        vals = {"Slope": s["slope"], "LOOMin": s["loo_kappa_min_bp2"], "LOOMax": s["loo_kappa_max_bp2"], "LOOMaxAbsChange": s["loo_max_abs_change_bp2"],
                "IHH": 1 / np.sum(a**2), "ProductIHH": (np.sum((g["n_A"] * g["n_B"]).to_numpy(float)) ** 2) / np.sum((g["n_A"] * g["n_B"]).to_numpy(float) ** 2),
                "MaxGeometricWeightPct": 100 * a.max(), "NegativeMeanCount": int((tt_["mean_bp"] < 0).sum()), "PositiveMeanCount": int((tt_["mean_bp"] > 0).sum()),
                "BroadHouseholdCount": int(len(tt_))}
        for k, v in vals.items():
            m = src[f"RoundTwo{name}{k}"]
            rows.append(row(f"{clock} Round 2 macro RoundTwo{name}{k}", m["value"], v, "count" if "Count" in k else "value", f"data/v4/round2/macro_sources.json ({m['source']})"))
    return rows


def inventory() -> pd.DataFrame:
    """Logical inputs of the closeout with resolved private paths, status, hashes, coverage and dependents."""
    from trading_on_disclosure_v4.final import cache_extension as cx
    from trading_on_disclosure_v4.final import history_reread as rr

    root = pm.project_root()
    items = [
        ("event table, trade date", cm.REPO_ROOT / "data/v4/restricted/data/events_T.parquet", "trading_on_disclosure_v4.data.build", "exports 1-5, C1-C5, C7, C9"),
        ("event table, disclosure date", cm.REPO_ROOT / "data/v4/restricted/data/events_P.parquet", "trading_on_disclosure_v4.data.build", "exports 1-5, C1-C5, C7"),
        ("event duplicate flags", cm.REPO_ROOT / "data/v4/restricted/data/event_duplicate_flags.parquet", "outcomes.event_filters", "all event frames"),
        ("forward-return cache (FF5+MOM, market-adjusted)", fcache.CACHE_PATH, "outcomes.forward_cache", "all outcomes; C7"),
        ("forward-cache extension (repaired securities)", cx.OUT_CACHE, "final.cache_extension", "extended sample; C7"),
        ("local daily CRSP panel with delisting returns", pm.daily_parquet_path(), "outcomes.panel.build_daily_parquet", "C7 window audit; spread completion"),
        ("CRSP daily 2014+ (Dropbox)", root / "data/processed/crsp_daily_2014plus.parquet", "scripts/v2/22_build_market_panel_2014plus.py", "source of the local panel"),
        ("benchmark daily 2014+ (Dropbox)", root / "data/processed/benchmark_daily_2014plus.parquet", "scripts/v2/22_build_market_panel_2014plus.py", "trading calendar; market-adjusted outcomes"),
        ("Fama-French five factors + momentum (Dropbox)", root / "data/raw/factors/factors_daily.csv", "Ken French data library", "factor fits (C7)"),
        ("committed pooled winsorization cutoffs", evf.POOLED_CSV, "outcomes.pooled", "winsorized exports, C1-C3"),
        ("centered reference panels, trade date (2,000 draws)", cm.REPO_ROOT / "data/v4/restricted/n/panels/T/N2/A0_T_purchase_AB_L20_both_keepdups", "null.run_null_panels", "C4 (draws 0-499), export 4"),
        ("centered reference panels, disclosure date (2,000 draws)", cm.REPO_ROOT / "data/v4/restricted/n/panels/P/N2/A0_P_purchase_AB_L20_both_alllines", "null.run_null_panels", "C4 (draws 0-499), export 4"),
        ("pooled-mean experiment checkpoints", cm.REPO_ROOT / "data/v4/restricted/c/checkpoints/activity_power", "ext.activity_power", "C4 verification"),
        ("signal frame (public clock)", cm.REPO_ROOT / "outputs/frames/signals.csv", "v2 pipeline", "spreads"),
        ("timed transactions", cm.REPO_ROOT / "outputs/frames/timed_transactions.csv", "v2 pipeline", "extended sample; reconstruction"),
        ("matched transactions", cm.REPO_ROOT / "outputs/frames/matched_transactions.csv", "v2 pipeline", "extended sample; reconstruction"),
        ("re-read history rows (two households)", rr.ROWS_PARQUET, "final.history_reread", "C8"),
        ("re-read filing table", rr.FILINGS_PARQUET, "final.history_reread", "C8"),
        ("frozen results", cm.FROZEN_RESULTS, "scripts/v4/run_results.py", "comparators"),
        ("frozen decomposition", cm.FROZEN_DECOMP, "ext.decompose", "comparators"),
        ("frozen initial spread tables", cm.REPO_ROOT / "data/v4/existing/rerun_tables.csv", "existing.rerun", "comparators"),
        ("frozen extended spreads", cm.REPO_ROOT / "data/v4/final/spread_corrected.csv", "final.cumulative", "comparators"),
        ("frozen reconstruction estimates", cm.REPO_ROOT / "data/v4/final/history_recovery_final_estimates.csv", "final.history_recovery_final", "C8 comparator"),
    ]
    rows = []
    for name, path, producer, deps in items:
        p = path
        exists = p.exists()
        rec = {"logical_input": name, "resolved_private_path": str(p), "exists": exists, "readable": exists and (p.is_dir() or p.stat().st_size > 0),
               "sha256": "", "bytes": np.nan, "schema_or_contents": "", "coverage": "", "producing_code": producer, "dependent_tasks": deps}
        if exists and p.is_file():
            rec["sha256"] = cm.sha256_file(p)
            rec["bytes"] = p.stat().st_size
            if p.suffix == ".parquet":
                import pyarrow.parquet as pq

                md = pq.read_metadata(p)
                rec["schema_or_contents"] = f"{md.num_rows} rows; columns " + ", ".join(md.schema.names[:12]) + ("..." if len(md.schema.names) > 12 else "")
                names = md.schema.names
                date_col = next((c for c in ("anchor_date", "date", "follower_entry_date") if c in names), None)
                if date_col:
                    col = pd.read_parquet(p, columns=[date_col])[date_col]
                    rec["coverage"] = f"{date_col} {pd.to_datetime(col).min().date()} to {pd.to_datetime(col).max().date()}"
            elif p.suffix == ".csv":
                head = pd.read_csv(p, nrows=0)
                rec["schema_or_contents"] = "columns " + ", ".join(head.columns[:12]) + ("..." if len(head.columns) > 12 else "")
        elif exists and p.is_dir():
            files = sorted(p.glob("*.parquet"))
            rec["schema_or_contents"] = f"{len(files)} parquet files"
            rec["sha256"] = cm.sha256_bytes("\n".join(f"{f.name}:{f.stat().st_size}" for f in files).encode()) + " (names and sizes)"
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="baseline reconciliation and input inventory")
    ap.parse_args(argv)
    log = cm.logger("baseline")
    t0 = time.time()
    ctx, _ = cm.load_context()
    key = pd.read_csv(cm.CROSSWALK).set_index("household_id")["household_index"]
    rows = persistence_rows(ctx, key, log) + spread_rows(log) + macro_rows()
    df = pd.DataFrame(rows)
    cm.write_csv(df, OUT)
    inv = inventory()
    cm.write_csv(inv, cm.RESTRICTED / "private" / "input_inventory_private.csv")
    public = inv.assign(resolved_private_path=inv["logical_input"].map(lambda s: "<private; see restricted run record>"))
    cm.write_csv(public, cm.RUN_DIR / "input_inventory.csv")
    log(f"baseline: {int((df['verdict'] == 'PASS').sum())} PASS, {int((df['verdict'] == 'FAIL').sum())} FAIL; max rel diff {df.loc[df['units'] != 'count', 'rel_diff'].max():.2e}; "
        f"{time.time() - t0:.0f} s")
    if (df["verdict"] == "FAIL").any():
        print(df[df["verdict"] == "FAIL"].to_string())


if __name__ == "__main__":
    main()
