"""Reproduce the two numbers the extension plan names by calling the estimators, then diff them
against the files on record: the de-duplicated public-clock spread (existing/rerun.py) and the
headline persistence covariance (estimators/results.py). Writes data/v4/ext/reproduction.csv."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as panel_mod

REPO_ROOT = panel_mod.REPO_ROOT
OUT_DIR = REPO_ROOT / "data" / "v4" / "ext"
OUT_CSV = OUT_DIR / "reproduction.csv"
RESULTS = REPO_ROOT / "data" / "v4" / "results.csv"
RERUN = REPO_ROOT / "data" / "v4" / "existing" / "rerun_tables.csv"


def spread_rows() -> list[dict]:
    """TW_P1_E20 on the original and de-duplicated samples with day and 20-day-block clusters."""
    calendar = rerun.load_calendar()
    signals = rerun.load_signals()
    dedup, report = rerun.dedup_filter(signals, rerun.duplicate_signal_uids())
    on_file = pd.read_csv(RERUN)
    test = next(t for t in rerun.TESTS if t.test_id == rerun.PRIMARY_TEST_ID)
    rows = []
    for sample, frame in (("original", signals), ("dedup", dedup)):
        for clusters in rerun.CLUSTERS:
            r = rerun.run_test(frame, test, clusters, calendar)
            ref = on_file[(on_file.test_id == test.test_id) & (on_file["sample"] == sample) & (on_file.clusters == clusters)].iloc[0]
            for stat, val, ref_val in (("estimate_bp", r["estimate"] * rerun.BP, ref["estimate"]), ("se_bp", r["se"] * rerun.BP, ref["se"]),
                                       ("n_signals", r["n"], ref["n"]), ("n_households", r["n_households"], ref["n_households"]),
                                       ("n_dates_or_blocks", r["n_dates_or_blocks"], ref["n_dates_or_blocks"])):
                rows.append({"quantity": f"public_clock_spread/{test.test_id}", "sample": sample, "clusters": clusters, "statistic": stat,
                             "recomputed": float(val), "on_file": float(ref_val), "abs_diff": abs(float(val) - float(ref_val)),
                             "source_on_file": "data/v4/existing/rerun_tables.csv"})
    rows.append({"quantity": "public_clock_spread/dedup_filter", "sample": "dedup", "clusters": "", "statistic": "n_dropped_duplicates",
                 "recomputed": report["n_dropped"], "on_file": 3352, "abs_diff": abs(report["n_dropped"] - 3352), "source_on_file": "plan §2"})
    return rows


def tau_rows(clocks=("T", "P")) -> list[dict]:
    """Headline tau2 (event-share weights, block clusters) on the actual event frames."""
    calendar = panel_mod.v2_trading_calendar()
    sp.register_household_exclusion(sp.EXCL_TWO_VENDOR, evf.blind_vendor_households(2))
    frames = {c: evf.load_event_frame(c, calendar=calendar) for c in clocks}
    ctx = rs.Context(frames=frames, calendar=calendar, cutoffs=evf.pooled_cutoffs, placebo_source=lambda *a: iter(()))
    on_file = pd.read_csv(RESULTS)
    rows = []
    for c in clocks:
        spec = sp.headline_spec(c)
        panel = ctx.panel(spec)
        pb = rs.block_cluster(panel, calendar)
        r = cgm.tau2_i1(pb, "ES", anchor="anchor_blk")
        table = tau_mod.household_table(panel, "ES")
        cov = tau_mod.tau2_household_covariance(table)
        ref = on_file[(on_file.clock == c) & (on_file.analysis == "tau") & (on_file.variant == "event_share")].set_index("statistic")
        checks = (("tau2_bp2", r["tau2_hat"], ref.loc["tau2", "value"]), ("tau_bp", r["tau_hat"], ref.loc["tau", "value"]),
                  ("se_block_bp2", r["var_wx"] * r["se_b"], ref.loc["tau2", "se_block"]), ("t_block", r["t_stat"], ref.loc["t_block", "value"]),
                  ("n_households", r["n_households"], ref.loc["tau2", "n_households"]), ("n_events", r["n_events_a"] + r["n_events_b"], ref.loc["tau2", "n_events"]),
                  ("tau2_household_covariance_identity", cov, r["tau2_hat"]))
        for stat, val, ref_val in checks:
            rows.append({"quantity": f"persistence_tau2/{c}", "sample": "default", "clusters": "block20", "statistic": stat, "recomputed": float(val),
                         "on_file": float(ref_val), "abs_diff": abs(float(val) - float(ref_val)), "source_on_file": "data/v4/results.csv"})
        rows.append({"quantity": f"persistence_tau2/{c}", "sample": "default", "clusters": "", "statistic": "n_events_A", "recomputed": r["n_events_a"],
                     "on_file": np.nan, "abs_diff": np.nan, "source_on_file": "notes column of the tau2 row"})
        rows.append({"quantity": f"persistence_tau2/{c}", "sample": "default", "clusters": "", "statistic": "n_events_B", "recomputed": r["n_events_b"],
                     "on_file": np.nan, "abs_diff": np.nan, "source_on_file": "notes column of the tau2 row"})
    return rows


def main() -> pd.DataFrame:
    t0 = time.time()
    rows = spread_rows() + tau_rows()
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    summary = {"max_abs_diff": float(out["abs_diff"].max()), "n_checks": int(out["abs_diff"].notna().sum()),
               "seconds": round(time.time() - t0, 1), "written": str(OUT_CSV.relative_to(REPO_ROOT))}
    (OUT_DIR / "reproduction.json").write_text(json.dumps(summary, indent=2))
    print(out.to_string(index=False))
    print(json.dumps(summary))
    return out


if __name__ == "__main__":
    main()
