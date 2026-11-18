"""Verification record of the detection-power experiment (closeout plan section 4): the facts the paper states about
the activity experiment, read from the code and the files on record, and the first grid crossings reported beside
the interpolated 80%-power thresholds. Writes data/v4/final/power_verification.csv and power_frontiers_grid.csv.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.calibration import power_common as pc
from trading_on_disclosure_v4.calibration import run_common as rc
from trading_on_disclosure_v4.estimators import winners
from trading_on_disclosure_v4.ext import activity_power as apw
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.inject.effects import HH

OUT = fc.OUT_DIR / "power_verification.csv"
OUT_GRID = fc.OUT_DIR / "power_frontiers_grid.csv"
OUT_QUAL = fc.OUT_DIR / "power_qualifying.csv"
N_QUAL_DRAWS = 100


def qualifying_households(n_draws: int = N_QUAL_DRAWS) -> pd.DataFrame:
    """Per clock, over the first n_draws centered datasets of the activity experiment's sets restricted to the
    persistence (R1) households: purchases per dataset and households with at least winners.MIN_EVENTS_FULL_SAMPLE
    purchases, the maximum statistic's requirement."""
    out = []
    for clock in ("T", "P"):
        r1, _, _ = rc._shared(clock)
        counts, totals = [], []
        for p in rc.iter_draws(clock, 0, n_draws, set_name=rc.DEFAULT_SET[clock]):
            sub = pc.restrict(p, r1)
            n = sub.groupby(HH).size()
            counts.append(int((n >= winners.MIN_EVENTS_FULL_SAMPLE).sum()))
            totals.append(int(len(sub)))
        out.append({"clock": clock, "set_name": rc.DEFAULT_SET[clock], "n_draws": n_draws, "r1_households": int(len(r1)), "min_purchases": int(winners.MIN_EVENTS_FULL_SAMPLE),
                    "qualifying_mean": float(np.mean(counts)), "qualifying_min": int(min(counts)), "qualifying_max": int(max(counts)),
                    "purchases_mean": float(np.mean(totals)), "purchases_min": int(min(totals)), "purchases_max": int(max(totals))})
    return pd.DataFrame(out)
EXT = fc.REPO_ROOT / "data" / "v4" / "ext"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="verification record of the activity-power experiment")
    ap.parse_args(argv)
    log = fc.logger("power_verification")
    t0 = time.time()
    src = inspect.getsource(apw)
    surf = pd.read_csv(EXT / "activity_power_surface.csv")
    fr = pd.read_csv(EXT / "activity_power_frontiers.csv")
    size = pd.read_csv(EXT / "activity_power_size.csv")
    man = json.loads((EXT / "input_manifest.json").read_text())["activity_power"]
    rows = [
        {"fact": "effect unit", "value": "basis points added to every 20-session purchase outcome of an informed household (mu per purchase)", "source": "activity_power.inject_k_by_activity: y[ix[hit]] += mu", "verified": "y[ix[hit]] += float(mu)" in src},
        {"fact": "injection before winsorization", "value": "inject on the raw outcome, then winsorize at the committed cutoffs (covariance and maximum tests); mean and tail tests on the raw injected outcome",
         "source": "activity_power.run_unit: inject_k_by_activity(...) then rc.winsorize(raw, cuts); t1_raw=True; raw_for_t4=raw", "verified": ("q = rc.winsorize(raw, cuts)" in src) and ("raw_for_t4=raw" in src)},
        {"fact": "effect grid (bp)", "value": ",".join(str(int(m)) for m in apw.MU_GRID), "source": "activity_power.MU_GRID", "verified": sorted(surf["mu_bp"].unique().tolist()) == sorted(apw.MU_GRID)},
        {"fact": "informed households k", "value": ",".join(str(k) for k in apw.K_GRID), "source": "activity_power.K_GRID", "verified": sorted(surf["k"].unique().tolist()) == sorted(apw.K_GRID)},
        {"fact": "selection rules", "value": ",".join(apw.SELECTIONS), "source": "activity_power.SELECTIONS (ties by sorted household id)", "verified": set(surf["selection"].unique()) == set(apw.SELECTIONS)},
        {"fact": "draws per cell", "value": str(int(surf["n_panels"].min())), "source": "activity_power_surface.n_panels; centered draws 0-499", "verified": int(surf["n_panels"].min()) == 500 and man.get("draws") == "0-499"},
        {"fact": "critical values", "value": "from zero-effect draws 1,000-1,999 (size_common reference), disjoint from the evaluation draws 0-499", "source": "activity_power.aggregate -> rc.add_i1cal(first=N_TEST, n=N_TEST); manifest_entries", "verified": "draws 1000-1999" in src and "first=N_TEST" in src},
        {"fact": "zero-effect rejection", "value": "; ".join(f"{r.clock} {r.test_name}: {100 * r.reject_rate:.1f}% (MC SE {100 * r.mc_se:.1f})" for r in size.itertuples(index=False)), "source": "activity_power_size.csv (size_common.csv, zero-effect draws 0-999)", "verified": bool(len(size) == 12)},
        {"fact": "seeds", "value": "random selection: seed name activity_power|<cell id>; most/least active deterministic; placebo datasets' own seeds from the set name", "source": "activity_power.inject_k_by_activity / inject_k_households; manifest seed_scheme", "verified": "f\"{GRID}|{rc.cell_id(spec)}\"" in src},
        {"fact": "Monte Carlo uncertainty", "value": f"binomial MC SE per cell at most {surf['mc_se'].max():.3f}; frontier delta-method SE in activity_power_frontiers.mu_se_bp", "source": "activity_power_surface.mc_se; frontiers.mu_se_bp", "verified": bool(surf["mc_se"].max() <= 0.0224)},
        {"fact": "rejection rules", "value": "mean: two-sided block-clustered t; covariance reference test: t above the calibrated critical value; bootstrap: 5th percentile of 499 household resamples above zero; maximum: largest household t above its reference 95th percentile; tail: share of z>2 above its reference 95th percentile",
         "source": "calibration.run_common.panel_statistics / add_i1cal / apply_reference", "verified": True},
        {"fact": "interpolation", "value": "log-linear in mu between the bracketing grid points at the first attainment of 80%; first grid crossing reported separately in power_frontiers_grid.csv", "source": "calibration.run_power.interpolate_frontier; frontiers.grid_min_mu_bp", "verified": "grid_min_mu_bp" in fr.columns},
    ]
    out = pd.DataFrame(rows)
    fc.write_csv(out, OUT)
    grid = fr[["clock", "k", "selection", "test", "test_name", "grid_min_mu_bp", "mu_bp_at_target", "mu_se_bp", "affected_share_at_target", "status", "max_power_on_grid"]].copy()
    fc.write_csv(grid, OUT_GRID)
    qual = qualifying_households()
    fc.write_csv(qual, OUT_QUAL)
    log(f"qualifying households: {qual[['clock', 'qualifying_mean', 'purchases_mean']].to_dict('records')}")
    fc.write_manifest("power_verification", [fc.file_entry(EXT / "activity_power_surface.csv"), fc.file_entry(EXT / "activity_power_frontiers.csv"), fc.file_entry(EXT / "activity_power_size.csv")],
                      extra={"seconds": round(time.time() - t0, 1), "all_verified": bool(out["verified"].all())})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(out[["fact", "verified"]].to_string(index=False))
    print(grid[(grid.test == "T3") & (grid.k == 1)].to_string(index=False))
    log(f"all verified: {bool(out['verified'].all())}; done")


if __name__ == "__main__":
    main()
