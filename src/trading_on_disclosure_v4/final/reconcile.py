"""Reconciliation of the corrected household panels with the record (verification pass of 7 November 2026).

The cumulative module reports the headline covariance on three panels per clock: the record (baseline), the
baseline households on the corrected frame (common cohort) and the corrected sample with eligibility reassessed.
This module rebuilds the three panels from the released event frame and the corrected frame with the committed
cutoffs, checks the recomputed covariances against persistence_corrected.csv, and accounts for every purchase that
enters or leaves between panels: whether it is a repaired line (an event id absent from the record) or a released
purchase, whether any outcome shared by two panels changed, and how the households that become eligible qualify.
The change in each decomposition component between panels is read from decomposition_corrected.csv and split into
the part on the common cohort (data corrections) and the part from the added households (eligibility).
Writes data/v4/final/reconciliation.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

OUT = fc.OUT_DIR / "reconciliation.csv"
PERSISTENCE = fc.OUT_DIR / "persistence_corrected.csv"
DECOMPOSITION = fc.OUT_DIR / "decomposition_corrected.csv"
COHORTS = ("baseline", "common_cohort", "corrected")
TERMS = ("total", "base", "adj", "cross_base_adj", "cross_adj_base")
TOL = 1e-6


def purchase_counts(frame: pd.DataFrame, household: str, ycol: str = "y_h20") -> dict[str, tuple[int, int, int]]:
    """{period: (purchases with a finite outcome, calendar quarters, securities)} of one household, duplicates removed."""
    f = frame.loc[(frame["household_id"].astype(str) == household) & (frame["direction"] == "purchase") & ~frame["dup_all_lines"].astype(bool)]
    f = f.loc[np.isfinite(pd.to_numeric(f[ycol], errors="coerce").to_numpy(dtype=float))]
    out = {}
    for period, g in f.groupby("period"):
        out[str(period)] = (int(len(g)), int(pd.DatetimeIndex(g["anchor_date"]).to_period("Q").nunique()), int(g["permno"].nunique()))
    return out


def panel_moves(base: pd.DataFrame, other: pd.DataFrame, released_ids: set[str], ycol: str = "y_h20") -> dict:
    """Purchases of `other` not in `base` (split by repaired vs released ids and by period), purchases of `base` not in
    `other`, and the largest change of a shared purchase's outcome."""
    b_ids, o_ids = set(base["event_id"].astype(str)), set(other["event_id"].astype(str))
    added = other.loc[~other["event_id"].astype(str).isin(b_ids)]
    dropped = base.loc[~base["event_id"].astype(str).isin(o_ids)]
    repaired = ~added["event_id"].astype(str).isin(released_ids)
    shared = base[["event_id", ycol]].merge(other[["event_id", ycol]], on="event_id", suffixes=("_b", "_o"))
    change = float(np.abs(shared[f"{ycol}_b"].astype(float) - shared[f"{ycol}_o"].astype(float)).max()) if len(shared) else 0.0
    return {"added": int(len(added)), "added_A": int((added["period"] == "A").sum()), "added_B": int((added["period"] == "B").sum()),
            "added_repaired": int(repaired.sum()), "added_released": int((~repaired).sum()), "dropped": int(len(dropped)),
            "shared": int(len(shared)), "shared_outcome_max_abs_change": change}


def tau2(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    return cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")


def reconcile_clock(clock: str, calendar: pd.DatetimeIndex, cutoffs, log) -> list[dict]:
    spec = sp.headline_spec(clock)
    cuts = cutoffs(clock, spec.outcome)
    old = evf.load_event_frame(clock, calendar=calendar)
    new = pd.read_parquet(fc.RESTRICTED / "cumulative" / f"frame_{clock}_corrected.parquet")
    released_ids = set(old["event_id"].astype(str))
    base = sp.to_panel(old, spec, cuts)
    corr = sp.to_panel(new, spec, cuts)
    base_hh = set(base["household_id"].astype(str))
    common = sp.to_panel(new.loc[new["household_id"].astype(str).isin(base_hh)], spec, cuts)
    panels = {"baseline": base, "common_cohort": common, "corrected": corr}
    file_tau = pd.read_csv(PERSISTENCE)
    rows = []
    max_diff = 0.0
    for label, panel in panels.items():
        r = tau2(panel, calendar)
        on_file = float(file_tau.loc[(file_tau["clock"] == clock) & (file_tau["variant"] == label) & (file_tau["statistic"] == "tau2"), "value"].iloc[0])
        max_diff = max(max_diff, abs(r["tau2_hat"] - on_file))
        n_hh = int(panel["household_id"].nunique())
        rows.append(rs.row(clock, "reconciliation", label, "tau2", r["tau2_hat"], "bp2", n_events=int(len(panel)), n_households=n_hh, notes=f"recomputed; on file {on_file:.6f}"))
        rows.append(rs.row(clock, "reconciliation", label, "purchases", int(len(panel)), "count", n_events=int(len(panel)), n_households=n_hh))
        rows.append(rs.row(clock, "reconciliation", label, "purchases_A", int((panel["period"] == "A").sum()), "count", n_events=int(len(panel)), n_households=n_hh))
        rows.append(rs.row(clock, "reconciliation", label, "purchases_B", int((panel["period"] == "B").sum()), "count", n_events=int(len(panel)), n_households=n_hh))
        rows.append(rs.row(clock, "reconciliation", label, "households", n_hh, "count", n_events=int(len(panel)), n_households=n_hh))
        log(f"{clock} {label}: tau2 {r['tau2_hat']:.3f} (file {on_file:.3f}), purchases {len(panel)}, households {n_hh}")
    rows.append(rs.row(clock, "reconciliation", "verification", "tau2_max_abs_diff_vs_file", max_diff, "bp2", notes=f"tolerance {TOL}"))
    if max_diff > TOL:
        raise ValueError(f"{clock}: recomputed covariances differ from persistence_corrected.csv by {max_diff}")
    # purchases that move between panels
    for variant, a, b in (("common_vs_baseline", base, common), ("corrected_vs_common", common, corr)):
        m = panel_moves(a, b, released_ids)
        for k, v in m.items():
            rows.append(rs.row(clock, "reconciliation", variant, k, v, "bp" if k.endswith("change") else "count",
                               notes="repaired = event id absent from the released frame" if k.startswith("added_re") else ""))
        log(f"{clock} {variant}: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    # households that enter or leave, and how the entrants qualify
    corr_hh = set(corr["household_id"].astype(str))
    added_hh = sorted(corr_hh - base_hh, key=lambda h: -int((corr["household_id"].astype(str) == h).sum()))
    dropped_hh = base_hh - corr_hh
    rows.append(rs.row(clock, "reconciliation", "households", "added", len(added_hh), "count"))
    rows.append(rs.row(clock, "reconciliation", "households", "dropped", len(dropped_hh), "count"))
    ad = corr.loc[corr["household_id"].astype(str).isin(added_hh)]
    rows.append(rs.row(clock, "reconciliation", "households", "added_purchases", int(len(ad)), "count"))
    rows.append(rs.row(clock, "reconciliation", "households", "added_purchases_A", int((ad["period"] == "A").sum()), "count"))
    rows.append(rs.row(clock, "reconciliation", "households", "added_purchases_B", int((ad["period"] == "B").sum()), "count"))
    rows.append(rs.row(clock, "reconciliation", "households", "added_purchases_repaired", int((~ad["event_id"].astype(str).isin(released_ids)).sum()), "count"))
    for i, h in enumerate(added_hh, start=1):
        before, after = purchase_counts(old, h), purchase_counts(new, h)
        for period in ("A", "B"):
            for when, counts in (("before", before), ("after", after)):
                n, q, s = counts.get(period, (0, 0, 0))
                rows.append(rs.row(clock, "reconciliation", f"added_household_{i}", f"{period}_purchases_{when}", n, "count", notes=f"{q} quarters, {s} securities"))
                rows.append(rs.row(clock, "reconciliation", f"added_household_{i}", f"{period}_quarters_{when}", q, "count"))
                rows.append(rs.row(clock, "reconciliation", f"added_household_{i}", f"{period}_securities_{when}", s, "count"))
    # decomposition components: change on the common cohort (corrections) and from the added households (eligibility)
    dec = pd.read_csv(DECOMPOSITION)
    for alt in ("no_alpha_omit", "market_adjusted"):
        for kind in ("raw", "wins"):
            vals = {}
            for cohort in COHORTS:
                sub = dec.loc[(dec["clock"] == clock) & (dec["variant"] == f"{cohort}|{alt}")]
                vals[cohort] = {t: float(sub.loc[sub["statistic"] == f"tau2_{t}_{kind}", "value"].iloc[0]) for t in TERMS}
            for t in TERMS:
                rows.append(rs.row(clock, "reconciliation", f"delta_{alt}", f"{t}_{kind}_cohort", vals["common_cohort"][t] - vals["baseline"][t], "bp2", notes="common cohort minus baseline"))
                rows.append(rs.row(clock, "reconciliation", f"delta_{alt}", f"{t}_{kind}_added", vals["corrected"][t] - vals["common_cohort"][t], "bp2", notes="corrected minus common cohort"))
    return rows


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="reconcile the corrected household panels with the record")
    ap.add_argument("--clocks", default="T,P")
    args = ap.parse_args(argv)
    log = fc.logger("reconcile")
    t0 = time.time()
    calendar = pm.v2_trading_calendar()
    ctx, _ = ec.load_headline_context(tuple(args.clocks.split(",")), with_placebo=False)
    rows = []
    for clock in args.clocks.split(","):
        rows += reconcile_clock(clock, calendar, ctx.cutoffs, log)
    out = fc.write_rows(rows, OUT)
    entries = [fc.file_entry(fc.RESTRICTED / "cumulative" / f"frame_{c}_corrected.parquet") for c in args.clocks.split(",")]
    entries += [fc.file_entry(PERSISTENCE), fc.file_entry(DECOMPOSITION)]
    fc.write_manifest("reconcile", entries, config={"clocks": args.clocks.split(","), "weights": "ES", "rule": "R1", "cutoffs": "committed pooled p1/p99 by clock and period", "tolerance": TOL},
                      extra={"seconds": round(time.time() - t0, 1), "n_rows": int(len(out))})
    pd.set_option("display.width", 250)
    print(out[["clock", "variant", "statistic", "value", "n_events", "n_households"]].to_string(index=False))
    log(f"written {OUT} ({len(out)} rows) in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
