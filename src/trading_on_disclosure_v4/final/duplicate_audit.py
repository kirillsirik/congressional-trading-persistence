"""Route-specific duplicate audit of the history recovery (closeout plan section 3c).

The recovery of 19 October reconciled cross-document repetitions on every resolved row (exact and near-exact
routes pooled) and then selected the exact-route subset, so an earlier near-exact row could in principle have
suppressed a later exact-route row. This module lists every suppressed row with the retained row's route and
filing, reports whether an exact-route row precedes each suppressed exact-route row, re-runs the
reconciliation separately per route, and reports the exact-route persistence estimate under both
reconciliations. Writes data/v4/final/duplicate_route_audit.csv and duplicate_route_audit_estimates.csv.
Identities stay in memory; the audit table carries no household, security or name.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data import events as evm
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.ext import history_recovery as hr
from trading_on_disclosure_v4.ext import ledgers as lg
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

OUT_AUDIT = fc.OUT_DIR / "duplicate_route_audit.csv"
OUT_EST = fc.OUT_DIR / "duplicate_route_audit_estimates.csv"
KEY = ["household", "permno", "date", "direction", "owner_std", "ink_bracket"]


def reconcile_routes(res: pd.DataFrame, routes: tuple[str, ...]) -> pd.DataFrame:
    """The released hierarchy applied to the rows of `routes` only (route-specific reconciliation)."""
    sub = res.loc[res["route"].isin(routes)]
    return hr.reconcile(sub)


def suppressed_table(rec_pooled: pd.DataFrame) -> pd.DataFrame:
    """One line per suppressed row: its route and filing sequence, the retained (earliest) row's route and filing,
    and whether any exact-route row with the same key precedes it."""
    d = rec_pooled.copy()
    d["_exact"] = d["route"].isin(hr.ROUTES["exact"])
    rows = []
    for _, g in d.groupby(KEY, sort=False):
        if len(g) < 2 or g["filing_seq"].nunique() < 2:
            continue  # repeats inside one document are separate lots, not cross-document repetitions
        g = g.sort_values(["filing_seq", "page"], kind="stable")
        first = g.iloc[0]
        for r in g.loc[g["filing_seq"] > first.filing_seq].itertuples(index=False):
            earlier = g[g["filing_seq"] < r.filing_seq]
            rows.append({"household": r.household, "suppressed_filing_seq": int(r.filing_seq), "suppressed_page": int(r.page),
                         "suppressed_route": r.route, "direction": r.direction, "owner": r.owner_std,
                         "transaction_year": int(pd.Timestamp(r.date).year), "retained_filing_seq": int(first.filing_seq),
                         "retained_route": first.route, "retained_filing_date_minus_suppressed_days": int((pd.Timestamp(first.filing_date) - pd.Timestamp(r.filing_date)).days),
                         "exact_route_row_precedes": bool(earlier["_exact"].any()),
                         "suppressed_exact_by_near_only": bool(r.route in hr.ROUTES["exact"] and not earlier["_exact"].any())})
    return pd.DataFrame(rows)


def exact_estimate(rec: pd.DataFrame, label_name: str, labels: dict, chamber: str, timed: pd.DataFrame, calendar: pd.DatetimeIndex,
                   ctx: rs.Context, frame_old: pd.DataFrame, ev_old: pd.DataFrame, loader) -> tuple[dict, pd.DataFrame]:
    """The expanded-cohort headline estimate on the exact route from a reconciled table (the recovery's own steps)."""
    d = rec.loc[rec["route"].isin(hr.ROUTES["exact"])]
    tf = hr.timed_frame(d, labels, chamber)
    union = pd.concat([timed, tf[[c for c in tf.columns if c in timed.columns]]], ignore_index=True)
    ev_new, _ = evm.build_events_T(union, calendar)
    new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))].copy()
    new_ev, n_constructed = hr.factor_outcomes(new_ev, calendar, loader)
    t_new, panel_new, _ = hr.expanded_estimate(ctx, calendar, frame_old, new_ev)
    return {"tau2": t_new["tau2_hat"], "se": t_new["var_wx"] * t_new["se_b"], "n_events": int(t_new["n_events_a"] + t_new["n_events_b"]),
            "n_households": int(t_new["n_households"]), "n_lines": int(len(tf)), "n_episodes": int(len(new_ev)),
            "n_with_outcome": int(np.isfinite(new_ev["y_h20"].astype(float)).sum()), "n_constructed": n_constructed,
            "n_purchases_A": int(((new_ev["direction"] == "purchase") & (new_ev["period"] == "A") & np.isfinite(new_ev["y_h20"].astype(float))).sum())}, new_ev


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="route-specific duplicate audit of the history recovery")
    ap.parse_args(argv)
    log = fc.logger("duplicate_audit")
    t0 = time.time()
    rows, filings = pd.read_parquet(hr.ROWS), pd.read_parquet(hr.FILINGS)
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    labels = {f"household_{i}": h for i, h in enumerate(households, start=1)}
    snapshot = pd.read_parquet(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT, columns=["issuer_name", "ticker_clean"])
    dictionary, ambiguous = hr.name_dictionary(matched, snapshot)
    by_sym = hr.symbol_ranges(lg.build_security_ranges())
    c = hr.candidates(rows)
    res = hr.resolve(c, dictionary, ambiguous, by_sym)
    rec_pooled = hr.reconcile(res)
    audit = suppressed_table(rec_pooled)
    log(f"suppressed rows (pooled reconciliation): {len(audit)}; exact-route rows suppressed by a near-exact-only predecessor: "
        f"{int(audit['suppressed_exact_by_near_only'].sum()) if len(audit) else 0}")
    rec_exact = reconcile_routes(res, hr.ROUTES["exact"])
    rec_near = reconcile_routes(res, hr.ROUTES["near_exact"])
    kept_pooled = set(rec_pooled.loc[~rec_pooled["amendment_exact_duplicate"] & rec_pooled["route"].isin(hr.ROUTES["exact"])].index)
    kept_exact = set(rec_exact.loc[~rec_exact["amendment_exact_duplicate"]].index)
    diff_rows = {"exact_rows_kept_pooled": len(kept_pooled), "exact_rows_kept_route_specific": len(kept_exact),
                 "rows_in_route_specific_not_pooled": len(kept_exact - kept_pooled), "rows_in_pooled_not_route_specific": len(kept_pooled - kept_exact),
                 "suppressed_pooled_all_routes": int(rec_pooled["amendment_exact_duplicate"].sum()),
                 "suppressed_route_specific_exact": int(rec_exact["amendment_exact_duplicate"].sum()),
                 "suppressed_route_specific_near": int(rec_near["amendment_exact_duplicate"].sum())}
    log(f"kept exact rows: pooled {len(kept_pooled)}, route-specific {len(kept_exact)}; symmetric difference {len(kept_exact ^ kept_pooled)}")
    # the exact-route estimate under both reconciliations
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    ctx, _ = ec.load_headline_context(("T",), with_placebo=False)
    frame_old = ctx.frames["T"]
    panel_old = ctx.panel(sp.headline_spec("T"))
    t_old = cgm.tau2_i1(rs.block_cluster(panel_old, calendar), "ES", anchor="anchor_blk")
    ev_old, _ = evm.build_events_T(timed, calendar)
    _store: dict = {}

    def loader():
        if "daily" not in _store:
            _store["daily"], _store["factors"] = pm.load_local_daily(), pm.load_factors()
        return _store["daily"], _store["factors"]

    est = [rs.row("T", "duplicate_route_audit", "original_cohort", "tau2_headline", t_old["tau2_hat"], "bp2", se_block=t_old["var_wx"] * t_old["se_b"],
                  n_events=int(t_old["n_events_a"] + t_old["n_events_b"]), n_households=int(t_old["n_households"]), notes="the headline")]
    results = {}
    for name, rec in (("pooled_reconciliation", rec_pooled), ("route_specific_reconciliation", rec_exact)):
        r, _ = exact_estimate(rec, name, labels, chamber, timed, calendar, ctx, frame_old, ev_old, loader)
        results[name] = r
        est.append(rs.row("T", "duplicate_route_audit", f"expanded_cohort_exact_{name}", "tau2_headline", r["tau2"], "bp2", se_block=r["se"],
                          n_events=r["n_events"], n_households=r["n_households"],
                          notes=f"provisional (recovered records unverified); exact route; {r['n_lines']} recovered lines, {r['n_episodes']} episodes, "
                                f"{r['n_with_outcome']} with outcome, {r['n_purchases_A']} first-period purchases; winsorized at the committed cutoffs; no placebo p"))
        log(f"{name}: tau2 {r['tau2']:.2f} (SE {r['se']:.1f}), lines {r['n_lines']}, episodes {r['n_episodes']}, A purchases {r['n_purchases_A']}")
    a, b = results["pooled_reconciliation"], results["route_specific_reconciliation"]
    est.append(rs.row("T", "duplicate_route_audit", "difference", "tau2_headline", b["tau2"] - a["tau2"], "bp2",
                      n_events=b["n_events"] - a["n_events"], n_households=b["n_households"] - a["n_households"],
                      notes="route-specific minus pooled reconciliation, exact route; n_events = difference in purchases, n_households = difference in households"))
    for k, v in diff_rows.items():
        est.append(rs.row("T", "duplicate_route_audit", "rows", k, float(v), "count"))
    if len(audit):
        fc.write_csv(audit, OUT_AUDIT)
    else:
        fc.write_csv(pd.DataFrame(columns=["household", "suppressed_filing_seq", "suppressed_route", "retained_route", "exact_route_row_precedes", "suppressed_exact_by_near_only"]), OUT_AUDIT)
    est_df = fc.write_rows(est, OUT_EST)
    fc.write_manifest("duplicate_audit", [fc.file_entry(hr.ROWS, rows=len(rows)), fc.file_entry(hr.FILINGS, rows=len(filings)),
                                          fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "matched_transactions.csv"),
                                          fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "timed_transactions.csv"),
                                          fc.file_entry(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT)],
                      config={"routes": {k: list(v) for k, v in hr.ROUTES.items()}, "near": hr.NEAR, "max_lag_days": hr.MAX_LAG_DAYS},
                      extra={"seconds": round(time.time() - t0, 1), **diff_rows})
    log(f"done in {time.time() - t0:.0f} s")
    pd.set_option("display.width", 250)
    print(audit.to_string(index=False) if len(audit) else "no suppressed rows")
    print(est_df[["variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
