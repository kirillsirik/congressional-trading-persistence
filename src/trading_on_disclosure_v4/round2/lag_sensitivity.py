"""Task C8 and Export 8: the provisional reconstruction of the two households' 2015-2020 histories re-run with a maximum
filing lag of 90, 180 and 400 calendar days, applied to the extracted rows before resolution, duplicate suppression,
aggregation and eligibility (history_recovery_final's sequence, run in memory; no frozen file is written).

python -m trading_on_disclosure_v4.round2.lag_sensitivity
Writes <run>/lag_sensitivity/ (ocr_lag_sensitivity.csv, household-level eligibility and cohort comparison, manifest) and
the record-level intermediate under the restricted run folder.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data import events as evm
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.data.eligibility import RULES
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.ext import history_recovery as hr
from trading_on_disclosure_v4.ext import ledgers as lg
from trading_on_disclosure_v4.final import history_recovery_final as hrf
from trading_on_disclosure_v4.final import history_reread as rr
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm
from trading_on_disclosure_v4.round2 import common as cm

OUT = cm.RUN_DIR / "lag_sensitivity"
LAGS = (400, 180, 90)
FROZEN = cm.REPO_ROOT / "data" / "v4" / "final" / "history_recovery_final_estimates.csv"


def failing_rule(fx: dict) -> str:
    na, qa, sa, nb, qb = RULES["R1"]
    out = []
    if fx["n_A"] < na:
        out.append(f"period-A purchases {fx['n_A']} < {na}")
    if fx["quarters_A"] < qa:
        out.append(f"period-A quarters {fx['quarters_A']} < {qa}")
    if fx["securities_A"] < sa:
        out.append(f"period-A securities {fx['securities_A']} < {sa}")
    if fx["n_B"] < nb:
        out.append(f"period-B purchases {fx['n_B']} < {nb}")
    if fx["quarters_B"] < qb:
        out.append(f"period-B quarters {fx['quarters_B']} < {qb}")
    return "; ".join(out) if out else ""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Task C8: filing-lag sensitivity of the provisional reconstruction")
    ap.parse_args(argv)
    log = cm.logger("lag_sensitivity")
    t0 = time.time()
    rows_all, filings = pd.read_parquet(rr.ROWS_PARQUET), pd.read_parquet(rr.FILINGS_PARQUET)
    rows = hrf.admissible_rows(rows_all, filings)
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from blind_vendor_households")
    labels = {f"household_{i}": h for i, h in enumerate(households, start=1)}
    snapshot = pd.read_parquet(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT, columns=["issuer_name", "ticker_clean"])
    dictionary, ambiguous = hr.name_dictionary(matched, snapshot)
    by_sym = hr.symbol_ranges(lg.build_security_ranges())
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    ctx, _ = cm.load_context(("T",))
    frame_old = ctx.frames["T"]
    panel_old = ctx.panel(sp.headline_spec("T"))
    ev_old, _ = evm.build_events_T(timed, calendar)
    old_ids = set(ev_old["event_id"])
    key = pd.read_csv(cm.CROSSWALK).set_index("household_id")["household_index"]
    store: dict = {}

    def loader():
        if "daily" not in store:
            store["daily"], store["factors"] = pm.load_local_daily(), pm.load_factors()
        return store["daily"], store["factors"]

    results, hh_rows, cohort, records = [], [], {}, []
    saved = hr.MAX_LAG_DAYS
    try:
        for lag in LAGS:
            hr.MAX_LAG_DAYS = lag
            c = hr.candidates(rows)
            res = hr.resolve(c, dictionary, ambiguous, by_sym)
            rec = hr.reconcile(res)
            fd = pd.to_datetime(c["filing_date"]).to_numpy().astype("datetime64[D]")
            td = pd.to_datetime(c["date"]).to_numpy().astype("datetime64[D]")
            lagdays = np.where(np.isnat(fd) | np.isnat(td), np.nan, (fd - td).astype("timedelta64[D]").astype(float))
            records.append(res.assign(max_lag_days=lag, filing_lag_days=lagdays).drop(columns=["page_image_path"], errors="ignore"))
            for route, routes in hr.ROUTES.items():
                d = rec.loc[rec["route"].isin(routes)]
                tf = hr.timed_frame(d, labels, chamber)
                union = pd.concat([timed, tf[[col for col in tf.columns if col in timed.columns]]], ignore_index=True)
                ev_new, _ = evm.build_events_T(union, calendar)
                new_ev = ev_new.loc[~ev_new["event_id"].isin(old_ids)].copy()
                if not new_ev["household_id"].isin(set(households)).all() or len(ev_new) != len(ev_old) + len(new_ev):
                    raise ValueError("recovered episodes touch a released episode")
                new_ev, n_constructed = hr.factor_outcomes(new_ev, calendar, loader)
                t_new, panel_new, frame_new = hr.expanded_estimate(ctx, calendar, frame_old, new_ev)
                table = tau_mod.household_table(panel_new, "ES")
                hid = pd.Index(table.household_id).astype(str)
                cohort[(lag, route)] = pd.DataFrame({"household_index": key.reindex(hid).to_numpy(), "n_A": table.n_a.astype(int), "n_B": table.n_b.astype(int),
                                                     "weight": table.w, "weight_share": table.w / table.w.sum()})
                pa = new_ev.loc[(new_ev["direction"] == "purchase") & (new_ev["period"] == "A") & np.isfinite(new_ev["y_h20"].astype(float))]
                for lab, h in labels.items():
                    fx = hr.household_facts(frame_new, h)
                    eligible = h in set(panel_new["household_id"])
                    hh_rows.append({"max_lag_days": lag, "route": route, "household": lab, "eligible": eligible, "n_A": fx["n_A"], "quarters_A": fx["quarters_A"],
                                    "securities_A": fx["securities_A"], "n_B": fx["n_B"], "quarters_B": fx["quarters_B"],
                                    "reason_not_eligible": "" if eligible else failing_rule(fx),
                                    "candidate_rows": int(c.loc[c["household"] == lab, "candidate"].sum()),
                                    "rows_outside_lag": int((c.loc[c["household"] == lab, "has_date"] & ~c.loc[c["household"] == lab, "date_plausible"]).sum()),
                                    "recovered_first_period_purchases": int((pa["household_id"] == h).sum()),
                                    "weight_share": float(cohort[(lag, route)].loc[cohort[(lag, route)]["household_index"] == key[h], "weight_share"].sum())})
                results.append({"max_lag_days": lag, "route": route, "tau2_bp2": t_new["tau2_hat"], "se_block_bp2": t_new["var_wx"] * t_new["se_b"],
                                "n_events": int(t_new["n_events_a"] + t_new["n_events_b"]), "n_events_A": int(t_new["n_events_a"]), "n_events_B": int(t_new["n_events_b"]),
                                "n_households": int(t_new["n_households"]), "n_anchor_blocks": int(t_new["n_anchor_dates"]), "dof": int(t_new["dof"]),
                                "psd_corrected": bool(t_new["psd_corrected"]), "rows_extracted": int(len(rows)), "rows_with_date": int(c["has_date"].sum()),
                                "rows_date_plausible": int(c["date_plausible"].sum()), "rows_candidate": int(c["candidate"].sum()),
                                "rows_resolved_route": int(d["route"].notna().sum()), "duplicates_suppressed": int(d["amendment_exact_duplicate"].sum()),
                                "recovered_lines": int(len(tf)), "episodes": int(len(new_ev)), "outcomes_from_released_constructor": n_constructed,
                                "first_period_purchases_with_outcome": int(len(pa)),
                                "recovered_purchase_mean_A_bp": float(pa["y_h20"].mean()) if len(pa) else np.nan,
                                "cutoffs": "committed pooled p1/p99 by period (T, ff5mom_h20)", "uncertainty": "household x 20-session block CGM SE of tau2 (the reported measure)",
                                "status": "provisional: recovered records unverified at the transaction level"})
                log(f"lag {lag} {route}: tau2 {t_new['tau2_hat']:.3f} (SE {t_new['var_wx'] * t_new['se_b']:.3f}), {int(t_new['n_households'])} households, "
                    f"{int(t_new['n_events_a'] + t_new['n_events_b'])} purchases, candidates {int(c['candidate'].sum())}")
    finally:
        hr.MAX_LAG_DAYS = saved
    res_df = pd.DataFrame(results)
    frozen = pd.read_csv(FROZEN)
    checks = []
    for route in hr.ROUTES:
        f = frozen[(frozen["variant"] == f"expanded_cohort_{route}") & (frozen["statistic"] == "tau2_headline")].iloc[0]
        r = res_df[(res_df["max_lag_days"] == 400) & (res_df["route"] == route)].iloc[0]
        for stat, a, b in (("tau2", r["tau2_bp2"], f["value"]), ("se_block", r["se_block_bp2"], f["se_block"]), ("n_events", r["n_events"], f["n_events"]),
                           ("n_households", r["n_households"], f["n_households"])):
            checks.append({"route": route, "statistic": stat, "reproduced_400": float(a), "frozen": float(b), "abs_diff": abs(float(a) - float(b))})
    chk = pd.DataFrame(checks)
    if (chk["abs_diff"] > 1e-8 * np.maximum(1.0, chk["frozen"].abs())).any():
        raise AssertionError(f"400-day baseline does not reproduce history_recovery_final_estimates.csv:\n{chk}")
    # cohort and within-household changes relative to 400 days
    comp = []
    for (lag, route), g in cohort.items():
        base = cohort[(400, route)].set_index("household_index")
        cur = g.set_index("household_index")
        both = base.index.intersection(cur.index)
        comp.append({"max_lag_days": lag, "route": route, "households": int(len(cur)), "households_dropped_vs_400": int(len(base.index.difference(cur.index))),
                     "households_added_vs_400": int(len(cur.index.difference(base.index))),
                     "retained_households_with_changed_counts": int(((base.loc[both, "n_A"] != cur.loc[both, "n_A"]) | (base.loc[both, "n_B"] != cur.loc[both, "n_B"])).sum()),
                     "purchases_A_change_in_retained": int(cur.loc[both, "n_A"].sum() - base.loc[both, "n_A"].sum()),
                     "purchases_B_change_in_retained": int(cur.loc[both, "n_B"].sum() - base.loc[both, "n_B"].sum())})
    cm.write_csv(res_df, OUT / "ocr_lag_sensitivity.csv")
    cm.write_csv(pd.DataFrame(hh_rows), OUT / "reconstructed_household_eligibility.csv")
    cm.write_csv(pd.DataFrame(comp), OUT / "cohort_changes_vs_400.csv")
    cm.write_csv(chk, OUT / "baseline_400_reproduction.csv")
    restricted = cm.RESTRICTED / "lag_sensitivity"
    restricted.mkdir(parents=True, exist_ok=True)
    recs = pd.concat(records, ignore_index=True)
    for c_ in ("permno",):
        if c_ in recs.columns:
            recs[c_] = pd.to_numeric(recs[c_], errors="coerce")
    recs.to_parquet(restricted / "record_level_intermediate.parquet", index=False)
    cm.write_json({"lags_days": list(LAGS), "restriction": "row kept as a candidate only when filing_date - max_lag <= transaction date <= filing_date, before "
                   "identity resolution, duplicate suppression, episode construction, aggregation and eligibility (ext.history_recovery.candidates)",
                   "unchanged": "extraction (closeout re-read rows), admissible-row rule for the 2021 filings, matching routes, duplicate priority, "
                                "released baseline households and their period-B histories, committed cutoffs, the headline estimator",
                   "routes": {k: list(v) for k, v in hr.ROUTES.items()},
                   "inputs": [cm.file_entry(rr.ROWS_PARQUET, rows=len(rows_all)), cm.file_entry(rr.FILINGS_PARQUET, rows=len(filings)), cm.file_entry(FROZEN)],
                   "record_level_intermediate": {**cm.file_entry(restricted / "record_level_intermediate.parquet", rows=len(recs)), "location": "restricted"},
                   "status": "provisional reconstruction; this run is not the stratified hand review of 453 records in 61 strata",
                   "seconds": round(time.time() - t0, 1)}, OUT / "lag_sensitivity_manifest.json")
    log(f"400-day baseline reproduced (max |diff| {chk['abs_diff'].max():.2e}); done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
