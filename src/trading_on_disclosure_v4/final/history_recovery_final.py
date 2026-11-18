"""Provisional recovery of the two households' histories on the re-read rows (closeout plan section 3b, second
step): the recovery of 19 October (identity without returns, the released hierarchy, episodes, the headline
estimate on the expanded cohort) applied to the rows of the closeout re-read, with the rows of the 2021 filings
admitted only when their transaction date falls in 2015-2020. Everything stays provisional and outside the
primary sample; the inventory reports completeness by household, year, filing and page.
Writes data/v4/final/history_recovery_final_inventory.csv, history_recovery_final_estimates.csv,
history_recovery_final_filings.csv; row-level output restricted under data/v4/restricted/final/history/.
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
from trading_on_disclosure_v4.final import history_reread as rr
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import panel as pm

OUT_INV = fc.OUT_DIR / "history_recovery_final_inventory.csv"
OUT_EST = fc.OUT_DIR / "history_recovery_final_estimates.csv"
OUT_FIL = fc.OUT_DIR / "history_recovery_final_filings.csv"
PERIOD_END = pd.Timestamp("2020-12-31")
LABEL = "provisional: recovered records unverified at the transaction level (closeout re-read)"


def admissible_rows(rows: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """The re-read rows of the 2015-2020 filings, plus the rows of the 2021 filings dated on or before 31 December 2020."""
    ext_keys = set(zip(filings.loc[filings["extended_inventory"], "household"], filings.loc[filings["extended_inventory"], "filing_seq"], strict=True))
    is_ext = np.array([k in ext_keys for k in zip(rows["household"], rows["filing_seq"], strict=True)])
    dated = pd.to_datetime(rows["date"], errors="coerce")
    keep = ~is_ext | (is_ext & dated.notna() & (dated <= PERIOD_END))
    out = rows.loc[keep].copy()
    out["from_extended_inventory"] = is_ext[keep]
    return out


def filing_table(filings: pd.DataFrame, res: pd.DataFrame, rec: pd.DataFrame, pages: pd.DataFrame) -> pd.DataFrame:
    """Completeness per filing (counts only): pages, readable pages, candidate rows, resolved rows, suppressed repeats."""
    rows = []
    for f in filings.itertuples(index=False):
        pg = pages[(pages["household"] == f.household) & (pages["filing_seq"] == f.filing_seq)]
        rg = res[(res["household"] == f.household) & (res["filing_seq"] == f.filing_seq)]
        dg = rec[(rec["household"] == f.household) & (rec["filing_seq"] == f.filing_seq)]
        rows.append({"household": f.household, "filing_seq": int(f.filing_seq), "filing_year": int(pd.Timestamp(f.filing_date).year),
                     "filing_month": int(pd.Timestamp(f.filing_date).month), "extended_inventory": bool(f.extended_inventory), "pages": int(f.pages),
                     "pages_readable": int(pg["readable_after"].sum()), "pages_unreadable": int((~pg["readable_after"]).sum()),
                     "rows_extracted": int(len(rg)), "rows_candidate": int(rg["candidate"].sum()), "rows_resolved_exact": int(rg["route"].isin(hr.ROUTES["exact"]).sum()),
                     "rows_resolved_near": int((rg["route"] == "name, near-exact").sum()), "rows_unresolved_candidates": int((rg["candidate"] & rg["route"].isna()).sum()),
                     "repeats_suppressed": int(dg["amendment_exact_duplicate"].sum()), "rows_dated_2015_2020": int(((pd.to_datetime(rg["date"], errors="coerce") >= "2015-01-01") & (pd.to_datetime(rg["date"], errors="coerce") <= PERIOD_END)).sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="provisional history recovery on the closeout re-read")
    ap.parse_args(argv)
    log = fc.logger("history_recovery_final")
    t0 = time.time()
    rows_all, filings = pd.read_parquet(rr.ROWS_PARQUET), pd.read_parquet(rr.FILINGS_PARQUET)
    pages = pd.read_csv(rr.PAGE_CSV)
    rows = admissible_rows(rows_all, filings)
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
    rec = hr.reconcile(res)
    log(f"rows {len(rows_all)} ({len(rows)} admissible); candidates {int(c['candidate'].sum())}; routes {res['route'].value_counts().to_dict()}; "
        f"repeats suppressed {int(rec['amendment_exact_duplicate'].sum())}")
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    ctx, _ = ec.load_headline_context(("T",), with_placebo=False)
    frame_old = ctx.frames["T"]
    panel_old = ctx.panel(sp.headline_spec("T"))
    t_old = cgm.tau2_i1(rs.block_cluster(panel_old, calendar), "ES", anchor="anchor_blk")
    ev_old, _ = evm.build_events_T(timed, calendar)
    est = [rs.row("T", "history_recovery_final", "original_cohort", "tau2_headline", t_old["tau2_hat"], "bp2", se_block=t_old["var_wx"] * t_old["se_b"],
                  n_events=int(t_old["n_events_a"] + t_old["n_events_b"]), n_households=int(t_old["n_households"]), notes="the headline")]
    est.append(rs.row("T", "history_recovery_final", "dictionary", "size", float(len(dictionary)), "count",
                      notes=f"unambiguous name-to-ticker dictionary from the parsed records and the vendor issuer names; {len(ambiguous)} ambiguous names excluded"))
    _store: dict = {}

    def loader():
        if "daily" not in _store:
            _store["daily"], _store["factors"] = pm.load_local_daily(), pm.load_factors()
        return _store["daily"], _store["factors"]

    episodes = {}
    for name, routes in hr.ROUTES.items():
        d = rec.loc[rec["route"].isin(routes)]
        tf = hr.timed_frame(d, labels, chamber)
        union = pd.concat([timed, tf[[col for col in tf.columns if col in timed.columns]]], ignore_index=True)
        ev_new, _ = evm.build_events_T(union, calendar)
        new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))].copy()
        if not new_ev["household_id"].isin(set(households)).all() or len(ev_new) != len(ev_old) + len(new_ev):
            raise ValueError("recovered episodes touch a released episode")
        new_ev, n_constructed = hr.factor_outcomes(new_ev, calendar, loader)
        t_new, panel_new, frame_new = hr.expanded_estimate(ctx, calendar, frame_old, new_ev)
        new_ev["household"] = new_ev["household_id"].map({v: k for k, v in labels.items()})
        episodes[name] = new_ev
        eligible = [lab for lab, hid in labels.items() if hid in set(panel_new["household_id"])]
        pa = new_ev.loc[(new_ev["direction"] == "purchase") & (new_ev["period"] == "A") & np.isfinite(new_ev["y_h20"].astype(float))]
        est.append(rs.row("T", "history_recovery_final", f"expanded_cohort_{name}", "tau2_headline", t_new["tau2_hat"], "bp2", se_block=t_new["var_wx"] * t_new["se_b"],
                          n_events=int(t_new["n_events_a"] + t_new["n_events_b"]), n_households=int(t_new["n_households"]),
                          notes=f"{LABEL}; {'exact route' if name == 'exact' else 'near-exact route added'}; {len(tf)} recovered lines, {len(new_ev)} episodes, "
                                f"{n_constructed} outcomes from the released constructor; households newly eligible {len(eligible)}; {len(pa)} first-period purchases with an outcome; "
                                f"winsorized at the committed cutoffs; no placebo p"))
        for lab, hid in labels.items():
            fx = hr.household_facts(frame_new, hid)
            est.append(rs.row("T", "history_recovery_final", f"eligibility_{lab}", name, float(lab in eligible), "flag", n_events=fx["n_A"],
                              notes=f"period A purchases with an outcome {fx['n_A']} over {fx['quarters_A']} quarters and {fx['securities_A']} securities; period B {fx['n_B']} over {fx['quarters_B']} quarters"))
        est.append(rs.row("T", "history_recovery_final", "recovered_purchase_mean_A", name, float(pa["y_h20"].mean()) if len(pa) else np.nan, "bp", n_events=int(len(pa)),
                          n_households=int(pa["household_id"].nunique()), notes="raw factor outcome of the recovered first-period purchases, unweighted"))
        log(f"{name}: lines {len(tf)}, episodes {len(new_ev)}, first-period purchases {len(pa)}, tau2 {t_new['tau2_hat']:.1f} (households {int(t_new['n_households'])})")
        new_ev.drop(columns=["household_id"]).to_parquet(rr.HISTORY_DIR / f"recovered_episodes_{name}.parquet", index=False)
    inv = hr.inventory(filings.rename(columns={"pages_with_headers": "pages_with_headers"}).assign(pages_with_headers=filings.get("pages_with_headers", pd.Series(0, index=filings.index)).fillna(0).astype(int),
                                                                                                    read_status=filings["read_status"]), res, rec, episodes)
    res.drop(columns=["page_image_path"], errors="ignore").to_parquet(rr.HISTORY_DIR / "recovered_transactions.parquet", index=False)
    ftab = filing_table(filings, res, rec, pages)
    for df, path in ((inv, OUT_INV), (ftab, OUT_FIL)):
        hr.th.guard(df, path)
        fc.write_csv(df, path)
    est_df = fc.write_rows(est, OUT_EST)
    hr.th.guard(est_df, OUT_EST)
    fc.write_manifest("history_recovery_final", [fc.file_entry(rr.ROWS_PARQUET, rows=len(rows_all)), fc.file_entry(rr.FILINGS_PARQUET, rows=len(filings)),
                                                 fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "matched_transactions.csv"), fc.file_entry(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT)],
                      config={"period_end": str(PERIOD_END.date()), "routes": {k: list(v) for k, v in hr.ROUTES.items()}, "near": hr.NEAR, "max_lag_days": hr.MAX_LAG_DAYS},
                      extra={"seconds": round(time.time() - t0, 1), "n_admissible_rows": int(len(rows)), "n_rows_from_2021_filings_admitted": int(rows["from_extended_inventory"].sum())})
    pd.set_option("display.width", 250)
    print(inv.to_string(index=False))
    print(est_df[["variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
