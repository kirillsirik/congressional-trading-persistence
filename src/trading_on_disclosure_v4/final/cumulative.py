"""The cumulative corrected sample (closeout plan section 3d) and its missing-outcome accounting (section 3f).

One construction, in the released code and order, from the union of the released timed lines and the
rule-repaired lines (identity by the design's ticker-and-date rule; the lines dated beyond the return calendar
and the lines with two date-valid candidates stay out; recovered histories stay out): entry dates, the signal
collapse with the mixed flag, the version-4 duplicate audit on the union, the 20-day market-adjusted outcome at
public entry, the July-2026 completion of endpoint-censored windows, the verified delisting paths with the
terminal payoff once (proceeds reinvested at the benchmark, cash holding beside it), transaction- and
public-clock events with factor outcomes from the released cache and its extension, eligibility reassessed,
winsorization at the committed cutoffs. Outputs (data/v4/final/): sample_flow_ledger.csv, spread_corrected.csv,
persistence_corrected.csv, decomposition_corrected.csv, missingness_corrected.csv; restricted frames under
data/v4/restricted/final/cumulative/ for the reference-panel and delisting modules.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data import events as evm
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data.calendar import HORIZON, window_end
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.estimators import tests_t1_t4 as tt
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import common as ec
from trading_on_disclosure_v4.ext import decompose as dc
from trading_on_disclosure_v4.ext import delisting as dl
from trading_on_disclosure_v4.ext import endpoint as ep
from trading_on_disclosure_v4.ext import inference as inf
from trading_on_disclosure_v4.ext import ledgers as lg
from trading_on_disclosure_v4.ext import repair
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.final import cache_extension as cx
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.inference import block_bootstrap as bb
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import forward_cache as fcache
from trading_on_disclosure_v4.outcomes import panel as pm
from trading_on_disclosure_v4.outcomes.seeds import seed_from

BP = fc.BP
H = fc.H
RESTRICTED = fc.RESTRICTED / "cumulative"
OUT_LEDGER = fc.OUT_DIR / "sample_flow_ledger.csv"
OUT_SPREAD = fc.OUT_DIR / "spread_corrected.csv"
OUT_TAU = fc.OUT_DIR / "persistence_corrected.csv"
OUT_DECOMP = fc.OUT_DIR / "decomposition_corrected.csv"
OUT_MISS = fc.OUT_DIR / "missingness_corrected.csv"
OUT_LADDER = fc.OUT_DIR / "cluster_ladder_corrected.csv"
SIGNAL_OUTCOME_COLS = ["signal_uid", "ar_E_h20", "ar_E_h20_reason", "ar_T_h20", "ar_T_h20_reason", "car_ff5mom_h20", "car_ff5mom_h20_reason"]
SPREAD_VARIANTS = ("baseline_record", "corrected_observed", "corrected_july", "corrected_july_delisting_reinvested", "corrected_july_delisting_cash",
                   "corrected_delisting_reinvested_no_july")
PRIMARY_SPREAD = "corrected_july_delisting_reinvested"


# ---------------------------------------------------------------- the union

def build_union(log) -> dict:
    """Records -> union timed lines -> signals on the union with the duplicate flag of the union audit."""
    calendar = pm.v2_trading_calendar()
    ranges = lg.build_security_ranges()
    cand = repair.resolve_candidates(ranges)
    resolved = cand.loc[cand["permno_repaired"].notna()].copy()
    timed = load.load_timed_transactions()
    union_timed, new_timed = repair.timed_union(timed, resolved, calendar)
    matched = load.load_matched_transactions()
    rep_matched = resolved.loc[:, [c for c in matched.columns if c in resolved.columns]].copy()
    rep_matched["permno"] = resolved["permno_repaired"].astype(int).to_numpy()
    matched_union = pd.concat([matched, rep_matched], ignore_index=True)
    sig_union = repair.signals_union(union_timed)
    sig_union["dup_all_lines_union"] = repair.dedup_flags_union(matched_union, sig_union)
    sig_old = load.load_signals()
    old_uids = set(sig_old["signal_uid"])
    repaired_uids = set(new_timed["transaction_uid"])
    parents = sig_union["parent_transaction_uids"].astype(str).str.split(";")
    sig_union["has_repaired_lot"] = parents.map(lambda ps: any(p in repaired_uids for p in ps)).to_numpy()
    sig_union["all_lots_repaired"] = parents.map(lambda ps: all(p in repaired_uids for p in ps)).to_numpy()
    sig_union["is_new_signal"] = ~sig_union["signal_uid"].isin(old_uids)
    old_flags = rerun.duplicate_signal_uids()
    log(f"timed lines: released {len(timed)}, repaired candidates {len(cand)}, resolved {len(resolved)}, beyond the calendar "
        f"{new_timed.attrs.get('n_beyond_calendar', 0)}, entering {len(new_timed)}; union {len(union_timed)}")
    log(f"signals on the union {len(sig_union)} (released {len(sig_old)}, new {int(sig_union['is_new_signal'].sum())}); "
        f"duplicate-flagged on the union {int(sig_union['dup_all_lines_union'].sum())} (released flags {len(old_flags)})")
    return {"calendar": calendar, "cand": cand, "resolved": resolved, "timed": timed, "union_timed": union_timed, "new_timed": new_timed,
            "sig_old": sig_old, "sig_union": sig_union, "old_flags": old_flags, "matched": matched}


# ---------------------------------------------------------------- the public-clock universe with outcomes

def public_universe_corrected(u: dict, log) -> pd.DataFrame:
    """Non-mixed signals of the union with an entry session and not duplicate-flagged on the union, with the released
    outcomes for released signals and the released constructors' outcomes for new signals."""
    s = u["sig_union"]
    keep = (~s["mixed_direction"].astype(bool)) & s["follower_entry_date"].notna() & (~s["dup_all_lines_union"].astype(bool))
    uni = s.loc[keep].copy()
    old = pd.read_csv(fc.REPO_ROOT / "outputs" / "frames" / "signals.csv", low_memory=False, usecols=SIGNAL_OUTCOME_COLS)
    new_sig = uni.loc[uni["is_new_signal"]].copy()
    daily, bench, factors = pm.load_local_daily(), pm.load_benchmark(), pm.load_factors()
    new_out = repair.outcomes_for(new_sig, daily, bench, factors, u["calendar"])[SIGNAL_OUTCOME_COLS]
    outcomes = pd.concat([old.loc[old["signal_uid"].isin(set(uni.loc[~uni["is_new_signal"], "signal_uid"]))], new_out], ignore_index=True)
    uni = uni.merge(outcomes, on="signal_uid", how="left", validate="1:1")
    uni["follower_entry_date"] = pd.to_datetime(uni["follower_entry_date"])
    cal = u["calendar"]
    uni["anchor_idx"] = cal.get_indexer(pd.DatetimeIndex(uni["follower_entry_date"]))
    if (uni["anchor_idx"] < 0).any():
        raise ValueError("public-clock anchors off the calendar")
    wend = window_end(uni["follower_entry_date"], cal, HORIZON)
    uni["period"] = evm.assign_period(uni["follower_entry_date"], wend).to_numpy()
    out = pd.DataFrame({"clock": "P", "universe": "corrected_dedup_nonmixed_with_entry_session", "event_id": "P:" + uni["signal_uid"].astype(str),
                        "signal_uid": uni["signal_uid"].astype(str), "household": uni["filer_uid"].astype(str), "permno": uni["permno"].astype(int),
                        "direction": uni["signal_direction"].astype(str), "period": uni["period"], "anchor_date": uni["follower_entry_date"],
                        "anchor_idx": uni["anchor_idx"].astype(int), "record_source": uni["record_source"].astype(str), "is_new": uni["is_new_signal"].to_numpy(),
                        "has_repaired_lot": uni["has_repaired_lot"].to_numpy(), "y_market_adjusted": uni["ar_E_h20"].to_numpy(dtype=float) * BP,
                        "status_market_adjusted": uni["ar_E_h20_reason"].astype(str), "y_ff5mom": uni["car_ff5mom_h20"].to_numpy(dtype=float) * BP,
                        "status_ff5mom": uni["car_ff5mom_h20_reason"].astype(str)})
    out["blk"] = out["anchor_idx"] // 20
    log(f"corrected public universe {len(out)} signals ({int(out['is_new'].sum())} new); observed market-adjusted outcomes "
        f"{int(np.isfinite(out['y_market_adjusted']).sum())}")
    return out.reset_index(drop=True)


def baseline_universe(u: dict) -> pd.DataFrame:
    """The record's de-duplicated universe (the window audit's construction), for the stage-consistent comparison."""
    uni, _ = w.public_universe(u["calendar"])
    uni["blk"] = uni["anchor_idx"] // 20
    return uni


def completions(uni: pd.DataFrame, market: w.Market, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """July completion of the endpoint-censored windows and the verified delisting paths of the universe."""
    trunc = uni.loc[uni["status_market_adjusted"] == "end_of_sample_truncation"]
    sessions, rows, bench_j = ep.july_slice(set(trunc["permno"].astype(int)))
    cal, bench, grids = ep.extended_grid(market, sessions, rows, bench_j)
    comp = ep.complete_windows(trunc, market, cal, bench, grids)
    y = uni["y_market_adjusted"].to_numpy(dtype=float)
    dl_idx = market.delist.reindex(uni["permno"].to_numpy()).to_numpy(dtype=float)
    has = w.delisting_in_window(uni["anchor_idx"].to_numpy(), dl_idx)
    led = dl.path_ledger(uni.loc[has & ~np.isfinite(y)], market, "P")
    log(f"endpoint-censored {len(trunc)}, completed {int(comp['completed'].sum())}; delisted windows without an outcome {len(led)}, verified {int(led['verified'].sum())}")
    return comp, led


def spread_rows(uni: pd.DataFrame, comp: pd.DataFrame, led: pd.DataFrame, base_uni: pd.DataFrame, calendar: pd.DatetimeIndex, log) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The de-duplicated spread under the stage sequence, baseline and corrected, with the outcome vectors."""
    y0 = uni["y_market_adjusted"].to_numpy(dtype=float)
    done = comp.loc[comp["completed"]].set_index("event_id")["y_completed_bp"]
    f_july = uni["event_id"].map(done)
    m_july = ~np.isfinite(y0) & f_july.notna().to_numpy()
    ver = led.loc[led["verified"]].set_index("event_id")
    vectors = {}
    y_july = y0.copy()
    y_july[m_july] = f_july.to_numpy(dtype=float)[m_july]
    vectors["corrected_observed"] = y0
    vectors["corrected_july"] = y_july
    for label, col in (("corrected_july_delisting_reinvested", "y_reinvest_bp"), ("corrected_july_delisting_cash", "y_cash_bp")):
        yv = y_july.copy()
        f = uni["event_id"].map(ver[col]) if len(ver) else pd.Series(np.nan, index=uni.index)
        m = ~np.isfinite(yv) & f.notna().to_numpy()
        yv[m] = f.to_numpy(dtype=float)[m]
        vectors[label] = yv
    yv = y0.copy()
    f = uni["event_id"].map(ver["y_reinvest_bp"]) if len(ver) else pd.Series(np.nan, index=uni.index)
    m = ~np.isfinite(yv) & f.notna().to_numpy()
    yv[m] = f.to_numpy(dtype=float)[m]
    vectors["corrected_delisting_reinvested_no_july"] = yv
    rows = []
    base = w.spread_estimate(base_uni, base_uni["y_market_adjusted"].to_numpy(dtype=float))
    ref = pd.read_csv(rerun.TABLES_PATH)
    ref = ref.loc[(ref["test_id"] == rerun.PRIMARY_TEST_ID) & (ref["sample"] == "dedup") & (ref["clusters"] == "block20")].iloc[0]
    if abs(base["estimate"] - ref["estimate"]) > 1e-6 or base["n"] != int(ref["n"]):
        raise AssertionError("the baseline spread does not reproduce data/v4/existing/rerun_tables.csv")
    rows.append(rs.row("P", "spread", "baseline_record", "spread", base["estimate"], "bp", se_block=base["se"], n_events=base["n"], n_households=base["n_households"],
                       notes=f"the record (reproduces rerun_tables.csv); {base['n_dates_or_blocks']} blocks"))
    notes = {"corrected_observed": "union sample, observed 20-day outcomes only",
             "corrected_july": f"as above, the {int(m_july.sum())} endpoint-censored windows completed on the July 2026 slice",
             "corrected_july_delisting_reinvested": f"as above plus the {int(len(ver))} verified delisting paths, terminal payoff once, proceeds reinvested at the benchmark (the primary corrected spread)",
             "corrected_july_delisting_cash": f"as above plus the {int(len(ver))} verified delisting paths, proceeds held in cash",
             "corrected_delisting_reinvested_no_july": "union sample with the verified delisting paths (reinvested) but without the July completion"}
    for label in SPREAD_VARIANTS[1:]:
        r = w.spread_estimate(uni, vectors[label])
        rows.append(rs.row("P", "spread", label, "spread", r["estimate"], "bp", se_block=r["se"], n_events=r["n"], n_households=r["n_households"],
                           notes=f"{notes[label]}; {r['n_dates_or_blocks']} blocks; dof {r['dof']}"))
        rows.append(rs.row("P", "spread", label, "n_imputed", int(np.isfinite(vectors[label]).sum() - np.isfinite(y0).sum()), "count", n_events=r["n"]))
        log(f"spread {label}: {r['estimate']:.3f} bp (SE {r['se']:.3f}), n {r['n']}, households {r['n_households']}")
    return rows, vectors


def ladder_rows(uni: pd.DataFrame, y: np.ndarray, calendar: pd.DatetimeIndex) -> list[dict]:
    """Household x {day, 20-, 40-, 60-day block, quarter} and household x block x security standard errors of the primary corrected spread."""
    ok = np.isfinite(y)
    f = uni.loc[ok]
    X = np.column_stack([np.ones(int(ok.sum())), (f["direction"].to_numpy() == "purchase").astype(float)])
    rows = []
    for name, r in inf.ladder(y[ok], X, None, f["household"].to_numpy(), f["anchor_date"], f["permno"].to_numpy(), calendar, 1):
        rows.append(rs.row("P", "spread_ladder", PRIMARY_SPREAD, name, r["estimate"], "bp",
                           se_block=r["se"], n_events=int(ok.sum()), n_households=int(f["household"].nunique()),
                           notes=f"t {r['t']:.3f}, two-sided p {r['p_two_sided']:.3f}, dof {r['dof']}; clusters " + "; ".join(f"{k} {v}" for k, v in r["cluster_counts"].items())))
    return rows


# ---------------------------------------------------------------- transaction- and public-clock event frames

def merged_cache(columns: list[str]) -> pd.DataFrame:
    cache = fcache.load_cache(columns=columns)
    ext = pd.read_parquet(cx.OUT_CACHE, columns=columns)
    if set(cache["permno"].unique()) & set(ext["permno"].unique()):
        raise ValueError("the cache extension overlaps the released cache")
    return pd.concat([cache, ext], ignore_index=True)


def outcomes_from_cache(events: pd.DataFrame, calendar: pd.DatetimeIndex, cache: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(e["anchor_date"]))
    if (e["anchor_idx"] < 0).any():
        raise ValueError("anchors off the calendar")
    e = e.merge(cache, on=["permno", "anchor_idx"], how="left")
    for y, col in evf.CACHE_Y.items():
        e[y] = e[col].astype(float) * BP
    e["sigma_event"] = e["resid_sd"].astype(float) * np.sqrt(evf.SIGMA_HORIZON) * BP
    return e.drop(columns=[*evf.CACHE_Y.values(), "resid_sd", "anchor_idx"])


def transaction_frames(u: dict, cache: pd.DataFrame, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(baseline T frame, corrected T frame, corrected T events) with the frame columns of the results run."""
    cal = u["calendar"]
    ev_old, _ = evm.build_events_T(u["timed"], cal)
    ev_new_all, _ = evm.build_events_T(u["union_timed"], cal)
    new_ev = ev_new_all.loc[~ev_new_all["event_id"].isin(set(ev_old["event_id"]))].copy()
    if not (len(ev_new_all) == len(ev_old) + len(new_ev)):
        raise ValueError("union episodes do not nest the released episodes")
    frame_old = evf.load_event_frame("T", calendar=cal)
    meta_new = new_ev.loc[new_ev["direction"].isin(["purchase", "sale"]) & new_ev["period"].isin(["A", "B"]), evf.META_COLUMNS].copy()
    meta_new["dup_any_line"], meta_new["dup_all_lines"] = False, False
    new_frame = outcomes_from_cache(meta_new, cal, cache)
    new_frame = new_frame[frame_old.columns]
    frame_new = pd.concat([frame_old, new_frame], ignore_index=True)
    log(f"transaction clock: released episodes {len(ev_old)}, union {len(ev_new_all)}, new {len(new_ev)}; frame {len(frame_old)} -> {len(frame_new)} "
        f"(new A/B purchases with a factor outcome {int(((new_frame['direction'] == 'purchase') & np.isfinite(new_frame['y_h20'].astype(float))).sum())})")
    return frame_old, frame_new, ev_new_all


def public_frames(u: dict, cache: pd.DataFrame, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(baseline P frame, corrected P frame, corrected P events): released events minus those newly mixed or newly
    duplicate on the union, plus the new signals' events."""
    cal = u["calendar"]
    s = u["sig_union"].copy()
    s["mixed_direction"] = s["mixed_direction"].astype(bool)
    ev_all, _ = evm.build_events_P(s, cal)
    dup = set(s.loc[s["dup_all_lines_union"].astype(bool), "signal_uid"].astype(str))
    ev_all["dup_all_lines"] = ev_all["signal_uid"].astype(str).isin(dup)
    ev_all["dup_any_line"] = ev_all["dup_all_lines"]
    frame_old = evf.load_event_frame("P", calendar=cal)
    old_ids = set(frame_old["event_id"])
    ids_union = set(ev_all["event_id"])
    gone = frame_old.loc[~frame_old["event_id"].isin(ids_union)]
    kept = frame_old.loc[frame_old["event_id"].isin(ids_union)].copy()
    flag = ev_all.set_index("event_id")["dup_all_lines"]
    kept["dup_all_lines"] = kept["event_id"].map(flag).astype(bool).to_numpy()
    kept["dup_any_line"] = kept["dup_all_lines"]
    newly_flagged = int((kept["dup_all_lines"] & ~frame_old.set_index("event_id").loc[kept["event_id"], "dup_all_lines"].astype(bool).to_numpy()).sum())
    new_meta = ev_all.loc[~ev_all["event_id"].isin(old_ids) & ev_all["direction"].isin(["purchase", "sale"]) & ev_all["period"].isin(["A", "B"]), [*evf.META_COLUMNS, "dup_any_line", "dup_all_lines"]].copy()
    new_frame = outcomes_from_cache(new_meta, cal, cache)[frame_old.columns]
    frame_new = pd.concat([kept, new_frame], ignore_index=True)
    log(f"public clock: released events in the frame {len(frame_old)}, no longer non-mixed {len(gone)}, newly duplicate-flagged {newly_flagged}, new {len(new_frame)}; frame -> {len(frame_new)}")
    return frame_old, frame_new, ev_all


# ---------------------------------------------------------------- estimates

def tau_block(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    return cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")


def persistence_rows(clock: str, frames: dict[str, pd.DataFrame], ctx_cutoffs, calendar: pd.DatetimeIndex, n_boot: int, log) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Baseline, corrected and common-cohort headline covariance with block SE, day SE and the household bootstrap."""
    spec = sp.headline_spec(clock)
    cuts = ctx_cutoffs(clock, spec.outcome)
    panels = {}
    rows = []
    base_panel = sp.to_panel(frames["baseline"], spec, cuts)
    base_hh = set(base_panel["household_id"].astype(str))
    for label, frame in (("baseline", frames["baseline"]), ("corrected", frames["corrected"])):
        panel = sp.to_panel(frame, spec, cuts)
        panels[label] = panel
    common = sp.to_panel(frames["corrected"].loc[frames["corrected"]["household_id"].astype(str).isin(base_hh)], spec, cuts)
    panels["common_cohort"] = common
    sample_cuts = evf.sample_cutoffs(pd.to_numeric(frames["corrected"].loc[(frames["corrected"]["direction"] == "purchase") & ~frames["corrected"]["dup_all_lines"].astype(bool), "y_h20"], errors="coerce").to_numpy(dtype=float),
                                     frames["corrected"].loc[(frames["corrected"]["direction"] == "purchase") & ~frames["corrected"]["dup_all_lines"].astype(bool), "period"].to_numpy())
    panels["corrected_own_cutoffs"] = sp.to_panel(frames["corrected"], spec, sample_cuts)
    notes = {"baseline": "the record: released frame, committed cutoffs", "corrected": "union frame, eligibility reassessed, committed cutoffs",
             "common_cohort": "union frame restricted to the households eligible on the record (their repaired purchases included), committed cutoffs",
             "corrected_own_cutoffs": "union frame, cutoffs re-estimated as the pooled 1st/99th percentiles of the corrected purchase panel by period (sensitivity)"}
    for label, panel in panels.items():
        r = tau_block(panel, calendar)
        rd = cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_date")
        b = bb.tau2_household_bootstrap(panel, "ES", n_boot, seed_from(spec.name, clock, "household bootstrap")) if n_boot > 0 else None
        n_ev, n_hh = int(r["n_events_a"] + r["n_events_b"]), int(r["n_households"])
        rows.append(rs.row(clock, "persistence", label, "tau2", r["tau2_hat"], "bp2", se_block=r["var_wx"] * r["se_b"],
                           boot_lo=b["tau2_low"] if b else np.nan, boot_hi=b["tau2_high"] if b else np.nan, n_events=n_ev, n_households=n_hh,
                           notes=f"{notes[label]}; events A {r['n_events_a']}, B {r['n_events_b']}; {r['n_anchor_dates']} blocks, dof {r['dof']}; "
                                 f"household bootstrap {n_boot} draws, percentile interval in boot_lo/boot_hi; reference p added by final.reference"))
        rows.append(rs.row(clock, "persistence", label, "tau", r["tau_hat"], "bp", boot_lo=b["tau_low"] if b else np.nan, boot_hi=b["tau_high"] if b else np.nan,
                           n_events=n_ev, n_households=n_hh, notes="signed square root"))
        rows.append(rs.row(clock, "persistence", label, "tau2_se_day", rd["var_wx"] * rd["se_b"], "bp2", n_events=n_ev, n_households=n_hh, notes="household x anchor day clustered"))
        rows.append(rs.row(clock, "persistence", label, "t_block", r["t_stat"], "t", n_events=n_ev, n_households=n_hh, notes=f"dof {r['dof']}"))
        if b:
            rows.append(rs.row(clock, "persistence", label, "tau_upper", b["tau_upper_one_sided"], "bp", n_events=n_ev, n_households=n_hh,
                               notes="one-sided 95% household-bootstrap upper bound (nominal; see the calibration of the bootstrap in the paper)"))
            rows.append(rs.row(clock, "persistence", label, "boot_valid_draws", b["n_valid"], "count", n_events=n_ev, n_households=n_hh))
        tb = tau_mod.household_table(panel, "ES")
        rows.append(rs.row(clock, "persistence", label, "n_events_A", int(tb.n_a.sum()), "count", n_events=n_ev, n_households=n_hh))
        rows.append(rs.row(clock, "persistence", label, "n_events_B", int(tb.n_b.sum()), "count", n_events=n_ev, n_households=n_hh))
        log(f"{clock} persistence {label}: tau2 {r['tau2_hat']:.2f} (block SE {r['var_wx'] * r['se_b']:.1f}), households {n_hh}, purchases {n_ev}")
    # the persistence households' raw purchase mean (the appendix statistic), baseline and corrected
    raw_spec = sp.Spec(**{**spec.__dict__, "winsorized": False})
    for label in ("baseline", "corrected"):
        rp = sp.to_panel(frames[label], raw_spec, None)
        m = tt.t1_clustered_mean(rs.block_cluster(rp, calendar), anchor="anchor_blk")
        rows.append(rs.row(clock, "persistence", label, "purchase_mean_raw", m["estimate"], "bp", se_block=m["se"], n_events=m["n_events"], n_households=m["n_households"],
                           notes="raw factor outcome of the persistence households' purchases, household x 20-day block clustered"))
    return rows, panels


def decomposition_rows(clock: str, frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> list[dict]:
    """The four components, raw and winsorized (D on the winsorized outcomes), for the intercept-omitted and
    market-adjusted alternatives on the corrected and common-cohort panels (the re-fitted alternative is on the baseline only)."""
    out = []
    spec = sp.headline_spec(clock)
    for label in ("corrected", "common_cohort", "baseline"):
        frame = frames[label]
        raw = sp.to_panel(frame, sp.Spec(**{**spec.__dict__, "winsorized": False}), None)
        fr = frame.set_index("event_id")
        base = raw[["event_id", "household_id", "period", "anchor_date"]].copy()
        for k, c in dc.COLS.items():
            base[k] = pd.to_numeric(fr.loc[base["event_id"], c], errors="coerce").to_numpy(dtype=float)
        for variant, yx in (("no_alpha_omit", "Y0"), ("market_adjusted", "Ym")):
            ok = np.isfinite(base[["Y1", yx]].to_numpy(dtype=float)).all(axis=1)
            panel = base.loc[ok].reset_index(drop=True)
            hh = dc.household_means(panel, ["Y1", yx])
            wts = tau_mod.household_weights(hh["n_A"].to_numpy(), hh["n_B"].to_numpy(), "ES")
            t_raw = dc.terms(hh, "Y1", yx, wts)
            pw_ = panel.copy()
            pw_["Y1w"] = dc.winsorized_column(panel, "Y1", clock)
            pw_[f"{yx}w"] = dc.winsorized_column(panel, yx, clock)
            hhw = dc.household_means(pw_, ["Y1w", f"{yx}w"])
            ww = tau_mod.household_weights(hhw["n_A"].to_numpy(), hhw["n_B"].to_numpy(), "ES")
            t_w = dc.terms(hhw, "Y1w", f"{yx}w", ww)
            n_ev, n_hh = int(len(panel)), int(panel["household_id"].nunique())
            for kind, t in (("raw", t_raw), ("wins", t_w)):
                for k in dc.TERMS:
                    out.append(rs.row(clock, "decomposition_corrected", f"{label}|{variant}", f"tau2_{k}_{kind}", t[k], "bp2", n_events=n_ev, n_households=n_hh,
                                      notes="each outcome winsorized at the committed cutoffs; D = Y1w - Yxw" if kind == "wins" else "raw outcomes"))
                out.append(rs.row(clock, "decomposition_corrected", f"{label}|{variant}", f"identity_gap_{kind}", t["identity_gap"], "bp2", n_events=n_ev, n_households=n_hh))
                out.append(rs.row(clock, "decomposition_corrected", f"{label}|{variant}", f"corr_D_AB_{kind}", t["corr_D_AB"], "correlation", n_events=n_ev, n_households=n_hh))
    return out


# ---------------------------------------------------------------- missingness

def missingness_rows(uni: pd.DataFrame, vectors: dict[str, np.ndarray], base_uni: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Counts of missing outcomes by stage, direction and source on the corrected universe (and the record), the
    residual set after completion, and the direction-mean and percentile assignments for the residual set."""
    rows = []
    for label, frame, y in (("baseline_record", base_uni, base_uni["y_market_adjusted"].to_numpy(dtype=float)),
                            ("corrected_observed", uni, vectors["corrected_observed"]),
                            ("corrected_july", uni, vectors["corrected_july"]),
                            (PRIMARY_SPREAD, uni, vectors[PRIMARY_SPREAD])):
        src = frame["record_source"].to_numpy() if "record_source" in frame.columns else np.array(["all"] * len(frame))
        for dirn in ("purchase", "sale", "all"):
            for s in ("official", "capitol_trades", "all"):
                m = np.ones(len(frame), dtype=bool)
                if dirn != "all":
                    m &= frame["direction"].to_numpy() == dirn
                if s != "all":
                    m &= src == s
                n = int(m.sum())
                obs = int(np.isfinite(y[m]).sum())
                rec = {"stage": label, "direction": dirn, "source": s, "n_signals": n, "n_observed": obs, "n_missing": n - obs, "retention": obs / n if n else np.nan}
                if label == "corrected_observed":
                    st = frame["status_market_adjusted"].to_numpy()
                    for reason in ("end_of_sample_truncation", "missing_return_days", "no_security_returns"):
                        rec[f"n_{reason}"] = int((m & (st == reason)).sum())
                rows.append(rec)
    # sensitivity for the residual set after completion: direction mean and the 5th/95th percentile assignments
    y = vectors[PRIMARY_SPREAD]
    frame = uni.assign(y=y, y_partial=np.nan)
    sens = []
    base = w.spread_estimate(frame, y)
    sens.append(w.sens_row("public_spread_dedup_corrected", "P", "(i) observed after completion", base["estimate"], "bp", base["se"], base["n"], 0,
                           float(np.isfinite(y[frame["direction"].to_numpy() == "purchase"]).mean()), float(np.isfinite(y[frame["direction"].to_numpy() == "sale"]).mean()),
                           f"{int((~np.isfinite(y)).sum())} residual missing outcomes"))
    for t in (w.T_MEAN, w.T_LOW, w.T_HIGH):
        yi, n_imp, rule = w.impute(y, frame["direction"].to_numpy(), t, None)
        e = w.spread_estimate(frame, yi)
        sens.append(w.sens_row("public_spread_dedup_corrected", "P", t, e["estimate"], "bp", e["se"], e["n"], n_imp, np.nan, np.nan,
                               f"residual set after July completion and verified delisting paths; {rule}"))
    return rows, sens


# ---------------------------------------------------------------- the ledger

def ledger_rows(u: dict, uni: pd.DataFrame, base_uni: pd.DataFrame, comp: pd.DataFrame, led: pd.DataFrame, frames_T: dict, frames_P: dict, panels_T: dict, panels_P: dict) -> list[dict]:
    s, so = u["sig_union"], u["sig_old"]
    new_t = u["new_timed"]
    rows = []

    def add(stage, unit, baseline, corrected, note=""):
        rows.append({"stage": stage, "unit": unit, "baseline": baseline, "corrected": corrected, "difference": (corrected - baseline) if (baseline is not None and corrected is not None) else np.nan, "note": note})

    add("matched, amendment-deduplicated, directional lines with entry dates (timed)", "line", len(u["timed"]), len(u["union_timed"]),
        f"repaired: {len(u['cand'])} eligible unresolved lines, {len(u['resolved'])} resolved by the ticker-and-date rule, {new_t.attrs.get('n_beyond_calendar', 0)} dated beyond the return calendar, {len(new_t)} entering")
    add("signals after the collapse (incl. mixed)", "signal", len(so), len(s), f"{int(s['is_new_signal'].sum())} new signal keys; {int((s['has_repaired_lot'] & ~s['all_lots_repaired']).sum())} repaired lots join existing keys")
    mixed_old, mixed_new = int(so["mixed_direction"].astype(bool).sum()), int(s["mixed_direction"].astype(bool).sum())
    add("mixed-direction signals excluded", "signal", mixed_old, mixed_new, "mixed on the union: released signals newly mixed plus new signals mixed on creation")
    nm_old, nm_new = so.loc[~so["mixed_direction"].astype(bool)], s.loc[~s["mixed_direction"].astype(bool)]
    add("non-mixed signals without an entry session", "signal", int(nm_old["follower_entry_date"].isna().sum()), int(nm_new["follower_entry_date"].isna().sum()))
    we_old, we_new = nm_old.loc[nm_old["follower_entry_date"].notna()], nm_new.loc[nm_new["follower_entry_date"].notna()]
    dup_old = int(we_old["signal_uid"].isin(u["old_flags"]).sum())
    dup_new = int(we_new["dup_all_lines_union"].astype(bool).sum())
    add("vendor duplicate signals excluded (all lines duplicate an official transaction)", "signal", dup_old, dup_new, "audit re-run on the union: repaired official lines reveal further vendor copies")
    add("de-duplicated non-mixed signals with an entry session (universe)", "signal", len(base_uni), len(uni))
    y_b, y_c = base_uni["y_market_adjusted"].to_numpy(dtype=float), uni["y_market_adjusted"].to_numpy(dtype=float)
    add("complete 20-day market-adjusted outcome on the record", "signal", int(np.isfinite(y_b).sum()), int(np.isfinite(y_c).sum()))
    for reason in ("end_of_sample_truncation", "missing_return_days", "no_security_returns"):
        add(f"incomplete: {reason}", "signal", int((base_uni["status_market_adjusted"] == reason).sum()), int((uni["status_market_adjusted"] == reason).sum()))
    add("endpoint-censored windows completed on the July 2026 slice", "signal", None, int(comp["completed"].sum()))
    add("delisting windows verified and valued (terminal payoff once)", "signal", None, int(led["verified"].sum()), f"{len(led)} delisted windows without an outcome; verified {int(led['verified'].sum())}")
    done = set(comp.loc[comp["completed"], "event_id"]) | set(led.loc[led["verified"], "event_id"])
    resid = int((~np.isfinite(y_c) & ~uni["event_id"].isin(done)).sum())
    add("residual missing outcomes after completion", "signal", None, resid)
    add("signals in the primary corrected spread", "signal", int(np.isfinite(y_b).sum()), int(np.isfinite(y_c).sum()) + len(done))
    add("households in the spread", "household", int(base_uni.loc[np.isfinite(y_b), "household"].nunique()), int(uni.loc[np.isfinite(y_c) | uni["event_id"].isin(done), "household"].nunique()))
    for clock, frames, panels in (("T", frames_T, panels_T), ("P", frames_P, panels_P)):
        name = "transaction" if clock == "T" else "public"
        fb, fcx = frames["baseline"], frames["corrected"]
        pb = fb.loc[(fb["direction"] == "purchase") & ~fb["dup_all_lines"].astype(bool)]
        pc = fcx.loc[(fcx["direction"] == "purchase") & ~fcx["dup_all_lines"].astype(bool)]
        add(f"{name}-clock purchases in periods A and B (de-duplicated)", "purchase", len(pb), len(pc))
        add(f"{name}-clock purchases with a factor outcome", "purchase", int(np.isfinite(pd.to_numeric(pb['y_h20'], errors='coerce')).sum()), int(np.isfinite(pd.to_numeric(pc['y_h20'], errors='coerce')).sum()))
        add(f"{name}-clock households with purchases in both periods", "household",
            int(pb.groupby('household_id')['period'].nunique().ge(2).sum()), int(pc.groupby('household_id')['period'].nunique().ge(2).sum()))
        add(f"{name}-clock persistence households (rule R1)", "household", int(panels["baseline"]["household_id"].nunique()), int(panels["corrected"]["household_id"].nunique()),
            f"common cohort {int(panels['common_cohort']['household_id'].nunique())}")
        add(f"{name}-clock persistence-sample purchases", "purchase", len(panels["baseline"]), len(panels["corrected"]), f"common cohort {len(panels['common_cohort'])}")
    return rows


# ---------------------------------------------------------------- driver

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="the cumulative corrected sample and its missing-outcome accounting")
    ap.add_argument("--n-boot", type=int, default=rs.N_BOOT)
    ap.add_argument("--clocks", default="T,P")
    args = ap.parse_args(argv)
    clocks = tuple(args.clocks.split(","))
    log = fc.logger("cumulative")
    t0 = time.time()
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    if not cx.OUT_CACHE.exists():
        raise FileNotFoundError(f"{cx.OUT_CACHE} missing: run final.cache_extension first")
    u = build_union(log)
    calendar = u["calendar"]
    uni = public_universe_corrected(u, log)
    base_uni = baseline_universe(u)
    market = w.Market(calendar, pm.load_benchmark(), pm.load_factors(), pm.daily_parquet_path())
    market.load(uni["permno"].unique())
    comp, led = completions(uni, market, log)
    spread, vectors = spread_rows(uni, comp, led, base_uni, calendar, log)
    ladder = ladder_rows(uni, vectors[PRIMARY_SPREAD], calendar)
    miss_rows, sens = missingness_rows(uni, vectors, base_uni)
    cache = merged_cache(["permno", "anchor_idx", "resid_sd", *evf.CACHE_Y.values()])
    ctx, _ = ec.load_headline_context(clocks, with_placebo=False)
    frames_T = frames_P = None
    panels_T = panels_P = None
    tau_rows, dec_rows = [], []
    if "T" in clocks:
        fb, fcx, ev_T = transaction_frames(u, cache, log)
        frames_T = {"baseline": fb, "corrected": fcx}
        r, panels_T = persistence_rows("T", frames_T, ctx.cutoffs, calendar, args.n_boot, log)
        tau_rows += r
        frames_T_all = {**frames_T, "common_cohort": fcx.loc[fcx["household_id"].astype(str).isin(set(panels_T["baseline"]["household_id"].astype(str)))]}
        dec_rows += decomposition_rows("T", frames_T_all, calendar)
        ev_T.to_parquet(RESTRICTED / "events_T_corrected.parquet", index=False)
        fcx.to_parquet(RESTRICTED / "frame_T_corrected.parquet", index=False)
    if "P" in clocks:
        fb, fcx, ev_P = public_frames(u, cache, log)
        frames_P = {"baseline": fb, "corrected": fcx}
        r, panels_P = persistence_rows("P", frames_P, ctx.cutoffs, calendar, args.n_boot, log)
        tau_rows += r
        frames_P_all = {**frames_P, "common_cohort": fcx.loc[fcx["household_id"].astype(str).isin(set(panels_P["baseline"]["household_id"].astype(str)))]}
        dec_rows += decomposition_rows("P", frames_P_all, calendar)
        ev_P.to_parquet(RESTRICTED / "events_P_corrected.parquet", index=False)
        fcx.to_parquet(RESTRICTED / "frame_P_corrected.parquet", index=False)
    led_rows = ledger_rows(u, uni, base_uni, comp, led, frames_T, frames_P, panels_T, panels_P)
    # restricted intermediates
    uni.assign(**{f"y_{k}": v for k, v in vectors.items()}).to_parquet(RESTRICTED / "public_universe_corrected.parquet", index=False)
    comp.to_parquet(RESTRICTED / "completed_windows.parquet", index=False)
    led.to_parquet(RESTRICTED / "delisting_paths.parquet", index=False)
    u["sig_union"].to_parquet(RESTRICTED / "signals_union.parquet", index=False)
    u["union_timed"].to_parquet(RESTRICTED / "timed_union.parquet", index=False)
    # committed outputs
    fc.write_csv(pd.DataFrame(led_rows), OUT_LEDGER)
    fc.write_rows(spread + ladder, OUT_SPREAD)
    fc.write_rows(tau_rows, OUT_TAU)
    fc.write_rows(dec_rows, OUT_DECOMP)
    fc.write_csv(pd.DataFrame(miss_rows), OUT_MISS)
    fc.write_csv(pd.DataFrame(sens)[w.SENS_COLUMNS], OUT_MISS.with_name("missingness_sensitivity_corrected.csv"))
    entries = [fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "timed_transactions.csv"), fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "matched_transactions.csv"),
               fc.file_entry(fc.REPO_ROOT / "outputs" / "frames" / "signals.csv"), fc.file_entry(repair.TIER_ROWS), fc.file_entry(cx.OUT_CACHE),
               fc.file_entry(fcache.MANIFEST_PATH), fc.file_entry(pm.daily_parquet_path()), fc.file_entry(pm.project_root() / "data" / "raw" / "crsp" / ep.SLICE),
               fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_T.parquet"), fc.file_entry(fc.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "events_P.parquet"),
               fc.file_entry(evf.POOLED_CSV)]
    fc.write_manifest("cumulative", entries, config={"n_boot": args.n_boot, "clocks": list(clocks), "primary_spread": PRIMARY_SPREAD, "weights": "ES", "rule": "R1",
                                                      "cutoffs": "committed pooled p1/p99 by clock and period"},
                      extra={"seconds": round(time.time() - t0, 1)})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(led_rows).to_string(index=False))
    print(rs.results_frame(spread)[["variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(3).to_string(index=False))
    print(rs.results_frame(tau_rows)[["clock", "variant", "statistic", "value", "se_block", "boot_lo", "boot_hi", "n_events", "n_households"]].round(2).to_string(index=False))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
