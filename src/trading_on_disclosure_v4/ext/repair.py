"""Identity repairs and residual return gaps (extension plan round 3). The unresolved lines classified as
eligible common stocks are re-matched without looking at any return (a unique date-valid common stock for the
ticker as written, normalized, or read from the asset name), then passed through the released stages in the
released code on the union of the released timed lines and the repaired ones: entry dates, signal collapse
with the mixed flag, the version-4 duplicate audit, the market-adjusted and factor outcomes, transaction-clock
episodes and household eligibility. Writes data/v4/ext/repair_reconciliation.csv, repair_estimates.csv,
residual_gaps.csv; row-level tables under data/v4/restricted/ext/repair/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v2.outcomes import returns as rt
from trading_on_disclosure_v2.signals.build import build_public_signals
from trading_on_disclosure_v2.timing import derive_two_clock_entries
from trading_on_disclosure_v4.data import duplicates as du
from trading_on_disclosure_v4.data import events as evm
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.estimators import tau as tau_mod
from trading_on_disclosure_v4.existing import rerun
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.ext import ledgers as lg
from trading_on_disclosure_v4.ext import windows as w
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel as pm

BP = 1e4
H = 20
RESTRICTED = cm.RESTRICTED_EXT / "repair"
OUT_RECON = cm.OUT_DIR / "repair_reconciliation.csv"
OUT_EST = cm.OUT_DIR / "repair_estimates.csv"
OUT_GAPS = cm.OUT_DIR / "residual_gaps.csv"
LOG = cm.LOG_DIR / "repair.log"
TIER_ROWS = cm.RESTRICTED_EXT / "ledgers" / "tier_d_rows.parquet"
OUTCOME_COLS = {"ar_E_h20": "follower_entry_date", "ar_T_h20": "transaction_benchmark_entry_date"}


def log(msg: str) -> None:
    cm.log_line(LOG, msg)


# ---------------------------------------------------------------- identity repair (no returns read)

def resolve_candidates(ranges: pd.DataFrame) -> pd.DataFrame:
    """The eligible unresolved lines with their unique date-valid common-stock PERMNO (NaN when none or several)."""
    rows = pd.read_parquet(TIER_ROWS)
    elig = rows.loc[rows["category"] == "unresolved_eligible_security"].copy()
    dedup = load.load_deduplicated_transactions()
    sr = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "security_resolution.csv", low_memory=False, usecols=["transaction_uid", "transaction_date_reported"])
    d = elig.merge(dedup, on="transaction_uid", how="left", suffixes=("", "_d")).merge(sr.rename(columns={"transaction_date_reported": "txn_date_sr"}), on="transaction_uid", how="left")
    d["txn_date"] = pd.to_datetime(d["txn_date_sr"], errors="coerce")
    tick = d["ticker_text"].astype("string").str.strip()
    paren = d["security_name_text"].astype("string").str.strip().str.extract(lg.PAREN_TICKER, expand=False)
    d["ticker_used"] = tick.where(tick.notna() & (tick.str.len() > 0), paren)
    r = ranges.copy()
    r["common"] = lg.is_common_stock(r)
    r = r.loc[r["common"]]
    by_sym: dict[str, pd.DataFrame] = {}
    for col in ("Ticker", "TradingSymbol"):
        for sym, g in r.groupby(col):
            if sym:
                by_sym.setdefault(sym, []).append(g)
    by_sym = {k: pd.concat(v).drop_duplicates(lg.RANGE_KEY) for k, v in by_sym.items()}
    permno, how = [], []
    for t, dt in zip(d["ticker_used"].astype(object), d["txn_date"], strict=True):
        if pd.isna(t) or pd.isna(dt):
            permno.append(np.nan)
            how.append("no ticker or date")
            continue
        raw, norm = lg.normalize_ticker(str(t))
        cand, used = by_sym.get(raw), "as written"
        if cand is None and norm != raw:
            cand, used = by_sym.get(norm), "normalized"
        if cand is None:
            permno.append(np.nan)
            how.append("no common stock with the ticker")
            continue
        cov = cand[(cand["first_date"] <= dt) & (dt <= cand["last_date"])]
        ps = sorted(set(cov["PERMNO"].astype(int)))
        if len(ps) == 1:
            permno.append(ps[0])
            how.append(used)
        elif len(ps) == 0:
            permno.append(np.nan)
            how.append("no name range covers the date")
        else:
            permno.append(np.nan)
            how.append(f"{len(ps)} date-valid common stocks")
    d["permno_repaired"] = permno
    d["resolution"] = how
    return d


# ---------------------------------------------------------------- the released stages on the union

def timed_union(timed: pd.DataFrame, repaired: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry dates for the repaired lines with the released rule; returns (union timed frame, the repaired timed rows)."""
    cols = [c for c in timed.columns if c in repaired.columns]
    new = repaired.loc[repaired["permno_repaired"].notna(), cols].copy()
    new["permno"] = repaired.loc[repaired["permno_repaired"].notna(), "permno_repaired"].astype(int).to_numpy()
    new["trade_date_analysis"] = pd.to_datetime(new["trade_date_analysis"])
    new["analysis_public_date"] = pd.to_datetime(new["analysis_public_date"])
    beyond = new["trade_date_analysis"] > calendar[-1]
    new = new.loc[~beyond].copy()
    new.attrs["n_beyond_calendar"] = int(beyond.sum())
    new = derive_two_clock_entries(new, calendar, transaction_col="trade_date_analysis", public_col="analysis_public_date")
    new.attrs["n_beyond_calendar"] = int(beyond.sum())
    # self-check: the released rule on released rows reproduces the released entry dates
    chk = timed.sample(min(2000, len(timed)), random_state=0).copy()
    chk["trade_date_analysis"] = pd.to_datetime(chk["trade_date_analysis"])
    chk["analysis_public_date"] = pd.to_datetime(chk["analysis_public_date"])
    rec = derive_two_clock_entries(chk.drop(columns=[c for c in chk.columns if "entry" in c]), calendar, transaction_col="trade_date_analysis", public_col="analysis_public_date")
    for col in ("follower_entry_date", "transaction_benchmark_entry_date"):
        a, b = pd.to_datetime(rec[col]), pd.to_datetime(chk[col])
        bad = int(((a != b) & ~(a.isna() & b.isna())).sum())
        if bad:
            raise ValueError(f"entry-date rule mismatch on released rows: {bad} of {len(chk)} ({col})")
    union = pd.concat([timed, new], ignore_index=True)
    return union, new


def signals_union(union_timed: pd.DataFrame) -> pd.DataFrame:
    """The released signal collapse on the union (purchases and sales, analysis public date key)."""
    t = union_timed.loc[union_timed["direction"].isin(["purchase", "sale"]) & union_timed["permno"].notna()].copy()
    t["permno"] = t["permno"].astype(int)
    keep = ["transaction_uid", "filer_uid", "document_uid", "permno", "direction", "analysis_public_date", "transaction_date_reported", "owner", "amount_bracket",
            "analysis_public_date_source", "analysis_public_date_is_proxy"]
    s = build_public_signals(t[keep])
    ent = t.groupby(["filer_uid", "document_uid", "permno", "direction", "analysis_public_date"], dropna=False).agg(
        follower_entry_date=("follower_entry_date", "first"), transaction_benchmark_entry_date=("transaction_benchmark_entry_date", "first"),
        record_source=("record_source", lambda x: "official" if (x == "official").any() else "capitol_trades"), chamber=("chamber", "first")).reset_index()
    return s.merge(ent, on=["filer_uid", "document_uid", "permno", "direction", "analysis_public_date"], how="left")


def outcomes_for(frame: pd.DataFrame, daily: pd.DataFrame, bench: pd.Series, factors: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Released constructors: 20-day market-adjusted outcome at both entries, FF5+MOM at the public entry."""
    d = daily.loc[daily["permno"].isin(frame["permno"].unique())]
    out = rt.compute_signal_outcomes(frame, d, bench, calendar, anchors=(("E", "follower_entry_date", (H,)), ("T", "transaction_benchmark_entry_date", (H,))))
    out = fm.compute_factor_model_cars(out, d, factors, calendar, "follower_entry_date", H, models=("ff5mom",))
    return out


def dedup_flags_union(matched_union: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    """`dup_primary_all_lines` per signal on the union, from the version-4 audit."""
    audit, _ = du.prepare_audit_frame(matched_union)
    rows = du.audit_rows(audit)
    pairs = du.explode_signal_parents(signals)
    dup = set(rows.loc[rows["dup_primary"], "transaction_uid"])
    tot = pairs.groupby("signal_uid").size()
    hit = pairs[pairs["transaction_uid"].isin(dup)].groupby("signal_uid").size()
    all_lines = (hit.reindex(tot.index).fillna(0) >= tot) & (hit.reindex(tot.index).fillna(0) > 0)
    return all_lines.reindex(signals["signal_uid"]).fillna(False).to_numpy()


# ---------------------------------------------------------------- residual gaps

def residual_gaps(calendar: pd.DatetimeIndex, market: w.Market) -> pd.DataFrame:
    u, _ = w.public_universe(calendar)
    y = u["y_market_adjusted"].to_numpy(float)
    market.load(u["permno"].unique())
    dl = market.delist.reindex(u["permno"].to_numpy()).to_numpy(float)
    has = w.delisting_in_window(u["anchor_idx"].to_numpy(), dl)
    rows = []
    for r in u.loc[u["status_market_adjusted"] == "no_security_returns"].itertuples(index=False):
        g = market.grid(int(r.permno))
        fin = np.flatnonzero(np.isfinite(g))
        a = int(r.anchor_idx)
        if len(fin) == 0:
            cls, gap = "security has no return in the panel", np.nan
        elif a > fin[-1]:
            cls, gap = "anchor after the security's last panel session", int(a - fin[-1])
        elif a < fin[0]:
            cls, gap = "anchor before the security's first panel session", int(fin[0] - a)
        else:
            cls, gap = "anchor inside the security's span, no return in the window", int(np.flatnonzero(np.isfinite(g[a + 1:]))[0] + 1) if np.isfinite(g[a + 1:]).any() else np.nan
        rows.append({"status": "no_security_returns", "class": cls, "direction": r.direction, "period": r.period, "gap_sessions": gap, "delisting_in_window": bool(has[u.index.get_loc(r.Index)]) if False else np.nan})
    nd = u.loc[(u["status_market_adjusted"] == "missing_return_days") & ~has & ~np.isfinite(y)]
    for r in nd.itertuples(index=False):
        g = market.grid(int(r.permno))
        lo, hi = int(r.anchor_idx) + 1, int(r.anchor_idx) + H
        rows.append({"status": "missing_return_days", "class": "interior missing session, no delisting: unverifiable", "direction": r.direction, "period": r.period,
                     "gap_sessions": int((~np.isfinite(g[lo:hi + 1])).sum()), "delisting_in_window": False})
    df = pd.DataFrame(rows)
    out = df.groupby(["status", "class", "direction"]).agg(n=("gap_sessions", "size"), gap_median=("gap_sessions", "median"), gap_min=("gap_sessions", "min"), gap_max=("gap_sessions", "max")).reset_index()
    tot = df.groupby(["status", "class"]).agg(n=("gap_sessions", "size"), gap_median=("gap_sessions", "median"), gap_min=("gap_sessions", "min"), gap_max=("gap_sessions", "max")).reset_index()
    tot["direction"] = "all"
    return pd.concat([out, tot[out.columns]], ignore_index=True)


# ---------------------------------------------------------------- driver

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="identity repairs on the eligible unresolved lines; residual gaps")
    ap.parse_args(argv)
    t0 = time.time()
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    calendar = pm.v2_trading_calendar()
    ranges = lg.build_security_ranges()
    cand = resolve_candidates(ranges)
    recon: list[dict] = []
    recon.append(dict(stage="eligible unresolved lines (section 2)", n=int(len(cand)), note="common_stock_date_valid + nonunique_date_valid_candidates"))
    for how, g in cand.groupby("resolution"):
        recon.append(dict(stage=f"resolution: {how}", n=int(len(g)), note="identity by ticker and date validity only; no return read"))
    resolved = cand.loc[cand["permno_repaired"].notna()]
    recon.append(dict(stage="resolved to one date-valid common stock", n=int(len(resolved)), note=f"{resolved['filer_uid'].nunique()} households, {resolved['document_uid'].nunique()} reports"))
    log(f"candidates {len(cand)}, resolved {len(resolved)}")
    # released stages on the union
    timed = load.load_timed_transactions()
    matched = load.load_matched_transactions()
    union_timed, new_timed = timed_union(timed, resolved, calendar)
    recon.append(dict(stage="repaired lines with a trade date beyond the return calendar (no episode possible)", n=int(new_timed.attrs.get("n_beyond_calendar", 0)), note="dropped from the union"))
    recon.append(dict(stage="repaired lines with entry dates", n=int(len(new_timed)), note=f"follower entry missing on {int(new_timed['follower_entry_date'].isna().sum())}"))
    sig_old = load.load_signals()
    sig_new_all = signals_union(union_timed)
    old_uids = set(sig_old["signal_uid"])
    repaired_uids = set(new_timed["transaction_uid"])
    parents = sig_new_all["parent_transaction_uids"].str.split(";")
    has_rep = parents.map(lambda ps: any(p in repaired_uids for p in ps))
    only_rep = parents.map(lambda ps: all(p in repaired_uids for p in ps))
    is_new = ~sig_new_all["signal_uid"].isin(old_uids)
    recon.append(dict(stage="signals on the union", n=int(len(sig_new_all)), note=f"released {len(sig_old)}"))
    recon.append(dict(stage="new signals (all lots repaired)", n=int((is_new & only_rep).sum()), note="signal keys that did not exist"))
    recon.append(dict(stage="repaired lots joining an existing signal key", n=int((has_rep & ~only_rep).sum()), note="lot added to an existing signal; no new signal"))
    old_mixed = sig_old.set_index("signal_uid")["mixed_direction"].astype(bool)
    new_mixed = sig_new_all.set_index("signal_uid")["mixed_direction"].astype(bool)
    common_uids = [u for u in sig_new_all["signal_uid"] if u in old_mixed.index]
    newly_mixed = int((new_mixed.reindex(common_uids).to_numpy() & ~old_mixed.reindex(common_uids).to_numpy()).sum())
    recon.append(dict(stage="released signals newly mixed", n=newly_mixed, note="a repaired line of the other direction under the same filer, report and security"))
    new_sig = sig_new_all.loc[is_new & only_rep].copy()
    recon.append(dict(stage="new signals mixed on creation", n=int(new_sig["mixed_direction"].astype(bool).sum()), note=""))
    # outcomes for the new signals with the released constructors
    daily = pm.load_local_daily()
    bench = pm.load_benchmark()
    factors = pm.load_factors()
    new_out = outcomes_for(new_sig, daily, bench, factors, calendar)
    recon.append(dict(stage="new signals with a 20-day market-adjusted outcome", n=int(new_out["ar_E_h20"].notna().sum()),
                      note="; ".join(f"{k} {v}" for k, v in new_out["ar_E_h20_reason"].value_counts().items())))
    recon.append(dict(stage="new signals with a factor outcome", n=int(new_out["car_ff5mom_h20"].notna().sum()), note=""))
    # duplicate audit on the union (matched frame + repaired lines)
    rep_matched = resolved.loc[:, [c for c in matched.columns if c in resolved.columns]].copy()
    rep_matched["permno"] = resolved["permno_repaired"].astype(int).to_numpy()
    matched_union = pd.concat([matched, rep_matched], ignore_index=True)
    flags = dedup_flags_union(matched_union, sig_new_all)
    sig_new_all["dup_all_lines_union"] = flags
    old_flags = rerun.duplicate_signal_uids()
    old_on_union = sig_new_all["signal_uid"].isin(old_flags).to_numpy()
    recon.append(dict(stage="released signals newly flagged as vendor duplicates on the union", n=int((flags & ~old_on_union & sig_new_all["signal_uid"].isin(old_uids).to_numpy()).sum()),
                      note="a repaired official line is the twin of a vendor-only signal"))
    recon.append(dict(stage="new signals flagged as vendor duplicates", n=int((flags & (is_new & only_rep).to_numpy()).sum()), note=""))
    # the public-clock spread: baseline vs repaired
    signals_old = rerun.load_signals()
    dedup_old, _ = rerun.dedup_filter(signals_old, old_flags)
    base = rerun.run_test(dedup_old, next(t for t in rerun.TESTS if t.test_id == "TW_P1_E20"), "block20", calendar)
    keep_old = dedup_old.loc[~dedup_old["signal_uid"].isin(set(sig_new_all.loc[sig_new_all["mixed_direction"].astype(bool), "signal_uid"]))]
    keep_old = keep_old.loc[~keep_old["signal_uid"].isin(set(sig_new_all.loc[flags, "signal_uid"]))]
    add = new_out.loc[~new_out["mixed_direction"].astype(bool) & ~new_out["signal_uid"].isin(set(sig_new_all.loc[flags, "signal_uid"])) & new_out["ar_E_h20"].notna()]
    rep_frame = pd.concat([keep_old[["signal_uid", "filer_uid", "direction", "follower_entry_date", "ar_E_h20"]],
                          add[["signal_uid", "filer_uid", "direction", "follower_entry_date", "ar_E_h20"]]], ignore_index=True)
    rep_frame["follower_entry_date"] = pd.to_datetime(rep_frame["follower_entry_date"])
    r = rerun.run_test(rep_frame, next(t for t in rerun.TESTS if t.test_id == "TW_P1_E20"), "block20", calendar)
    est = [rs.row("P", "repair", "public_spread_dedup", "baseline", base["estimate"] * BP, "bp", se_block=base["se"] * BP, n_events=base["n"], n_households=base["n_households"], notes="the record"),
           rs.row("P", "repair", "public_spread_dedup", "repaired", r["estimate"] * BP, "bp", se_block=r["se"] * BP, n_events=r["n"], n_households=r["n_households"],
                  notes=f"{len(add)} new signals with an outcome added; {len(dedup_old) - len(keep_old)} released signals removed (newly mixed or newly duplicate)")]
    recon.append(dict(stage="public-clock spread sample: released signals removed (newly mixed or duplicate)", n=int(len(dedup_old) - len(keep_old)), note=""))
    recon.append(dict(stage="public-clock spread sample: new signals added", n=int(len(add)), note=""))
    # transaction clock: episodes, outcomes, eligibility, tau2
    ev_old, _ = evm.build_events_T(timed, calendar)
    ev_new, _ = evm.build_events_T(union_timed, calendar)
    new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))]
    recon.append(dict(stage="transaction-clock episodes on the union", n=int(len(ev_new)), note=f"released {len(ev_old)}; new {len(new_ev)}"))
    ctx, _ = cm.load_headline_context(("T",), with_placebo=False)
    frame_old = ctx.frames["T"]
    cache = fc.load_cache(columns=["permno", "anchor_idx", "bhar_ff5mom_h20"])
    ne = new_ev.copy()
    ne["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ne["anchor_date"]))
    ne = ne.merge(cache, on=["permno", "anchor_idx"], how="left")
    miss = ne["bhar_ff5mom_h20"].isna() & ~ne["permno"].isin(set(cache["permno"].unique()))
    if miss.any():
        sub = ne.loc[miss, ["event_id", "permno", "anchor_date"]].rename(columns={"anchor_date": "follower_entry_date"})
        o = fm.compute_factor_model_cars(sub, daily.loc[daily["permno"].isin(sub["permno"].unique())], factors, calendar, "follower_entry_date", H, models=("ff5mom",))
        ne.loc[miss, "bhar_ff5mom_h20"] = o["car_ff5mom_h20"].to_numpy()
    ne["y_h20"] = ne["bhar_ff5mom_h20"].astype(float) * BP
    ne["dup_all_lines"] = False
    ne["record_source"] = ne["record_source"].astype(str)
    recon.append(dict(stage="new transaction-clock purchases in periods A and B with a factor outcome", n=int(((ne["direction"] == "purchase") & ne["period"].isin(["A", "B"]) & ne["y_h20"].notna()).sum()),
                      note=f"securities outside the forward cache computed with the released constructor: {int(miss.sum())}"))
    cols = ["event_id", "household_id", "clock", "period", "anchor_date", "anchor_year", "window_end_date", "record_source", "permno", "direction", "dup_all_lines", "y_h20"]
    for c in cols:
        if c not in frame_old.columns and c in ne.columns:
            pass
    ext_cols = [c for c in frame_old.columns if c in ne.columns]
    frame_new = pd.concat([frame_old, ne[ext_cols]], ignore_index=True)
    for c in frame_old.columns:
        if c not in frame_new.columns:
            frame_new[c] = np.nan
    spec = sp.headline_spec("T")
    panel_old = ctx.panel(spec)
    ctx2 = rs.Context(frames={"T": frame_new}, calendar=calendar, cutoffs=ctx.cutoffs, placebo_source=lambda *a: iter(()))
    panel_new = ctx2.panel(spec)
    hh_old, hh_new = set(panel_old["household_id"]), set(panel_new["household_id"])
    recon.append(dict(stage="persistence households: baseline", n=len(hh_old), note=""))
    recon.append(dict(stage="persistence households: repaired", n=len(hh_new), note=f"newly eligible {len(hh_new - hh_old)}, no longer eligible {len(hh_old - hh_new)}"))
    recon.append(dict(stage="persistence-sample purchases: baseline / repaired", n=int(len(panel_new)), note=f"baseline {len(panel_old)}"))
    t_old = cgm.tau2_i1(rs.block_cluster(panel_old, calendar), "ES", anchor="anchor_blk")
    t_new = cgm.tau2_i1(rs.block_cluster(panel_new, calendar), "ES", anchor="anchor_blk")
    est += [rs.row("T", "repair", "tau2_headline", "baseline", t_old["tau2_hat"], "bp2", se_block=t_old["var_wx"] * t_old["se_b"], n_events=int(t_old["n_events_a"] + t_old["n_events_b"]), n_households=int(t_old["n_households"]), notes="the headline"),
            rs.row("T", "repair", "tau2_headline", "repaired", t_new["tau2_hat"], "bp2", se_block=t_new["var_wx"] * t_new["se_b"], n_events=int(t_new["n_events_a"] + t_new["n_events_b"]), n_households=int(t_new["n_households"]),
                   notes="repaired purchases added, eligibility re-applied; winsorized at the committed cutoffs; no placebo p (the stored placebo datasets hold no repaired events)")]
    for lab, pnl in (("baseline", panel_old), ("repaired", panel_new)):
        tb = tau_mod.household_table(pnl, "ES")
        est.append(rs.row("T", "repair", "tau2_headline", f"{lab}_households_events", int(tb.n_a.sum() + tb.n_b.sum()), "count", n_households=int(len(tb.household_id)), notes=f"period A {int(tb.n_a.sum())}, period B {int(tb.n_b.sum())}"))
    # residual gaps
    market = w.Market(calendar, bench, factors, pm.daily_parquet_path())
    gaps = residual_gaps(calendar, market)
    gaps.to_csv(OUT_GAPS, index=False, lineterminator="\n")
    # write
    pd.DataFrame(recon).to_csv(OUT_RECON, index=False, lineterminator="\n")
    cm.write_rows(est, OUT_EST)
    cand.drop(columns=[c for c in cand.columns if c in ("security_name_text",)]).to_parquet(RESTRICTED / "candidates.parquet", index=False)
    new_out.to_parquet(RESTRICTED / "new_signals_outcomes.parquet", index=False)
    ne.to_parquet(RESTRICTED / "new_episodes.parquet", index=False)
    cm.write_manifest("repair", [cm.file_entry(TIER_ROWS), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "timed_transactions.csv"), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "matched_transactions.csv"),
                                 cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv"), cm.file_entry(pm.daily_parquet_path())], {"seconds": round(time.time() - t0, 1)})
    log(f"done in {time.time() - t0:.0f} s")
    print(pd.DataFrame(recon).to_string(index=False))
    print(rs.results_frame(est)[["clock", "variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
