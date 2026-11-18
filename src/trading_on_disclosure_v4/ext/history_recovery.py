"""History recovery for the two households (extension plan round 3): the extracted schedule rows become
candidate transactions; identities are resolved without any return (a parenthesized ticker, else the cleaned asset
text looked up in a name dictionary built from the project's own parsed records, then the design's date-valid
exact-ticker rule); amendments and cross-document duplicates follow the released hierarchy; the recovered purchases
and sales become transaction-clock episodes with the released rules, the factor outcome and the eligibility rule,
and the headline persistence estimate is reported for the original cohort and the expanded one. Writes
data/v4/ext/history_recovery_inventory.csv and history_recovery_estimates.csv; transactions restricted.
"""

from __future__ import annotations

import argparse
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes import factor_models as fm
from trading_on_disclosure_v2.securities.matching import normalize_name
from trading_on_disclosure_v4.data import events as evm
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.ext import ledgers as lg
from trading_on_disclosure_v4.ext import two_households as th
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel as pm

BP = 1e4
H = 20
HIST = cm.RESTRICTED_EXT / "two_households" / "history"
ROWS = HIST / "extracted_rows.parquet"
FILINGS = HIST / "filings.parquet"
OUT_INV = cm.OUT_DIR / "history_recovery_inventory.csv"
OUT_EST = cm.OUT_DIR / "history_recovery_estimates.csv"
LOG = cm.LOG_DIR / "history_recovery.log"
OWNER = {"SP": "spouse", "JT": "joint", "DC": "dependent_child"}
TYPE = {"P": "purchase", "S": "sale"}
NEAR = 0.90
MAX_LAG_DAYS = 400
ROUTES = {"exact": ("ticker in parentheses", "name, exact"), "near_exact": ("ticker in parentheses", "name, exact", "name, near-exact")}
LABEL = "provisional, validated on the gold standard only"
DECOR_TAG = re.compile(r"\[[A-Za-z]{2}\]")
DECOR_PAREN = re.compile(r"\(([A-Za-z][A-Za-z0-9.\-]{0,6})\)")
DECOR_DOLLAR = re.compile(r"\$\s?[\d,]+(?:\s*-\s*\$\s?[\d,]+)?")
DECOR_WORDS = re.compile(r"\b(COMMON STOCK|COMMON|STOCK|CMN|COM|CAPITAL|SHARES|SHS|ORDINARY|ORD|NEW|ADR|ADS)\b")


def log(msg: str) -> None:
    cm.log_line(LOG, msg)


# ---------------------------------------------------------------- identity without returns

def clean_key(text) -> str:
    """The matcher's name normalization after the form's decorations are removed (parenthesized ticker, asset-type
    code, dollar amounts, security-type words); the share class survives as one token."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = DECOR_DOLLAR.sub(" ", DECOR_PAREN.sub(" ", DECOR_TAG.sub(" ", str(text))))
    t = re.sub(r"\bCLASS ([A-C])\b", r"CLASS\1", normalize_name(t))
    return re.sub(r"\s+", " ", DECOR_WORDS.sub(" ", t)).strip()


def ocr_key(text) -> str:
    """clean_key of an extracted asset text, less a leading owner code and one-character tokens at either end."""
    toks = clean_key(text).split()
    while toks and (toks[0] in OWNER or len(toks[0]) == 1):
        toks.pop(0)
    while toks and len(toks[-1]) == 1:
        toks.pop()
    return " ".join(toks)


def name_dictionary(matched: pd.DataFrame, snapshot: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Cleaned name -> ticker from the released matched lines (both sources) and the vendor snapshot's issuer names;
    returns (the dictionary, the names that map to several tickers and are therefore not in it)."""
    off = matched.loc[matched["permno"].notna()]
    pairs = pd.concat([pd.DataFrame({"key": off["security_name_text"].map(clean_key), "ticker": off["ticker_text"].astype(object)}),
                       pd.DataFrame({"key": snapshot["issuer_name"].map(clean_key), "ticker": snapshot["ticker_clean"].astype(object)})], ignore_index=True)
    pairs["ticker"] = pairs["ticker"].map(lambda x: str(x).strip().upper() if isinstance(x, str) and str(x).strip() else None)
    pairs = pairs.loc[(pairs["key"] != "") & pairs["ticker"].notna()]
    n = pairs.groupby("key")["ticker"].nunique()
    ambiguous = set(n.index[n > 1])
    d = pairs.drop_duplicates("key").set_index("key")["ticker"]
    return {k: v for k, v in d.items() if k not in ambiguous}, ambiguous


def near_match(key: str, by_prefix: dict[str, list[str]], dictionary: dict[str, str]) -> tuple[str | None, str]:
    """The unique ticker among dictionary names sharing the key's first three characters with similarity >= NEAR."""
    if len(key) < 4:
        return None, "near-exact: key too short"
    hits = []
    for k in by_prefix.get(key[:3], ()):
        m = SequenceMatcher(None, key, k)
        if m.real_quick_ratio() >= NEAR and m.quick_ratio() >= NEAR and m.ratio() >= NEAR:
            hits.append(k)
    if not hits:
        return None, "name not in the dictionary"
    ticks = {dictionary[k] for k in hits}
    if len(ticks) != 1:
        return None, "near-exact: several tickers"
    return ticks.pop(), "resolved"


def symbol_ranges(ranges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Common-stock name ranges by ticker and by trading symbol (the repair module's lookup)."""
    r = ranges.loc[lg.is_common_stock(ranges)]
    by: dict[str, list[pd.DataFrame]] = {}
    for col in ("Ticker", "TradingSymbol"):
        for sym, g in r.groupby(col):
            if sym:
                by.setdefault(sym, []).append(g)
    return {k: pd.concat(v).drop_duplicates(lg.RANGE_KEY) for k, v in by.items()}


def ticker_permno(by_sym: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp) -> tuple[float, str]:
    """The design's date-valid exact-ticker rule: one common stock whose name range covers the date."""
    raw, norm = lg.normalize_ticker(str(ticker))
    cand = by_sym.get(raw)
    if cand is None and norm != raw:
        cand = by_sym.get(norm)
    if cand is None:
        return np.nan, "no common stock with the ticker"
    cov = cand[(cand["first_date"] <= date) & (date <= cand["last_date"])]
    ps = sorted(set(cov["PERMNO"].astype(int)))
    if len(ps) == 1:
        return float(ps[0]), "resolved"
    return np.nan, ("no name range covers the date" if not ps else f"{len(ps)} date-valid common stocks")


def candidates(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows with a plausible date (on or before the filing date, within MAX_LAG_DAYS of it), an asset text, an ink-based
    purchase or sale type and an ink-based bracket."""
    r = rows.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r["filing_date"] = pd.to_datetime(r["filing_date"])
    r["has_date"] = r["date"].notna()
    r["date_plausible"] = r["has_date"] & (r["date"] <= r["filing_date"]) & (r["date"] >= r["filing_date"] - pd.Timedelta(days=MAX_LAG_DAYS))
    r["has_asset"] = r["asset_text"].map(lambda s: isinstance(s, str) and ocr_key(s) != "")
    r["direction"] = r["ink_type"].map(lambda s: TYPE.get(str(s).strip().upper()) if isinstance(s, str) else None)
    r["has_type"] = r["direction"].notna()
    r["has_bracket"] = r["ink_bracket"].map(lambda s: isinstance(s, str) and s.strip() != "")
    r["candidate"] = r["date_plausible"] & r["has_asset"] & r["has_type"] & r["has_bracket"]
    return r


def resolve(c: pd.DataFrame, dictionary: dict[str, str], ambiguous: set[str], by_sym: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Identity for the candidate rows: a parenthesized ticker, else the cleaned name in the dictionary (exact, then
    near-exact), then the date-valid exact-ticker rule; no return is read."""
    by_prefix: dict[str, list[str]] = {}
    for k in dictionary:
        by_prefix.setdefault(k[:3], []).append(k)
    route, ticker, permno, reason = [], [], [], []
    for r in c.itertuples(index=False):
        if not r.candidate:
            route.append(None), ticker.append(None), permno.append(np.nan), reason.append("not a candidate")
            continue
        key = ocr_key(r.asset_text)
        paren = DECOR_PAREN.search(str(r.asset_text))
        if paren:
            t, how, why = paren.group(1).upper(), "ticker in parentheses", "resolved"
        elif key in dictionary:
            t, how, why = dictionary[key], "name, exact", "resolved"
        elif key in ambiguous:
            t, how, why = None, None, "name maps to several tickers"
        else:
            t, why = near_match(key, by_prefix, dictionary)
            how = "name, near-exact" if t else None
        if t is not None:
            p, why = ticker_permno(by_sym, t, r.date)
        else:
            p = np.nan
        route.append(how if np.isfinite(p) else None), ticker.append(t), permno.append(p), reason.append(why)
    out = c.copy()
    out["route"], out["ticker_resolved"], out["permno"], out["resolution"] = route, ticker, permno, reason
    return out


# ---------------------------------------------------------------- the released hierarchy and the episodes

def reconcile(res: pd.DataFrame) -> pd.DataFrame:
    """Released hierarchy on the resolved rows: a row whose security, date, direction, owner and bracket reproduce a
    row of an earlier-filed document of the household is suppressed; the public date stays the row's own filing date."""
    d = res.loc[res["permno"].notna()].copy()
    d["owner_std"] = d["owner"].map(lambda o: OWNER.get(str(o).strip().upper(), "self") if isinstance(o, str) else "self")
    d = d.sort_values(["household", "filing_date", "filing_seq", "page"], kind="stable")
    key = ["household", "permno", "date", "direction", "owner_std", "ink_bracket"]
    first = d.groupby(key)["filing_seq"].transform("min")
    d["amendment_exact_duplicate"] = (d["filing_seq"] != first).to_numpy()
    amended = d["is_amendment"].astype(bool)
    d["lineage"] = np.where(amended & d["amends_seq"].notna(), "explicit", np.where(amended, "unresolved", "original"))
    return d


def timed_frame(d: pd.DataFrame, labels: dict[str, str], chamber: str) -> pd.DataFrame:
    """A timed-transactions frame for the recovered lines with the released columns the episode builder reads."""
    keep = d.loc[~d["amendment_exact_duplicate"]]
    uid = [f"REC:{r.household}:{r.filing_seq}:{r.page}:{i}" for i, r in enumerate(keep.itertuples(index=False))]
    return pd.DataFrame({
        "transaction_uid": uid, "filer_uid": keep["household"].map(labels).to_numpy(),
        "document_uid": ("REC:" + keep["household"].astype(str) + ":" + keep["filing_seq"].astype(str)).to_numpy(),
        "record_source": "official", "direction": keep["direction"].to_numpy(), "permno": keep["permno"].astype("int64").to_numpy(),
        "trade_date_analysis": pd.to_datetime(keep["date"]).to_numpy(), "transaction_date_reported": pd.to_datetime(keep["date"]).to_numpy(),
        "analysis_public_date": pd.to_datetime(keep["filing_date"]).to_numpy(), "analysis_public_date_source": evm.OFFICIAL_OBSERVED,
        "analysis_public_date_is_proxy": False, "owner": keep["owner_std"].to_numpy(), "amount_bracket": keep["ink_bracket"].to_numpy(),
        "chamber": chamber, "is_amendment": keep["is_amendment"].astype(bool).to_numpy(), "report_year": keep["report_year"].to_numpy(),
        "ticker_text": keep["ticker_resolved"].to_numpy(), "route": keep["route"].to_numpy()})


def factor_outcomes(ev: pd.DataFrame, calendar: pd.DatetimeIndex, loader) -> tuple[pd.DataFrame, int]:
    """FF5+MOM 20-session outcome from the forward cache, or from the released constructor for securities outside it."""
    cache = fc.load_cache(columns=["permno", "anchor_idx", "bhar_ff5mom_h20"])
    e = ev.copy()
    e["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(e["anchor_date"]))
    e = e.merge(cache, on=["permno", "anchor_idx"], how="left")
    miss = e["bhar_ff5mom_h20"].isna() & ~e["permno"].isin(set(cache["permno"].unique()))
    if miss.any():
        daily, factors = loader()
        sub = e.loc[miss, ["event_id", "permno", "anchor_date"]].rename(columns={"anchor_date": "follower_entry_date"})
        o = fm.compute_factor_model_cars(sub, daily.loc[daily["permno"].isin(sub["permno"].unique())], factors, calendar, "follower_entry_date", H, models=("ff5mom",))
        e.loc[miss, "bhar_ff5mom_h20"] = o["car_ff5mom_h20"].to_numpy()
    e["y_h20"] = e["bhar_ff5mom_h20"].astype(float) * BP
    return e, int(miss.sum())


def expanded_estimate(ctx: rs.Context, calendar: pd.DatetimeIndex, frame_old: pd.DataFrame, new_ev: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The headline specification on the released frame plus the recovered episodes; eligibility re-applied."""
    ne = new_ev.copy()
    ne["dup_all_lines"], ne["dup_any_line"] = False, False
    ne["record_source"] = ne["record_source"].astype(str)
    frame_new = pd.concat([frame_old, ne[[c for c in frame_old.columns if c in ne.columns]]], ignore_index=True)
    for c in frame_old.columns:
        if c not in frame_new.columns:
            frame_new[c] = np.nan
    spec = sp.headline_spec("T")
    ctx2 = rs.Context(frames={"T": frame_new}, calendar=calendar, cutoffs=ctx.cutoffs, placebo_source=lambda *a: iter(()))
    panel = ctx2.panel(spec)
    t = cgm.tau2_i1(rs.block_cluster(panel, calendar), "ES", anchor="anchor_blk")
    return t, panel, frame_new


def household_facts(frame: pd.DataFrame, hid: str) -> dict:
    """Purchases with a finite outcome by period, with quarters and securities (the eligibility rule's inputs)."""
    sub = frame.loc[(frame["household_id"] == hid) & (frame["direction"] == "purchase") & np.isfinite(frame["y_h20"].astype(float))]
    out = {}
    for p in ("A", "B"):
        g = sub.loc[sub["period"] == p]
        out[f"n_{p}"], out[f"quarters_{p}"], out[f"securities_{p}"] = int(len(g)), int(pd.DatetimeIndex(g["anchor_date"]).to_period("Q").nunique()), int(g["permno"].nunique())
    return out


def tau_row(variant: str, t: dict, notes: str) -> dict:
    return rs.row("T", "history_recovery", "tau2_headline", variant, t["tau2_hat"], "bp2", se_block=t["var_wx"] * t["se_b"],
                  n_events=int(t["n_events_a"] + t["n_events_b"]), n_households=int(t["n_households"]), notes=notes)


# ---------------------------------------------------------------- driver

def inventory(filings: pd.DataFrame, res: pd.DataFrame, rec: pd.DataFrame, episodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts by household and year (filing-date year for filings and rows, trade year for episodes), with totals."""
    f = filings.copy()
    f["year"] = pd.to_datetime(f["filing_date"]).dt.year
    r = res.copy()
    r["year"] = pd.to_datetime(r["filing_date"]).dt.year
    d = rec.copy()
    d["year"] = pd.to_datetime(d["filing_date"]).dt.year
    rows = []
    keys = sorted({(h, int(y)) for h, y in zip(f["household"], f["year"], strict=True)} | {(h, int(y)) for h, y in zip(r["household"], r["year"], strict=True)})
    groups = [(h, y, f[(f.household == h) & (f.year == y)], r[(r.household == h) & (r.year == y)], d[(d.household == h) & (d.year == y)]) for h, y in keys]
    groups += [(h, "all", f[f.household == h], r[r.household == h], d[d.household == h]) for h in sorted(f["household"].unique())]
    groups += [("total", "all", f, r, d)]
    for h, y, fg, rg, rec_g in groups:
        row = {"household": h, "year": y, "filings": int(len(fg)), "filings_read": int((fg["read_status"] == "read").sum()), "pages": int(fg["pages"].sum()),
               "pages_with_headers": int(fg["pages_with_headers"].sum()), "rows_extracted": int(len(rg)), "rows_with_date": int(rg["has_date"].sum()),
               "rows_date_plausible": int(rg["date_plausible"].sum()), "rows_with_ink_type": int(rg["has_type"].sum()), "rows_with_ink_bracket": int(rg["has_bracket"].sum()),
               "rows_candidate": int(rg["candidate"].sum()), "resolved_ticker": int((rg["route"] == "ticker in parentheses").sum()),
               "resolved_name_exact": int((rg["route"] == "name, exact").sum()), "resolved_name_near": int((rg["route"] == "name, near-exact").sum()),
               "unresolved_name": int(rg["candidate"].sum() - rg["route"].notna().sum() - rg["resolution"].isin(["no common stock with the ticker", "no name range covers the date"]).sum()
                                      - rg["resolution"].str.contains("date-valid common stocks", na=False).sum()),
               "unresolved_ticker_rule": int(rg["resolution"].isin(["no common stock with the ticker", "no name range covers the date"]).sum()
                                             + rg["resolution"].str.contains("date-valid common stocks", na=False).sum()),
               "duplicates_suppressed_exact": int((rec_g["amendment_exact_duplicate"] & rec_g["route"].isin(ROUTES["exact"])).sum()),
               "duplicates_suppressed_near": int(rec_g["amendment_exact_duplicate"].sum())}
        for name, ev in episodes.items():
            e = ev.loc[ev["household"].isin(set(fg["household"]))] if y == "all" else ev.loc[ev["household"].isin(set(fg["household"])) & (pd.DatetimeIndex(ev["anchor_date"]).year == y)]
            row[f"episodes_{name}"] = int(len(e))
            row[f"episodes_with_outcome_{name}"] = int(np.isfinite(e["y_h20"].astype(float)).sum())
            row[f"purchases_A_with_outcome_{name}"] = int(((e["direction"] == "purchase") & (e["period"] == "A") & np.isfinite(e["y_h20"].astype(float))).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="history recovery for the two households from the extracted schedule rows")
    ap.parse_args(argv)
    t0 = time.time()
    rows, filings = pd.read_parquet(ROWS), pd.read_parquet(FILINGS)
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    labels = {f"household_{i}": h for i, h in enumerate(households, start=1)}
    snapshot = pd.read_parquet(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT, columns=["issuer_name", "ticker_clean"])
    dictionary, ambiguous = name_dictionary(matched, snapshot)
    by_sym = symbol_ranges(lg.build_security_ranges())
    log(f"rows {len(rows)}, filings {len(filings)}; dictionary {len(dictionary)} names ({len(ambiguous)} ambiguous dropped); tickers with ranges {len(by_sym)}")
    c = candidates(rows)
    res = resolve(c, dictionary, ambiguous, by_sym)
    rec = reconcile(res)
    log(f"candidates {int(c['candidate'].sum())}; routes {res['route'].value_counts().to_dict()}; resolution {res.loc[res['candidate'], 'resolution'].value_counts().to_dict()}; "
        f"exact duplicates {int(rec['amendment_exact_duplicate'].sum())}; lineage {rec['lineage'].value_counts().to_dict()}")
    timed = load.load_timed_transactions()
    chamber = str(timed.loc[timed["filer_uid"].isin(households), "chamber"].mode().iloc[0])
    calendar = pm.v2_trading_calendar()
    ctx, _ = cm.load_headline_context(("T",), with_placebo=False)
    frame_old = ctx.frames["T"]
    panel_old = ctx.panel(sp.headline_spec("T"))
    t_old = cgm.tau2_i1(rs.block_cluster(panel_old, calendar), "ES", anchor="anchor_blk")
    ev_old, _ = evm.build_events_T(timed, calendar)
    est = [tau_row("original_cohort", t_old, "the headline; the two households hold no first-period purchase on the record")]
    for lab, hid in labels.items():
        fx = household_facts(frame_old, hid)
        est.append(rs.row("T", "history_recovery", f"eligibility_{lab}", "record", float(hid in set(panel_old["household_id"])), "flag", n_events=fx["n_A"],
                          notes=f"period A purchases with an outcome {fx['n_A']} over {fx['quarters_A']} quarters and {fx['securities_A']} securities; period B {fx['n_B']} over {fx['quarters_B']} quarters"))
    _store: dict = {}

    def loader():
        if "daily" not in _store:
            _store["daily"], _store["factors"] = pm.load_local_daily(), pm.load_factors()
        return _store["daily"], _store["factors"]

    episodes: dict[str, pd.DataFrame] = {}
    for name, routes in ROUTES.items():
        d = rec.loc[rec["route"].isin(routes)]
        tf = timed_frame(d, labels, chamber)
        union = pd.concat([timed, tf[[col for col in tf.columns if col in timed.columns]]], ignore_index=True)
        ev_new, _ = evm.build_events_T(union, calendar)
        new_ev = ev_new.loc[~ev_new["event_id"].isin(set(ev_old["event_id"]))].copy()
        if not new_ev["household_id"].isin(set(households)).all() or len(ev_new) != len(ev_old) + len(new_ev):
            raise ValueError("recovered episodes touch a released episode")
        new_ev, n_constructed = factor_outcomes(new_ev, calendar, loader)
        t_new, panel_new, frame_new = expanded_estimate(ctx, calendar, frame_old, new_ev)
        inv_label = {label: hid for label, hid in labels.items()}
        new_ev["household"] = new_ev["household_id"].map({v: k for k, v in inv_label.items()})
        episodes[name] = new_ev
        eligible = [lab for lab, hid in labels.items() if hid in set(panel_new["household_id"])]
        note = ("identities by a parenthesized ticker or an exact name" if name == "exact" else "near-exact names added (not the plan's rule; recorded in DEVIATIONS.md)")
        est.append(tau_row(f"expanded_cohort_{name}", t_new, f"{LABEL}; {note}; {len(new_ev)} recovered episodes, {n_constructed} outcomes from the released constructor; "
                           f"households newly eligible {len(eligible)}; winsorized at the committed cutoffs; no placebo p (the stored placebo datasets hold no recovered events)"))
        for lab, hid in labels.items():
            fx = household_facts(frame_new, hid)
            est.append(rs.row("T", "history_recovery", f"eligibility_{lab}", name, float(lab in eligible), "flag", n_events=fx["n_A"],
                              notes=f"period A purchases with an outcome {fx['n_A']} over {fx['quarters_A']} quarters and {fx['securities_A']} securities; period B {fx['n_B']} over {fx['quarters_B']} quarters"))
        pa = new_ev.loc[(new_ev["direction"] == "purchase") & (new_ev["period"] == "A") & np.isfinite(new_ev["y_h20"].astype(float))]
        est.append(rs.row("T", "history_recovery", "recovered_purchase_mean_A", name, float(pa["y_h20"].mean()) if len(pa) else np.nan, "bp", n_events=int(len(pa)),
                          n_households=int(pa["household_id"].nunique()), notes="raw factor outcome of the recovered first-period purchases, unweighted"))
        new_ev.drop(columns=["household_id"]).to_parquet(HIST / f"recovered_episodes_{name}.parquet", index=False)
        log(f"{name}: lines {len(tf)}, episodes {len(new_ev)}, with outcome {int(np.isfinite(new_ev['y_h20']).sum())}, constructed {n_constructed}, eligible {eligible}, "
            f"tau2 {t_new['tau2_hat']:.1f} (households {int(t_new['n_households'])})")
    inv = inventory(filings, res, rec, episodes)
    res.drop(columns=["page_image_path"], errors="ignore").to_parquet(HIST / "recovered_transactions.parquet", index=False)
    th.guard(inv, OUT_INV)
    inv.to_csv(OUT_INV, index=False, lineterminator="\n")
    est_df = cm.write_rows(est, OUT_EST)
    th.guard(est_df, OUT_EST)
    cm.write_manifest("history_recovery", [cm.file_entry(ROWS, rows=len(rows)), cm.file_entry(FILINGS, rows=len(filings)),
                                           cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "matched_transactions.csv"), cm.file_entry(vs.dropbox_project_root() / vs.VENDOR_SNAPSHOT)],
                      {"seconds": round(time.time() - t0, 1), "name_dictionary_size": len(dictionary), "ambiguous_names_dropped": len(ambiguous)})
    log(f"done in {time.time() - t0:.0f} s")
    pd.set_option("display.width", 250)
    print(inv.to_string(index=False))
    print(est_df[["variant", "statistic", "value", "se_block", "n_events", "n_households"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
