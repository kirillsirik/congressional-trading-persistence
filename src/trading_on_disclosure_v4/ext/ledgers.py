"""Coverage and attrition ledgers at the report, record, signal and household levels (extension plan
2026-10-19, section 2), with the unresolved security matches split into intended exclusions, unresolved
eligible securities, date-invalid tickers and securities not on CRSP. Counts only.

Outputs under data/v4/ext/: ledger_reports.csv, ledger_records.csv, ledger_tier_d_classification.csv,
ledger_signals.csv, ledger_households.csv. The unscreened CRSP security ranges are cached under
data/v4/restricted/ext/ledgers/ (built once from the raw daily files by streaming).
"""

from __future__ import annotations

import argparse
import gzip
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from congressional_trading.official.hybrid import identity_key
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vos
from trading_on_disclosure_v4.data.eligibility import eligibility_flags
from trading_on_disclosure_v4.ext import common as cm
from trading_on_disclosure_v4.outcomes import event_filters as ef

OUT = cm.OUT_DIR
RESTRICTED = cm.RESTRICTED_EXT / "ledgers"
RANGES = RESTRICTED / "crsp_security_ranges_2012plus.parquet"
LOG = cm.LOG_DIR / "ledgers.log"
RAW_CRSP = ("crsp_v2_daily_1980_present.csv.gz", "crsp_v2_daily_202604_202606.csv.gz", "crsp_v2_daily_202607_202607.csv.gz")
RANGE_KEY = ["PERMNO", "Ticker", "TradingSymbol", "PrimaryExch", "SecurityType", "ShareType"]
COMMON_EXCH = ("N", "A", "Q")
CATEGORIES = ("retained", "intended_exclusion", "unresolved_eligible_security", "duplicate", "aggregation", "incomplete_outcome", "unparsed_source")
#: fixed name patterns for unresolved lines without a ticker (plan section 2; one list, applied in order after the
#: House asset-type tag, e.g. "[GS]", when the name carries one)
NAME_PATTERNS = (
    ("bond_note_treasury_muni", r"\b(bond|bonds|note|notes|treasury|treasuries|t-bill|bills?|muni|municipal|debenture|cusip|matur|due 20\d\d|\d+(\.\d+)?%|"
                                r"state of|cnty|county|city of|twp|sch dist|school district|school districts|isd|usd|cusd|tpk|turnpike|commn|hsg|housing|wtr|pwr|swr|util|"
                                r"rfdg|impt|unltd|g/o|b/e|ctfs|cops|build america|taxable|ser [a-z]|rev|go|auth|agy|dist|fin agy|fin auth|dev auth|edl|hlth|hosp|univ|"
                                r"pub|pwr auth|redev|wks|wtrwks|swr|transn|trans auth|arpt|port auth|toll|brdg|hwy|capital appreciation|zero cpn|strips)\b"),
    ("fund_etf_trust_index", r"\b(fund|funds|etf|etfs|etn|trust|index|ishares|vanguard|spdr|portfolio|mutual|money market|reit|invesco|fidelity|schwab|proshares|direxion|"
                             r"wisdomtree|vaneck|first trust|global x|sector|ucits|unit investment|closed-end|inst|instl|admiral|investor class|class [a-z] shares)\b"),
    ("preferred_or_depositary", r"\b(pfd|preferred|preference|depositary|depository|cum|cumulative|perpetual|series [a-z]|ser [a-z] pfd)\b"),
    ("option_warrant", r"\b(call|calls|put|puts|option|options|warrant|warrants|strike|expir|exp \d|leaps)\b"),
    ("crypto", r"\b(bitcoin|ethereum|crypto|cryptocurrency|btc|eth|coin|token|solana|ripple|xrp|litecoin|dogecoin|usdc|stablecoin)\b"),
    ("partnership_private", r"\b(llc|l\.p\.|lp|partnership|partners|private|holdings|ventures|capital|fund i+|series [a-z]|limited|gp|sfr|investment club|real estate|rental|farm|acres)\b"),
    ("cash_annuity_insurance", r"\b(annuity|annuities|insurance|certificate of deposit|cd|cds|cash|savings|deposit|checking|money mkt|sweep|brokerage account|ira|401k|403b|529)\b"),
    ("commodity_currency_future", r"\b(gold|silver|platinum|palladium|bullion|crude|soybean|corn|wheat|cattle|futures|future|forex|currency|fx|cme|comex|nymex)\b"),
)
#: House asset-type tags in square brackets (official form codes); [ST] (stocks incl. ADRs) is the only tag that leaves a name eligible
ASSET_TAG = re.compile(r"\[([A-Za-z]{2})\]")
ASSET_TAG_CLASS = {"ST": "stock_tag", "PS": "partnership_private", "RS": "preferred_or_depositary", "SA": "preferred_or_depositary",
                   "AB": "bond_note_treasury_muni", "CS": "bond_note_treasury_muni", "GS": "bond_note_treasury_muni", "DO": "bond_note_treasury_muni", "ET": "bond_note_treasury_muni",
                   "EF": "fund_etf_trust_index", "MF": "fund_etf_trust_index", "RE": "fund_etf_trust_index", "TR": "fund_etf_trust_index", "HE": "fund_etf_trust_index",
                   "HN": "partnership_private", "IC": "partnership_private", "OI": "partnership_private", "OL": "partnership_private", "DS": "partnership_private",
                   "BK": "cash_annuity_insurance", "BA": "cash_annuity_insurance", "IH": "cash_annuity_insurance", "IR": "cash_annuity_insurance", "MA": "cash_annuity_insurance",
                   "FN": "cash_annuity_insurance", "VA": "cash_annuity_insurance", "VI": "cash_annuity_insurance", "WU": "cash_annuity_insurance", "PE": "cash_annuity_insurance",
                   "DB": "cash_annuity_insurance", "EQ": "cash_annuity_insurance", "OP": "option_warrant", "FU": "commodity_currency_future", "FE": "commodity_currency_future",
                   "PM": "commodity_currency_future", "CT": "crypto", "CO": "partnership_private", "FA": "partnership_private", "RP": "partnership_private",
                   "MO": "partnership_private", "IP": "partnership_private", "OT": "other_tag"}
PAREN_TICKER = re.compile(r"\(([A-Za-z]{1,5}(?:[.\-/][A-Za-z]{1,2})?)\)(?!.*\([A-Za-z]{1,5}(?:[.\-/][A-Za-z]{1,2})?\))")
PARSED_STATES, IMAGE_STATES, OCR_STATES, FAILED_STATES = vos.PARSED_STATES, vos.IMAGE_ONLY_STATES, vos.OCR_REVIEW_STATES, vos.FAILED_STATES


def log(msg: str) -> None:
    cm.log_line(LOG, msg)


def dropbox_root() -> Path:
    return vos.dropbox_project_root()


# ---------------------------------------------------------------- CRSP security ranges (unscreened)

def build_security_ranges(path: Path = RANGES, since: str = "2012-01-01", chunk_rows: int = 4_000_000) -> pd.DataFrame:
    """Per (PERMNO, ticker, trading symbol, exchange, security type, share type): first and last session on or
    after `since`, streamed from the raw unscreened CRSP daily files; cached as parquet."""
    if path.exists():
        return pd.read_parquet(path)
    raw_dir = dropbox_root() / "data" / "raw" / "crsp"
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for name in RAW_CRSP:
        f = raw_dir / name
        n_rows = 0
        with gzip.open(f, "rt") as fh:
            for chunk in pd.read_csv(fh, usecols=["PERMNO", "DlyCalDt", "Ticker", "TradingSymbol", "PrimaryExch", "SecurityType", "ShareType"],
                                     dtype=str, chunksize=chunk_rows, keep_default_na=False):
                n_rows += len(chunk)
                chunk = chunk[chunk["DlyCalDt"] >= since]
                if chunk.empty:
                    continue
                g = chunk.groupby(RANGE_KEY, sort=False)["DlyCalDt"].agg(["min", "max", "size"]).reset_index()
                parts.append(g)
                log(f"{name}: {n_rows:,} rows read, {len(parts)} partial tables, {time.time() - t0:.0f} s")
    allp = pd.concat(parts, ignore_index=True)
    out = allp.groupby(RANGE_KEY, sort=True).agg(first_date=("min", "min"), last_date=("max", "max"), n_sessions=("size", "sum")).reset_index()
    out["first_date"] = pd.to_datetime(out["first_date"])
    out["last_date"] = pd.to_datetime(out["last_date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log(f"security ranges: {len(out):,} rows for {out['PERMNO'].nunique():,} PERMNOs in {time.time() - t0:.0f} s")
    return out


def is_common_stock(r: pd.DataFrame) -> pd.Series:
    """The design's screen (scripts/v2/22_build_market_panel_2014plus.py): SecurityType EQTY, ShareType NS, exchange N/A/Q."""
    return (r["SecurityType"] == "EQTY") & (r["ShareType"] == "NS") & r["PrimaryExch"].isin(COMMON_EXCH)


def subclass(r: pd.Series) -> str:
    if r["ShareType"] == "AD":
        return "adr"
    if r["SecurityType"] == "FUND":
        return "fund_or_etf"
    if r["ShareType"] in ("SB", "UG"):
        return "other_share_type"
    if r["SecurityType"] == "EQTY" and r["ShareType"] == "NS" and r["PrimaryExch"] not in COMMON_EXCH:
        return "other_exchange"
    return "no_crsp_type"


def normalize_ticker(t: str) -> tuple[str, str]:
    """(as written upper-cased, with dots / hyphens / slashes / spaces removed)."""
    raw = str(t).strip().upper()
    return raw, re.sub(r"[.\-/ ]", "", raw)


def classify_ticker_rows(rows: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """One classification per unresolved line with a ticker: unresolved_eligible_security (a common stock whose name
    range covers the transaction date), intended_exclusion with a sub-class, date_invalid (gap in days), not_on_crsp."""
    r = ranges.copy()
    r["common"] = is_common_stock(r)
    r["sub"] = r.apply(subclass, axis=1)
    by_sym: dict[str, pd.DataFrame] = {}
    for col in ("Ticker", "TradingSymbol"):
        for sym, g in r.groupby(col):
            if sym:
                by_sym.setdefault(sym, []).append(g)
    by_sym = {k: pd.concat(v).drop_duplicates(RANGE_KEY) for k, v in by_sym.items()}
    out = []
    for row in rows.itertuples(index=False):
        raw, norm = normalize_ticker(row.ticker)
        cand = by_sym.get(raw)
        if cand is None and norm != raw:
            cand = by_sym.get(norm)
        d = pd.Timestamp(row.transaction_date_reported)
        rec = {"transaction_uid": row.transaction_uid, "ticker_as_written": raw, "ticker_normalized": norm, "category": "", "subclass": "", "gap_days": np.nan,
               "n_candidates": 0 if cand is None else int(len(cand))}
        if cand is None or pd.isna(d):
            rec["category"], rec["subclass"] = ("not_on_crsp", "ticker_absent_2012plus") if cand is None else ("not_on_crsp", "no_transaction_date")
        else:
            covers = cand[(cand["first_date"] <= d) & (d <= cand["last_date"])]
            if len(covers) and covers["common"].any():
                rec["category"], rec["subclass"] = "unresolved_eligible_security", "common_stock_date_valid"
            elif len(covers):
                rec["category"], rec["subclass"] = "intended_exclusion", subclass(covers.iloc[0]) if len(covers) == 1 else "|".join(sorted(set(covers["sub"])))
            else:
                gap = np.minimum((cand["first_date"] - d).dt.days.abs(), (d - cand["last_date"]).dt.days.abs()).min()
                rec["gap_days"] = float(gap)
                rec["category"], rec["subclass"] = ("date_invalid", "common_stock_other_dates" if cand["common"].any() else "non_common_other_dates")
        out.append(rec)
    return pd.DataFrame(out)


def classify_name_rows(names: pd.Series) -> pd.Series:
    """The House asset-type tag when the name carries one, else the first matching fixed pattern per unresolved line
    without a ticker; 'unclassified' otherwise. A '[ST]' tag (stock) leaves the name unclassified unless a pattern fires."""
    raw = names.fillna("").astype(str)
    low = raw.str.lower()
    out = pd.Series("unclassified", index=names.index, dtype=object)
    tag = raw.str.extract(ASSET_TAG, expand=False).str.upper().map(ASSET_TAG_CLASS)
    done = tag.notna() & ~tag.isin(["stock_tag"])
    out[done] = tag[done]
    for label, pat in NAME_PATTERNS:
        hit = low.str.contains(re.sub(r"\((?!\?)", "(?:", pat), regex=True) & ~done
        out[hit] = label
        done |= hit
    stock = (tag == "stock_tag") & ~done
    out[stock] = "stock_tag_no_ticker"
    return out


# ---------------------------------------------------------------- record level

def record_ledger() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Released stages (categorized) plus the split of the unresolved security matches."""
    att = pd.read_csv(cm.REPO_ROOT / "outputs" / "canonical" / "attrition_ledger.csv")
    stage_cat = {"no_usable_public_date": "intended_exclusion", "before_sample_start": "intended_exclusion", "amendment_exact_duplicate": "duplicate",
                 "no_purchase_or_sale_interpretation": "intended_exclusion", "unresolved_security_match": "split_below",
                 "failed_common_stock_screen": "intended_exclusion", "collapsed_repeated_lots": "aggregation"}
    rows = []
    for r in att.itertuples(index=False):
        for col, cat in stage_cat.items():
            n = float(getattr(r, col))
            if n > 0:
                rows.append({"level": "record", "stage": r.stage, "reason": col, "category": cat, "n": int(n), "n_in": int(r.n_in), "n_out": int(r.n_out),
                             "source": "outputs/canonical/attrition_ledger.csv"})
    rows.append({"level": "record", "stage": "collapse_to_signals", "reason": "signals_out", "category": "retained", "n": int(att["n_out"].iloc[-1]),
                 "n_in": int(att["n_in"].iloc[0]), "n_out": int(att["n_out"].iloc[-1]), "source": "outputs/canonical/attrition_ledger.csv"})
    dedup = load.load_deduplicated_transactions()
    sr = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "security_resolution.csv", low_memory=False)
    d = dedup.merge(sr[["transaction_uid", "match_tier", "resolution_basis", "transaction_date_reported"]].rename(columns={"transaction_date_reported": "txn_date_sr"}),
                    on="transaction_uid", how="inner")
    d = d[d["direction"].isin(["purchase", "sale"])]
    un = d[d["match_tier"] == "D"].copy()
    un["transaction_date_reported"] = pd.to_datetime(un["txn_date_sr"], errors="coerce")
    un["ticker"] = un["ticker_text"].astype("string").str.strip()
    has_t = un["ticker"].notna() & (un["ticker"].str.len() > 0)
    paren = un.loc[~has_t, "security_name_text"].astype("string").str.strip().str.extract(PAREN_TICKER, expand=False)
    un.loc[~has_t, "ticker"] = paren
    un["ticker_source"] = np.where(has_t, "ticker_text", np.where(un["ticker"].notna(), "parenthesized_in_name", "none"))
    ranges = build_security_ranges()
    with_t = un[un["ticker"].notna()]
    cls_t = classify_ticker_rows(with_t[["transaction_uid", "ticker", "transaction_date_reported"]], ranges)
    cls_t = cls_t.merge(un[["transaction_uid", "ticker_source", "resolution_basis"]], on="transaction_uid")
    nonunique = cls_t["resolution_basis"] == "nonunique_candidates"
    cls_t.loc[nonunique, ["category", "subclass"]] = ["unresolved_eligible_security", "nonunique_date_valid_candidates"]
    no_t = un[un["ticker"].isna()].copy()
    no_t["subclass"] = classify_name_rows(no_t["security_name_text"])
    no_t["category"] = np.where(no_t["subclass"].isin(["unclassified", "stock_tag_no_ticker", "other_tag"]), "unclassified_no_ticker", "intended_exclusion")
    no_t["ticker_source"] = "none"
    cls = pd.concat([cls_t[["transaction_uid", "category", "subclass", "ticker_source", "gap_days"]],
                     no_t[["transaction_uid", "category", "subclass", "ticker_source"]].assign(gap_days=np.nan)], ignore_index=True)
    cls = cls.merge(un[["transaction_uid", "direction", "record_source"]], on="transaction_uid")
    cls.to_parquet(RESTRICTED / "tier_d_rows.parquet", index=False)
    tier = cls.groupby(["category", "subclass", "ticker_source", "direction", "record_source"]).size().rename("n").reset_index()
    tier["gap_days_median"] = cls.groupby(["category", "subclass", "ticker_source", "direction", "record_source"])["gap_days"].median().to_numpy()
    tier["level"] = "record"
    tier["stage"] = "security_matching"
    # the eligible-missing rate per direction and source: unresolved eligible / (resolved + unresolved eligible)
    resolved = d[d["match_tier"] != "D"].groupby(["direction", "record_source"]).size()
    unres = cls[cls["category"] == "unresolved_eligible_security"].groupby(["direction", "record_source"]).size()
    rate_rows = []
    for key in sorted(set(resolved.index) | set(unres.index)):
        n_res, n_un = int(resolved.get(key, 0)), int(unres.get(key, 0))
        rate_rows.append({"level": "record", "direction": key[0], "record_source": key[1], "n_resolved": n_res, "n_unresolved_eligible": n_un,
                          "missing_eligible_rate": n_un / (n_res + n_un) if n_res + n_un else np.nan,
                          "n_unresolved_total": int(len(cls[(cls["direction"] == key[0]) & (cls["record_source"] == key[1])])),
                          "unresolved_share_of_directional": float(len(cls[(cls["direction"] == key[0]) & (cls["record_source"] == key[1])]) /
                                                               len(d[(d["direction"] == key[0]) & (d["record_source"] == key[1])]))})
    for direction in ("purchase", "sale", "all"):
        m_res = resolved[[k for k in resolved.index if direction in ("all", k[0])]].sum()
        m_un = unres[[k for k in unres.index if direction in ("all", k[0])]].sum() if len(unres) else 0
        tot = len(cls) if direction == "all" else int((cls["direction"] == direction).sum())
        dtot = len(d) if direction == "all" else int((d["direction"] == direction).sum())
        rate_rows.append({"level": "record", "direction": direction, "record_source": "all", "n_resolved": int(m_res), "n_unresolved_eligible": int(m_un),
                          "missing_eligible_rate": m_un / (m_res + m_un) if m_res + m_un else np.nan, "n_unresolved_total": tot,
                          "unresolved_share_of_directional": tot / dtot})
    return pd.DataFrame(rows), tier, rate_rows


# ---------------------------------------------------------------- report level

def report_ledger() -> pd.DataFrame:
    """Official PTR census by chamber x report year x parse state (with pages) and vendor snapshot records by year."""
    official = dropbox_root() / "data" / "official"
    con = sqlite3.connect(f"file:{official / 'state.sqlite'}?mode=ro", uri=True)
    try:
        f = pd.read_sql_query("SELECT f.filing_id, f.chamber, f.report_type, f.report_year, f.state, d.doc_format, d.page_count "
                              "FROM filings f LEFT JOIN filing_documents d ON d.filing_id = f.filing_id AND d.doc_role = 'primary_pdf'", con)
    finally:
        con.close()
    f = f.drop_duplicates("filing_id")
    f["chamber"] = f["chamber"].str.lower()
    st = f["state"]
    f["state_group"] = np.select([st.isin(PARSED_STATES), st.isin(IMAGE_STATES), st.isin(OCR_STATES), st.isin(FAILED_STATES), st == "excluded_report_type"],
                                 ["parsed", "image_only", "ocr_review_gated", "failed", "excluded_report_type"], "other")
    cat = {"parsed": "retained", "image_only": "unparsed_source", "ocr_review_gated": "unparsed_source", "failed": "unparsed_source",
           "excluded_report_type": "intended_exclusion", "other": "unparsed_source"}
    canon = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "canonical_transactions.csv", low_memory=False, usecols=["document_uid", "record_source"])
    with_lines = set(canon.loc[canon["record_source"] == "official", "document_uid"])
    f["has_canonical_line"] = f["filing_id"].isin(with_lines)
    g = f[f["report_type"] == "ptr"].groupby(["chamber", "report_year", "state_group"]).agg(
        n_filings=("filing_id", "size"), n_scanned_pdf=("doc_format", lambda s: int((s == "scanned_pdf").sum())),
        n_text_pdf=("doc_format", lambda s: int((s == "text_pdf").sum())), pages=("page_count", "sum"),
        n_with_canonical_lines=("has_canonical_line", "sum")).reset_index()
    g["category"] = g["state_group"].map(cat)
    g["level"] = "report"
    g["source"] = "official census (state.sqlite)"
    # vendor snapshot
    snap = pd.read_parquet(dropbox_root() / vos.VENDOR_SNAPSHOT, columns=["published_date", "stable_record_key"])
    yr = pd.to_datetime(snap["published_date"], errors="coerce").dt.year
    v = snap.assign(year=yr).groupby("year").agg(n_rows=("stable_record_key", "size"), n_distinct_records=("stable_record_key", "nunique")).reset_index()
    v = v.rename(columns={"year": "report_year"})
    v["chamber"], v["state_group"], v["category"], v["level"], v["source"] = "vendor", "snapshot_rows", "retained_before_assembly", "report", "vendor snapshot 20260721_221640"
    diag = pd.read_json(cm.REPO_ROOT / "paper" / "data" / "source_construction_diagnostics.json", typ="series")
    fates = [("vendor_duplicates_of_parsed_official", "duplicate"), ("repeat_scrape_duplicates", "duplicate"),
             ("scan_linked_vendor_rows_retained", "retained"), ("vendor_only_rows_retained", "retained")]
    fr = pd.DataFrame([{"level": "report", "chamber": "vendor", "report_year": np.nan, "state_group": k, "category": c, "n_filings": int(diag[k]),
                        "source": "paper/data/source_construction_diagnostics.json"} for k, c in fates])
    return pd.concat([g, v, fr], ignore_index=True)


# ---------------------------------------------------------------- signal level

def signal_ledger() -> pd.DataFrame:
    """Public clock from the signals frame; transaction clock from the timed lines, the episodes and the cache reasons."""
    rows = []
    s = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv", low_memory=False,
                    usecols=["signal_uid", "direction", "mixed_direction", "follower_entry_date", "ar_E_h20_reason", "car_ff5mom_h20_reason", "record_source"])
    mixed = s["mixed_direction"].astype(bool)
    rows.append(dict(level="signal", clock="P", stage="signals", reason="all", category="input", direction="all", period="all", n=int(len(s))))
    for dirn, g in s[mixed].groupby("direction"):
        rows.append(dict(level="signal", clock="P", stage="mixed_direction", reason="mixed_direction_signal", category="intended_exclusion", direction=dirn, period="all", n=int(len(g))))
    nm = s[~mixed]
    no_entry = nm["follower_entry_date"].isna()
    for dirn, g in nm[no_entry].groupby("direction"):
        rows.append(dict(level="signal", clock="P", stage="entry_session", reason="no_entry_session_after_public_date", category="incomplete_outcome", direction=dirn, period="all", n=int(len(g))))
    ev = pd.read_parquet(ef.EVENTS_DIR / "events_P.parquet", columns=["event_id", "signal_uid", "direction", "period"])
    flags = pd.read_parquet(ef.FLAGS_PATH, columns=["event_id", "clock", "dup_primary_all_lines"])
    dup_ids = set(flags.loc[(flags["clock"] == "P") & flags["dup_primary_all_lines"], "event_id"])
    ev["dup"] = ev["event_id"].isin(dup_ids)
    for (dirn, per), g in ev[ev["dup"]].groupby(["direction", "period"]):
        rows.append(dict(level="signal", clock="P", stage="duplicates", reason="vendor_copy_of_official_all_lines", category="duplicate", direction=dirn, period=per, n=int(len(g))))
    kept = ev[~ev["dup"]].merge(nm[["signal_uid", "ar_E_h20_reason", "car_ff5mom_h20_reason"]], on="signal_uid", how="left")
    for (dirn, per, reason), g in kept.groupby(["direction", "period", "ar_E_h20_reason"]):
        rows.append(dict(level="signal", clock="P", stage="outcome_h20_market_adjusted", reason=reason, category="retained" if reason == "ok" else "incomplete_outcome",
                         direction=dirn, period=per, n=int(len(g))))
    for (dirn, per, reason), g in kept.groupby(["direction", "period", "car_ff5mom_h20_reason"]):
        rows.append(dict(level="signal", clock="P", stage="outcome_h20_ff5mom", reason=reason, category="retained" if reason == "ok" else "incomplete_outcome",
                         direction=dirn, period=per, n=int(len(g))))
    # transaction clock
    timed = load.load_timed_transactions()
    directional = timed["direction"].isin(["purchase", "sale"])
    rows.append(dict(level="signal", clock="T", stage="timed_lines", reason="all", category="input", direction="all", period="all", n=int(len(timed))))
    rows.append(dict(level="signal", clock="T", stage="timed_lines", reason="non_directional", category="intended_exclusion", direction="other", period="all", n=int((~directional).sum())))
    rows.append(dict(level="signal", clock="T", stage="timed_lines", reason="null_permno", category="intended_exclusion", direction="all", period="all",
                     n=int((directional & timed["permno"].isna()).sum())))
    evt = pd.read_parquet(ef.EVENTS_DIR / "events_T.parquet", columns=["event_id", "direction", "period", "n_lines", "permno", "anchor_date", "complete_window_h20"])
    rows.append(dict(level="signal", clock="T", stage="episodes", reason="lines_collapsed_into_episodes", category="aggregation", direction="all", period="all",
                     n=int(evt["n_lines"].sum() - len(evt))))
    for (dirn, per), g in evt.groupby(["direction", "period"]):
        rows.append(dict(level="signal", clock="T", stage="episodes", reason="period_" + per, category="intended_exclusion" if per == "X" else "retained",
                         direction=dirn, period=per, n=int(len(g))))
    from trading_on_disclosure_v4.outcomes import forward_cache as fc
    from trading_on_disclosure_v4.outcomes import panel as panel_mod
    cal = panel_mod.v2_trading_calendar()
    cache = fc.load_cache(columns=["permno", "anchor_idx", "reason_h20", "reason_mkt_h20"])
    ab = evt[evt["period"].isin(["A", "B"])].copy()
    ab["anchor_idx"] = cal.get_indexer(pd.DatetimeIndex(ab["anchor_date"]))
    ab = ab.merge(cache, on=["permno", "anchor_idx"], how="left")
    ab["reason_h20"] = ab["reason_h20"].fillna("security_not_in_cache")
    ab["reason_mkt_h20"] = ab["reason_mkt_h20"].fillna("security_not_in_cache")
    for col, stage in (("reason_h20", "outcome_h20_ff5mom"), ("reason_mkt_h20", "outcome_h20_market_adjusted")):
        for (dirn, per, reason), g in ab.groupby(["direction", "period", col]):
            rows.append(dict(level="signal", clock="T", stage=stage, reason=reason, category="retained" if reason == "ok" else "incomplete_outcome",
                             direction=dirn, period=per, n=int(len(g))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- household level

def household_ledger() -> pd.DataFrame:
    rows = []
    canon = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "canonical_transactions.csv", low_memory=False, usecols=["filer_uid", "record_source", "direction"])
    matched = load.load_matched_transactions()
    s = pd.read_csv(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv", low_memory=False, usecols=["filer_uid", "mixed_direction", "ar_E_h20", "direction"])
    hh_canon, hh_matched, hh_sig = set(canon["filer_uid"]), set(matched["filer_uid"]), set(s["filer_uid"])
    hh_test = set(s.loc[(~s["mixed_direction"].astype(bool)) & s["ar_E_h20"].notna(), "filer_uid"])
    rows += [dict(level="household", clock="P", stage="canonical_lines", reason="households_with_any_line", category="input", n=len(hh_canon)),
             dict(level="household", clock="P", stage="security_matching", reason="no_line_matched_to_a_permno", category="unresolved_eligible_or_intended", n=len(hh_canon - hh_matched)),
             dict(level="household", clock="P", stage="signals", reason="households_with_a_signal", category="retained", n=len(hh_sig)),
             dict(level="household", clock="P", stage="outcome", reason="no_signal_with_a_20_day_outcome", category="incomplete_outcome", n=len(hh_sig - hh_test)),
             dict(level="household", clock="P", stage="spread_test", reason="households_in_the_spread", category="retained", n=len(hh_test))]
    counts = pd.read_parquet(cm.REPO_ROOT / "data" / "v4" / "restricted" / "data" / "household_period_counts.parquet")
    flags_all = pd.read_parquet(ef.FLAGS_PATH, columns=["event_id", "clock", "dup_primary_all_lines"])
    for clock in ("T", "P"):
        ev = pd.read_parquet(ef.EVENTS_DIR / f"events_{clock}.parquet", columns=["event_id", "household_id", "direction", "period", "record_source"])
        dup_ids = set(flags_all.loc[(flags_all["clock"] == clock) & flags_all["dup_primary_all_lines"], "event_id"])
        for stream, sub in (("default", ev[~ev["event_id"].isin(dup_ids)]), ("official_only", ev[ev["record_source"] == "official"])):
            pur = sub[(sub["direction"] == "purchase") & sub["period"].isin(["A", "B"])]
            per_hh = pur.groupby("household_id")["period"].agg(lambda p: set(p))
            n_a = int(sum("A" in p for p in per_hh))
            n_b = int(sum("B" in p for p in per_hh))
            n_ab = int(sum(("A" in p) and ("B" in p) for p in per_hh))
            rows += [dict(level="household", clock=clock, stage=f"purchases_{stream}", reason="households_with_purchases_2015_2020", category="retained", n=n_a),
                     dict(level="household", clock=clock, stage=f"purchases_{stream}", reason="households_with_purchases_2021_2025", category="retained", n=n_b),
                     dict(level="household", clock=clock, stage=f"purchases_{stream}", reason="households_with_purchases_in_both_periods", category="retained", n=n_ab),
                     dict(level="household", clock=clock, stage=f"purchases_{stream}", reason="households_with_any_purchase_2015_2025", category="retained", n=int(len(per_hh)))]
        fl = eligibility_flags(counts, clock)
        a_half = (fl["n_events_A"] >= 10) & (fl["n_quarters_A"] >= 4) & (fl["n_securities_A"] >= 3)
        rows += [dict(level="household", clock=clock, stage="persistence_rule_default", reason="first_period_half_met", category="retained", n=int(a_half.sum())),
                 dict(level="household", clock=clock, stage="persistence_rule_default", reason="both_halves_met", category="retained", n=int(fl["R1"].sum())),
                 dict(level="household", clock=clock, stage="persistence_rule_default", reason="first_half_met_second_not", category="intended_exclusion", n=int((a_half & ~fl["R1"]).sum()))]
    classes = pd.read_csv(cm.REPO_ROOT / "data" / "v4" / "diagnostics" / "vendor_household_classes.csv")
    for r in classes.itertuples(index=False):
        rows.append(dict(level="household", clock="both", stage="vendor_class", reason=r.household_class, category="retained", n=int(r.n_households)))
    rows.append(dict(level="household", clock="both", stage="vendor_class", reason="official_only_households", category="retained",
                     n=int(len(set(matched.loc[matched["record_source"] == "official", "filer_uid"]) - set(matched.loc[matched["record_source"] == "capitol_trades", "filer_uid"])))))
    # census identities that the parsed record cannot see at all
    official = dropbox_root() / "data" / "official"
    con = sqlite3.connect(f"file:{official / 'state.sqlite'}?mode=ro", uri=True)
    try:
        f = pd.read_sql_query("SELECT filing_id, chamber, filer_name_raw, report_year, state FROM filings WHERE report_type = 'ptr'", con)
    finally:
        con.close()
    f["identity"] = [identity_key(n, c) for n, c in zip(f["filer_name_raw"], f["chamber"], strict=True)]
    f["parsed"] = f["state"].isin(PARSED_STATES)
    g = f.groupby("identity").agg(n=("filing_id", "size"), n_parsed=("parsed", "sum"), first=("report_year", "min"), last=("report_year", "max"))
    unseen = g[g["n_parsed"] == 0]
    rows += [dict(level="household", clock="both", stage="census_identities", reason="filer_identities_with_ptr_filings", category="input", n=int(len(g))),
             dict(level="household", clock="both", stage="census_identities", reason="identities_with_no_parsed_filing", category="unparsed_source", n=int(len(unseen))),
             dict(level="household", clock="both", stage="census_identities", reason="identities_with_no_parsed_filing_and_filings_2015_2020", category="unparsed_source",
                  n=int((f[f["identity"].isin(unseen.index)].groupby("identity")["report_year"].apply(lambda y: y.between(2015, 2020).any())).sum())),
             dict(level="household", clock="both", stage="census_identities", reason="ptr_filings_of_identities_with_no_parsed_filing", category="unparsed_source", n=int(unseen["n"].sum()))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="coverage and attrition ledgers")
    ap.add_argument("--only", default="", help="comma list of levels to rebuild (reports,records,signals,households); default all")
    a = ap.parse_args(argv)
    levels = set(a.only.split(",")) if a.only else {"reports", "records", "signals", "households"}
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "records" in levels:
        stages, tier, rates = record_ledger()
        stages.to_csv(OUT / "ledger_records.csv", index=False, lineterminator="\n")
        tier.to_csv(OUT / "ledger_tier_d_classification.csv", index=False, lineterminator="\n")
        pd.DataFrame(rates).to_csv(OUT / "ledger_missing_eligible_rates.csv", index=False, lineterminator="\n")
        log(f"records: {len(stages)} stage rows, {len(tier)} tier-D rows, {time.time() - t0:.0f} s")
    if "reports" in levels:
        report_ledger().to_csv(OUT / "ledger_reports.csv", index=False, lineterminator="\n")
        log(f"reports done, {time.time() - t0:.0f} s")
    if "signals" in levels:
        signal_ledger().to_csv(OUT / "ledger_signals.csv", index=False, lineterminator="\n")
        log(f"signals done, {time.time() - t0:.0f} s")
    if "households" in levels:
        household_ledger().to_csv(OUT / "ledger_households.csv", index=False, lineterminator="\n")
        log(f"households done, {time.time() - t0:.0f} s")
    root = dropbox_root()
    inputs = [cm.file_entry(cm.REPO_ROOT / "outputs" / "canonical" / "attrition_ledger.csv"), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "security_resolution.csv"),
              cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "deduplicated_transactions.csv"), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "signals.csv"),
              cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "canonical_transactions.csv"), cm.file_entry(cm.REPO_ROOT / "outputs" / "frames" / "timed_transactions.csv"),
              cm.file_entry(ef.EVENTS_DIR / "events_T.parquet"), cm.file_entry(ef.EVENTS_DIR / "events_P.parquet"), cm.file_entry(ef.FLAGS_PATH),
              cm.file_entry(root / "data" / "official" / "state.sqlite"), cm.file_entry(root / vos.VENDOR_SNAPSHOT)]
    inputs += [cm.file_entry(root / "data" / "raw" / "crsp" / n) for n in RAW_CRSP]
    if RANGES.exists():
        inputs.append(cm.file_entry(RANGES))
    cm.write_manifest("ledgers", inputs, {"seconds": round(time.time() - t0, 1)})
    log(f"ledgers written to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
