"""Cross-source duplicate audit: vendor-only transactions that reproduce an official one.

The canonical pipeline removes vendor rows only at the filing level (linked filing
with parsed transactions) and its exact-duplicate key compares bracket labels that
differ across sources, so record-level cross-source duplicates can survive. This
module harmonizes the bracket vocabularies and counts such survivors; it never
touches outcomes.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

_OFFICIAL_RANGE = re.compile(r"^\$?([\d,]+)\s*-\s*\$?([\d,]+)$")
_OFFICIAL_OVER = re.compile(r"^over\s+\$?([\d,]+)$", re.IGNORECASE)
_VENDOR_RANGE = re.compile(r"^([\d.]+)([KM])\s*[–-]\s*([\d.]+)([KM])$")
_VENDOR_UNDER = re.compile(r"^<\s*([\d.]+)([KM])$")
_MULT = {"K": 1_000, "M": 1_000_000}


def _num(text: str, unit: str | None = None) -> float:
    value = float(text.replace(",", ""))
    return value * _MULT[unit] if unit else value


def harmonize_bracket(label) -> tuple[float, float]:
    """Map an official or vendor bracket label to (lower, upper) dollars.

    Official ``$1,001 - $15,000`` -> (1001, 15000); ``Over $1,000,000`` -> (1000001, inf).
    Vendor ``1K–15K`` -> (1001, 15000) (vendor lower bounds are exclusive round
    numbers, as in the snapshot's ``size_lower``); ``< 1K`` -> (1, 1000).
    ``unknown``, empty or unparseable -> (nan, nan).
    """
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return (math.nan, math.nan)
    text = str(label).strip()
    if not text or text.lower() == "unknown":
        return (math.nan, math.nan)
    m = _OFFICIAL_RANGE.match(text)
    if m:
        return (_num(m.group(1)), _num(m.group(2)))
    m = _OFFICIAL_OVER.match(text)
    if m:
        return (_num(m.group(1)) + 1, math.inf)
    m = _VENDOR_RANGE.match(text)
    if m:
        return (_num(m.group(1), m.group(2)) + 1, _num(m.group(3), m.group(4)))
    m = _VENDOR_UNDER.match(text)
    if m:
        return (1.0, _num(m.group(1), m.group(2)))
    return (math.nan, math.nan)


def harmonize_brackets(labels: pd.Series) -> pd.DataFrame:
    """Vectorized :func:`harmonize_bracket` over distinct labels -> columns lower, upper."""
    uniq = {lab: harmonize_bracket(lab) for lab in labels.dropna().unique()}
    lower = labels.map(lambda x: uniq.get(x, (math.nan, math.nan))[0] if pd.notna(x) else math.nan)
    upper = labels.map(lambda x: uniq.get(x, (math.nan, math.nan))[1] if pd.notna(x) else math.nan)
    return pd.DataFrame({"bracket_lower": lower.astype(float), "bracket_upper": upper.astype(float)})


AUDIT_KEY = ["filer_uid", "permno", "direction"]
VARIANTS = {
    "primary": {"date": "analysis", "shift": (0,), "bracket": True},
    "raw_vendor_date": {"date": "raw", "shift": (0,), "bracket": True},
    "pm1_day": {"date": "analysis", "shift": (-1, 0, 1), "bracket": True},
    "no_bracket": {"date": "analysis", "shift": (0,), "bracket": False},
    "pm1_day_no_bracket": {"date": "analysis", "shift": (-1, 0, 1), "bracket": False},
}


def prepare_audit_frame(matched: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Harmonize brackets, recompute lags and exclude implausible dates; returns (frame, counts)."""
    f = matched.loc[matched["direction"].isin(["purchase", "sale"]) & matched["permno"].notna()].copy()
    f["permno"] = f["permno"].astype("int64")
    for col in ("trade_date_analysis", "trade_date_official", "trade_date_capitol_trades_raw", "analysis_public_date"):
        f[col] = pd.to_datetime(f[col], errors="coerce")
    f[["bracket_lower", "bracket_upper"]] = harmonize_brackets(f["amount_bracket"])
    f["lag_recomputed"] = (f["analysis_public_date"] - f["trade_date_analysis"]).dt.days
    last_public = f["analysis_public_date"].max()
    neg_lag = f["lag_recomputed"] < 0
    future_official = f["trade_date_official"] > last_public
    counts = {
        "rows_in": int(len(f)),
        "stored_reporting_lag_negative": int((f["reporting_lag_days"] < 0).sum()),
        "recomputed_lag_negative": int(neg_lag.sum()),
        "official_trade_date_after_last_public_date": int(future_official.sum()),
        "bracket_unparsed_nonunknown": int(
            (f["bracket_lower"].isna() & ~f["amount_bracket"].fillna("unknown").str.lower().eq("unknown")).sum()
        ),
        "bracket_unknown": int(f["amount_bracket"].fillna("unknown").str.lower().eq("unknown").sum()),
    }
    excluded = neg_lag | future_official
    counts["rows_excluded_implausible_dates"] = int(excluded.sum())
    f = f.loc[~excluded].copy()
    counts["rows_audited"] = int(len(f))
    counts["vendor_rows"] = int(f["record_source"].eq("capitol_trades").sum())
    counts["official_rows"] = int(f["record_source"].eq("official").sum())
    return f, counts


def _key_frame(rows: pd.DataFrame, date: pd.Series, bracket: bool) -> pd.DataFrame:
    key = rows[AUDIT_KEY].copy()
    key["trade_date"] = pd.to_datetime(date).to_numpy()
    if bracket:
        key["bracket_lower"] = rows["bracket_lower"].fillna(-1.0).to_numpy()
        key["bracket_upper"] = rows["bracket_upper"].fillna(-1.0).to_numpy()
    return key


def match_vendor_to_official(audit: pd.DataFrame, variant: str = "primary") -> pd.Series:
    """Boolean per vendor row: an official row shares the variant's key.

    Vendor date: ``trade_date_analysis`` (pipeline's CT+1 correction) or the raw
    CT date; ``shift`` adds calendar days to the vendor date before lookup.
    Rows with an unparseable/unknown bracket never match when ``bracket`` is on.
    """
    spec = VARIANTS[variant]
    vendor = audit[audit["record_source"] == "capitol_trades"]
    official = audit[audit["record_source"] == "official"]
    off_key = _key_frame(official, official["trade_date_analysis"], spec["bracket"])
    cols = list(off_key.columns)
    off_index = pd.MultiIndex.from_frame(off_key.drop_duplicates())
    base_date = vendor["trade_date_analysis" if spec["date"] == "analysis" else "trade_date_capitol_trades_raw"]
    hit = pd.Series(False, index=vendor.index)
    for shift in spec["shift"]:
        key = _key_frame(vendor, base_date + pd.Timedelta(days=shift), spec["bracket"])
        idx = pd.MultiIndex.from_frame(key[cols])
        hit |= pd.Series(idx.isin(off_index), index=vendor.index)
    if spec["bracket"]:
        hit &= vendor["bracket_lower"].notna()
    hit &= base_date.notna()
    return hit.reindex(audit.index, fill_value=False)


def audit_rows(audit: pd.DataFrame) -> pd.DataFrame:
    """Restricted row-level table: every vendor row with one match flag per variant."""
    vendor = audit[audit["record_source"] == "capitol_trades"]
    out = vendor[
        ["transaction_uid", "filer_uid", "document_uid", "permno", "direction", "trade_date_analysis",
         "trade_date_capitol_trades_raw", "amount_bracket", "bracket_lower", "bracket_upper", "chamber"]
    ].copy()
    for variant in VARIANTS:
        out[f"dup_{variant}"] = match_vendor_to_official(audit, variant).loc[vendor.index].to_numpy()
    return out.reset_index(drop=True)


def explode_signal_parents(signals: pd.DataFrame) -> pd.DataFrame:
    """(transaction_uid, signal_uid) pairs from ``parent_transaction_uids`` (';'-joined)."""
    pairs = signals[["signal_uid", "parent_transaction_uids"]].copy()
    pairs["transaction_uid"] = pairs["parent_transaction_uids"].astype(str).str.split(";")
    return pairs.explode("transaction_uid")[["transaction_uid", "signal_uid"]].reset_index(drop=True)


def affected_signals_by_year(
    rows: pd.DataFrame, pairs: pd.DataFrame, events_p: pd.DataFrame, variant: str
) -> pd.DataFrame:
    """Public-clock events (all directions) containing >= 1 duplicate vendor line, by anchor year."""
    dup_uids = set(rows.loc[rows[f"dup_{variant}"], "transaction_uid"])
    hit_signals = set(pairs.loc[pairs["transaction_uid"].isin(dup_uids), "signal_uid"])
    ev = events_p[["signal_uid", "anchor_year", "direction"]].copy()
    ev["affected"] = ev["signal_uid"].isin(hit_signals)
    by_year = ev.groupby("anchor_year").agg(n_signals=("signal_uid", "size"), n_affected=("affected", "sum"))
    by_year["share_affected"] = by_year["n_affected"] / by_year["n_signals"]
    by_year = by_year.reset_index()
    total = pd.DataFrame(
        [{"anchor_year": "all", "n_signals": int(len(ev)), "n_affected": int(ev["affected"].sum()),
          "share_affected": float(ev["affected"].mean())}]
    )
    out = pd.concat([by_year, total], ignore_index=True)
    out.insert(0, "variant", variant)
    return out


def audit_summary(rows: pd.DataFrame, pairs: pd.DataFrame, events_p: pd.DataFrame) -> pd.DataFrame:
    """One row per variant: matched vendor rows, affected households, affected P events."""
    out = []
    for variant in VARIANTS:
        hit = rows[rows[f"dup_{variant}"]]
        dup_uids = set(hit["transaction_uid"])
        hit_signals = set(pairs.loc[pairs["transaction_uid"].isin(dup_uids), "signal_uid"])
        ev_hit = events_p["signal_uid"].isin(hit_signals)
        purch = events_p["direction"].eq("purchase")
        out.append(
            {
                "variant": variant,
                "vendor_rows": int(len(rows)),
                "vendor_rows_matched": int(len(hit)),
                "share_vendor_rows_matched": float(len(hit) / len(rows)) if len(rows) else np.nan,
                "households_affected": int(hit["filer_uid"].nunique()),
                "p_events_total": int(len(events_p)),
                "p_events_affected": int(ev_hit.sum()),
                "share_p_events_affected": float(ev_hit.mean()),
                "p_purchase_events_affected": int((ev_hit & purch).sum()),
                "share_p_purchase_events_affected": float(ev_hit[purch].mean()),
            }
        )
    return pd.DataFrame(out)


def flagged_line_counts(events_t: pd.DataFrame, dup_uids: set) -> pd.Series:
    """Per T episode: how many of its ``transaction_uids`` are in ``dup_uids``."""
    parts = events_t["transaction_uids"].astype(str).str.split(";")
    return parts.map(lambda lst: sum(u in dup_uids for u in lst)).astype("int64")


def t_episode_exposure(rows: pd.DataFrame, events_t: pd.DataFrame, variant: str = "primary") -> dict:
    """T-clock exposure to flagged vendor lines: the episode key equals the audit key minus
    the bracket, so a flagged line collapses INTO the episode of its official twin. Reports
    episodes containing a flagged line, episodes that would vanish if flagged lines were
    dropped (all lines flagged) and episodes whose ``n_lines`` alone would change."""
    dup_uids = set(rows.loc[rows[f"dup_{variant}"], "transaction_uid"])
    n_flagged = flagged_line_counts(events_t, dup_uids)
    n_lines = events_t["n_lines"].astype("int64")
    with_line = n_flagged > 0
    removed = with_line & (n_flagged >= n_lines)
    return {
        "variant": variant,
        "t_events_total": int(len(events_t)),
        "t_events_with_flagged_line": int(with_line.sum()),
        "share_t_events_with_flagged_line": float(with_line.mean()) if len(events_t) else np.nan,
        "t_lines_flagged": int(n_flagged.sum()),
        "t_events_removed_if_flagged_lines_dropped": int(removed.sum()),
        "t_events_n_lines_changed_only": int((with_line & ~removed).sum()),
    }


PAIR_DIFF_BINS = [-np.inf, -1.5, -0.5, 0.5, 1.5, 7.5, np.inf]
PAIR_DIFF_LABELS = ["< -1", "-1", "0", "+1", "+2..+7", "> +7"]
OWNER_MAP = {"dependent_child": "child"}


def pairing_diagnostic(audit: pd.DataFrame, rows: pd.DataFrame, variant: str = "primary") -> pd.DataFrame:
    """Count-only table over vendor-official pairs that share the variant's key.

    Metrics: pair counts; histogram of (vendor public-date proxy - official public
    date) in calendar days; owner-label agreement after mapping ``dependent_child``
    to ``child``; the owner-label pair distribution. Columns: metric, bin, n_pairs, share.
    """
    spec = VARIANTS[variant]
    if spec["shift"] != (0,) or spec["date"] != "analysis":
        raise ValueError("pairing_diagnostic supports unshifted analysis-date variants only")
    key_cols = AUDIT_KEY + ["trade_date"] + (["bracket_lower", "bracket_upper"] if spec["bracket"] else [])
    official = audit[audit["record_source"] == "official"]
    vendor = audit.loc[audit["transaction_uid"].isin(rows.loc[rows[f"dup_{variant}"], "transaction_uid"])]

    def _with_key(frame: pd.DataFrame) -> pd.DataFrame:
        key = _key_frame(frame, frame["trade_date_analysis"], spec["bracket"])
        key["public_date"] = frame["analysis_public_date"].to_numpy()
        key["owner"] = frame["owner"].astype(str).str.strip().str.lower().replace(OWNER_MAP).to_numpy()
        key["uid"] = frame["transaction_uid"].to_numpy()
        return key

    pairs = _with_key(vendor).merge(_with_key(official), on=key_cols, suffixes=("_v", "_o"))
    n = len(pairs)
    out = [
        {"metric": "pairs", "bin": "total", "n_pairs": n},
        {"metric": "pairs", "bin": "distinct_vendor_lines", "n_pairs": int(pairs["uid_v"].nunique())},
        {"metric": "pairs", "bin": "distinct_official_lines", "n_pairs": int(pairs["uid_o"].nunique())},
    ]
    diff = (pairs["public_date_v"] - pairs["public_date_o"]).dt.days
    hist = pd.cut(diff, PAIR_DIFF_BINS, labels=PAIR_DIFF_LABELS).value_counts(sort=False)
    out += [{"metric": "public_date_diff_days", "bin": str(b), "n_pairs": int(c)} for b, c in hist.items()]
    out.append({"metric": "public_date_diff_days", "bin": "in [-1, +7]", "n_pairs": int(diff.between(-1, 7).sum())})
    agree = pairs["owner_v"] == pairs["owner_o"]
    out += [{"metric": "owner_agreement", "bin": "agree", "n_pairs": int(agree.sum())},
            {"metric": "owner_agreement", "bin": "disagree", "n_pairs": int((~agree).sum())}]
    combo = (pairs["owner_v"] + " vs " + pairs["owner_o"]).value_counts()
    out += [{"metric": "owner_pair_vendor_vs_official", "bin": str(b), "n_pairs": int(c)} for b, c in combo.items()]
    table = pd.DataFrame(out)
    table.insert(0, "variant", variant)
    table["share"] = table["n_pairs"] / n if n else np.nan
    return table
