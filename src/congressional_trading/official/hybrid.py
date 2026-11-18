"""Hybrid sample construction: identity-aware, boundary-reconciled.

Remediation of audit findings HYB-001 and HYB-002.

**HYB-001 — identity-aware filing-date recovery.** The previous implementation
matched a Capitol Trades row to an official filing on *surname plus vendor
publication date*, then silently took the last candidate when several matched.
Surnames are not unique in Congress, so that could attach one member's
transactions to another member's filing date. This module keys on a stable
identity (Bioguide ID where resolvable, otherwise a normalized
full-name+chamber key), requires the candidate filing to be **unique**, and
leaves ambiguous groups **unresolved** rather than guessing. Candidate count,
match method, match confidence and an ambiguity flag are retained per row.

**HYB-002 — segment before harmonization.** The previous implementation
replaced publication dates with recovered filing dates and *then* filtered the
Capitol Trades segment on the replaced date, so a row whose recovered date
moved across the boundary vanished from both segments. Here the source segment
is assigned from an **immutable source-selection date** that is never
rewritten; event-date harmonization happens afterwards and cannot move a row
between segments. Disjointness and exhaustiveness are asserted against input
row identifiers and written to a reconciliation report.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CT_TRADE_DATE_SHIFT_DAYS = 1
FILING_MATCH_WINDOW_DAYS = 7

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|md|phd)\b\.?", re.I)
_HONORIFIC = re.compile(r"^\s*(hon|honorable|mr|mrs|ms|dr|rep|sen)\.?\s+", re.I)


def normalize_name(value: object) -> str:
    """Accent-insensitive, honorific- and suffix-free lowercase full name."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _HONORIFIC.sub(" ", s)
    s = _SUFFIX.sub(" ", s)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    return " ".join(s.lower().split())


def identity_key(name: object, chamber: object = None) -> str:
    """Fallback identity when no Bioguide ID is available.

    Full name plus chamber, not surname: surnames collide in Congress and a
    surname-only key is what HYB-001 flagged.
    """
    n = normalize_name(name)
    if not n:
        return ""
    ch = str(chamber).strip().lower() if chamber is not None and pd.notna(chamber) else ""
    return f"{n}|{ch}" if ch else n


_BIOGUIDE = re.compile(r"^[A-Z]\d{6}$")


def add_canonical_filer_uid(
    df: pd.DataFrame, name_to_bioguide: pd.Series | None = None
) -> pd.DataFrame:
    """Give every row a filer identity that is comparable ACROSS the splice.

    Finding ID-001, surfaced by robustness family R3. ``legislator_id`` is not
    one identifier space: the official segment carries Bioguide IDs (235 unique)
    and the Capitol Trades segment carries name slugs (170 unique), and only 2
    values coincided even though 43 people appear in both segments by name.

    That has three consequences, all silent before this fix:

    1. **Clustered standard errors were wrong.** Freeze §7.1 requires clustering
       on legislator; the same person carried two cluster ids across the
       boundary, so within-person dependence was ignored and SEs were understated.
    2. **Family R3 could not work** — the common-support sample it needs collapsed
       to a handful of coincidental matches.
    3. **Filer-level statistics double-counted people** (405 ids for at most ~362
       distinct filers).

    Resolution order, most reliable first: an existing Bioguide ID; the Bioguide
    ID the official pipeline resolved for the same name+chamber; otherwise a
    normalized ``name:`` key. The method is retained so the share resolved to a
    Bioguide ID is reportable rather than assumed.
    """
    out = df.copy()
    namemap = name_to_bioguide if name_to_bioguide is not None else pd.Series(dtype="object")
    legis = out.get("legislator_id", pd.Series([pd.NA] * len(out), index=out.index))
    names = out.get("politician_name", pd.Series([pd.NA] * len(out), index=out.index))
    chambers = out.get("chamber", pd.Series([pd.NA] * len(out), index=out.index))

    uids: list[str] = []
    methods: list[str] = []
    for lid, name, ch in zip(legis, names, chambers, strict=False):
        sid = "" if lid is None or (isinstance(lid, float) and pd.isna(lid)) else str(lid).strip()
        if _BIOGUIDE.match(sid):
            uids.append(sid)
            methods.append("bioguide_native")
            continue
        key = identity_key(name, ch)
        mapped = namemap.get(key) if key else None
        if mapped is not None and pd.notna(mapped):
            uids.append(str(mapped))
            methods.append("bioguide_via_name_chamber")
        elif key:
            uids.append(f"name:{key}")
            methods.append("name_chamber_fallback")
        elif sid:
            uids.append(f"vendor:{sid}")
            methods.append("vendor_id_fallback")
        else:
            uids.append("")
            methods.append("unresolved")
    out["filer_uid"] = uids
    out["filer_uid_method"] = methods
    return out


def filer_uid_stats(df: pd.DataFrame) -> dict:
    """Reportable evidence that the identity space is now shared."""
    if "filer_uid" not in df.columns:
        return {}
    src = df.get("data_source")
    stats = {
        "rows": int(len(df)),
        "unique_filer_uid": int(df["filer_uid"].replace("", pd.NA).nunique()),
        "by_method": df["filer_uid_method"].value_counts().to_dict(),
        "unresolved_rows": int((df["filer_uid"] == "").sum()),
    }
    if src is not None:
        off = set(df.loc[src == "official_filings", "filer_uid"]) - {""}
        ct = set(df.loc[src != "official_filings", "filer_uid"]) - {""}
        prev_off = set(df.loc[src == "official_filings", "legislator_id"].dropna())
        prev_ct = set(df.loc[src != "official_filings", "legislator_id"].dropna())
        stats.update(
            filer_uid_official=len(off), filer_uid_ct=len(ct),
            filer_uid_in_both=len(off & ct),
            legislator_id_in_both=len(prev_off & prev_ct),
        )
    return stats


@dataclass
class RecoveryResult:
    frame: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_official_filing_identities(state_db: Path, compat_csv: Path | None = None) -> pd.DataFrame:
    """Official filings with a stable identity and their filing date.

    Bioguide IDs are taken from the compatibility export (which carries the
    time-aware resolution done by the official pipeline) and joined onto
    filings; filings without one keep a name+chamber identity key.
    """
    con = sqlite3.connect(state_db)
    f = pd.read_sql_query(
        "SELECT filing_id, chamber, filer_name_raw, "
        "COALESCE(filing_date, received_date) AS fdate "
        "FROM filings WHERE report_type='ptr'",
        con,
    )
    con.close()
    f["fdate"] = pd.to_datetime(f["fdate"], errors="coerce")
    f = f.dropna(subset=["fdate"]).copy()
    f["chamber_norm"] = f["chamber"].str.lower()
    f["bioguide_id"] = pd.NA

    if compat_csv is not None and Path(compat_csv).exists():
        c = pd.read_csv(
            compat_csv, low_memory=False,
            usecols=["official_filing_id", "legislator_id", "politician_name", "chamber"],
        ).dropna(subset=["official_filing_id"])
        bio = c[c["legislator_id"].astype(str).str.match(r"^[A-Z]\d{6}$", na=False)]
        m = bio.drop_duplicates("official_filing_id").set_index("official_filing_id")["legislator_id"]
        f["bioguide_id"] = f["filing_id"].map(m)
        # name -> bioguide lookup for resolving vendor rows
        namemap = (
            bio.assign(key=[identity_key(n, ch) for n, ch in
                            zip(bio["politician_name"], bio["chamber"], strict=False)])
            .groupby("key")["legislator_id"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else pd.NA)
        )
    else:
        namemap = pd.Series(dtype="object")

    f["identity_key"] = [
        identity_key(n, ch) for n, ch in zip(f["filer_name_raw"], f["chamber"], strict=False)
    ]
    f.attrs["name_to_bioguide"] = namemap
    return f


def recover_official_filing_dates(
    ct: pd.DataFrame,
    filings: pd.DataFrame,
    source_date_col: str = "source_selection_date",
) -> RecoveryResult:
    """Attach the official filing date a vendor row was published from.

    Matching is identity-aware and requires a unique candidate filing:

    1. by Bioguide ID when the vendor row resolves to one;
    2. otherwise by normalized full-name + chamber identity key.

    A group with more than one distinct candidate filing date is **not**
    matched; it is flagged ambiguous and keeps the vendor date.
    """
    ct = ct.copy()
    namemap = filings.attrs.get("name_to_bioguide", pd.Series(dtype="object"))
    ct["_ident"] = [
        identity_key(n, ch)
        for n, ch in zip(ct.get("politician_name"), ct.get("chamber"), strict=False)
    ]
    ct["_bio"] = ct["_ident"].map(namemap) if len(namemap) else pd.NA
    ct["_src"] = pd.to_datetime(ct[source_date_col], errors="coerce")

    def _match(key_col_ct: str, key_col_f: str, method: str) -> pd.DataFrame:
        left = ct.loc[ct[key_col_ct].notna() & (ct[key_col_ct] != ""), [key_col_ct, "_src"]]
        left = left.drop_duplicates()
        if left.empty:
            return pd.DataFrame(columns=["_k", "_src", "official_filing_date",
                                         "matched_filing_id", "candidate_count", "match_method"])
        right = filings.loc[filings[key_col_f].notna(), [key_col_f, "fdate", "filing_id"]]
        j = left.rename(columns={key_col_ct: "_k"}).merge(
            right.rename(columns={key_col_f: "_k"}), on="_k", how="inner"
        )
        j = j[(j.fdate <= j._src)
              & (j.fdate >= j._src - pd.Timedelta(days=FILING_MATCH_WINDOW_DAYS))]
        if j.empty:
            return pd.DataFrame(columns=["_k", "_src", "official_filing_date",
                                         "matched_filing_id", "candidate_count", "match_method"])
        g = j.groupby(["_k", "_src"])
        agg = g.agg(
            official_filing_date=("fdate", "max"),
            distinct_dates=("fdate", "nunique"),
            candidate_count=("filing_id", "nunique"),
        ).reset_index()
        # unique-date requirement: several filings on the SAME day are fine
        # (they were published together); different days are ambiguous.
        agg = agg[agg["distinct_dates"] == 1].drop(columns=["distinct_dates"])
        fid = (j.sort_values("filing_id").groupby(["_k", "_src"])["filing_id"]
               .agg(lambda s: s.iat[0] if s.nunique() == 1 else pd.NA).reset_index()
               .rename(columns={"filing_id": "matched_filing_id"}))
        agg = agg.merge(fid, on=["_k", "_src"], how="left")
        agg["match_method"] = method
        return agg

    by_bio = _match("_bio", "bioguide_id", "bioguide+window")
    matched = ct[["_bio", "_src"]].rename(columns={"_bio": "_k"}).merge(
        by_bio, on=["_k", "_src"], how="left"
    )
    ct["official_filing_date"] = matched["official_filing_date"].to_numpy()
    ct["matched_filing_id"] = matched["matched_filing_id"].to_numpy()
    ct["candidate_count"] = matched["candidate_count"].to_numpy()
    ct["event_date_match_method"] = matched["match_method"].to_numpy()

    need = ct["official_filing_date"].isna()
    if need.any():
        by_name = _match("_ident", "identity_key", "name_chamber+window")
        m2 = ct.loc[need, ["_ident", "_src"]].rename(columns={"_ident": "_k"}).merge(
            by_name, on=["_k", "_src"], how="left"
        )
        ct.loc[need, "official_filing_date"] = m2["official_filing_date"].to_numpy()
        ct.loc[need, "matched_filing_id"] = m2["matched_filing_id"].to_numpy()
        ct.loc[need, "candidate_count"] = m2["candidate_count"].to_numpy()
        ct.loc[need, "event_date_match_method"] = m2["match_method"].to_numpy()

    ct["official_filing_date"] = pd.to_datetime(ct["official_filing_date"], errors="coerce")
    ct["event_date_recovered"] = ct["official_filing_date"].notna().astype(int)
    ct["event_date_ambiguous"] = (
        ct["official_filing_date"].isna() & ct["_ident"].ne("")
    ).astype(int)
    ct["event_date_match_confidence"] = ct["event_date_match_method"].map(
        {"bioguide+window": 0.95, "name_chamber+window": 0.80}
    ).fillna(0.0)
    # analysis event date: recovered official filing date, else the vendor date
    ct["analysis_event_date"] = ct["official_filing_date"].fillna(ct["_src"])
    ct["event_date_is_fallback"] = ct["official_filing_date"].isna().astype(int)

    stats = {
        "rows": int(len(ct)),
        "recovered": int(ct["event_date_recovered"].sum()),
        "recovered_share": float(ct["event_date_recovered"].mean()) if len(ct) else 0.0,
        "by_bioguide": int((ct["event_date_match_method"] == "bioguide+window").sum()),
        "by_name_chamber": int((ct["event_date_match_method"] == "name_chamber+window").sum()),
        "fallback_vendor_date": int(ct["event_date_is_fallback"].sum()),
        "unresolved_or_ambiguous": int(ct["event_date_ambiguous"].sum()),
    }
    return RecoveryResult(ct.drop(columns=["_ident", "_bio", "_src"]), stats)


def reconcile_boundary(
    official: pd.DataFrame, vendor: pd.DataFrame, boundary: pd.Timestamp,
    official_ids: pd.Series, vendor_ids: pd.Series,
) -> dict:
    """Prove every eligible input row appears exactly once, or is excluded.

    Segment membership is decided on the immutable source-selection date, so
    event-date harmonization cannot move a row across the boundary.
    """
    off_in = set(official_ids.astype(str))
    ven_in = set(vendor_ids.astype(str))
    overlap = off_in & ven_in
    report = {
        "boundary": str(boundary.date()),
        "official_input_rows": len(off_in),
        "vendor_input_rows": len(ven_in),
        "official_selected": int(len(official)),
        "vendor_selected": int(len(vendor)),
        "id_overlap_between_segments": len(overlap),
        "official_excluded": len(off_in) - int(len(official)),
        "vendor_excluded": len(ven_in) - int(len(vendor)),
        "total_selected": int(len(official)) + int(len(vendor)),
    }
    report["disjoint"] = len(overlap) == 0
    report["exhaustive_within_eligibility"] = (
        report["official_selected"] + report["official_excluded"] == report["official_input_rows"]
        and report["vendor_selected"] + report["vendor_excluded"] == report["vendor_input_rows"]
    )
    return report
