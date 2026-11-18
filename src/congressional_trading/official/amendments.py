"""Amendment reconciliation and transaction versioning.

Originals are immutable: an amendment never erases the original filing or its
rows. Amendment groups are built per filer; within a group, rows are aligned
by content key, occurrence-indexed so legitimate repeated identical
transactions survive. Output views:

- transaction_history: every observed row version with lineage fields;
- transactions_current: the latest filing version's interpretation per
  canonical transaction (superseded rows carry is_current=0).

Uncertain relationships are kept unlinked with low confidence rather than
collapsed.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from . import ids

RECONCILIATION_VERSION = "1"


def build_amendment_groups(filings: pd.DataFrame) -> pd.DataFrame:
    """Group filings by (chamber, filer, report period) and order versions.

    House e-filed amendments reference no machine-readable parent; Senate
    amendment titles mark amended reports. Grouping key: chamber + normalized
    filer name + report_year + (period signature = sorted transaction dates
    span, supplied by caller as period_key). Confidence is 'high' only when an
    explicit amendment marker exists and exactly one earlier filing matches.
    """
    filings = filings.copy()
    filings["group_key"] = (
        filings["chamber"].fillna("")
        + "|" + filings["filer_name_raw"].fillna("").str.lower().str.strip()
        + "|" + filings["period_key"].fillna("")
    )
    filings["version_number"] = 0
    filings["amends_filing_id"] = None
    filings["amendment_relationship_confidence"] = None
    filings["group_root_filing_id"] = filings["filing_id"]

    for _, group in filings.groupby("group_key"):
        if len(group) == 1:
            continue
        ordered = group.sort_values(["filing_date", "filing_id"], na_position="last")
        root_id = ordered.iloc[0]["filing_id"]
        prev_id = None
        for version, (idx, row) in enumerate(ordered.iterrows()):
            filings.loc[idx, "version_number"] = version
            filings.loc[idx, "group_root_filing_id"] = root_id
            if version > 0:
                is_marked = bool(row.get("is_amendment"))
                filings.loc[idx, "amends_filing_id"] = prev_id
                filings.loc[idx, "amendment_relationship_confidence"] = (
                    "high" if is_marked else "low"
                )
            prev_id = row["filing_id"]
    return filings


def reconcile_transactions(
    raw_tx: pd.DataFrame, filings_with_groups: pd.DataFrame
) -> pd.DataFrame:
    """Assign canonical IDs and current/superseded status.

    raw_tx must contain: raw_transaction_id, filing_id, and the raw content
    fields used by ids.transaction_content_key.
    """
    fmap = filings_with_groups.set_index("filing_id")
    rows = []
    for _, tx in raw_tx.iterrows():
        fid = tx["filing_id"]
        if fid not in fmap.index:
            continue
        filing = fmap.loc[fid]
        content_key = ids.transaction_content_key(
            filing["chamber"],
            filing.get("filer_name_raw") or "",
            tx.get("owner_raw") or "",
            tx.get("asset_description_raw") or "",
            tx.get("transaction_type_raw") or "",
            tx.get("transaction_date_raw") or "",
            tx.get("amount_raw") or "",
        )
        rows.append(
            {
                **tx.to_dict(),
                "content_key": content_key,
                "group_root_filing_id": filing["group_root_filing_id"],
                "version_number": int(filing["version_number"]),
                "is_amendment": int(bool(filing.get("is_amendment"))),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # correction key: same owner/asset/type/date but e.g. corrected amount —
    # used only to link versions, never to merge rows within one filing
    df["correction_key"] = [
        ids.transaction_content_key(
            r.get("chamber") or "", r.get("filer_name_raw") or "",
            r.get("owner_raw") or "", r.get("asset_description_raw") or "",
            r.get("transaction_type_raw") or "", r.get("transaction_date_raw") or "", "",
        )
        for r in df.to_dict("records")
    ]
    # occurrence index: repeated identical content within one filing is real
    df["occurrence"] = df.groupby(["filing_id", "content_key"]).cumcount()

    # a row is current iff it belongs to the latest version of its filing group
    group_max = df.groupby("group_root_filing_id")["version_number"].transform("max")
    df["is_current"] = (df["version_number"] == group_max).astype(int)

    # canonical IDs: link rows across versions (exact content first, then
    # correction key for leftovers); unmatched rows get their own identity
    df["canonical_transaction_id"] = None
    df["superseded_by"] = None
    for _, group in df.groupby("group_root_filing_id"):
        assigned: dict[tuple[str, int], str] = {}     # (content_key, occ) -> cid
        by_correction: dict[tuple[str, int], str] = {}
        prev_rows: dict[str, list] = {}
        for version in sorted(group["version_number"].unique()):
            sub = group[group["version_number"] == version]
            corr_occ: dict[str, int] = {}
            for idx, row in sub.iterrows():
                exact = (row["content_key"], row["occurrence"])
                occ_c = corr_occ.get(row["correction_key"], 0)
                corr_occ[row["correction_key"]] = occ_c + 1
                corr = (row["correction_key"], occ_c)
                if exact in assigned:
                    cid = assigned[exact]
                elif corr in by_correction and version > 0:
                    cid = by_correction[corr]
                else:
                    cid = ids.canonical_transaction_id(
                        row["group_root_filing_id"], row["content_key"], row["occurrence"]
                    )
                assigned[exact] = cid
                by_correction.setdefault(corr, cid)
                df.loc[idx, "canonical_transaction_id"] = cid
                for prev_idx in prev_rows.get(cid, []):
                    df.loc[prev_idx, "superseded_by"] = row["raw_transaction_id"]
                prev_rows.setdefault(cid, []).append(idx)

    df["reconciliation_status"] = "reconciled"
    df["reconciliation_version"] = RECONCILIATION_VERSION
    return df


def period_key_from_transactions(raw_tx: pd.DataFrame) -> pd.Series:
    """Per-filing period signature: min/max transaction date raw text; used to
    group original+amendment filings covering the same reporting period."""
    def _key(sub: pd.DataFrame) -> str:
        dates = sorted(d for d in sub.get("transaction_date_raw", pd.Series()) if d)
        if not dates:
            return "no-dates"
        return json.dumps([dates[0], dates[-1]])

    return raw_tx.groupby("filing_id").apply(_key, include_groups=False)


def summarize_groups(filings_with_groups: pd.DataFrame) -> dict:
    groups = defaultdict(int)
    multi = filings_with_groups.groupby("group_root_filing_id").size()
    groups["amendment_groups_total"] = int((multi > 1).sum())
    groups["filings_in_groups"] = int(multi[multi > 1].sum())
    return dict(groups)
