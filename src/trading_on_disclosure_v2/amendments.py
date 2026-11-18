"""Deterministic amendment and duplicate handling (final directive §7).

No manual review of amendments is required for the canonical run. The
hierarchy is:

7.1 explicit official lineage — used verbatim when the official system
    supplies an amendment-parent identifier;
7.2 exact repeated transactions — within the same filer and amendment
    lineage, a later row whose critical fields exactly reproduce an already
    publicly disclosed row is suppressed as a second public signal (retained
    in provenance with a flag);
7.3 new or changed amendment information keeps the AMENDMENT's public date —
    never backdated to the original filing (enforced structurally: this
    module never modifies any public-date column);
7.4 unresolved lineage — the document stays a separate source document,
    only exact duplicates are suppressed, and the row is flagged
    AMENDMENT_LINEAGE_UNRESOLVED.
"""

from __future__ import annotations

import pandas as pd

from trading_on_disclosure_v2.dates import ANALYSIS_PUBLIC_COL

SECURITY_FIELD_CANDIDATES = (
    "permno",
    "normalized_asset",
    "ticker_text",
    "security_name_text",
)

CRITICAL_FIELD_CANDIDATES = (
    *SECURITY_FIELD_CANDIDATES,
    "transaction_date_reported",
    "direction",
    "owner",
    "amount_bracket",
)

MINIMUM_CRITICAL = {"transaction_date_reported", "direction"}

DUPLICATE_FLAG = "amendment_exact_duplicate"
LINEAGE_STATUS_COL = "amendment_lineage_status"
UNRESOLVED_FLAG = "AMENDMENT_LINEAGE_UNRESOLVED"


class AmendmentError(RuntimeError):
    pass


def apply_amendment_hierarchy(txns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag exact cross-document repeats and lineage status.

    Requires ``filer_uid``, ``document_uid``, ``analysis_public_date`` plus at
    least one security field and the minimum critical fields. Returns the
    frame with ``amendment_exact_duplicate`` and ``amendment_lineage_status``
    columns added, and a machine-readable summary. Rows are never deleted
    here; the sample stage excludes flagged duplicates with an attrition code.
    """
    for col in ("filer_uid", "document_uid", ANALYSIS_PUBLIC_COL):
        if col not in txns.columns:
            raise AmendmentError(f"amendment handling requires column {col!r}")
    security_fields = [f for f in SECURITY_FIELD_CANDIDATES if f in txns.columns]
    if not security_fields:
        raise AmendmentError(
            "amendment handling requires a security identifier or a "
            "normalized asset description"
        )
    missing_min = MINIMUM_CRITICAL - set(txns.columns)
    if missing_min:
        raise AmendmentError(f"missing minimum critical fields: {sorted(missing_min)}")
    critical = [security_fields[0]] + [
        f
        for f in CRITICAL_FIELD_CANDIDATES
        if f in txns.columns and f != security_fields[0] and f not in security_fields[1:]
    ]

    out = txns.copy()
    public_before = out[ANALYSIS_PUBLIC_COL].copy()

    # 7.1 lineage status: explicit parent id > standalone original > unresolved
    has_parent_col = "amendment_parent_uid" in out.columns
    is_amendment = (
        out["is_amendment"].fillna(False).astype(bool)
        if "is_amendment" in out.columns
        else pd.Series(False, index=out.index)
    )
    parent = (
        out["amendment_parent_uid"]
        if has_parent_col
        else pd.Series(pd.NA, index=out.index, dtype=object)
    )
    linked = is_amendment & parent.notna() & (parent.astype(str).str.len() > 0)
    status = pd.Series("ORIGINAL", index=out.index, dtype=object)
    status[linked] = "OFFICIAL_LINEAGE"
    status[is_amendment & ~linked] = UNRESOLVED_FLAG
    out[LINEAGE_STATUS_COL] = status

    # 7.2 exact repeated transactions within the same filer: order by public
    # date (stable tie-break on document then row order); every later row in a
    # different document that exactly reproduces the critical fields of an
    # already-disclosed row is a duplicate public signal.
    ordered = out.sort_values(
        [ANALYSIS_PUBLIC_COL, "document_uid"], kind="mergesort"
    )
    dup_key = ["filer_uid", *critical]
    first_doc = ordered.groupby(dup_key, dropna=False)["document_uid"].transform("first")
    repeated = ordered.duplicated(subset=dup_key, keep="first")
    cross_document = ordered["document_uid"] != first_doc
    duplicate = (repeated & cross_document).reindex(out.index)
    out[DUPLICATE_FLAG] = duplicate.fillna(False).astype(bool)

    # 7.3 structural no-backdating guard
    if not public_before.equals(out[ANALYSIS_PUBLIC_COL]):
        raise AmendmentError(
            "amendment handling modified analysis_public_date (backdating "
            "is forbidden)"
        )

    summary = {
        "critical_fields_used": dup_key,
        "n_rows": int(len(out)),
        "n_official_lineage": int((out[LINEAGE_STATUS_COL] == "OFFICIAL_LINEAGE").sum()),
        "n_unresolved_lineage": int((out[LINEAGE_STATUS_COL] == UNRESOLVED_FLAG).sum()),
        "n_exact_duplicates_suppressed": int(out[DUPLICATE_FLAG].sum()),
        "unresolved_exclusion_sensitivity_recommended": bool(
            (out[LINEAGE_STATUS_COL] == UNRESOLVED_FLAG).mean() > 0.01
        ),
    }
    return out, summary
