"""Deterministic identifiers for filings, documents, and transaction rows.

Raw transaction IDs are tied to the exact source document version (sha256),
page, and row, so a re-download with different bytes yields new raw IDs.
Canonical transaction IDs identify a transaction across filing amendments via
the amendment-group root filing plus a content key plus an occurrence index —
never member+ticker+date alone, which would collapse legitimate same-day
transactions.
"""

from __future__ import annotations

import hashlib


def _h(parts: list[str]) -> str:
    joined = "|".join("" if p is None else str(p).strip().lower() for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def filing_id(chamber: str, source_doc_id: str) -> str:
    """Stable filing identifier from the official document identifier.

    House: Clerk index DocID (string, digits not assumed fixed-width).
    Senate: eFD report UUID, lowercased (paper GUIDs appear uppercased on the
    site; casing is normalized here but the raw value is preserved elsewhere).
    """
    return f"{chamber.lower()}-{str(source_doc_id).strip().lower()}"


def raw_transaction_id(
    filing: str, doc_sha256: str, page: int | None, row_index: int, raw_fields: dict
) -> str:
    field_sig = _h([f"{k}={raw_fields[k]}" for k in sorted(raw_fields)])
    return "rt-" + _h([filing, doc_sha256, str(page), str(row_index), field_sig])


def transaction_content_key(
    chamber: str,
    filer_name_raw: str,
    owner_raw: str,
    asset_description_raw: str,
    transaction_type_raw: str,
    transaction_date_raw: str,
    amount_raw: str,
) -> str:
    """Content fingerprint used to align rows across filing versions."""
    return _h(
        [
            chamber,
            filer_name_raw,
            owner_raw,
            asset_description_raw,
            transaction_type_raw,
            transaction_date_raw,
            amount_raw,
        ]
    )


def canonical_transaction_id(group_root_filing_id: str, content_key: str, occurrence: int) -> str:
    """Identity of a transaction across amendments.

    ``occurrence`` disambiguates identical rows within the same filing group
    (repeated same-day same-asset transactions are legitimate and preserved).
    """
    return "ct-" + _h([group_root_filing_id, content_key, str(occurrence)])
