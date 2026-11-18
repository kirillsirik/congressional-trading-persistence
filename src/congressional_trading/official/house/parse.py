"""Parser for e-filed (text-native) House PTR PDFs.

Form family "house-eptr" (electronic PTRs, DocIDs beginning "2", 2013→present;
layout anchors: header row "ID | Owner | Asset | Transaction Type | Date |
Notification Date | Amount | Cap. Gains > $200?"). Scanned/paper PTRs
(8-digit DocIDs) are routed to the OCR/review path by the cascade, never
parsed here.

Strategy:
1. pdfplumber table extraction; rows with proper column splits parsed directly
   (high confidence).
2. Rows collapsed into a single cell ("blob" rows) reconstructed by regex
   (lower confidence).
3. Subfield continuation rows (``F S:``, ``S O:``, ``D:``, ``C:``) are
   attached to the preceding transaction, never dropped: filing status,
   subholding, description, comments live there.
Every row keeps its page, row index, and raw text evidence.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field

from ..parse_failures import ParseFailure

PARSER_NAME = "house_eptr"
PARSER_VERSION = "1.0"

_DATE = r"\d{1,2}/\d{1,2}/\d{4}"
# Older House PDFs render the range separator as U+00AD (soft hyphen) and
# other Unicode dashes, not ASCII '-'. Some also emit the form's checkbox
# glyph ("gfedc"/"gfedcb") between the range bounds.
_DASH = r"[-­‐-―−]"
_CHECKBOX = r"(?:\s*gfedcb?\s*)?"
# A dollar figure may be sub-dollar and rendered without a leading zero
# ("$.01" on penny-priced options), so the integer part is optional.
_MONEY = r"\$[\d,]*\.?\d+"
_AMOUNT = rf"{_MONEY}(?:\s*{_DASH}\s*{_CHECKBOX}\$?[\d,]*\.?\d+|\s*\+)?|None"
# Some House PDFs have broken font encodings that render capitals as lowercase
# throughout ("sP Hill International, Inc. (HIl) s 12/26/2013 ..."), so the
# owner and transaction-type codes are matched case-insensitively. The
# normalizers already fold case, and the surrounding date+amount context makes
# a false positive implausible.
_TXTYPE = r"(?i:[PSE])\s*(?i:\(partial\))?"
_OWNER_CODE = r"(?i:SP|DC|JT)"

# Blob row: owner? asset ... type date date amount
# The owner code must be a standalone token — without the lookahead, "SP"
# would be eaten off the front of asset names like "SPDR ..." or "SPY".
_BLOB_ROW = re.compile(
    rf"^(?P<owner>{_OWNER_CODE}(?=\s))?\s*(?P<asset>.+?)\s+"
    rf"(?P<tx_type>{_TXTYPE})\s+"
    rf"(?P<tx_date>{_DATE})\s+(?P<notif_date>{_DATE})\s+"
    rf"(?P<amount>{_AMOUNT})\s*(?P<gains>.*)$",
    re.S,
)

# Older House PDFs (roughly 2014-2016) put the owner in a column AFTER the two
# dates and render it as words, then wrap the asset name and the amount around
# it: "Henderson Enterprises Inc. S 12/31/2014 12/31/2014 Spouse/DC over
# $1,000,000". `_BLOB_ROW` requires the amount immediately after the
# notification date, so every row in that layout failed to reconstruct and was
# recorded as `unreconstructed_row` (526 review items; 62 of them real
# transactions in this layout).
_OWNER_WORD = r"(?i:spouse/dc|spouse|dependent child|joint|self|sp|dc|jt)"
_BLOB_ROW_OWNER_MID = re.compile(
    rf"^(?P<owner>{_OWNER_CODE}(?=\s))?\s*(?P<asset>.+?)\s+"
    rf"(?P<tx_type>{_TXTYPE})\s+"
    rf"(?P<tx_date>{_DATE})\s+(?P<notif_date>{_DATE})\s+"
    rf"(?P<owner_mid>{_OWNER_WORD})\s+"
    rf"(?P<mid>.*?)\s*(?P<amount>{_AMOUNT})\s*(?P<gains>.*)$",
    re.S,
)

_SUBFIELD = re.compile(
    r"^\s*(?:F\x00*\s*S\x00*|F S|FILING STATUS)\s*\x00*:|"
    r"^\s*(?:S\x00*\s*O\x00*|S O|SUBHOLDING OF)\s*\x00*:|"
    r"^\s*(?:D\x00*|DESCRIPTION)\s*\x00*:|"
    r"^\s*(?:C\x00*|COMMENTS)\s*\x00*:|"
    r"^\s*(?:L\x00*|LOCATION)\s*\x00*:",
    re.I,
)

# The negative lookbehind protects the asset cell's OWN label: bond and ABS rows
# read "SP Security Description: VENDEE MTG TR S 03/24/2017 ...", and treating
# that "Description:" as a subfield marker truncated the row to "SP Security",
# so the transaction could not be reconstructed at all.
_SUBFIELD_SPLIT = re.compile(
    r"(?:^|(?<=\s))(?<!Security )(F\x00*\s*S\x00*|FILING STATUS|S\x00*\s*O\x00*|"
    r"SUBHOLDING OF|D\x00*|DESCRIPTION|C\x00*|COMMENTS|L\x00*|LOCATION)\s*\x00*:\s*",
    re.I,
)

_SUBFIELD_KEY = {
    "fs": "filing_status_raw", "filingstatus": "filing_status_raw",
    "so": "subholding_raw", "subholdingof": "subholding_raw",
    "d": "description_raw", "description": "description_raw",
    "c": "comment_raw", "comments": "comment_raw",
    "l": "location_raw", "location": "location_raw",
}


@dataclass
class HouseParseOutput:
    filer_meta: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)


def _clean(cell: str | None) -> str:
    if cell is None:
        return ""
    return " ".join(cell.replace("\x00", "").split())


def _norm_key(label: str) -> str:
    return re.sub(r"[^a-z]", "", label.lower())


def _parse_subfields(text: str) -> dict:
    out: dict[str, str] = {}
    parts = _SUBFIELD_SPLIT.split(text)
    # parts = [prefix, label1, value1, label2, value2, ...]
    for i in range(1, len(parts) - 1, 2):
        key = _SUBFIELD_KEY.get(_norm_key(parts[i]))
        if key:
            val = _clean(parts[i + 1])
            if val:
                out[key] = val
    return out


def _is_header_row(cells: list[str]) -> bool:
    """Strict header detection: a real header carries the form's own labels
    and never a date+amount. A loose substring test would discard genuine
    transactions in assets like 'Asset Management Owner Corp'."""
    if not cells:
        return False
    joined = " ".join(cells).lower()
    if re.search(_DATE, joined) and "$" in joined:
        return False
    if cells[0].strip().upper() == "ID":
        return True
    labels = ("transaction type", "notification date", "cap. gains", "cap gains")
    return sum(lbl in joined for lbl in labels) >= 2


def _looks_like_txrow(cells: list[str]) -> bool:
    """Proper split-column transaction row: type code + two dates in place."""
    if len(cells) < 7:
        return False
    joined = cells[4] + " " + cells[5] if len(cells) > 5 else ""
    return (
        bool(re.fullmatch(_TXTYPE, cells[3].strip()))
        and len(re.findall(_DATE, joined)) >= 2
    )


def _parse_amount(text: str) -> str:
    return _clean(text)


def parse_house_ptr_pdf(content: bytes) -> HouseParseOutput:
    import pdfplumber

    out = HouseParseOutput()
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        first_text = pdf.pages[0].extract_text() or ""
        out.filer_meta = _parse_filer_meta(first_text)
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for table in tables:
                _parse_table_rows(table, page_no, out)
            # pdfplumber's table detection misses rows that fall outside the
            # detected table bbox (typically the last row on a page). Reconcile
            # against the page text so no transaction is dropped silently.
            _reconcile_page_text(page, page_no, out)
    return out


# A transaction as it appears in the page text: type, both dates, amount.
_TX_SIGNATURE = re.compile(
    rf"(?<![\w/])(?P<tx_type>{_TXTYPE})\s+(?P<tx_date>{_DATE})\s+(?P<notif_date>{_DATE})"
    rf"\s+(?P<amount>\$[\d,]+)"
)


def _row_signature(tx_type: str, tx_date: str, notif_date: str, amount: str) -> tuple:
    first_amount = re.search(r"\$[\d,]+", amount or "")
    return (
        _clean(tx_type).replace(" ", "").lower(),
        _clean(tx_date),
        _clean(notif_date),
        first_amount.group(0).replace(",", "") if first_amount else "",
    )


def _reconcile_page_text(page, page_no: int, out: HouseParseOutput) -> None:
    text = page.extract_text() or ""
    if not text.strip():
        return
    have = Counter(
        _row_signature(
            r.get("transaction_type_raw", ""), r.get("transaction_date_raw", ""),
            r.get("notification_date_raw", ""), r.get("amount_raw", ""),
        )
        for r in out.rows
        if r.get("page") == page_no
    )
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for m in _TX_SIGNATURE.finditer(line):
            sig = _row_signature(
                m.group("tx_type"), m.group("tx_date"), m.group("notif_date"),
                m.group("amount"),
            )
            if have.get(sig, 0) > 0:
                have[sig] -= 1
                continue
            # not captured by any table row on this page: recover it
            candidate = _candidate_from_lines(lines, i, m)
            parsed = _parse_blob_text(candidate, page_no, confidence=0.70,
                                      extraction="pdf_text_reconciled")
            if parsed is not None:
                out.rows.append(parsed)
                have[sig] = 0
            else:
                out.failures.append(ParseFailure(
                    "unreconstructed_row",
                    f"transaction present in page text but not reconstructable: "
                    f"{line.strip()[:200]}",
                    page_no,
                ))


def _candidate_from_lines(lines: list[str], i: int, m: re.Match) -> str:
    """Rebuild one transaction's text: the asset name may wrap onto the line
    above, and the amount's upper bound onto the line below."""
    parts = []
    prefix = lines[i][: m.start()].strip()
    if not prefix and i > 0:
        prev = lines[i - 1].strip()
        if prev and not _TX_SIGNATURE.search(prev) and not _SUBFIELD.match(prev):
            parts.append(prev)
    parts.append(lines[i].strip())
    tail = lines[i][m.end():].strip()
    if i + 1 < len(lines) and (not tail or tail.lower().startswith("gfedc")):
        nxt = lines[i + 1].strip()
        if nxt and not _TX_SIGNATURE.search(nxt):
            parts.append(nxt)
    return " ".join(parts)


def _parse_filer_meta(text: str) -> dict:
    meta: dict[str, str] = {}
    m = re.search(r"Name:\s*(.+)", text)
    if m:
        meta["filer_name_doc"] = _clean(m.group(1))
    m = re.search(r"Status:\s*(.+)", text)
    if m:
        meta["filer_status_raw"] = _clean(m.group(1))
    m = re.search(r"State/District:\s*([A-Z]{2}\d{0,2})", text)
    if m:
        meta["state_district_doc"] = m.group(1)
    m = re.search(r"[Ff]iling\s*ID\s*[#:]?\s*(\d+)", text)
    if m:
        meta["filing_id_doc"] = m.group(1)
    # NOTE: do not flag amendments from page text — the e-filed PTR template
    # itself contains the word "Amendment"; index/search labels and per-row
    # "F S: Amended" are the reliable signals.
    return meta


def _parse_table_rows(table: list[list[str | None]], page_no: int, out: HouseParseOutput) -> None:
    for raw_cells in table:
        cells = [(c if c is not None else "") for c in raw_cells]
        cleaned = [_clean(c) for c in cells]
        raw_text = " | ".join(c.replace("\x00", "?") for c in cells if c)
        if _is_header_row(cleaned) or not any(cleaned):
            continue
        # The subfield block does not always start in the first cell: when the
        # extractor leaves leading cells empty, the block lands in a later one.
        # Testing only cells[0] sent 442 pure metadata blocks down the blob path,
        # where they were recorded as `unreconstructed_row` because they happen
        # to contain a date — no transaction was ever missing from them.
        first_text = next((c for c in cells if c and c.strip()), "")
        if _SUBFIELD.match(first_text):
            sub = _parse_subfields("\n".join(c for c in cells if c))
            if out.rows and sub:
                out.rows[-1].update({k: v for k, v in sub.items() if k not in out.rows[-1]})
            elif sub:
                # continuation with no preceding transaction (page break or a
                # dropped row): record it rather than discarding the text
                out.failures.append(ParseFailure(
                    "orphan_subfield_block",
                    f"subfield block with no preceding transaction row: "
                    f"{raw_text[:200]}",
                    page_no,
                ))
            continue
        if _looks_like_txrow(cleaned):
            row = {
                "row_id_doc": cleaned[0] or None,
                "owner_raw": cleaned[1],
                "asset_description_raw": cleaned[2],
                "transaction_type_raw": cleaned[3],
                "transaction_date_raw": cleaned[4],
                "notification_date_raw": cleaned[5],
                "amount_raw": _parse_amount(cleaned[6]),
                "capital_gains_over_200_raw": (
                    _SUBFIELD_SPLIT.split(cleaned[7])[0].strip() if len(cleaned) > 7 else ""
                ),
                "page": page_no,
                "extraction": "pdf_table",
                "confidence": 0.95,
                "raw_text": raw_text,
            }
            _extract_inline_subfields(row)
            out.rows.append(row)
            continue
        # Blob row: everything jammed into the first cell(s)
        blob = "\n".join(c for c in cells if c and c.strip())
        blob_clean = " ".join(blob.replace("\x00", "").split())
        sub = _parse_subfields(blob)
        parsed = _parse_blob_text(blob_clean, page_no, 0.75, "pdf_text_blob", raw_text)
        if parsed is not None:
            parsed.update({k: v for k, v in sub.items() if k not in parsed})
            out.rows.append(parsed)
        elif len(blob_clean) > 12 and re.search(_DATE, blob_clean):
            # A fragment carrying a transaction signature is a real missing row.
            # One without a signature is wrapped asset text (a bond maturity
            # line, a trust interest, a page-break continuation) — its text is
            # kept for review, but nothing is missing and it must not block a
            # release.
            kind = ("unreconstructed_row" if _TX_SIGNATURE.search(blob_clean)
                    else "asset_description_continuation")
            out.failures.append(ParseFailure(kind, blob_clean[:200], page_no))


def _parse_blob_text(
    text: str,
    page_no: int,
    confidence: float,
    extraction: str,
    raw_text: str | None = None,
) -> dict | None:
    """Reconstruct one transaction from a run of text, or None."""
    cleaned = " ".join(str(text).replace("\x00", "").split())
    noise_cut = _SUBFIELD_SPLIT.split(cleaned)[0]
    m = _BLOB_ROW.match(noise_cut.strip())
    owner_mid = mid_text = ""
    if not m:
        # Older layout: owner as words between the dates and the amount.
        m = _BLOB_ROW_OWNER_MID.match(noise_cut.strip())
        if not m:
            return None
        owner_mid = _clean(m.group("owner_mid"))
        mid_text = _clean(m.group("mid") or "")
    asset = _clean(m.group("asset"))
    # Text after the amount is wrapped asset-name continuation (ticker,
    # exchange line), not the cap-gains checkbox column.
    trailer = _SUBFIELD_SPLIT.split(_clean(m.group("gains")))[0].strip()
    trailer = re.sub(r"\b(gfedcb?|X)\b\s*$", "", trailer, flags=re.I).strip()
    # A trailer that is itself a transaction means two rows were jammed into
    # one cell; keep the asset clean and record the remainder for review.
    extra_tx = _TX_SIGNATURE.search(trailer) if trailer else None
    if trailer and not extra_tx:
        asset = f"{asset} {trailer}".strip()

    amount = m.group("amount")
    if owner_mid:
        # The text between the owner word and the dollar figure is the
        # open-ended bracket's "Over" plus wrapped asset-name continuation. The
        # "Over" belongs to the amount — without it `normalize_amount` cannot
        # tell "$1,000,000" (no such bracket) from "Over $1,000,000".
        rest = re.sub(r"(?i)\bov[e2c]r\b", "", mid_text).strip()
        if re.search(r"(?i)\bov[e2c]r\b", mid_text) and "over" not in amount.lower():
            amount = f"Over {amount}"
        if rest:
            asset = f"{asset} {rest}".strip()

    row = {
        "row_id_doc": None,
        "owner_raw": _clean(m.group("owner") or "") or owner_mid,
        "asset_description_raw": asset,
        "transaction_type_raw": _clean(m.group("tx_type")),
        "transaction_date_raw": m.group("tx_date"),
        "notification_date_raw": m.group("notif_date"),
        "amount_raw": _parse_amount(amount),
        "capital_gains_over_200_raw": None,
        "page": page_no,
        "extraction": extraction,
        "confidence": confidence,
        "raw_text": raw_text if raw_text is not None else cleaned[:500],
    }
    row.update(_parse_subfields(text))
    if extra_tx:
        row["unparsed_trailer"] = trailer[:300]
    return row


def _extract_inline_subfields(row: dict) -> None:
    """Asset cells often carry the subfield block inside the same cell."""
    asset = row["asset_description_raw"]
    parts = _SUBFIELD_SPLIT.split(asset)
    if len(parts) > 1:
        row["asset_description_raw"] = _clean(parts[0])
        row.update(
            {k: v for k, v in _parse_subfields(asset[len(parts[0]):]).items() if k not in row}
        )
