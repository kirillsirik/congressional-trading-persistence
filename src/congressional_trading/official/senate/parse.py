"""Parser for structured (electronic) Senate eFD PTR pages.

Semantic DOM parsing: transaction tables are located by their header labels
(never by CSS position alone), header meanings are preserved, and row order,
comments, and report metadata are retained. Paper filings never reach this
parser — the cascade routes them to OCR/review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from ..parse_failures import ParseFailure

PARSER_NAME = "senate_efd_html"
PARSER_VERSION = "1.0"

# Canonical header labels → raw field names. Matching is fuzzy on
# lowercased/alnum text so cosmetic markup changes don't break the mapping.
HEADER_MAP = {
    "transactiondate": "transaction_date_raw",
    "date": "transaction_date_raw",
    "owner": "owner_raw",
    "ticker": "ticker_raw",
    "assetname": "asset_description_raw",
    "assettype": "asset_type_raw",
    "type": "transaction_type_raw",
    "transactiontype": "transaction_type_raw",
    "amount": "amount_raw",
    "comment": "comment_raw",
    "comments": "comment_raw",
    "id": "row_id_doc",
}


@dataclass
class SenateParseOutput:
    filer_meta: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)


def _norm_header(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def parse_senate_ptr_html(html: str) -> SenateParseOutput:
    out = SenateParseOutput()
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    if h1:
        out.filer_meta["report_title_raw"] = " ".join(h1.get_text(" ", strip=True).split())
        if "amendment" in out.filer_meta["report_title_raw"].lower():
            out.filer_meta["is_amendment_doc"] = "true"
    filed = soup.find(string=re.compile(r"Filed\s+\d{2}/\d{2}/\d{4}"))
    if filed:
        m = re.search(r"Filed\s+(\d{2}/\d{2}/\d{4})\s*(?:@\s*([\d:]+\s*[APM]{2}))?", str(filed))
        if m:
            out.filer_meta["filed_date_raw"] = m.group(1)
            if m.group(2):
                out.filer_meta["filed_time_raw"] = m.group(2)
    h2 = soup.find("h2", class_="filedReport")
    if h2:
        out.filer_meta["filer_name_doc"] = " ".join(h2.get_text(" ", strip=True).split())

    for table in soup.find_all("table"):
        headers = [
            _norm_header(th.get_text(" ", strip=True)) for th in table.find_all("th")
        ]
        if not headers:
            continue
        mapped = [HEADER_MAP.get(h) for h in headers]
        # a PTR transaction table must expose at least asset + type + amount
        essentials = {"asset_description_raw", "transaction_type_raw", "amount_raw"}
        if not essentials.issubset({m for m in mapped if m}):
            continue
        body = table.find("tbody") or table
        for row_index, tr in enumerate(body.find_all("tr")):
            cells = tr.find_all("td")
            if not cells:
                continue
            row: dict = {"extraction": "html_dom", "confidence": 0.97}
            texts = []
            for i, td in enumerate(cells):
                # keep sub-line details (option/bond info in .text-muted) as text
                text = " ".join(td.get_text(" ", strip=True).split())
                texts.append(text)
                if i < len(mapped) and mapped[i]:
                    row[mapped[i]] = text
            row["raw_text"] = " | ".join(texts)
            row["row_index_table"] = row_index
            if row.get("ticker_raw") in {"--", "—", ""}:
                row["ticker_raw"] = None
            _extract_asset_details(cells, mapped, row)
            out.rows.append(row)
    if not out.rows and soup.find(string=re.compile(r"[Tt]ransaction")):
        out.failures.append(ParseFailure(
            "no_parseable_table_rows",
            "transaction section present but no parseable table rows",
        ))
    return out


def _extract_asset_details(cells, mapped, row: dict) -> None:
    """Options/bonds carry structured sub-lines inside the asset cell."""
    try:
        asset_idx = mapped.index("asset_description_raw")
    except ValueError:
        return
    if asset_idx >= len(cells):
        return
    cell_text = cells[asset_idx].get_text("\n", strip=True)
    for pattern, key in [
        (r"Option Type:\s*([^\n]+?)(?:\s+Strike price|\n|$)", "option_type_raw"),
        (r"Strike price:\s*\$?([\d.,]+)", "option_strike_raw"),
        (r"Expires:\s*([\d/]+)", "option_expiration_raw"),
        (r"Rate/Coupon:\s*([\d.]+%?)", "bond_rate_raw"),
        (r"Matures:\s*([\d/]+)", "bond_maturity_raw"),
    ]:
        m = re.search(pattern, cell_text)
        if m:
            row[key] = m.group(1).strip()
    first_line = cell_text.split("\n", 1)[0].strip()
    if first_line and any(k in row for k in
                          ("option_type_raw", "bond_rate_raw", "bond_maturity_raw")):
        row["asset_description_raw"] = first_line
