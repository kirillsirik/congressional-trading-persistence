"""Supplementary House discovery via the Clerk's search endpoint.

The yearly {YEAR}FD.zip indexes omit some PTR filings entirely — verified
live 2026-07-24: the 2012 index contains zero FilingType=P rows, while the
Clerk search UI returns 190 "PTR Original" + 7 "PTR Amendment" + 2 "PTR
Waiver" for Filing Year 2012.
This adapter POSTs the same endpoint the search UI uses
(/FinancialDisclosure/ViewMemberSearchResult, one request per year) and
unions the results with the ZIP-index discovery by DocID.

Search "Filing Year" is the report coverage year; the ZIP index "Year" is the
filed year — both are recorded.
"""

from __future__ import annotations

import logging
import re

from .. import ids
from ..config import OfficialConfig
from ..db import StateStore
from ..http import PoliteSession

logger = logging.getLogger(__name__)

SEARCH_PATH = "/FinancialDisclosure/ViewMemberSearchResult"

_ROW = re.compile(r"<tr role=\"row\">(.*?)</tr>", re.S)
_HREF = re.compile(r'href="([^"]+)"[^>]*>\s*([^<]+?)\s*</a>')
_CELL = re.compile(r'data-label="([^"]+)">([^<]*)<')
_DOCID = re.compile(r"/(\d{4})/(\d+)\.pdf")

PTR_LABELS = {"PTR Original", "PTR Amendment", "PTR Waiver"}


def discover_year_via_search(
    config: OfficialConfig, store: StateStore, session: PoliteSession, run_id: str, year: int
) -> dict:
    url = config.house_base + SEARCH_PATH
    result = session.post(
        url,
        data={"LastName": "", "FilingYear": str(year), "State": "", "District": ""},
        headers={"Referer": config.house_base + "/FinancialDisclosure"},
    )
    store.record_fetch(
        run_id=run_id, source="house_clerk_search", chamber="house", url=url, method="POST",
        params_json=f'{{"FilingYear": {year}}}', http_status=result.status,
        final_url=result.final_url, content_type=result.content_type,
        retry_count=result.retry_count,
        result_status="failed" if result.error_category else "ok",
        error_category=result.error_category, error_detail=result.error_detail,
    )
    if result.error_category or result.content is None:
        raise RuntimeError(f"house search {year} failed: {result.error_detail}")
    html = result.content.decode("utf-8", errors="replace")
    rows = _ROW.findall(html)
    if not rows and "Search Results" not in html:
        raise RuntimeError(f"house search {year}: unexpected response layout (source drift?)")

    label_counts: dict[str, int] = {}
    n_new = 0
    n_ptr = 0
    for row_html in rows:
        href_m = _HREF.search(row_html)
        cells = dict(_CELL.findall(row_html))
        label = (cells.get("Filing") or "").strip()
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in PTR_LABELS or not href_m:
            continue
        href, name = href_m.group(1), " ".join(href_m.group(2).split())
        doc_m = _DOCID.search(href)
        if not doc_m:
            continue
        url_year, doc_id = doc_m.group(1), doc_m.group(2)
        n_ptr += 1
        fid = ids.filing_id("house", doc_id)
        existing = store.conn.execute(
            "SELECT filing_id FROM filings WHERE filing_id=?", (fid,)
        ).fetchone()
        last, first = (name.split(",", 1) + [""])[:2] if "," in name else (name, "")
        full_url = config.house_base + "/" + href.lstrip("/")
        fields = dict(
            chamber="house",
            source_system="house_clerk",
            source_doc_id=doc_id,
            report_type="ptr",
            report_type_raw=label,
            is_amendment=1 if label == "PTR Amendment" else 0,
            source_url=full_url,
        )
        if not existing:
            fields.update(
                filer_name_raw=name,
                first_name_raw=first.strip().replace("HON.", "").strip(),
                last_name_raw=last.strip().title(),
                state_district_raw=(cells.get("Office") or "").strip(),
                report_year=int(url_year),
                discovered_run_id=run_id,
            )
            n_new += 1
        store.upsert_filing(fid, **fields)
    logger.info(
        "house search %s: %d result rows, %d PTR rows (%d new); labels %s",
        year, len(rows), n_ptr, n_new, label_counts,
    )
    return {"year": year, "rows": len(rows), "ptr_rows": n_ptr, "new_filings": n_new,
            "labels": label_counts}
