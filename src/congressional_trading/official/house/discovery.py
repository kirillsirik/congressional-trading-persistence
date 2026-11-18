"""House yearly-index discovery.

Official entry point: https://disclosures-clerk.house.gov/FinancialDisclosure
Yearly archives: /public_disc/financial-pdfs/{YEAR}FD.zip, each containing a
{YEAR}FD.xml (and .txt) index of every filing that year. Verified live
2026-07-24 for 2012 and 2026.

FilingType codes observed in official indexes (validated empirically per year
by cli.py discover, not assumed): P = periodic transaction report. Other
codes (O/A/C/D/W/X...) are preserved in the raw layer and excluded from the
PTR universe with state=excluded_report_type.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile

from .. import ids
from ..config import OfficialConfig
from ..db import StateStore, utc_now_iso
from ..http import PoliteSession
from ..storage import store_bytes

logger = logging.getLogger(__name__)

PTR_FILING_TYPE = "P"

INDEX_FIELDS = [
    "Prefix", "First", "Middle", "Last", "Suffix", "FilingType",
    "StateDst", "Year", "FilingDate", "DocID",
]


def index_zip_url(config: OfficialConfig, year: int) -> str:
    return f"{config.house_base}/public_disc/financial-pdfs/{year}FD.zip"


def parse_index_zip(content: bytes) -> list[dict]:
    """Parse every row of the yearly XML index (no report-type filter here)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"no XML member in archive (members: {zf.namelist()})")
        xml_bytes = zf.read(xml_names[0])
    root = ET.fromstring(xml_bytes)
    rows = []
    for member in root:
        row = {child.tag: (child.text or "").strip() for child in member}
        rows.append(row)
    return rows


def _iso_date(us_date: str) -> str | None:
    """'1/9/2012' or '01/09/2012' -> '2012-01-09'; None when unparseable."""
    parts = us_date.strip().split("/")
    if len(parts) != 3:
        return None
    try:
        m, d, y = (int(p) for p in parts)
        if not (1 <= m <= 12 and 1 <= d <= 31 and 1900 < y < 2200):
            return None
        return f"{y:04d}-{m:02d}-{d:02d}"
    except ValueError:
        return None


def discover_year(
    config: OfficialConfig, store: StateStore, session: PoliteSession, run_id: str, year: int
) -> dict:
    """Download + parse one yearly index; upsert filings. Returns counts."""
    url = index_zip_url(config, year)
    prev = store.last_fetch_meta(url)
    result = session.get(
        url,
        etag=prev["etag"] if prev else None,
        last_modified=prev["last_modified"] if prev else None,
        allow_not_modified=True,
    )
    manifest_common = dict(
        run_id=run_id, source="house_clerk_index", chamber="house", url=url, method="GET",
        http_status=result.status, final_url=result.final_url,
        content_type=result.content_type, retry_count=result.retry_count,
        etag=result.etag, last_modified=result.last_modified,
    )
    if result.not_modified:
        store.record_fetch(**manifest_common, result_status="not_modified",
                           sha256=prev["sha256"] if prev else None)
        logger.info("house index %s unchanged (304)", year)
        return {"year": year, "status": "not_modified", "rows": 0, "ptrs": 0}
    if result.error_category or result.content is None:
        store.record_fetch(**manifest_common, result_status="failed",
                           error_category=result.error_category, error_detail=result.error_detail)
        raise RuntimeError(f"house index {year} fetch failed: {result.error_detail}")

    content = result.content
    if content[:2] != b"PK":
        store.record_fetch(**manifest_common, result_status="failed",
                           error_category="invalid_archive",
                           error_detail=f"not a ZIP (starts {content[:8]!r})")
        raise RuntimeError(f"house index {year}: response is not a ZIP archive")

    digest, _ = store_bytes(config.raw_store, content, ".zip")

    rows = parse_index_zip(content)
    if not rows:
        raise RuntimeError(f"house index {year}: parsed zero rows — source drift, failing loudly")

    n_ptr = 0
    filing_type_counts: dict[str, int] = {}
    for row in rows:
        ftype = row.get("FilingType", "")
        filing_type_counts[ftype] = filing_type_counts.get(ftype, 0) + 1
        doc_id = row.get("DocID", "")
        if not doc_id:
            continue
        fid = ids.filing_id("house", doc_id)
        is_ptr = ftype == PTR_FILING_TYPE
        if is_ptr:
            n_ptr += 1
        name = " ".join(
            p for p in (row.get("Prefix"), row.get("First"), row.get("Middle"),
                        row.get("Last"), row.get("Suffix")) if p
        )
        prior = store.conn.execute(
            "SELECT report_type FROM filings WHERE filing_id=?", (fid,)
        ).fetchone()
        # a PTR identified by the search endpoint is never downgraded here
        report_type = "ptr" if (is_ptr or (prior and prior["report_type"] == "ptr")) else "other"
        store.upsert_filing(
            fid,
            chamber="house",
            source_system="house_clerk",
            source_doc_id=doc_id,
            filer_name_raw=name,
            first_name_raw=row.get("First"),
            last_name_raw=row.get("Last"),
            prefix_raw=row.get("Prefix"),
            suffix_raw=row.get("Suffix"),
            state_district_raw=row.get("StateDst"),
            report_type_raw=ftype,
            report_type=report_type,
            report_year=int(row["Year"]) if row.get("Year", "").isdigit() else year,
            filing_date=_iso_date(row.get("FilingDate", "")),
            discovered_run_id=run_id,
        )
        if not is_ptr:
            # Only exclude a filing no other source claims is a PTR, and never
            # move one that has already been fetched or parsed: the Clerk
            # search endpoint types some filings as PTRs that the ZIP index
            # types otherwise (e.g. "PTR Waiver").
            current = store.conn.execute(
                "SELECT state, report_type FROM filings WHERE filing_id=?", (fid,)
            ).fetchone()
            if current and (
                current["report_type"] == "ptr" or current["state"] != "discovered"
            ):
                continue
            store.set_filing_state(fid, "excluded_report_type", f"FilingType={ftype}")

    # The manifest row is written only after every index row is persisted, so
    # an interrupted pass is not masked by a 304 on the next run.
    store.record_fetch(**manifest_common, result_status="ok", sha256=digest,
                       stored_suffix=".zip", content_length=len(content))
    logger.info(
        "house %s: %d index rows, %d PTRs, filing types %s",
        year, len(rows), n_ptr, dict(sorted(filing_type_counts.items())),
    )
    return {
        "year": year,
        "status": "ok",
        "rows": len(rows),
        "ptrs": n_ptr,
        "filing_types": filing_type_counts,
        "fetched_at": utc_now_iso(),
    }
