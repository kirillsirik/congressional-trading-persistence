"""House PTR document retrieval with response validation.

Filing URL convention (verified in reference research + live):
/public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf for PTRs. Some historical documents
live under /public_disc/financial-pdfs/{YEAR}/{DocID}.pdf; the fetcher tries
the PTR path first and falls back once, recording which succeeded.
"""

from __future__ import annotations

import logging

from ..config import OfficialConfig
from ..db import StateStore
from ..formats import classify_document
from ..http import PoliteSession
from ..storage import store_bytes

logger = logging.getLogger(__name__)


def ptr_pdf_urls(config: OfficialConfig, year: int, doc_id: str) -> list[str]:
    return [
        f"{config.house_base}/public_disc/ptr-pdfs/{year}/{doc_id}.pdf",
        f"{config.house_base}/public_disc/financial-pdfs/{year}/{doc_id}.pdf",
    ]


def fetch_filing(
    config: OfficialConfig,
    store: StateStore,
    session: PoliteSession,
    run_id: str,
    filing_row,
) -> str:
    """Fetch one House PTR PDF; returns the resulting filing state."""
    fid = filing_row["filing_id"]
    year = filing_row["report_year"]
    doc_id = filing_row["source_doc_id"]

    existing_docs = [d for d in store.documents_for(fid) if d["doc_role"] == "primary_pdf"]
    if existing_docs and not config.force_refresh:
        # The document is already archived. If a previous run was interrupted
        # between storing it and recording the state, the filing would
        # otherwise stay 'discovered' forever and never be parsed.
        if filing_row["state"] in {"discovered", "fetch_retry_pending"}:
            doc = existing_docs[-1]
            store.set_filing_state(
                fid, "fetched", f"{doc['doc_format']} (already archived)"
            )
            return "fetched"
        return filing_row["state"]

    last_error = None
    for url in ptr_pdf_urls(config, year, doc_id):
        result = session.get(url)
        manifest = dict(
            run_id=run_id, source="house_clerk_pdf", chamber="house", url=url, method="GET",
            http_status=result.status, final_url=result.final_url,
            content_type=result.content_type, retry_count=result.retry_count,
            etag=result.etag, last_modified=result.last_modified, filing_id=fid,
        )
        if result.error_category == "not_found":
            store.record_fetch(**manifest, result_status="failed",
                               error_category="not_found", error_detail=result.error_detail)
            last_error = ("official_not_found", f"404 at {url}")
            continue
        if result.error_category or result.content is None:
            store.record_fetch(**manifest, result_status="failed",
                               error_category=result.error_category,
                               error_detail=result.error_detail)
            last_error = ("fetch_retry_pending", f"{result.error_category}: {result.error_detail}")
            continue

        fmt = classify_document(result.content, result.content_type)
        if fmt.doc_format in {"structured_html", "malformed", "unavailable"}:
            # An HTML error page saved with a .pdf suffix is a fetch failure.
            store.record_fetch(**manifest, result_status="failed",
                               error_category="invalid_document",
                               error_detail=f"expected PDF, got {fmt.doc_format}: {fmt.evidence}")
            last_error = ("invalid_document", f"{fmt.doc_format} at {url}: {fmt.evidence}")
            continue

        digest, _ = store_bytes(config.raw_store, result.content, ".pdf")
        store.record_fetch(**manifest, result_status="ok", sha256=digest, stored_suffix=".pdf",
                           content_length=len(result.content))
        store.add_document(
            filing_id=fid, doc_role="primary_pdf", url=url, sha256=digest,
            stored_suffix=".pdf", content_type=result.content_type,
            doc_format=fmt.doc_format, format_confidence=fmt.confidence,
            format_evidence=fmt.evidence, page_count=fmt.page_count,
        )
        store.upsert_filing(fid, source_url=url)
        store.set_filing_state(fid, "fetched", f"{fmt.doc_format} ({fmt.page_count} pages)")
        return "fetched"

    state, detail = last_error if last_error else ("fetch_retry_pending", "no URL succeeded")
    store.set_filing_state(fid, state, detail)
    return state
