"""Senate report retrieval: landing pages plus linked scanned-page images.

Every filing's landing page is preserved even when it only points elsewhere.
Paper filings (/search/view/paper/{GUID}/) expose scanned page images
(img.filingImage); each image is fetched and stored. Only official
efdsearch.senate.gov URLs are followed (allowlist).

SEN-001: a paper filing is ``fetched`` only when **every** expected page image
was retrieved. Previously any single successful page marked the whole filing
fetched, so an incomplete scan was indistinguishable from a complete document.
The expected page set is declared in ``filing_page_ledger`` before any image is
requested, so a partial fetch is recoverable and a retry re-requests only the
pages that are missing.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

from ..config import OfficialConfig
from ..db import StateStore
from ..formats import classify_document
from ..http import PoliteSession
from ..storage import load_bytes, store_bytes

logger = logging.getLogger(__name__)

#: State for a paper filing whose page set is incomplete. Nonterminal, so
#: coverage reporting counts it as unfinished work and the release gate refuses
#: to publish while any filing sits here (freeze §9).
PARTIAL_FETCH_STATE = "fetch_partial_pending"


def _official_senate_host(host: str) -> bool:
    """Official Senate domains only (efdsearch, efd-media-public, ...)."""
    return host == "senate.gov" or host.endswith(".senate.gov")

_FILING_IMAGE = re.compile(
    r'<img[^>]+class=["\'][^"\']*filingImage[^"\']*["\'][^>]+src=["\']([^"\']+)["\']|'
    r'<img[^>]+src=["\']([^"\']+)["\'][^>]+class=["\'][^"\']*filingImage[^"\']*["\']'
)


def _expected_page_urls(html: str, base_url: str) -> list[str]:
    """Unique, order-preserving list of official page-image URLs in a landing page.

    SEN-001 requires the expected count to be the number of *unique* image URLs:
    the same scan can appear twice in the markup, and counting duplicates would
    make a complete filing look short.
    """
    raw = [next(g for g in m.groups() if g) for m in _FILING_IMAGE.finditer(html)]
    urls: list[str] = []
    for img_url in raw:
        full = urljoin(base_url, img_url)
        if not _official_senate_host(urlparse(full).netloc):
            logger.warning("skipping non-official image host: %s", full)
            continue
        if full not in urls:
            urls.append(full)
    return urls


def _finish_paper_filing(store: StateStore, fid: str) -> str:
    """Set the filing state from the page ledger. Fails closed on a short set."""
    expected, fetched = store.page_counts(fid)
    if expected and fetched < expected:
        store.set_filing_state(
            fid, PARTIAL_FETCH_STATE,
            f"paper filing: {fetched}/{expected} page images fetched",
        )
        return PARTIAL_FETCH_STATE
    detail = (
        f"landing page saved; {fetched}/{expected} page images" if expected
        else "landing page saved"
    )
    store.set_filing_state(fid, "fetched", detail)
    return "fetched"


def _fetch_pages(
    config: OfficialConfig,
    store: StateStore,
    session: PoliteSession,
    run_id: str,
    fid: str,
    urls: list[str],
) -> None:
    """Fetch the given page images, recording each outcome in the ledger."""
    for full in urls:
        img_res = session.get(full)
        img_manifest = dict(
            run_id=run_id, source="senate_efd_page_image", chamber="senate", url=full,
            method="GET", http_status=img_res.status, final_url=img_res.final_url,
            content_type=img_res.content_type, retry_count=img_res.retry_count,
            filing_id=fid,
        )
        if img_res.error_category or img_res.content is None:
            store.record_fetch(**img_manifest, result_status="failed",
                               error_category=img_res.error_category,
                               error_detail=img_res.error_detail)
            store.record_page_result(
                fid, full, "missing", error_category=img_res.error_category,
                error_detail=img_res.error_detail,
            )
            continue
        ext = ".gif" if img_res.content[:3] == b"GIF" else ".img"
        img_digest, _ = store_bytes(config.raw_store, img_res.content, ext)
        store.record_fetch(**img_manifest, result_status="ok", sha256=img_digest,
                           stored_suffix=ext, content_length=len(img_res.content))
        store.add_document(
            filing_id=fid, doc_role="page_image", url=full, sha256=img_digest,
            stored_suffix=ext, content_type=img_res.content_type, doc_format="image",
            format_confidence=1.0, format_evidence="filingImage element",
        )
        store.record_page_result(fid, full, "ok", sha256=img_digest, stored_suffix=ext)


def fetch_filing(
    config: OfficialConfig,
    store: StateStore,
    session: PoliteSession,
    run_id: str,
    filing_row,
) -> str:
    fid = filing_row["filing_id"]
    url = filing_row["source_url"]
    if not url:
        store.set_filing_state(fid, "fetch_retry_pending", "no source_url recorded")
        return "fetch_retry_pending"
    if not _official_senate_host(urlparse(url).netloc):
        store.set_filing_state(fid, "quarantined", f"non-allowlisted host: {url}")
        return "quarantined"

    existing = [d for d in store.documents_for(fid) if d["doc_role"] == "landing_html"]
    if existing and not config.force_refresh:
        # Landing page already archived. Reconcile against the expected page set
        # rather than trusting the recorded state: a run interrupted midway
        # through the images used to come back as a complete filing.
        if filing_row["state"] in {"discovered", "fetch_retry_pending", PARTIAL_FETCH_STATE}:
            expected, _ = store.page_counts(fid)
            if not expected:
                # Ledger predates SEN-001 (or the landing page had no images):
                # rebuild the expectation from the archived landing HTML.
                landing = existing[-1]
                try:
                    html = load_bytes(
                        config.raw_store, landing["sha256"], landing["stored_suffix"] or ""
                    ).decode("utf-8", errors="replace")
                except OSError:
                    html = ""
                urls = _expected_page_urls(html, url)
                if urls:
                    store.record_expected_pages(fid, urls)
                    # mark pages already on disk as ok
                    have = {d["url"] for d in store.documents_for(fid)
                            if d["doc_role"] == "page_image"}
                    for page_url in urls:
                        if page_url in have:
                            store.record_page_result(fid, page_url, "ok")
            missing = store.missing_pages(fid)
            if missing:
                _fetch_pages(config, store, session, run_id, fid, missing)
            return _finish_paper_filing(store, fid)
        return filing_row["state"]

    result = session.get(url)
    manifest = dict(
        run_id=run_id, source="senate_efd_report", chamber="senate", url=url, method="GET",
        http_status=result.status, final_url=result.final_url,
        content_type=result.content_type, retry_count=result.retry_count, filing_id=fid,
    )
    if result.error_category or result.content is None:
        store.record_fetch(**manifest, result_status="failed",
                           error_category=result.error_category, error_detail=result.error_detail)
        state = (
            "official_not_found" if result.error_category == "not_found"
            else "fetch_retry_pending"
        )
        store.set_filing_state(fid, state, f"{result.error_category}: {result.error_detail}")
        return state

    html = result.content.decode("utf-8", errors="replace")
    digest, _ = store_bytes(config.raw_store, result.content, ".html")
    store.record_fetch(**manifest, result_status="ok", sha256=digest, stored_suffix=".html",
                       content_length=len(result.content))
    fmt = classify_document(result.content, result.content_type)
    store.add_document(
        filing_id=fid, doc_role="landing_html", url=url, sha256=digest, stored_suffix=".html",
        content_type=result.content_type, doc_format=fmt.doc_format,
        format_confidence=fmt.confidence, format_evidence=fmt.evidence,
    )

    # Paper filings: the expected page set is declared BEFORE any image is
    # requested, so an interrupted run leaves a recoverable record of what is
    # still owed rather than a filing that looks complete.
    image_urls = _expected_page_urls(html, url)
    if image_urls:
        store.record_expected_pages(fid, image_urls)
        _fetch_pages(config, store, session, run_id, fid, image_urls)
    return _finish_paper_filing(store, fid)
