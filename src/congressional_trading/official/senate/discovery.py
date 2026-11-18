"""Senate eFD search discovery: paginated, date-windowed PTR search.

Uses the site's DataTables JSON endpoint (POST /search/report/data/) after a
lawful agreement session. Discovery walks bounded received-date windows
(default quarterly) and recursively splits any window whose result count hits
the truncation guard, so completeness is provable from the window ledger.

Report-type code 11 = Periodic Transaction Report; filer types 1=Senator,
4=Candidate, 5=Former Senator (validated against the live form).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

from .. import ids
from ..config import OfficialConfig
from ..db import StateStore, utc_now_iso
from ..http import PoliteSession
from .session import csrf_token

logger = logging.getLogger(__name__)

DATA_PATH = "/search/report/data/"
SEARCH_REFERER = "/search/"

PTR_REPORT_TYPE = "11"
PAGE_SIZE = 100
WINDOW_SPLIT_THRESHOLD = 2500  # recursively split windows bigger than this

_LINK = re.compile(r'href=["\']([^"\']+)["\'][^>]*>(.*?)<', re.S)
_VIEW_KIND = re.compile(r"/search/view/(\w[\w-]*)/([0-9a-fA-F-]+)/")


def _fmt(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year} 00:00:00"


def search_page(
    config: OfficialConfig,
    session: PoliteSession,
    start: int,
    window_start: date,
    window_end: date,
    filer_types: str = "[]",
) -> dict:
    token = csrf_token(session, config)
    payload = {
        "start": str(start),
        "length": str(PAGE_SIZE),
        "report_types": f"[{PTR_REPORT_TYPE}]",
        "filer_types": filer_types,
        "submitted_start_date": _fmt(window_start),
        "submitted_end_date": _fmt(window_end),
        "candidate_state": "",
        "senator_state": "",
        "office_id": "",
        "first_name": "",
        "last_name": "",
        "csrfmiddlewaretoken": token or "",
    }
    result = session.post(
        config.senate_base + DATA_PATH,
        data=payload,
        headers={
            "Referer": config.senate_base + SEARCH_REFERER,
            "X-CSRFToken": token or "",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    if result.error_category or result.content is None:
        raise RuntimeError(f"senate search failed: HTTP {result.status} {result.error_detail}")
    try:
        return json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"senate search returned non-JSON (source drift?): {result.content[:200]!r}"
        ) from exc


def _parse_result_row(row: list) -> dict | None:
    """DataTables row: [first_name, last_name, office, report_link_html, date]."""
    if len(row) < 5:
        return None
    first, last, office, link_html, date_received = row[0], row[1], row[2], row[3], row[4]
    m = _LINK.search(link_html or "")
    if not m:
        return None
    href, title = m.group(1), " ".join(m.group(2).split())
    kind_m = _VIEW_KIND.search(href)
    kind, doc_uuid = (kind_m.group(1), kind_m.group(2)) if kind_m else (None, None)
    return {
        "first_name": (first or "").strip(),
        "last_name": (last or "").strip(),
        "office": (office or "").strip(),
        "href": href,
        "title": title,
        "view_kind": kind,        # ptr | paper | annual | extension-notice | ...
        "doc_uuid": doc_uuid,
        "date_received": (date_received or "").strip(),
    }


def _iso_us_date(text: str) -> str | None:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text.strip())
    if not m:
        return None
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{yy:04d}-{mm:02d}-{dd:02d}"


def discover_window(
    config: OfficialConfig,
    store: StateStore,
    session: PoliteSession,
    run_id: str,
    window_start: date,
    window_end: date,
    depth: int = 0,
) -> dict:
    """Discover all PTR filings received in [window_start, window_end)."""
    first_page = search_page(config, session, 0, window_start, window_end)
    total = int(first_page.get("recordsTotal", 0))
    store.record_fetch(
        run_id=run_id, source="senate_efd_search", chamber="senate",
        url=config.senate_base + DATA_PATH, method="POST",
        params_json=json.dumps({"window": [str(window_start), str(window_end)], "start": 0}),
        http_status=200, result_status="ok", content_length=None,
    )
    if total > WINDOW_SPLIT_THRESHOLD and (window_end - window_start).days > 1:
        mid = window_start + (window_end - window_start) / 2
        logger.info(
            "senate window %s→%s has %d results; splitting at %s",
            window_start, window_end, total, mid,
        )
        a = discover_window(config, store, session, run_id, window_start, mid, depth + 1)
        b = discover_window(config, store, session, run_id, mid, window_end, depth + 1)
        return {
            "window": [str(window_start), str(window_end)], "total": total, "split": True,
            "rows_upserted": a["rows_upserted"] + b["rows_upserted"],
        }

    seen: set[str] = set()
    undecodable: list[str] = []
    upserted = 0
    start = 0
    page = first_page
    while True:
        data = page.get("data", [])
        if not data and start == 0 and total > 0:
            raise RuntimeError("senate search: nonzero recordsTotal but empty first page")
        for row in data:
            parsed = _parse_result_row(row)
            if parsed is None or not parsed["doc_uuid"]:
                # an undecodable result row is a coverage gap, not a no-op
                undecodable.append(str(row)[:300])
                continue
            if parsed["doc_uuid"].lower() in seen:
                continue  # duplicates can appear across page boundaries
            seen.add(parsed["doc_uuid"].lower())
            fid = ids.filing_id("senate", parsed["doc_uuid"])
            title_l = parsed["title"].lower()
            store.upsert_filing(
                fid,
                chamber="senate",
                source_system="senate_efd",
                source_doc_id=parsed["doc_uuid"],
                filer_name_raw=f"{parsed['first_name']} {parsed['last_name']}".strip(),
                first_name_raw=parsed["first_name"],
                last_name_raw=parsed["last_name"],
                office_raw=parsed["office"],
                report_type_raw=parsed["title"],
                report_type="ptr",
                report_year=int(parsed["date_received"][-4:])
                if re.search(r"\d{4}$", parsed["date_received"]) else None,
                received_date=_iso_us_date(parsed["date_received"]),
                is_amendment=1 if "amendment" in title_l else 0,
                source_url=config.senate_base + parsed["href"],
                discovered_run_id=run_id,
            )
            upserted += 1
        start += PAGE_SIZE
        if start >= total:
            break
        page = search_page(config, session, start, window_start, window_end)

    # A shortfall means rows the official search reported were not collected
    # (result set changed mid-pagination, or markup drift). Re-query the
    # window once; if it persists, record it as a coverage gap rather than
    # letting a silent shortfall pass as success.
    shortfall = total - len(seen)
    if shortfall > 0 or undecodable:
        recheck = search_page(config, session, 0, window_start, window_end)
        recheck_total = int(recheck.get("recordsTotal", 0))
        still_short = recheck_total > len(seen)
        if still_short or undecodable:
            detail = (
                f"window {window_start}→{window_end}: recordsTotal={total} "
                f"(recheck {recheck_total}) but {len(seen)} unique rows collected; "
                f"{len(undecodable)} undecodable result rows"
            )
            logger.error("senate discovery shortfall — %s", detail)
            store.conn.execute(
                "INSERT INTO source_contract_checks "
                "(source, checked_at_utc, contract_sha256, detail, ok) VALUES (?,?,?,?,0)",
                ("senate_efd_search", utc_now_iso(), None, detail),
            )
            store.conn.commit()
            for raw in undecodable[:20]:
                store.add_review(
                    filing_id=f"senate-window-{window_start}",
                    chamber="senate",
                    reason="undecodable_search_result_row",
                    raw_text=raw,
                    severity="high",
                )
            raise RuntimeError(
                f"senate discovery incomplete — {detail}. Fix or narrow the window; "
                "a silent shortfall would understate coverage."
            )
    return {
        "window": [str(window_start), str(window_end)], "total": total,
        "split": False, "rows_upserted": upserted, "unique": len(seen),
        "undecodable_rows": len(undecodable),
    }


def quarterly_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cur = start
    while cur < end:
        nxt_month = ((cur.month - 1) // 3 + 1) * 3 + 1
        nxt = date(cur.year + (1 if nxt_month > 12 else 0), 1 if nxt_month > 12 else nxt_month, 1)
        windows.append((cur, min(nxt, end)))
        cur = nxt
    return windows


def discover_range(
    config: OfficialConfig,
    store: StateStore,
    session: PoliteSession,
    run_id: str,
    start: date,
    end: date | None = None,
) -> list[dict]:
    end = end or (date.today() + timedelta(days=1))
    results = []
    for w_start, w_end in quarterly_windows(start, end):
        results.append(discover_window(config, store, session, run_id, w_start, w_end))
    return results
