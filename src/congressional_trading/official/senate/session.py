"""Senate eFD session: lawful agreement acceptance + CSRF handling.

The public search home (https://efdsearch.senate.gov/search/home/) requires
agreeing to the Ethics in Government Act §105(c) restrictions before search.
This adapter:

- refuses to run unless ``SENATE_EFD_USAGE_ACKNOWLEDGED`` is set;
- preserves the agreement page and a hash of its text with every run;
- submits the site's own agreement form exactly as a browser would
  (``prohibition_agreement=1`` + Django CSRF token), with no evasion.

Source contract verified live 2026-07-24.
"""

from __future__ import annotations

import hashlib
import logging
import re

from ..config import OfficialConfig
from ..db import StateStore, utc_now_iso
from ..http import PoliteSession
from ..storage import store_bytes

logger = logging.getLogger(__name__)

HOME_PATH = "/search/home/"
SEARCH_PATH = "/search/"

_CSRF_INPUT = re.compile(r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']')
_AGREEMENT_MARKER = "prohibition_agreement"


class SenateAccessNotAcknowledged(RuntimeError):
    pass


def establish_session(
    config: OfficialConfig, store: StateStore, session: PoliteSession, run_id: str
) -> None:
    """Accept the eFD agreement on an authorized run; raises otherwise."""
    if not config.senate_usage_acknowledged:
        raise SenateAccessNotAcknowledged(
            "Senate eFD collection requires explicit acknowledgment of the statutory "
            "restrictions on use of financial-disclosure reports (5 U.S.C. app. §105(c)). "
            "Read the Senate eFD terms of use, confirm the project's use is "
            "permitted, then set SENATE_EFD_USAGE_ACKNOWLEDGED=true."
        )

    home_url = config.senate_base + HOME_PATH
    result = session.get(home_url)
    if result.error_category or result.content is None:
        raise RuntimeError(f"senate home fetch failed: {result.error_detail}")
    html = result.content.decode("utf-8", errors="replace")

    digest, _ = store_bytes(config.raw_store, result.content, ".html")
    agreement_hash = hashlib.sha256(html.encode()).hexdigest()
    store.record_fetch(
        run_id=run_id, source="senate_efd_home", chamber="senate", url=home_url, method="GET",
        http_status=result.status, final_url=result.final_url, content_type=result.content_type,
        retry_count=result.retry_count, sha256=digest, stored_suffix=".html",
        result_status="ok",
    )

    if _AGREEMENT_MARKER not in html:
        store.conn.execute(
            "INSERT INTO source_contract_checks (source, checked_at_utc, contract_sha256, detail, ok) "
            "VALUES (?,?,?,?,0)",
            ("senate_efd_home", utc_now_iso(), agreement_hash,
             "agreement form marker missing — source layout drift", ),
        )
        store.conn.commit()
        raise RuntimeError(
            "senate eFD home page no longer contains the expected agreement form; "
            "source contract drift — failing loudly (see source_contract_checks)"
        )

    m = _CSRF_INPUT.search(html)
    if not m:
        raise RuntimeError("senate eFD: csrfmiddlewaretoken not found on home page")
    token = m.group(1)

    post = session.post(
        home_url,
        data={"csrfmiddlewaretoken": token, "prohibition_agreement": "1"},
        headers={"Referer": home_url},
    )
    if post.error_category or post.status >= 400:
        raise RuntimeError(f"senate agreement submission failed: HTTP {post.status}")

    store.conn.execute(
        "INSERT INTO source_contract_checks (source, checked_at_utc, contract_sha256, detail, ok) "
        "VALUES (?,?,?,?,1)",
        ("senate_efd_home", utc_now_iso(), agreement_hash,
         "agreement accepted under documented research acknowledgment"),
    )
    store.conn.commit()
    logger.info("senate eFD session established (agreement text sha256=%s…)", agreement_hash[:12])


def csrf_token(session: PoliteSession, config: OfficialConfig) -> str | None:
    for cookie in session.session.cookies:
        if cookie.name == "csrftoken" and "senate.gov" in (cookie.domain or ""):
            return cookie.value
    return None
