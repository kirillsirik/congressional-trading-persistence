"""Polite HTTP client with per-host rate limiting, retries, conditional GET,
and a fetch-manifest record for every request.

Conduct rules: honest user agent with a
contact address, conservative delays, exponential backoff with jitter,
Retry-After respected, no evasion of access controls. 403/CAPTCHA responses
are recorded as terminal access failures, never retried around.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import requests

from .config import OfficialConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content: bytes | None
    content_type: str
    etag: str | None
    last_modified: str | None
    retry_count: int
    error_category: str | None  # None on success
    error_detail: str | None
    not_modified: bool = False
    elapsed_seconds: float = 0.0
    request_headers: dict = field(default_factory=dict)


class PoliteSession:
    """requests.Session wrapper enforcing per-host minimum request spacing."""

    def __init__(self, config: OfficialConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.user_agent
        self._last_request_at: dict[str, float] = {}

    def _throttle(self, host: str) -> None:
        last = self._last_request_at.get(host)
        interval = self.config.request_interval_seconds
        if last is not None:
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait + random.uniform(0, 0.25 * interval))
        self._last_request_at[host] = time.monotonic()

    def request(
        self,
        method: str,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        headers: dict | None = None,
        data: dict | None = None,
        allow_not_modified: bool = False,
    ) -> FetchResult:
        host = requests.utils.urlparse(url).netloc
        req_headers = dict(headers or {})
        if allow_not_modified and not self.config.force_refresh:
            if etag:
                req_headers["If-None-Match"] = etag
            if last_modified:
                req_headers["If-Modified-Since"] = last_modified

        attempt = 0
        start = time.monotonic()
        while True:
            self._throttle(host)
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=req_headers,
                    data=data,
                    timeout=(self.config.connect_timeout, self.config.read_timeout),
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                if attempt < self.config.max_retries:
                    self._backoff(attempt)
                    attempt += 1
                    continue
                return FetchResult(
                    url, url, 0, None, "", None, None, attempt,
                    "network_error", f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=time.monotonic() - start,
                    request_headers=req_headers,
                )

            if resp.status_code in RETRYABLE_STATUS and attempt < self.config.max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 300))
                else:
                    self._backoff(attempt)
                attempt += 1
                continue

            error_category = None
            error_detail = None
            if resp.status_code == 304:
                return FetchResult(
                    url, resp.url, 304, None, resp.headers.get("Content-Type", ""),
                    etag, last_modified, attempt, None, None, not_modified=True,
                    elapsed_seconds=time.monotonic() - start, request_headers=req_headers,
                )
            if resp.status_code == 404:
                error_category, error_detail = "not_found", "HTTP 404"
            elif resp.status_code == 403:
                error_category, error_detail = "access_denied", "HTTP 403 (never retried)"
            elif resp.status_code == 429:
                error_category, error_detail = "rate_limited", "HTTP 429 after retries"
            elif resp.status_code >= 400:
                error_category, error_detail = "http_error", f"HTTP {resp.status_code}"

            return FetchResult(
                url,
                resp.url,
                resp.status_code,
                resp.content if error_category is None else resp.content,
                resp.headers.get("Content-Type", ""),
                resp.headers.get("ETag"),
                resp.headers.get("Last-Modified"),
                attempt,
                error_category,
                error_detail,
                elapsed_seconds=time.monotonic() - start,
                request_headers=req_headers,
            )

    def _backoff(self, attempt: int) -> None:
        base = self.config.backoff_base_seconds * (2**attempt)
        time.sleep(base + random.uniform(0, base / 2))

    def get(self, url: str, **kwargs) -> FetchResult:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> FetchResult:
        return self.request("POST", url, **kwargs)
