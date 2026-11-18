"""Parsing for Capitol Trades records: DOM row → raw dict → canonical record.

Two layers:

* **DOM layer** (``parse_trade_row`` and the ``parse_*_cell`` helpers): ported
  from the legacy scraper. They operate on duck-typed element handles (any
  object with ``inner_text`` / ``query_selector`` / ``get_attribute``), so this
  module imports fine without Playwright installed.
* **Canonical layer** (``canonicalize_records``): raw text fields → typed
  canonical schema (``schemas.CANONICAL_COLUMNS``), including relative-date
  resolution ("09:01 Yesterday"), size-bracket parsing ("1K–15K", "50M+",
  "$1,001-$15,000"), ticker cleaning, transaction/owner normalization and
  disclosure-lag cross-checking. Anything unusual is flagged in
  ``parser_warnings`` rather than silently dropped.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..config import SIZE_BRACKETS
from ..utils.identifiers import add_record_keys
from .schemas import (
    CANONICAL_COLUMNS,
    CONTENT_HASH_FIELDS,
    RAW_FIELDNAMES,
    SOURCE_NAME,
)

logger = logging.getLogger(__name__)

# Capitol Trades displays US Eastern local times; relative dates are resolved
# in this zone (falling back to UTC if tz data is unavailable).
_SITE_TIMEZONE = "America/New_York"

_MISSING_TOKENS = {"", "n/a", "na", "--", "-", "none", "null", "undisclosed"}

# ── DOM-layer helpers (ported from the original collector) ────────────────


def parse_politician_cell(cell: Any) -> dict[str, str]:
    """Politician cell: name link + party/chamber/state text."""
    result = {"politician_name": "", "party": "", "chamber": "", "state": "", "politician_url": ""}
    link = cell.query_selector("a")
    if link:
        result["politician_name"] = (link.inner_text() or "").strip()
        result["politician_url"] = link.get_attribute("href") or ""
    full_text = (cell.inner_text() or "").strip()
    for party in ("Democrat", "Republican", "Independent"):
        if party in full_text:
            result["party"] = party
            break
    for chamber in ("House", "Senate"):
        if chamber in full_text:
            result["chamber"] = chamber
            break
    state_match = re.search(r"\b([A-Z]{2})\s*$", full_text)
    if state_match:
        result["state"] = state_match.group(1)
    return result


def parse_issuer_cell(cell: Any) -> dict[str, str]:
    """Issuer cell: issuer name link + ticker like ``AAPL:US``."""
    result = {"issuer_name": "", "ticker": "", "issuer_url": ""}
    link = cell.query_selector("a")
    if link:
        result["issuer_name"] = (link.inner_text() or "").strip()
        result["issuer_url"] = link.get_attribute("href") or ""
    full_text = (cell.inner_text() or "").strip()
    ticker_match = re.search(r"([A-Z0-9.]+:[A-Z]{2})", full_text)
    if ticker_match:
        result["ticker"] = ticker_match.group(1)
    return result


def _cell_text(cell: Any) -> str:
    """Single-line text of a cell (date cells put time and date on two lines)."""
    return (cell.inner_text() or "").strip().replace("\n", " ")


def parse_trade_row(
    cells: list[Any],
    page_number: int,
    source_url: str,
    scraped_at_utc: str,
) -> dict[str, Any] | None:
    """One table row (list of ``<td>`` handles) → raw record dict.

    Column layout (legacy-verified): 0 politician, 1 issuer, 2 published,
    3 traded, 4 filed-after, 5 owner, 6 type, 7 size, 8 price. Returns None
    for header/malformed rows (fewer than 8 cells).
    """
    if len(cells) < 8:
        return None
    raw: dict[str, Any] = dict.fromkeys(RAW_FIELDNAMES, "")
    raw.update(parse_politician_cell(cells[0]))
    raw.update(parse_issuer_cell(cells[1]))
    raw["published_date"] = _cell_text(cells[2])
    raw["traded_date"] = _cell_text(cells[3])
    raw["filed_after_days"] = _cell_text(cells[4])
    raw["owner"] = _cell_text(cells[5])
    raw["tx_type"] = _cell_text(cells[6])
    raw["size"] = _cell_text(cells[7]) if len(cells) > 7 else ""
    raw["price"] = _cell_text(cells[8]) if len(cells) > 8 else ""
    raw["page"] = page_number
    raw["source_url"] = source_url
    raw["scraped_at_utc"] = scraped_at_utc
    return raw


# ── Scalar parsers (raw text → typed values) ─────────────────────────────────

_TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_AGO_RE = re.compile(r"^(\d+)\s*(min(?:ute)?s?|hours?|hrs?)\s+ago$", re.IGNORECASE)
_ABSOLUTE_FORMATS = (
    "%Y-%m-%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%m/%d/%Y",
)


def _to_site_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return dt.astimezone(ZoneInfo(_SITE_TIMEZONE))
    except Exception:  # pragma: no cover - tz database unavailable
        return dt


def parse_scraped_date(
    text: str | None, reference_utc: datetime | None = None
) -> tuple[str | None, str | None]:
    """Parse a scraped date cell to an ISO date string.

    Handles absolute forms ("2026-02-09", "5 Feb 2026", possibly with a
    leading/trailing "HH:MM" time) and relative forms ("Today", "Yesterday",
    "09:01 Yesterday", "23 min ago", "3 hours ago") resolved against
    ``reference_utc`` (the scrape timestamp) in the site's timezone.

    Returns ``(iso_date | None, warning | None)``; relative forms always carry
    a warning so downstream analysis knows the date was imputed.
    """
    if text is None:
        return None, None
    s = " ".join(str(text).split())
    if s.lower() in _MISSING_TOKENS:
        return None, None
    low = s.lower()

    def _resolve_relative(delta: timedelta, kind: str) -> tuple[str | None, str | None]:
        if reference_utc is None:
            return None, f"relative_date_unresolved({kind}):{s!r}"
        local = _to_site_tz(reference_utc) - delta
        return local.date().isoformat(), f"relative_date({kind}):{s!r}"

    ago = _AGO_RE.match(low)
    if ago:
        n, unit = int(ago.group(1)), ago.group(2)
        delta = timedelta(minutes=n) if unit.lower().startswith("min") else timedelta(hours=n)
        return _resolve_relative(delta, "ago")
    if "yesterday" in low:
        return _resolve_relative(timedelta(days=1), "yesterday")
    if "today" in low:
        return _resolve_relative(timedelta(days=0), "today")

    stripped = " ".join(_TIME_TOKEN_RE.sub(" ", s).split())
    for fmt in _ABSOLUTE_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).date().isoformat(), None
        except ValueError:
            continue
    parsed = pd.to_datetime(stripped, errors="coerce")
    if pd.notna(parsed):
        return parsed.date().isoformat(), None
    return None, f"unparseable_date:{s!r}"


_DASH_RE = re.compile(r"[‐‑‒–—―−-]")
_MULTIPLIERS = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
_NUM = r"\$?\s*([\d,]+(?:\.\d+)?)\s*([KMBkmb]?)"
_RANGE_RE = re.compile(rf"^{_NUM}\s*(?:–|to)\s*{_NUM}$")
_PLUS_RE = re.compile(rf"^{_NUM}\s*\+$")
_LT_RE = re.compile(rf"^<\s*{_NUM}$")


def _to_dollars(number: str, suffix: str) -> float:
    return float(number.replace(",", "")) * _MULTIPLIERS[suffix.upper()]


def parse_size_bracket(text: str | None) -> dict[str, Any]:
    """Parse a trade-size bracket into bounds, midpoints and a top-code flag.

    Supports the Capitol Trades display style ("1K–15K", "50M+", "< 1K",
    en-dash or hyphen) and the statutory dollar style ("$1,001-$15,000",
    "$50,000,001+"). Display brackets are first resolved through the
    methodology mapping ``config.SIZE_BRACKETS`` (so "1K–15K" → statutory
    bounds 1,001–15,000); unknown shapes fall back to a generic parse.
    """
    out: dict[str, Any] = {
        "size_lower": None,
        "size_upper": None,
        "size_midpoint_arithmetic": None,
        "size_midpoint_geometric": None,
        "size_top_coded_flag": 0,
        "warning": None,
    }
    if text is None:
        return out
    s = " ".join(str(text).split())
    if s.lower() in _MISSING_TOKENS:
        return out
    norm = _DASH_RE.sub("–", s.upper().replace(" ", "").replace("$", "").replace(",", ""))

    lower: float | None = None
    upper: float | None = None
    top_coded = False

    if norm in SIZE_BRACKETS:
        lo, hi = SIZE_BRACKETS[norm]
        lower, upper = float(lo), float(hi)
        top_coded = norm.endswith("+")
    else:
        compact = _DASH_RE.sub("–", s)
        if m := _RANGE_RE.match(compact):
            lower = _to_dollars(m.group(1), m.group(2))
            upper = _to_dollars(m.group(3), m.group(4))
        elif m := _PLUS_RE.match(compact):
            lower = upper = _to_dollars(m.group(1), m.group(2))
            top_coded = True
        elif m := _LT_RE.match(compact):
            lower, upper = 1.0, _to_dollars(m.group(1), m.group(2))
        else:
            out["warning"] = f"unparseable_size:{s!r}"
            return out

    if lower is not None and upper is not None and lower > upper:
        out["warning"] = f"inverted_size_bounds:{s!r}"
        lower, upper = upper, lower
    out["size_lower"] = lower
    out["size_upper"] = upper
    out["size_midpoint_arithmetic"] = (lower + upper) / 2.0
    out["size_midpoint_geometric"] = math.sqrt(max(lower, 1.0) * max(upper, 1.0))
    out["size_top_coded_flag"] = int(top_coded)
    return out


def map_transaction_type(text: str | None) -> tuple[str, int, str | None]:
    """Raw type → (clean type, buy/sell indicator +1/-1/0, warning)."""
    s = ("" if text is None else str(text)).strip().lower()
    if s.startswith(("buy", "purchase")):
        return "buy", 1, None
    if s.startswith(("sell", "sale")):
        return "sell", -1, None
    if s.startswith("exchange"):
        return "exchange", 0, None
    if s.startswith("receive"):
        return "receive", 0, None
    if s in _MISSING_TOKENS:
        return "other", 0, "missing_tx_type"
    return "other", 0, f"unmapped_tx_type:{s!r}"


def map_owner(text: str | None) -> tuple[str, str | None]:
    """Raw owner → (clean owner, warning)."""
    s = ("" if text is None else str(text)).strip().lower()
    for known in ("self", "spouse", "joint", "child", "dependent"):
        if s.startswith(known):
            return known, None
    if s in _MISSING_TOKENS or s.startswith("not disclosed"):
        return "undisclosed", None
    return "other", f"unmapped_owner:{s!r}"


def clean_ticker(text: str | None) -> str:
    """Strip the exchange suffix (``AAPL:US`` → ``AAPL``); missing → ''."""
    s = ("" if text is None else str(text)).strip()
    if s.lower() in _MISSING_TOKENS:
        return ""
    return s.split(":", 1)[0].strip()


def parse_price(text: str | None) -> float | None:
    s = ("" if text is None else str(text)).strip()
    if s.lower() in _MISSING_TOKENS:
        return None
    try:
        return float(s.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def slug_from_url(url: str | None) -> str:
    """Last path segment of a Capitol Trades URL (politician/issuer slug)."""
    s = ("" if url is None else str(url)).split("?", 1)[0].split("#", 1)[0].strip().rstrip("/")
    return s.rsplit("/", 1)[-1] if s else ""


_ETF_KEYWORDS = (
    "etf", "trust", "fund", "ishares", "spdr", "vanguard",
    "proshares", "vaneck", "wisdomtree", "invesco",
)  # fmt: skip


def derive_asset_type(ticker_clean: str, issuer_name: str) -> str:
    """Best-effort asset class: stock / etf_or_fund / unknown (no ticker)."""
    if not ticker_clean:
        return "unknown"
    issuer_low = (issuer_name or "").lower()
    if any(kw in issuer_low for kw in _ETF_KEYWORDS):
        return "etf_or_fund"
    return "stock"


def parse_disclosure_lag(
    filed_after_text: str | None,
    trade_date_iso: str | None,
    published_date_iso: str | None,
) -> tuple[int | None, str | None]:
    """Disclosure lag in days, cross-checked against published - traded.

    Prefers the site's "filed after" column when numeric (interface
    convention); warns when it disagrees with the date difference by more
    than one day, and falls back to the computed difference when absent.
    """
    raw_days: int | None = None
    if filed_after_text is not None:
        m = re.search(r"(\d+)", str(filed_after_text))
        if m:
            raw_days = int(m.group(1))
    computed: int | None = None
    if trade_date_iso and published_date_iso:
        computed = (
            datetime.fromisoformat(published_date_iso) - datetime.fromisoformat(trade_date_iso)
        ).days
    if raw_days is not None:
        if computed is not None and abs(computed - raw_days) > 1:
            return raw_days, f"lag_mismatch:filed_after={raw_days};published_minus_traded={computed}"
        return raw_days, None
    return computed, None


# ── Canonicalization ─────────────────────────────────────────────────────────


def _parse_reference_timestamp(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def canonicalize_record(raw: Mapping[str, Any], scrape_run_id: str) -> dict[str, Any]:
    """One raw record dict → canonical dict (without the hash key columns)."""
    warnings: list[str] = []

    def _warn(prefix: str, msg: str | None) -> None:
        if msg:
            warnings.append(f"{prefix}:{msg}")

    scraped_at = _parse_reference_timestamp(raw.get("scraped_at_utc"))

    trade_date, w = parse_scraped_date(raw.get("traded_date"), scraped_at)
    _warn("traded", w)
    published_date, w = parse_scraped_date(raw.get("published_date"), scraped_at)
    _warn("published", w)

    ticker_raw = ("" if raw.get("ticker") is None else str(raw.get("ticker"))).strip()
    ticker_clean = clean_ticker(ticker_raw)

    tx_clean, buy_sell, w = map_transaction_type(raw.get("tx_type"))
    _warn("tx_type", w)
    owner_clean, w = map_owner(raw.get("owner"))
    _warn("owner", w)

    size = parse_size_bracket(raw.get("size"))
    _warn("size", size["warning"])

    lag_days, w = parse_disclosure_lag(raw.get("filed_after_days"), trade_date, published_date)
    _warn("lag", w)
    if lag_days is not None and lag_days < 0:
        warnings.append(f"lag:negative_disclosure_lag:{lag_days}")

    page_raw = str(raw.get("page", "")).strip()
    try:
        page_number: int | None = int(float(page_raw)) if page_raw else None
    except ValueError:
        page_number = None

    return {
        "source": SOURCE_NAME,
        "source_url": str(raw.get("source_url", "") or ""),
        "scrape_run_id": scrape_run_id,
        "scrape_timestamp_utc": str(raw.get("scraped_at_utc", "") or ""),
        "page_number": page_number,
        "politician_name": str(raw.get("politician_name", "") or "").strip(),
        "politician_slug": slug_from_url(raw.get("politician_url")),
        "chamber": str(raw.get("chamber", "") or "").strip(),
        "party": str(raw.get("party", "") or "").strip(),
        "state": str(raw.get("state", "") or "").strip(),
        "issuer_name": str(raw.get("issuer_name", "") or "").strip(),
        "issuer_slug": slug_from_url(raw.get("issuer_url")),
        "ticker_raw": ticker_raw,
        "ticker_clean": ticker_clean,
        "asset_type": derive_asset_type(ticker_clean, str(raw.get("issuer_name", "") or "")),
        "transaction_type_raw": str(raw.get("tx_type", "") or "").strip(),
        "transaction_type_clean": tx_clean,
        "buy_sell_indicator": buy_sell,
        "owner_raw": str(raw.get("owner", "") or "").strip(),
        "owner_clean": owner_clean,
        "trade_date": trade_date,
        "published_date": published_date,
        "disclosure_lag_raw": str(raw.get("filed_after_days", "") or "").strip(),
        "disclosure_lag_days": lag_days,
        "size_raw": str(raw.get("size", "") or "").strip(),
        "size_lower": size["size_lower"],
        "size_upper": size["size_upper"],
        "size_midpoint_arithmetic": size["size_midpoint_arithmetic"],
        "size_midpoint_geometric": size["size_midpoint_geometric"],
        "size_top_coded_flag": size["size_top_coded_flag"],
        "price_raw": str(raw.get("price", "") or "").strip(),
        "price_numeric": parse_price(raw.get("price")),
        "filing_url": "",  # not exposed on the Capitol Trades list page
        "parser_warnings": ";".join(warnings),
    }


def canonicalize_records(
    raw_records: Iterable[Mapping[str, Any]], scrape_run_id: str
) -> pd.DataFrame:
    """Raw records → canonical DataFrame with stable keys and content hashes."""
    rows = [canonicalize_record(r, scrape_run_id) for r in raw_records]
    base_cols = [c for c in CANONICAL_COLUMNS if c not in ("stable_record_key", "row_content_hash")]
    df = pd.DataFrame(rows, columns=base_cols)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = add_record_keys(df, content_fields=CONTENT_HASH_FIELDS)
    n_warn = int((df["parser_warnings"] != "").sum())
    if n_warn:
        logger.info("canonicalize: %d of %d records carry parser warnings", n_warn, len(df))
    return df[CANONICAL_COLUMNS]
