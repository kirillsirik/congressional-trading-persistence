"""Clean the raw Capitol Trades scrape into canonical trade records.

Input: the 16-column raw scrape CSV (legacy scraper) or the canonical schema
produced by the new scraper (mapped via ``column_map``). Output: the canonical
trade-record columns.

Fixes over the legacy step scripts (see audit notes):

* Relative published dates ("Yesterday"/"Today"/"09:01 Yesterday") are imputed
  as ``trade_date + filed_after_days`` and flagged in ``parser_warning``
  instead of being silently dropped.
* Duplicates are *reported*: exact duplicates (same identity key and content)
  are dropped; conflicting duplicates (same identity key, different content —
  amended records) are flagged and kept.
* Power-committee augmentation has no hardcoded paths and generalizes the
  Congress-number computation.
* Nothing except unparseable-trade-date rows and exact duplicates is dropped
  here; all inclusion criteria become flags consumed by
  :mod:`sample_filters`, with a full attrition trail.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SIZE_BRACKETS, TOP_CODED_CAP
from ..utils.identifiers import add_record_keys
from ..utils.validation import require_columns

logger = logging.getLogger(__name__)

# ── Input schemas ────────────────────────────────────────────────────────────

RAW_SCRAPE_COLUMNS = [
    "politician_name", "party", "chamber", "state", "issuer_name", "ticker",
    "published_date", "traded_date", "filed_after_days", "owner", "tx_type",
    "size", "price", "politician_url", "issuer_url", "page",
]

# Canonical new-scraper column name -> legacy raw name consumed internally.
# (scraping/schemas.py may not exist yet; this mapping is the contract.)
CANONICAL_SCRAPER_COLUMN_MAP = {
    "trade_date": "traded_date",
    "transaction_type_raw": "tx_type",
    "size_raw": "size",
    "owner_raw": "owner",
    "ticker_raw": "ticker",
    "price_raw": "price",
    "disclosure_lag_days": "filed_after_days",
}

REQUIRED_RAW_COLUMNS = [
    "politician_name", "party", "chamber", "issuer_name", "ticker",
    "published_date", "traded_date", "owner", "tx_type", "size",
]

# ── Transaction type / owner mappings ────────────────────────────────────────

TX_TYPE_MAP = {
    "buy": "buy",
    "purchase": "buy",
    "sell": "sell",
    "sale": "sell",
    "sell (partial)": "sell",
    "sell (full)": "sell",
    "sell_partial": "sell",
    "sell_full": "sell",
    "exchange": "exchange",
    "receive": "receive",
}

OWNER_VALUES = {"self", "spouse", "joint", "child"}

# ── ETF / fund / non-equity screens (ported from legacy step2:180-203) ──────

ETF_FUND_KEYWORDS = [
    "ETF", "Fund", "Trust", "iShares", "Vanguard", "SPDR", "ProShares",
    "Invesco", "Schwab", "WisdomTree", "VanEck", " LP", " LLC",
    "BOND", "Treasury", "Municipal", "Fixed Income",
    "Futures", "Option", "Warrant",
]

COMMON_ETF_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA",
    "AGG", "BND", "LQD", "TLT", "HYG", "GLD", "SLV", "USO", "XLF",
    "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    "IVV", "IJH", "IJR", "VB", "VO", "VTV", "VUG", "ARKK", "ARKG",
    "SOXL", "TQQQ", "SQQQ", "SPXL", "TNA",
}

# Fields hashed to detect amended (conflicting-duplicate) records.
CONTENT_HASH_FIELDS = [
    "politician_name", "party", "chamber", "state", "issuer_name",
    "ticker_raw", "trade_date", "published_date", "transaction_type_raw",
    "owner_raw", "size_raw", "price_raw",
]

# ── Power committees (ported from legacy add_power_committee.py) ─────────────
# Key: (congress_number, chamber, committee_name) -> set of legislator names.
# 118th: Jan 3 2023 – Jan 3 2025; 119th: Jan 3 2025 – Jan 3 2027.

POWER_COMMITTEES: dict[tuple[int, str, str], set[str]] = {
    (118, "House", "Appropriations"): {
        "Tom Cole", "Hal Rogers", "Robert Aderholt", "Mike Simpson",
        "Chuck Fleischmann", "David Joyce", "Guy Reschenthaler",
        "Dan Newhouse", "Stephanie Bice", "Scott Franklin",
        "Michael Guest", "Mike Garcia",
        "Chellie Pingree", "Debbie Wasserman Schultz", "Ed Case",
        "Julia Letlow", "Lois Frankel", "John Rutherford",
        "Susie Lee", "Debbie Lesko",
    },
    (118, "House", "Ways and Means"): {
        "Adrian Smith", "Mike Kelly", "David Kustoff", "Ron Estes",
        "Kevin Hern", "Brad Schneider", "French Hill",
        "Terri Sewell", "Suzan DelBene", "Lloyd Doggett",
        "Linda Sanchez", "Judy Chu", "Don Beyer",
        "Earl Blumenauer", "Brad Sherman",
    },
    (118, "House", "Armed Services"): {
        "Adam Smith", "Rob Wittman", "Mike Garcia", "Doug Lamborn",
        "Austin Scott", "Michael Waltz", "Mark Green",
        "Scott DesJarlais", "Don Beyer", "Rick Larsen",
        "Seth Moulton", "Mikie Sherrill", "Pat Ryan",
        "Jared Moskowitz", "Jim Banks", "Dan Crenshaw",
        "Mike Flood", "Dale Strong", "Rich McCormick",
        "John James", "Gus Bilirakis",
    },
    (118, "House", "Energy and Commerce"): {
        "Cathy McMorris Rodgers", "Bob Latta", "Gus Bilirakis",
        "Michael Burgess", "Buddy Carter", "Neal Dunn",
        "Diana Harshbarger", "Debbie Lesko", "Greg Steube",
        "Rich McCormick", "Debbie Dingell", "Kathy Castor",
        "Doris Matsui", "Scott Peters", "Sean Casten",
        "Nanette Barragán", "Ann Wagner", "Darrell Issa",
        "Laurel Lee", "Rick Allen",
    },
    (118, "Senate", "Appropriations"): {
        "Susan Collins", "Mitch McConnell", "Lindsey Graham",
        "Jerry Moran", "John Boozman", "Shelley Moore Capito",
        "John Neely Kennedy", "Bill Hagerty", "Katie Britt",
        "Marco Rubio", "Pete Ricketts",
        "Chris Coons", "Gary Peters", "Joe Manchin",
    },
    (118, "Senate", "Finance"): {
        "Ron Wyden", "Michael Bennet", "Mark Warner",
        "Sheldon Whitehouse", "Tom Carper", "Tina Smith",
        "Rick Scott", "Bill Hagerty",
    },
    (118, "Senate", "Armed Services"): {
        "Dan Sullivan", "Tommy Tuberville", "Markwayne Mullin",
        "Rick Scott", "Katie Britt",
        "Richard Blumenthal", "Tammy Duckworth",
        "Angus King", "Mark Warner",
    },
    (118, "Senate", "Commerce"): {
        "Ted Cruz", "Cynthia Lummis", "Dan Sullivan",
        "Jerry Moran", "Shelley Moore Capito",
        "John Neely Kennedy", "Rick Scott",
        "Tammy Duckworth", "Gary Peters",
        "Richard Blumenthal", "John Hickenlooper",
        "Tina Smith", "John Fetterman",
    },
    (119, "House", "Appropriations"): {
        "Tom Cole", "Hal Rogers", "Robert Aderholt", "Mike Simpson",
        "Chuck Fleischmann", "David Joyce", "Guy Reschenthaler",
        "Dan Newhouse", "Stephanie Bice", "Scott Franklin",
        "Michael Guest", "Julia Letlow",
        "Chellie Pingree", "Debbie Wasserman Schultz", "Ed Case",
        "Lois Frankel", "Susie Lee", "John Rutherford",
    },
    (119, "House", "Ways and Means"): {
        "Adrian Smith", "Mike Kelly", "David Kustoff", "Ron Estes",
        "Kevin Hern", "Brad Schneider", "French Hill",
        "Terri Sewell", "Suzan DelBene", "Lloyd Doggett",
        "Linda Sanchez", "Judy Chu", "Don Beyer",
    },
    (119, "House", "Armed Services"): {
        "Adam Smith", "Rob Wittman", "Mike Garcia", "Doug Lamborn",
        "Austin Scott", "Mark Green",
        "Scott DesJarlais", "Don Beyer", "Rick Larsen",
        "Seth Moulton", "Mikie Sherrill", "Pat Ryan",
        "Dan Crenshaw", "Mike Flood", "Dale Strong",
        "Rich McCormick", "John James",
    },
    (119, "House", "Energy and Commerce"): {
        "Bob Latta", "Gus Bilirakis", "Michael Burgess",
        "Buddy Carter", "Neal Dunn", "Diana Harshbarger",
        "Greg Steube", "Rich McCormick",
        "Debbie Dingell", "Kathy Castor", "Doris Matsui",
        "Scott Peters", "Sean Casten", "Nanette Barragán",
        "Ann Wagner", "Darrell Issa", "Laurel Lee", "Rick Allen",
    },
    (119, "Senate", "Appropriations"): {
        "Susan Collins", "Mitch McConnell", "Lindsey Graham",
        "Jerry Moran", "John Boozman", "Shelley Moore Capito",
        "John Neely Kennedy", "Bill Hagerty", "Katie Britt",
        "Markwayne Mullin",
        "Chris Coons", "Gary Peters",
    },
    (119, "Senate", "Finance"): {
        "Ron Wyden", "Michael Bennet", "Mark Warner",
        "Sheldon Whitehouse", "Tina Smith",
        "Rick Scott", "Bill Hagerty",
    },
    (119, "Senate", "Armed Services"): {
        "Dan Sullivan", "Tommy Tuberville", "Markwayne Mullin",
        "Rick Scott", "Jim Banks", "Ashley Moody",
        "Richard Blumenthal", "Tammy Duckworth",
        "Angus King", "Mark Warner",
    },
    (119, "Senate", "Commerce"): {
        "Ted Cruz", "Cynthia Lummis", "Dan Sullivan",
        "Jerry Moran", "Shelley Moore Capito",
        "John Neely Kennedy", "Rick Scott",
        "John Curtis",
        "Tammy Duckworth", "Gary Peters",
        "Richard Blumenthal", "Tina Smith",
        "John Fetterman", "John Hickenlooper",
    },
}


@dataclass
class CleanTradesResult:
    """Output bundle of :func:`clean_capitol_trades`."""

    trades: pd.DataFrame
    attrition: pd.DataFrame
    duplicate_report: pd.DataFrame


# ── Date parsing ─────────────────────────────────────────────────────────────

_RELATIVE_DATE_RE = re.compile(r"\b(yesterday|today)\b", re.IGNORECASE)
_DATE_FORMATS = ("%d %b %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_date_scalar(value: object) -> pd.Timestamp:
    """Parse one Capitol Trades date string; relative dates return NaT."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip().replace("Sept", "Sep")
    if not s or _RELATIVE_DATE_RE.search(s):
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(s, dayfirst=True)
    except (ValueError, TypeError):
        return pd.NaT


def parse_date_series(series: pd.Series) -> pd.Series:
    """Vectorized mixed-format date parsing via a unique-value cache."""
    cache = {v: _parse_date_scalar(v) for v in series.dropna().unique()}
    return pd.to_datetime(series.map(cache))


# ── Ticker cleaning ──────────────────────────────────────────────────────────

_SHARE_CLASS_RE = re.compile(r"^([A-Z0-9]{2,})[.\-/ ]([A-Z])$")


def clean_ticker(raw: object) -> str | None:
    """Normalize a scraped ticker to CRSP style, or None if not a US listing.

    Rules: strip the ``:US`` country suffix; any other ``:XX`` suffix marks a
    non-US listing (None); uppercase; normalize share-class separators so that
    ``BRK/B``, ``BRK-B``, ``BRK B`` and ``BRK.B`` all become ``BRK.B``.

    Note on share classes: CRSP v2 stores share classes inconsistently across
    the ``Ticker`` and ``TradingSymbol`` columns, so the matcher builds its
    lookup from BOTH columns and the hand-curated recovery map covers the
    remaining corporate-action cases (e.g. Capitol Trades "B" for BRK.B).
    """
    if pd.isna(raw):
        return None
    t = str(raw).strip().upper()
    if not t:
        return None
    if t.endswith(":US"):
        t = t[:-3].strip()
    if ":" in t:
        return None
    m = _SHARE_CLASS_RE.match(t)
    if m:
        t = f"{m.group(1)}.{m.group(2)}"
    if not t or not t.replace(".", "").isalnum():
        return None
    return t


# ── Size bracket parsing ─────────────────────────────────────────────────────


def _normalize_bracket(raw: object) -> str:
    s = re.sub(r"\s+", "", str(raw))
    return (
        s.replace("–", "-").replace("—", "-")
        .replace("$", "").replace(",", "").upper()
    )


def _known_brackets() -> dict[str, tuple[float, float, bool]]:
    known = {
        _normalize_bracket(label): (float(lo), float(hi), label.endswith("+"))
        for label, (lo, hi) in SIZE_BRACKETS.items()
    }
    # Official dollar-form aliases ("$1,001 - $15,000" → "1001-15000") map to
    # the same brackets so both source formats parse without warnings.
    for label, (lo, hi) in SIZE_BRACKETS.items():
        top = label.endswith("+")
        known[f"{int(lo)}-{int(hi)}"] = (float(lo), float(hi), top)
        if top:
            known[f"{int(lo)}+"] = (float(lo), float(hi), True)
    # Rare sub-1K bracket seen in legacy data ("< 1K"): bounds $1–$1,000.
    known["<1K"] = (1.0, 1000.0, False)
    return known


_BRACKET_RE = re.compile(
    r"^(?P<lt><)?(?P<lo>\d+(?:\.\d+)?)(?P<lo_unit>[KM]?)"
    r"(?:-(?P<hi>\d+(?:\.\d+)?)(?P<hi_unit>[KM]?))?(?P<plus>\+)?$"
)
_UNIT_MULT = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0}


def parse_size_bracket(raw: object) -> tuple[float, float, int, bool]:
    """Parse a size bracket into ``(lower, upper, top_coded_flag, warned)``.

    Supports both the scraped ``1K–15K`` format (en-dash or hyphen) and the
    official ``$1,001 - $15,000`` format. Known brackets come from
    ``config.SIZE_BRACKETS``; anything else falls through to a generic regex
    (``warned=True``). Open-ended brackets (``…+``) are top-coded: unknown
    open-ended brackets get ``upper = max(lower, TOP_CODED_CAP)``.
    """
    if pd.isna(raw) or not str(raw).strip():
        return (np.nan, np.nan, 0, False)
    norm = _normalize_bracket(raw)
    known = _KNOWN_BRACKETS.get(norm)
    if known is not None:
        lo, hi, top = known
        return (lo, hi, int(top), False)
    m = _BRACKET_RE.match(norm)
    if not m:
        return (np.nan, np.nan, 0, True)
    lo = float(m.group("lo")) * _UNIT_MULT[m.group("lo_unit")]
    if m.group("lt"):
        return (1.0, lo, 0, True)
    if m.group("hi") is not None:
        hi = float(m.group("hi")) * _UNIT_MULT[m.group("hi_unit")]
        return (lo, hi, 0, True)
    if m.group("plus"):
        return (lo, max(lo, float(TOP_CODED_CAP)), 1, True)
    return (lo, lo, 0, True)


_KNOWN_BRACKETS = _known_brackets()


# ── Power committee helpers ──────────────────────────────────────────────────


def normalize_person_name(name: object) -> str:
    """Accent-insensitive, whitespace-collapsed lowercase name key."""
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def get_congress_number(date: pd.Timestamp) -> int:
    """Congress number for a date (sessions start Jan 3 of odd years)."""
    year = date.year
    if year % 2 == 1 and (date.month, date.day) < (1, 3):
        year -= 1
    return (year - 1789) // 2 + 1


def build_power_lookup() -> dict[tuple[int, str], list[str]]:
    """Flatten POWER_COMMITTEES to (congress, normalized name) -> committees."""
    lookup: dict[tuple[int, str], list[str]] = {}
    for (congress, _chamber, committee), members in POWER_COMMITTEES.items():
        for name in members:
            lookup.setdefault((congress, normalize_person_name(name)), []).append(committee)
    return lookup


def add_power_committee(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    """Attach ``power_committee`` (0/1) and ``power_committee_names``.

    Membership is evaluated at the trade date's Congress (contract:
    "committee membership at trade date").
    """
    lookup = build_power_lookup()
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    congresses = dates.map(lambda d: get_congress_number(d) if pd.notna(d) else -1)
    names = df["politician_name"].map(normalize_person_name)
    committees = [
        lookup.get((c, n), []) for c, n in zip(congresses, names, strict=True)
    ]
    df["power_committee"] = [int(bool(c)) for c in committees]
    df["power_committee_names"] = ["|".join(c) for c in committees]
    return df


# ── Attrition helper ─────────────────────────────────────────────────────────


def attrition_row(stage: str, n_dropped: int, n_remaining: int, reason: str) -> dict:
    return {
        "stage": stage,
        "n_dropped": int(n_dropped),
        "n_remaining": int(n_remaining),
        "reason": reason,
    }


# ── Main entry point ─────────────────────────────────────────────────────────


def clean_capitol_trades(
    raw: pd.DataFrame,
    config: dict,
    column_map: dict[str, str] | None = None,
) -> CleanTradesResult:
    """Clean a raw Capitol Trades scrape into canonical trade records.

    Parameters
    ----------
    raw : raw scrape DataFrame (16-column legacy schema or new-scraper schema).
    config : merged run config (used for size-bracket constants indirectly and
        kept in the signature per the module-interface convention).
    column_map : optional mapping ``{input column -> legacy raw column}``. If
        omitted and the frame looks like the new scraper schema, the built-in
        :data:`CANONICAL_SCRAPER_COLUMN_MAP` is applied automatically.
    """
    del config  # cleaning is fully determined by methodology constants
    df = raw.copy()

    if column_map:
        df = df.rename(columns=column_map)
    elif "traded_date" not in df.columns and "trade_date" in df.columns:
        df = df.rename(columns=CANONICAL_SCRAPER_COLUMN_MAP)
    require_columns(df, REQUIRED_RAW_COLUMNS, context="clean_capitol_trades input")
    if "state" not in df.columns:
        df["state"] = pd.NA
    if "price" not in df.columns:
        df["price"] = pd.NA

    attrition: list[dict] = [attrition_row("raw_scrape_rows", 0, len(df), "raw input")]
    warn = pd.Series("", index=df.index, dtype=object)

    def add_warning(mask: pd.Series, tag: str) -> None:
        idx = warn.index[mask.fillna(False).astype(bool)]
        warn.loc[idx] = np.where(warn.loc[idx] == "", tag, warn.loc[idx] + ";" + tag)

    # 1. Dates ---------------------------------------------------------------
    df["trade_date"] = parse_date_series(df["traded_date"].astype(str))
    df["published_date"] = parse_date_series(df["published_date"].astype(str))
    filed_after_raw = (
        df["filed_after_days"]
        if "filed_after_days" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    filed_after = pd.to_numeric(filed_after_raw, errors="coerce")

    # Impute relative published dates ("Yesterday"/"Today"/"09:01 Yesterday")
    # as trade_date + filed_after_days; flag them.
    raw_published = raw.get("published_date", pd.Series("", index=df.index)).astype(str)
    relative_mask = (
        df["published_date"].isna()
        & raw_published.str.contains(_RELATIVE_DATE_RE, na=False)
        & df["trade_date"].notna()
        & filed_after.notna()
    )
    if relative_mask.any():
        df.loc[relative_mask, "published_date"] = df.loc[relative_mask, "trade_date"] + pd.to_timedelta(
            filed_after[relative_mask], unit="D"
        )
        add_warning(relative_mask, "published_date_imputed_from_trade_date_plus_lag")
        logger.info("Imputed %d relative published dates", int(relative_mask.sum()))

    # Drop rows whose trade date cannot be parsed (nothing downstream works).
    bad_trade_date = df["trade_date"].isna()
    if bad_trade_date.any():
        df = df.loc[~bad_trade_date].copy()
        warn = warn.loc[df.index]
        filed_after = filed_after.loc[df.index]
    attrition.append(
        attrition_row(
            "drop_unparseable_trade_date", int(bad_trade_date.sum()), len(df),
            "traded_date could not be parsed",
        )
    )
    add_warning(df["published_date"].isna(), "published_date_unresolved")

    # 2. Canonical fields ----------------------------------------------------
    df["ticker_raw"] = df["ticker"]
    ticker_cache = {v: clean_ticker(v) for v in df["ticker"].dropna().unique()}
    df["ticker_clean"] = df["ticker"].map(ticker_cache)

    df["politician_name"] = df["politician_name"].astype(str).str.strip()
    party = df["party"].astype(str).str.strip().str.title()
    df["party"] = party.where(party.isin(["Democrat", "Republican"]), "Other")
    df["chamber"] = df["chamber"].astype(str).str.strip().str.title()
    df["issuer_name"] = df["issuer_name"].astype(str).str.strip()

    df["transaction_type_raw"] = df["tx_type"]
    tx_norm = df["tx_type"].astype(str).str.strip().str.lower()
    df["transaction_type_clean"] = tx_norm.map(TX_TYPE_MAP).fillna("other")
    df["buy_indicator"] = (df["transaction_type_clean"] == "buy").astype(int)

    df["owner_raw"] = df["owner"]
    owner_norm = df["owner"].astype(str).str.strip().str.lower()
    df["owner_clean"] = owner_norm.where(owner_norm.isin(OWNER_VALUES), "undisclosed")

    # Size brackets (both "1K–15K" and "$1,001-$15,000" formats).
    df["size_raw"] = df["size"]
    size_cache = {v: parse_size_bracket(v) for v in df["size"].dropna().unique()}
    parsed = df["size"].map(lambda v: size_cache.get(v, (np.nan, np.nan, 0, False)))
    df["size_lower"] = [p[0] for p in parsed]
    df["size_upper"] = [p[1] for p in parsed]
    df["size_top_coded_flag"] = [int(p[2]) for p in parsed]
    generic = pd.Series([bool(p[3]) for p in parsed], index=df.index)
    if generic.any():
        add_warning(generic, "size_bracket_generic_parse")
        logger.warning(
            "Parsed %d rows with non-standard size brackets generically: %s",
            int(generic.sum()),
            sorted(df.loc[generic, "size_raw"].astype(str).unique())[:10],
        )
    lower = df["size_lower"].clip(lower=1.0)
    upper = df["size_upper"].clip(lower=1.0)
    df["size_midpoint_geometric"] = np.sqrt(lower * upper)
    df["size_midpoint_arithmetic"] = (df["size_lower"] + df["size_upper"]) / 2.0

    # Legislator id from the Capitol Trades politician URL slug.
    slug = (
        df.get("politician_url", pd.Series(pd.NA, index=df.index))
        .astype(str)
        .str.extract(r"/politicians/([^/?#\s]+)", expand=False)
    )
    fallback = df["politician_name"].map(normalize_person_name).str.replace(" ", "_")
    df["legislator_id"] = slug.fillna(fallback)
    df.loc[df["legislator_id"].astype(str).str.strip().eq(""), "legislator_id"] = fallback

    # Disclosure lag: Capitol Trades filed_after when present (paper), else
    # calendar days between publication and execution.
    computed_lag = (df["published_date"] - df["trade_date"]).dt.days
    df["disclosure_lag_days"] = filed_after.fillna(computed_lag).astype("Int64")

    df["filing_date"] = pd.NaT  # official filing date not separately scraped
    df["senate_indicator"] = (df["chamber"] == "Senate").astype(int)
    df["republican_indicator"] = (df["party"] == "Republican").astype(int)
    df["price_raw"] = df["price"]

    # ETF / fund / non-equity screen (flag only; filtered in sample_filters).
    keyword_hit = df["issuer_name"].str.contains(
        "|".join(ETF_FUND_KEYWORDS), case=False, na=False, regex=True
    )
    etf_ticker_hit = df["ticker_clean"].isin(COMMON_ETF_TICKERS)
    df["is_common_stock_candidate"] = (
        df["ticker_clean"].notna() & ~keyword_hit & ~etf_ticker_hit
    ).astype(int)

    df = add_power_committee(df, date_col="trade_date")
    df["parser_warning"] = warn

    # 3. Duplicates ----------------------------------------------------------
    df = add_record_keys(df, content_fields=[c for c in CONTENT_HASH_FIELDS if c in df.columns])
    exact_dup = df.duplicated(subset=["stable_record_key", "row_content_hash"], keep="first")
    hash_variety = df.groupby("stable_record_key")["row_content_hash"].transform("nunique")
    conflicting = hash_variety > 1
    df["conflicting_duplicate_flag"] = conflicting.astype(int)

    report_cols = [
        "stable_record_key", "row_content_hash", "politician_name", "ticker_raw",
        "ticker_clean", "issuer_name", "trade_date", "published_date",
        "transaction_type_raw", "owner_raw", "size_raw", "price_raw",
    ]
    duplicate_report = df.loc[exact_dup | conflicting, report_cols].copy()
    duplicate_report["duplicate_type"] = np.where(
        exact_dup[exact_dup | conflicting], "exact", "conflicting"
    )
    duplicate_report = duplicate_report.sort_values(
        ["stable_record_key", "duplicate_type"]
    ).reset_index(drop=True)

    n_exact = int(exact_dup.sum())
    df = df.loc[~exact_dup].copy()
    attrition.append(
        attrition_row(
            "drop_exact_duplicates", n_exact, len(df),
            "identical stable_record_key and content hash (scrape overlap)",
        )
    )
    if conflicting.any():
        logger.warning(
            "%d records share an identity key with different content (kept, flagged)",
            int(df["conflicting_duplicate_flag"].sum()),
        )

    # 4. Final layout ----------------------------------------------------------
    df = df.sort_values(["published_date", "trade_date", "stable_record_key"], na_position="last")
    df = df.reset_index(drop=True)
    df["trade_id"] = np.arange(len(df), dtype=np.int64)

    canonical_cols = [
        "trade_id", "stable_record_key", "politician_name", "legislator_id",
        "party", "chamber", "state", "issuer_name", "ticker_raw", "ticker_clean",
        "trade_date", "published_date", "filing_date", "disclosure_lag_days",
        "transaction_type_raw", "transaction_type_clean", "buy_indicator",
        "owner_raw", "owner_clean", "size_raw", "size_lower", "size_upper",
        "size_midpoint_geometric", "size_midpoint_arithmetic",
        "size_top_coded_flag", "senate_indicator", "republican_indicator",
        "power_committee", "is_common_stock_candidate",
    ]
    extra_cols = [
        "power_committee_names", "price_raw", "parser_warning",
        "row_content_hash", "conflicting_duplicate_flag",
    ]
    trades = df[canonical_cols + extra_cols].copy()

    logger.info(
        "Clean trades: %d rows (%d exact duplicates dropped, %d conflicting flagged)",
        len(trades), n_exact, int(trades["conflicting_duplicate_flag"].sum()),
    )
    return CleanTradesResult(
        trades=trades,
        attrition=pd.DataFrame(attrition),
        duplicate_report=duplicate_report,
    )


def geometric_midpoint(lower: float, upper: float) -> float:
    """Geometric midpoint of a bracket (paper's primary size measure)."""
    return math.sqrt(max(lower, 1.0) * max(upper, 1.0))
