"""Normalization: owners, transaction types, amounts, dates, tickers.

Raw values are always preserved next to normalized values; unknown beats
guessed. Amounts are ranges — a midpoint is always labeled derived.
"""

from __future__ import annotations

import re
from datetime import date, datetime

NORMALIZATION_VERSION = "1"

# ── Owner codes (House form codes + Senate labels; official codebooks) ───────
OWNER_MAP = {
    "": "self",
    "self": "self",
    "sp": "spouse",
    "spouse": "spouse",
    "dc": "dependent_child",
    "child": "dependent_child",
    "dependent child": "dependent_child",
    "jt": "joint",
    "joint": "joint",
    "dj": "joint",  # observed rare House variant
    "trust": "trust",
}

OWNER_VALUES = {
    "self", "spouse", "dependent_child", "joint", "trust", "multiple",
    "not_reported", "unknown",
}


def normalize_owner(raw: str | None, chamber: str | None = None) -> str:
    """House PTR forms mark only non-self owners (SP/DC/JT); a blank owner
    cell on a House form means the filer (official form convention). Senate
    forms label owners explicitly, so blank stays not_reported."""
    if raw is None:
        return "self" if chamber == "house" else "not_reported"
    key = " ".join(str(raw).replace("\x00", "").split()).lower()
    if key == "":
        return "self" if chamber == "house" else "not_reported"
    return OWNER_MAP.get(key, "unknown")


# ── Transaction types ────────────────────────────────────────────────────────
def normalize_transaction_type(raw: str | None) -> tuple[str, bool]:
    """Returns (normalized_type, is_partial_sale)."""
    if raw is None:
        return "unknown", False
    text = " ".join(str(raw).replace("\x00", "").split()).lower()
    partial = "partial" in text
    head = text.split("(")[0].strip()
    if head in {"p", "purchase", "buy"}:
        return "purchase", False
    if head in {"s", "sale", "sell", "sale (full)"} or text.startswith("sale"):
        return ("sale_partial" if partial else "sale_full"), partial
    if head in {"e", "exchange"}:
        return "exchange", False
    if text:
        return "other", partial
    return "unknown", False


# ── Amount ranges ────────────────────────────────────────────────────────────
# Canonical categories: (lower, upper, open_ended). Bounds per the official
# form brackets; None upper = open-ended.
AMOUNT_CATEGORIES: dict[str, tuple[int, int | None, bool]] = {
    "$1 - $1,000": (1, 1000, False),
    "$1,001 - $15,000": (1001, 15000, False),
    "$15,001 - $50,000": (15001, 50000, False),
    "$50,001 - $100,000": (50001, 100000, False),
    "$100,001 - $250,000": (100001, 250000, False),
    "$250,001 - $500,000": (250001, 500000, False),
    "$500,001 - $1,000,000": (500001, 1000000, False),
    "$1,000,001 - $5,000,000": (1000001, 5000000, False),
    "$5,000,001 - $25,000,000": (5000001, 25000000, False),
    "$25,000,001 - $50,000,000": (25000001, 50000000, False),
    "Over $1,000,000": (1000001, None, True),  # historical House top bracket
    "Over $5,000,000": (5000001, None, True),
    "Over $25,000,000": (25000001, None, True),
    "Over $50,000,000": (50000001, None, True),
    "$1,000,000 +": (1000001, None, True),
    "$1,000,001 +": (1000001, None, True),
    "$5,000,001 +": (5000001, None, True),
    "$50,000,001 +": (50000001, None, True),
}

# Existing-project bracket labels (config.SIZE_BRACKETS) keyed by lower bound.
_EXISTING_BRACKET_BY_LOWER = {
    1001: "1K–15K", 15001: "15K–50K", 50001: "50K–100K", 100001: "100K–250K",
    250001: "250K–500K", 500001: "500K–1M", 1000001: "1M–5M",
    5000001: "5M–25M", 25000001: "25M–50M", 50000001: "50M+",
}

# Source PDFs use several Unicode dashes (incl. U+00AD soft hyphen) as the
# range separator; all are normalized before matching.
_DASHES = "-­‐‑‒–—―−"
_DASH_RE = re.compile(f"[{_DASHES}]")

# Owner-column header text bleeds into the amount cell on some older House
# forms ("Spouse/DC Over $1,000,000"); the owner itself is captured from its
# own column, so the bleed is stripped rather than interpreted.
_OWNER_BLEED = re.compile(r"^(?:spouse\s*/\s*dc|self|spouse|joint)\s+(?=over\b|\$)", re.I)

# The House form's checkbox glyph renders as literal "gfedc"/"gfedcb" text and
# can land between the range bounds.
_CHECKBOX_RE = re.compile(r"\bgfedcb?\b", re.I)

_AMOUNT_RANGE = re.compile(r"\$\s*([\d,]+)\s*-\s*\$?\s*([\d,]+)")
_AMOUNT_OPEN = re.compile(r"(?:Over\s*)?\$\s*([\d,]+)\s*\+?", re.I)


def normalize_amount(raw: str | None) -> dict:
    """Returns amount_category, amount_min, amount_max, amount_open_ended.

    Unknown categories return all-null with category 'unknown' — callers must
    route those rows to review, never guess.
    """
    out = {"amount_category": None, "amount_min": None, "amount_max": None,
           "amount_open_ended": None}
    if raw is None:
        return out
    text = " ".join(str(raw).replace("\x00", "").split())
    text = _DASH_RE.sub("-", text)
    text = _CHECKBOX_RE.sub(" ", text)
    # The House ePTR amount column's only spouse/DC top-code option reads
    # "Spouse/DC Over $1,000,000"; pdf_table extraction can lose the wrapped
    # amount, leaving the bare prefix. Restoring the fixed form label is a
    # form-vocabulary fact, not an amount inference.
    if _canon_amount(text) == "spouse/dcover":
        text = "Spouse/DC Over $1,000,000"
    text = _OWNER_BLEED.sub("", text).strip()
    text = " ".join(text.split())
    if not text or text.lower() == "none":
        return out
    for label, (lo, hi, open_ended) in AMOUNT_CATEGORIES.items():
        if _canon_amount(label) == _canon_amount(text):
            out.update(amount_category=label, amount_min=lo, amount_max=hi,
                       amount_open_ended=open_ended)
            return out
    m = _AMOUNT_RANGE.search(text)
    if m:
        lo = int(m.group(1).replace(",", ""))
        hi = int(m.group(2).replace(",", ""))
        if lo <= hi:
            out.update(amount_category=f"${lo:,} - ${hi:,}", amount_min=lo, amount_max=hi,
                       amount_open_ended=False)
            return out
    m = _AMOUNT_OPEN.fullmatch(text.strip())
    if m and ("over" in text.lower() or "+" in text):
        lo = int(m.group(1).replace(",", ""))
        out.update(amount_category=text, amount_min=lo, amount_max=None,
                   amount_open_ended=True)
        return out
    out["amount_category"] = "unknown"
    return out


def _canon_amount(text: str) -> str:
    return re.sub(r"[\s,]", "", _DASH_RE.sub("-", text)).lower()


def existing_bracket_label(amount_min: int | None) -> str | None:
    """Existing-project size_raw label for a canonical bracket, else None."""
    if amount_min is None:
        return None
    return _EXISTING_BRACKET_BY_LOWER.get(amount_min)


def amount_midpoints(amount_min, amount_max, top_coded_cap: int = 5_000_000) -> dict:
    """Derived midpoints matching the existing project's conventions.

    Open-ended brackets are top-coded at the existing TOP_CODED_CAP for the
    compat export; the flag records it. Never presented as a reported value.
    """
    out = {"size_midpoint_arithmetic": None, "size_midpoint_geometric": None,
           "size_top_coded_flag": 0}
    if amount_min is None:
        return out
    if amount_max is None:
        out["size_top_coded_flag"] = 1
        cap = max(top_coded_cap, amount_min)
        out["size_midpoint_arithmetic"] = float(cap)
        out["size_midpoint_geometric"] = float(cap)
        return out
    out["size_midpoint_arithmetic"] = (amount_min + amount_max) / 2
    out["size_midpoint_geometric"] = (amount_min * amount_max) ** 0.5
    return out


# ── Dates ────────────────────────────────────────────────────────────────────
_DATE_FORMATS = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y"]


def normalize_date(raw: str | None, min_year: int = 1990, max_year: int = 2100) -> str | None:
    if raw is None:
        return None
    text = " ".join(str(raw).replace("\x00", "").split())
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(text, fmt).date()
            if min_year <= d.year <= max_year:
                return d.isoformat()
            return None
        except ValueError:
            continue
    return None


def _as_iso(value: object) -> str | None:
    """Accept str/None/NaN/date-like; return an ISO date string or None."""
    if value is None or not isinstance(value, str):
        # pandas hands us float('nan') for missing values, and Timestamps
        # when a column was parsed as datetime
        text = "" if value is None else str(value)
        if text in {"", "nan", "NaT", "None"}:
            return None
        text = text[:10]
    else:
        text = value.strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def plausible_transaction_date(iso_date: object, filing_date: object) -> bool:
    """A transaction cannot postdate its filing by more than a few days
    (notification quirks); bond maturities misread as dates fail this."""
    tx = _as_iso(iso_date)
    if tx is None:
        return False
    filed = _as_iso(filing_date)
    if filed is None:
        return date.fromisoformat(tx) <= date.today()
    return tx <= filed or (date.fromisoformat(tx) - date.fromisoformat(filed)).days <= 7


# ── Tickers (source-native extraction only; resolution is enrichment) ────────
_HOUSE_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)")
# Some House PDFs have broken font encodings that render capitals as lowercase
# ("apple Inc. (aaPL) [ST]"). A parenthetical in the form's trailing
# ticker+asset-code position is a printed ticker regardless of its casing;
# matching it case-insensitively recovers it without inferring anything.
_HOUSE_TICKER_TRAILING = re.compile(
    r"\(([A-Za-z][A-Za-z0-9.\-]{0,9})\)\s*(?:\[[A-Za-z]{2,4}\])?\s*$"
)
_TICKER_STOPWORDS = {
    "INC", "LLC", "LP", "ETF", "CORP", "CO", "PLC", "SA", "AG", "NV",
    "THE", "AND", "FKA", "AKA", "JR", "SR", "II", "III", "IV", "ET", "AL",
    "LTD", "TRUST", "FUND", "CLASS", "SERIES", "NA", "NONE",
}
_ASSET_TYPE_CODE = re.compile(r"\[([A-Za-z]{2,4})\]")


def extract_house_ticker(asset_text: str | None) -> str | None:
    """Ticker printed in the filing, else None. Never inferred from a name."""
    if not asset_text:
        return None
    text = asset_text.strip()
    m = _HOUSE_TICKER_TRAILING.search(text)
    if m and m.group(1).upper() not in _TICKER_STOPWORDS:
        return m.group(1)
    candidates = [t for t in _HOUSE_TICKER.findall(text)
                  if t.upper() not in _TICKER_STOPWORDS]
    return candidates[-1] if candidates else None


def extract_asset_type_code(asset_text: str | None) -> str | None:
    if not asset_text:
        return None
    m = _ASSET_TYPE_CODE.search(asset_text)
    return m.group(1) if m else None


def clean_ticker(raw: object) -> str | None:
    """Ticker normalization, delegated to the existing project's cleaner.

    Single source of truth: the existing `clean_capitol_trades.clean_ticker`
    is NaN-safe, strips the ``:US`` venue suffix, rejects non-US listings and
    non-ticker text, and normalizes share classes to CRSP style (``BRK.B``).
    Reimplementing it here would let the two datasets drift apart.
    """
    from ..data.clean_capitol_trades import clean_ticker as _existing

    return _existing(raw)


# ── House asset-type codes (official Clerk code list; facts, reimplemented) ──
HOUSE_ASSET_TYPE_CODES = {
    "4K": "401K and Other Non-Federal Retirement Accounts",
    "5C": "529 College Savings Plan",
    "5F": "529 Prepaid Tuition Plan",
    "5P": "529 Prepaid Tuition Plan",
    "AB": "Asset-Backed Securities",
    "BA": "Bank Accounts, Money Market Accounts and CDs",
    "BK": "Brokerage Accounts",
    "CO": "Collectibles",
    "CS": "Corporate Securities (Bonds and Notes)",
    "CT": "Cryptocurrency",
    "DB": "Defined Benefit Pension",
    "DO": "Debts Owed to the Filer",
    "DS": "Delaware Statutory Trust",
    "EF": "Exchange Traded Funds (ETF)",
    "EQ": "Excepted/Qualified Blind Trust",
    "ET": "Exchange Traded Notes",
    "FA": "Farms",
    "FE": "Foreign Exchange Position (Currency)",
    "FN": "Fixed Annuity",
    "FU": "Futures",
    "GS": "Government Securities and Agency Debt",
    "HE": "Hedge Funds & Private Equity Funds (EIF)",
    "HN": "Hedge Funds & Private Equity Funds (non-EIF)",
    "IC": "Investment Club",
    "IH": "IRA (Held in Cash)",
    "IP": "Intellectual Property & Royalties",
    "IR": "IRA",
    "MA": "Managed Accounts (e.g., SMA and UMA)",
    "MF": "Mutual Funds",
    "MO": "Mineral/Oil/Solar Energy Rights",
    "OI": "Ownership Interest (Holding Investments)",
    "OL": "Ownership Interest (Engaged in a Trade or Business)",
    "OP": "Options",
    "OT": "Other",
    "PE": "Pensions",
    "PM": "Precious Metals",
    "PS": "Stock (Not Publicly Traded)",
    "RE": "Real Estate Invest. Trust (REIT)",
    "RP": "Real Property",
    "RS": "Restricted Stock Units (RSUs)",
    "SA": "Stock Appreciation Right",
    "ST": "Stocks (including ADRs)",
    "TR": "Trust",
    "VA": "Variable Annuity",
    "VI": "Variable Insurance",
    "WU": "Whole/Universal Insurance",
}
