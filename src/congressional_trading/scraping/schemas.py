"""Column schemas for the Capitol Trades scraping layer.

Three layers of record shape:

1. **Raw** (``RAW_FIELDNAMES``): exactly what the browser layer captures from
   one table row, plus scrape context (page, URL, timestamp). Text only, no
   parsing beyond ``inner_text`` extraction.
2. **Canonical** (``CANONICAL_COLUMNS``): the parsed, typed record written to
   ``canonical_trades.*`` in every snapshot. Downstream cleaning
   (``congressional_trading.data.clean_capitol_trades``) starts from this.
3. **Legacy** (``LEGACY_FIELDNAMES`` / ``LEGACY_COLUMN_MAP``): the 16-column
   CSV produced by the project's original collector, mapped into the
   raw layer by ``snapshots.import_legacy_csv``.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"

# ── Raw layer: one dict per scraped table row ────────────────────────────────
RAW_FIELDNAMES: list[str] = [
    "politician_name",
    "party",
    "chamber",
    "state",
    "politician_url",
    "issuer_name",
    "ticker",
    "issuer_url",
    "published_date",
    "traded_date",
    "filed_after_days",
    "owner",
    "tx_type",
    "size",
    "price",
    "page",
    "source_url",
    "scraped_at_utc",
]

# ── Canonical layer: parsed record schema (project spec order) ───────────────
CANONICAL_COLUMNS: list[str] = [
    "source",
    "source_url",
    "scrape_run_id",
    "scrape_timestamp_utc",
    "page_number",
    "politician_name",
    "politician_slug",
    "chamber",
    "party",
    "state",
    "issuer_name",
    "issuer_slug",
    "ticker_raw",
    "ticker_clean",
    "asset_type",
    "transaction_type_raw",
    "transaction_type_clean",
    "buy_sell_indicator",
    "owner_raw",
    "owner_clean",
    "trade_date",
    "published_date",
    "disclosure_lag_raw",
    "disclosure_lag_days",
    "size_raw",
    "size_lower",
    "size_upper",
    "size_midpoint_arithmetic",
    "size_midpoint_geometric",
    "size_top_coded_flag",
    "price_raw",
    "price_numeric",
    "filing_url",
    "parser_warnings",
    "stable_record_key",
    "row_content_hash",
]

# Substantive fields hashed into ``row_content_hash``. Run-specific context
# (run id, timestamp, page number, source URL, warnings) is excluded so the
# hash is stable across scrape runs and only changes when the *disclosure
# content* changes (amendment detection).
CONTENT_HASH_FIELDS: list[str] = [
    "politician_name",
    "politician_slug",
    "chamber",
    "party",
    "state",
    "issuer_name",
    "issuer_slug",
    "ticker_raw",
    "ticker_clean",
    "asset_type",
    "transaction_type_raw",
    "transaction_type_clean",
    "buy_sell_indicator",
    "owner_raw",
    "owner_clean",
    "trade_date",
    "published_date",
    "disclosure_lag_raw",
    "disclosure_lag_days",
    "size_raw",
    "size_lower",
    "size_upper",
    "size_top_coded_flag",
    "price_raw",
    "price_numeric",
    "filing_url",
]

# ── Legacy layer: the original 16-column capitol_trades.csv ──────────────────
LEGACY_FIELDNAMES: list[str] = [
    "politician_name",
    "party",
    "chamber",
    "state",
    "issuer_name",
    "ticker",
    "published_date",
    "traded_date",
    "filed_after_days",
    "owner",
    "tx_type",
    "size",
    "price",
    "politician_url",
    "issuer_url",
    "page",
]

# Legacy column → raw fieldname. The raw layer deliberately reuses the legacy
# names (identity mapping) so a legacy CSV becomes a valid raw table once the
# three context fields (source_url, scraped_at_utc and a normalized page) are
# added by ``snapshots.import_legacy_csv``.
LEGACY_COLUMN_MAP: dict[str, str] = {name: name for name in LEGACY_FIELDNAMES}

# Canonical values for parsed categorical fields.
TRANSACTION_TYPES_CLEAN = ["buy", "sell", "exchange", "receive", "other"]
OWNER_TYPES_CLEAN = ["self", "spouse", "joint", "child", "dependent", "undisclosed", "other"]

SOURCE_NAME = "capitoltrades.com"
