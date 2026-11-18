"""Vendor (Capitol Trades) snapshot layer: schemas, record canonicalization, validation and snapshots.

Modules
-------
- ``schemas``: raw / canonical / legacy column schemas.
- ``capitol_trades_parser``: table row -> raw dict -> canonical record.
- ``validation``: quality report over canonical data.
- ``snapshots``: immutable snapshot folders, diffs, latest pointer, legacy import.

The collector that downloaded the vendor table is not distributed (see data/README.md).
"""

from .capitol_trades_parser import canonicalize_records
from .schemas import CANONICAL_COLUMNS, LEGACY_FIELDNAMES, RAW_FIELDNAMES
from .snapshots import (
    SCRAPER_VERSION,
    ScrapeCheckpoint,
    build_scrape_manifest,
    create_snapshot_dir,
    diff_vs_previous,
    import_legacy_csv,
    load_latest_canonical,
    load_latest_manifest,
    write_snapshot,
)
from .validation import report_to_markdown, validate_canonical

__all__ = [
    "CANONICAL_COLUMNS",
    "LEGACY_FIELDNAMES",
    "RAW_FIELDNAMES",
    "SCRAPER_VERSION",
    "ScrapeCheckpoint",
    "build_scrape_manifest",
    "canonicalize_records",
    "create_snapshot_dir",
    "diff_vs_previous",
    "import_legacy_csv",
    "load_latest_canonical",
    "load_latest_manifest",
    "report_to_markdown",
    "validate_canonical",
    "write_snapshot",
]
