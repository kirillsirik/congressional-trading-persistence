"""Snapshot lifecycle for Capitol Trades scrapes.

Every scrape (or legacy import) produces one immutable folder::

    raw/capitol_trades/<UTC YYYYMMDD_HHMMSS>/
        raw_trades.csv / raw_trades.parquet
        canonical_trades.csv / canonical_trades.parquet
        scrape_manifest.json
        scrape_log.txt
        scrape_quality_report.md
        new_changed_removed_report.csv

plus one **mutable pointer file** ``raw/capitol_trades/latest_manifest.json``
that always names the latest finalized snapshot. Snapshot folders are never
modified after finalization and this module refuses to write into an existing
snapshot directory; the pointer is the only file that gets overwritten.

Mid-run, page batches are appended to ``_checkpoint_raw.csv`` inside the
run's own (new) snapshot folder, so a crash keeps all completed pages — this
fixes the legacy scraper's full-overwrite checkpointing. A folder containing
``_checkpoint_raw.csv`` but no ``scrape_manifest.json`` is a crashed run.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import pandas as pd

from ..io.local_io import atomic_write_text, write_dataframe
from ..io.manifests import git_state, hash_files, utc_run_id
from ..paths import REPO_ROOT, ProjectPaths
from .capitol_trades_parser import canonicalize_records
from .schemas import LEGACY_FIELDNAMES, RAW_FIELDNAMES, SCHEMA_VERSION
from .validation import report_to_markdown, validate_canonical

# Version of the vendor collector that wrote the snapshots; the collector itself is not distributed.
SCRAPER_VERSION = "2.0.0"

logger = logging.getLogger(__name__)

POINTER_FILENAME = "latest_manifest.json"
MANIFEST_FILENAME = "scrape_manifest.json"
CHECKPOINT_FILENAME = "_checkpoint_raw.csv"
SNAPSHOT_DIR_FORMAT = "%Y%m%d_%H%M%S"

DIFF_DISPLAY_COLUMNS = [
    "politician_name",
    "ticker_clean",
    "issuer_name",
    "trade_date",
    "published_date",
    "transaction_type_clean",
    "size_raw",
]


# ── Snapshot directory lifecycle ─────────────────────────────────────────────


def create_snapshot_dir(paths: ProjectPaths, when: datetime | None = None) -> Path:
    """Create a new immutable snapshot folder; refuse if it already exists."""
    when = when or datetime.now(timezone.utc)
    snapshot_dir = paths.raw_capitol_trades / when.strftime(SNAPSHOT_DIR_FORMAT)
    if snapshot_dir.exists():
        raise FileExistsError(
            f"Snapshot directory {snapshot_dir} already exists. Snapshots are "
            "immutable: never write into an existing snapshot folder."
        )
    snapshot_dir.mkdir(parents=True)
    logger.info("created snapshot directory %s", snapshot_dir)
    return snapshot_dir


class ScrapeCheckpoint:
    """Append-only raw-record sink inside a *new* snapshot folder.

    Each page batch is flushed to ``_checkpoint_raw.csv`` immediately, so a
    crash mid-scrape keeps every completed page. ``write_snapshot`` removes
    the checkpoint file once the finalized ``raw_trades.csv`` is in place.
    """

    def __init__(self, snapshot_dir: Path):
        self.path = snapshot_dir / CHECKPOINT_FILENAME
        if self.path.exists():
            raise FileExistsError(f"Checkpoint file already exists: {self.path}")
        self._fh: IO[str] | None = None
        self._writer: csv.DictWriter | None = None
        self.records_written = 0

    def append_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if self._fh is None:
            self._fh = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fh, fieldnames=RAW_FIELDNAMES, extrasaction="ignore"
            )
            self._writer.writeheader()
        assert self._writer is not None and self._fh is not None
        self._writer.writerows(records)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.records_written += len(records)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def __enter__(self) -> ScrapeCheckpoint:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ── Diff vs previous canonical dataset ───────────────────────────────────────


def diff_vs_previous(
    canonical_df: pd.DataFrame, previous_df: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify records vs the previous canonical dataset.

    Uses ``stable_record_key`` for identity and ``row_content_hash`` for
    content: new (key only now), changed (key in both, hash differs),
    unchanged (key and hash in both), disappeared (key only before —
    expected as records age out of Capitol Trades' ~3-year window).
    """
    keep = ["stable_record_key", "row_content_hash", *DIFF_DISPLAY_COLUMNS]

    def _slim(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in keep if c in df.columns]
        return df[cols].drop_duplicates("stable_record_key", keep="first")

    cur = _slim(canonical_df)
    if previous_df is None or previous_df.empty:
        diff = cur.rename(columns={"row_content_hash": "row_content_hash_current"}).copy()
        diff["status"] = "new"
        diff["row_content_hash_previous"] = ""
        out_cols = [
            "stable_record_key",
            "status",
            *[c for c in DIFF_DISPLAY_COLUMNS if c in diff.columns],
            "row_content_hash_previous",
            "row_content_hash_current",
        ]
        counts = {"new": len(diff), "changed": 0, "unchanged": 0, "disappeared": 0}
        return diff[out_cols], counts

    prev = _slim(previous_df)
    merged = cur.merge(
        prev,
        on="stable_record_key",
        how="outer",
        suffixes=("_current", "_previous"),
        indicator=True,
    )

    def _status(row: pd.Series) -> str:
        if row["_merge"] == "left_only":
            return "new"
        if row["_merge"] == "right_only":
            return "disappeared"
        if row["row_content_hash_current"] != row["row_content_hash_previous"]:
            return "changed"
        return "unchanged"

    merged["status"] = merged.apply(_status, axis=1)
    for col in DIFF_DISPLAY_COLUMNS:
        cur_col, prev_col = f"{col}_current", f"{col}_previous"
        if cur_col in merged.columns:
            merged[col] = merged[cur_col].where(merged[cur_col].notna(), merged.get(prev_col))
    out_cols = [
        "stable_record_key",
        "status",
        *[c for c in DIFF_DISPLAY_COLUMNS if c in merged.columns],
        "row_content_hash_previous",
        "row_content_hash_current",
    ]
    diff = merged[out_cols].fillna("")
    counts = {
        status: int((diff["status"] == status).sum())
        for status in ("new", "changed", "unchanged", "disappeared")
    }
    return diff, counts


# ── Manifest assembly ────────────────────────────────────────────────────────


def build_scrape_manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    mode: str,
    source_mode: str,
    raw_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    diff_counts: dict[str, int],
    quality_report: dict[str, Any],
    scrape_result: Any | None = None,
    runtime_seconds: float | None = None,
    extra_errors: list[str] | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``scrape_manifest.json`` payload (project spec fields)."""
    errors = list(extra_errors or [])
    warnings = list(extra_warnings or [])
    pages: dict[str, Any] = {}
    if scrape_result is not None:
        pages = {
            "attempted": scrape_result.pages_attempted,
            "succeeded": scrape_result.pages_succeeded,
            "total_detected": scrape_result.total_pages_detected,
            "page_size_requested": scrape_result.page_size_requested,
            "page_size_effective": scrape_result.page_size_effective,
            "stopped_reason": scrape_result.stopped_reason,
        }
        errors += scrape_result.errors
        warnings += scrape_result.warnings
        runtime_seconds = runtime_seconds or scrape_result.runtime_seconds
    warnings += [f"quality: {w}" for w in quality_report.get("warnings", [])]

    duplicates = quality_report.get("duplicates", {})
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "scraper_version": SCRAPER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_path": config.get("_config_path"),
        "mode": mode,
        "source_mode": source_mode,
        "pages": pages,
        "records": {
            "raw": int(len(raw_df)),
            "canonical": int(len(canonical_df)),
            "new": diff_counts.get("new", 0),
            "changed": diff_counts.get("changed", 0),
            "unchanged": diff_counts.get("unchanged", 0),
            "disappeared": diff_counts.get("disappeared", 0),
            "duplicate_key_rows": duplicates.get("duplicate_key_rows", 0),
            "conflicting_duplicate_groups": duplicates.get("conflicting_duplicate_groups", 0),
        },
        "date_ranges": quality_report.get("date_ranges", {}),
        "runtime_seconds": runtime_seconds,
        "errors": errors,
        "warnings": warnings,
    }


# ── Snapshot finalization and pointer ────────────────────────────────────────


def write_snapshot(
    *,
    snapshot_dir: Path,
    paths: ProjectPaths,
    raw_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
    diff_df: pd.DataFrame,
    log_text: str = "",
    formats: list[str] | None = None,
) -> Path:
    """Finalize a snapshot folder and update the latest pointer.

    Refuses to touch a folder that already contains a manifest (immutable).
    Parquet output degrades gracefully to CSV-only when no parquet engine is
    installed (recorded in the manifest warnings).
    """
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(
            f"{snapshot_dir} already contains {MANIFEST_FILENAME}; snapshots are immutable."
        )
    formats = formats or ["csv", "parquet"]

    written: list[Path] = []
    for name, df in (("raw_trades", raw_df), ("canonical_trades", canonical_df)):
        if "csv" in formats:
            written.append(write_dataframe(df, snapshot_dir / f"{name}.csv"))
        if "parquet" in formats:
            try:
                written.append(write_dataframe(df, snapshot_dir / f"{name}.parquet"))
            except ImportError as exc:
                msg = f"parquet skipped for {name} (no parquet engine installed: {exc})"
                logger.warning(msg)
                manifest.setdefault("warnings", []).append(msg)

    written.append(write_dataframe(diff_df, snapshot_dir / "new_changed_removed_report.csv"))
    quality_md = report_to_markdown(quality_report)
    atomic_write_text(snapshot_dir / "scrape_quality_report.md", quality_md)
    written.append(snapshot_dir / "scrape_quality_report.md")
    atomic_write_text(snapshot_dir / "scrape_log.txt", log_text or "(no log captured)\n")
    written.append(snapshot_dir / "scrape_log.txt")

    manifest["files"] = {Path(k).name: v for k, v in hash_files(written).items()}
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, default=str))

    checkpoint = snapshot_dir / CHECKPOINT_FILENAME
    if checkpoint.exists():
        checkpoint.unlink()

    update_latest_pointer(paths, snapshot_dir, manifest)
    logger.info(
        "snapshot finalized: %s (%d canonical records)", snapshot_dir, len(canonical_df)
    )
    return snapshot_dir


def update_latest_pointer(
    paths: ProjectPaths, snapshot_dir: Path, manifest: dict[str, Any]
) -> Path:
    """Overwrite the pointer file — the only mutable file at this level."""
    pointer = paths.raw_capitol_trades / POINTER_FILENAME
    payload = {
        "snapshot": snapshot_dir.name,
        "snapshot_path": str(snapshot_dir),
        "run_id": manifest.get("run_id"),
        "source_mode": manifest.get("source_mode"),
        "records_canonical": manifest.get("records", {}).get("canonical"),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(pointer, json.dumps(payload, indent=2))
    return pointer


# ── Loading the latest snapshot ──────────────────────────────────────────────


def _latest_snapshot_dir(paths: ProjectPaths) -> Path | None:
    base = paths.raw_capitol_trades
    pointer = base / POINTER_FILENAME
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text())
            candidate = base / str(payload.get("snapshot", ""))
            if (candidate / MANIFEST_FILENAME).exists():
                return candidate
            logger.warning("pointer %s names a missing/unfinalized snapshot", pointer)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read pointer %s: %s", pointer, exc)
    if not base.exists():
        return None
    # Fallback: newest finalized folder by timestamped name.
    finalized = sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / MANIFEST_FILENAME).exists()),
        key=lambda d: d.name,
        reverse=True,
    )
    return finalized[0] if finalized else None


def load_latest_manifest(paths: ProjectPaths) -> dict[str, Any] | None:
    snapshot_dir = _latest_snapshot_dir(paths)
    if snapshot_dir is None:
        return None
    return json.loads((snapshot_dir / MANIFEST_FILENAME).read_text())


def load_latest_canonical(paths: ProjectPaths) -> pd.DataFrame | None:
    """Canonical trades from the latest finalized snapshot (None if absent)."""
    snapshot_dir = _latest_snapshot_dir(paths)
    if snapshot_dir is None:
        return None
    parquet = snapshot_dir / "canonical_trades.parquet"
    csv_path = snapshot_dir / "canonical_trades.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except ImportError:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"trade_date": str, "published_date": str})
    logger.warning("snapshot %s has no canonical trades file", snapshot_dir)
    return None


# ── Legacy import ────────────────────────────────────────────────────────────


def import_legacy_csv(
    path: str | Path,
    paths: ProjectPaths,
    config: dict[str, Any] | None = None,
    scraped_at_utc: datetime | None = None,
) -> Path:
    """Convert the legacy 16-column ``capitol_trades.csv`` into snapshot #1.

    Reads the original scraper's output, lifts it into the raw schema (adding
    scrape context), canonicalizes, validates, diffs against any previous
    snapshot and writes a full immutable snapshot tagged
    ``source_mode="legacy_import"`` — so the February 2026 scrape enters the
    pipeline exactly like any future scrape.

    ``scraped_at_utc`` anchors relative published dates ("09:01 Yesterday");
    it defaults to the CSV file's modification time.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Legacy CSV not found: {src}")
    config = config or {}

    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    missing = [c for c in LEGACY_FIELDNAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{src} does not look like the legacy 16-column scrape: missing {missing}"
        )
    logger.info("legacy import: %d rows from %s", len(df), src)

    if scraped_at_utc is None:
        scraped_at_utc = datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc)
        logger.info(
            "using file mtime %s as scrape timestamp for relative-date resolution",
            scraped_at_utc.isoformat(timespec="seconds"),
        )
    scraped_iso = scraped_at_utc.isoformat(timespec="seconds")

    base_url = config.get("scraper", {}).get("base_url", "https://www.capitoltrades.com")
    trades_path = config.get("scraper", {}).get("trades_path", "/trades")
    source_url = base_url.rstrip("/") + trades_path

    raw_records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rec = dict.fromkeys(RAW_FIELDNAMES, "")
        for legacy_col in LEGACY_FIELDNAMES:
            rec[legacy_col] = row.get(legacy_col, "")
        rec["source_url"] = source_url
        rec["scraped_at_utc"] = scraped_iso
        raw_records.append(rec)
    raw_df = pd.DataFrame(raw_records, columns=RAW_FIELDNAMES)

    run_id = utc_run_id("scrape_capitoltrades_legacy_import")
    canonical_df = canonicalize_records(raw_records, scrape_run_id=run_id)
    quality_report = validate_canonical(canonical_df, config)
    previous = load_latest_canonical(paths)
    diff_df, diff_counts = diff_vs_previous(canonical_df, previous)

    manifest = build_scrape_manifest(
        run_id=run_id,
        config=config,
        mode="legacy_import",
        source_mode="legacy_import",
        raw_df=raw_df,
        canonical_df=canonical_df,
        diff_counts=diff_counts,
        quality_report=quality_report,
        extra_warnings=[f"imported from legacy CSV {src}"],
    )
    log_text = (
        f"Legacy import of {src}\n"
        f"run_id: {run_id}\nrows in: {len(df)}\ncanonical records: {len(canonical_df)}\n"
        f"scrape timestamp used for relative dates: {scraped_iso}\n"
        f"diff vs previous: {diff_counts}\n"
    )
    snapshot_dir = create_snapshot_dir(paths)
    return write_snapshot(
        snapshot_dir=snapshot_dir,
        paths=paths,
        raw_df=raw_df,
        canonical_df=canonical_df,
        manifest=manifest,
        quality_report=quality_report,
        diff_df=diff_df,
        log_text=log_text,
        formats=config.get("snapshot", {}).get("formats"),
    )
