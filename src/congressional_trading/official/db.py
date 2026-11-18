"""SQLite state store: fetch manifest, filings, documents, raw transactions,
failures, review queue, and run records.

The store is the pipeline's resumability and coverage backbone: every
discovered filing must reach an accountable terminal state recorded here.
Raw bytes live in the content-addressed store (storage.py); this DB holds
metadata and extraction results only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Tables that constitute the *evidentiary* state: what was discovered, fetched,
# parsed and queued. ``source_runs`` is deliberately excluded — it is a run log
# that changes on every invocation, so including it would make the digest
# useless as an input identity (finding REL-003).
EVIDENTIARY_TABLES: tuple[tuple[str, str], ...] = (
    ("filings", "filing_id"),
    ("filing_documents", "id"),
    ("filing_transactions_raw", "raw_transaction_id"),
    ("fetch_manifest", "id"),
    ("parse_failures", "id"),
    ("review_queue", "id"),
    ("source_contract_checks", "id"),
    ("filing_page_ledger", "filing_id, url"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_runs (
    run_id TEXT PRIMARY KEY,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    command TEXT,
    code_commit TEXT,
    pipeline_version TEXT,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS fetch_manifest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    chamber TEXT NOT NULL,
    url TEXT NOT NULL,
    method TEXT NOT NULL,
    params_json TEXT,
    fetched_at_utc TEXT NOT NULL,
    http_status INTEGER,
    final_url TEXT,
    content_type TEXT,
    content_length INTEGER,
    etag TEXT,
    last_modified TEXT,
    retry_count INTEGER,
    sha256 TEXT,
    stored_suffix TEXT,
    filing_id TEXT,
    result_status TEXT NOT NULL,   -- ok | not_modified | failed
    error_category TEXT,
    error_detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_manifest_url ON fetch_manifest(url);
CREATE INDEX IF NOT EXISTS ix_manifest_filing ON fetch_manifest(filing_id);

CREATE TABLE IF NOT EXISTS filings (
    filing_id TEXT PRIMARY KEY,          -- ids.filing_id()
    chamber TEXT NOT NULL,               -- house | senate
    source_system TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,         -- Clerk DocID / eFD report UUID (raw casing)
    filer_name_raw TEXT,
    first_name_raw TEXT,
    last_name_raw TEXT,
    prefix_raw TEXT,
    suffix_raw TEXT,
    state_district_raw TEXT,
    office_raw TEXT,
    filer_type_raw TEXT,
    report_type_raw TEXT,
    report_type TEXT,                    -- ptr | annual | amendment | other
    report_year INTEGER,
    filing_date TEXT,                    -- ISO; House index FilingDate
    received_date TEXT,                  -- ISO; Senate date received
    is_amendment INTEGER,
    amends_filing_id TEXT,
    amendment_relationship_confidence TEXT,
    source_url TEXT,                     -- official landing/document URL
    discovered_run_id TEXT,
    state TEXT NOT NULL DEFAULT 'discovered',
    -- terminal states: fetched_and_parsed | fetched_no_transactions |
    -- fetched_partial | fetched_unsupported | fetch_retry_pending |
    -- official_not_found | official_retention_unavailable | invalid_document |
    -- excluded_report_type | excluded_filer_type | quarantined | manually_verified
    state_detail TEXT,
    updated_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_filings_state ON filings(state);
CREATE INDEX IF NOT EXISTS ix_filings_year ON filings(chamber, report_year);

CREATE TABLE IF NOT EXISTS filing_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id TEXT NOT NULL REFERENCES filings(filing_id),
    doc_role TEXT NOT NULL,             -- primary_pdf | landing_html | page_image | index_zip
    url TEXT,
    sha256 TEXT,
    stored_suffix TEXT,
    content_type TEXT,
    fetched_at_utc TEXT,
    doc_format TEXT,                    -- formats.py classification
    format_confidence REAL,
    format_evidence TEXT,
    page_count INTEGER,
    UNIQUE(filing_id, doc_role, sha256)
);

CREATE TABLE IF NOT EXISTS filing_transactions_raw (
    raw_transaction_id TEXT PRIMARY KEY,
    filing_id TEXT NOT NULL REFERENCES filings(filing_id),
    doc_sha256 TEXT,
    page INTEGER,
    row_index INTEGER NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    extraction_method TEXT NOT NULL,    -- html_dom | pdf_text | pdf_table | ocr | manual
    parse_confidence REAL,
    raw_json TEXT NOT NULL,             -- all raw fields exactly as extracted
    source_text_evidence TEXT,
    parsed_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_rawtx_filing ON filing_transactions_raw(filing_id);

CREATE TABLE IF NOT EXISTS parse_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id TEXT NOT NULL,
    doc_sha256 TEXT,
    failure_type TEXT NOT NULL,  -- no_text | unsupported_template | ocr_unavailable | error
    detail TEXT,
    parser_name TEXT,
    parser_version TEXT,
    occurred_at_utc TEXT,
    resolved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id TEXT NOT NULL,
    raw_transaction_id TEXT,
    chamber TEXT,
    report_year INTEGER,
    reason TEXT NOT NULL,
    source_url TEXT,
    page INTEGER,
    raw_text TEXT,
    proposed_json TEXT,
    confidence REAL,
    severity TEXT DEFAULT 'medium',
    review_status TEXT DEFAULT 'pending',   -- pending | accepted | rejected
    reviewer TEXT,
    decision_json TEXT,
    decided_at_utc TEXT,
    created_at_utc TEXT
);

-- SEN-001: expected-vs-observed page ledger for multi-page (paper) filings.
-- A filing is complete only when every expected page has an 'ok' row here, so
-- a partial page set can never be mistaken for a whole document, and a retry
-- can fetch only what is missing.
CREATE TABLE IF NOT EXISTS filing_page_ledger (
    filing_id TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    url TEXT NOT NULL,
    status TEXT NOT NULL,          -- 'ok' | 'missing' | 'skipped_non_official'
    sha256 TEXT,
    stored_suffix TEXT,
    error_category TEXT,
    error_detail TEXT,
    last_attempt_utc TEXT,
    PRIMARY KEY (filing_id, url)
);

CREATE TABLE IF NOT EXISTS source_contract_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    checked_at_utc TEXT NOT NULL,
    contract_sha256 TEXT,
    detail TEXT,
    ok INTEGER NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ReadOnlyStateError(RuntimeError):
    """A write was attempted on a store opened read-only."""


class StateStore:
    """The pipeline's state DB.

    REL-003: opening the store used to run the schema DDL and commit
    unconditionally, so *reading* the state (validate, coverage, release)
    changed the file's bytes and no stable input identity existed. Read-only
    callers must pass ``read_only=True``; they then cannot write, and the file
    is left byte-for-byte untouched.
    """

    def __init__(self, path: Path, read_only: bool = False):
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"state DB not found: {self.path}")
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            self.conn.row_factory = sqlite3.Row
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path))
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _require_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyStateError(
                f"state store {self.path} is open read-only; this command must not "
                f"modify the evidentiary state (REL-003)"
            )

    # ── content identity ────────────────────────────────────────────────────
    def content_digest(self) -> dict:
        """Deterministic digest of the evidentiary tables.

        Stable across SQLite page reshuffles and across the run log, so it can
        serve as the input identity that release gating compares (freeze §9).
        """
        present = {
            r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        overall = hashlib.sha256()
        tables: dict[str, dict] = {}
        for table, order_by in EVIDENTIARY_TABLES:
            # A table added by a later schema version may be absent from an
            # older DB opened read-only; record that rather than crashing, and
            # keep "absent" distinct from "present but empty".
            if table not in present:
                tables[table] = {"rows": None, "sha256": None, "absent": True}
                overall.update(f"{table}:absent".encode())
                continue
            h = hashlib.sha256()
            n = 0
            for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
                h.update(repr(tuple(row)).encode())
                n += 1
            digest = h.hexdigest()
            tables[table] = {"rows": n, "sha256": digest}
            overall.update(f"{table}:{n}:{digest}".encode())
        return {"sha256": overall.hexdigest(), "tables": tables}

    # ── runs ────────────────────────────────────────────────────────────────
    def start_run(self, run_id: str, command: str, code_commit: str, version: str, params: dict):
        self._require_writable()
        self.conn.execute(
            "INSERT OR REPLACE INTO source_runs VALUES (?,?,?,?,?,?,?)",
            (run_id, utc_now_iso(), None, command, code_commit, version, json.dumps(params)),
        )
        self.conn.commit()

    def finish_run(self, run_id: str) -> None:
        self._require_writable()
        self.conn.execute(
            "UPDATE source_runs SET finished_at_utc=? WHERE run_id=?", (utc_now_iso(), run_id)
        )
        self.conn.commit()

    # ── fetch manifest ──────────────────────────────────────────────────────
    def record_fetch(self, **kw) -> None:
        cols = (
            "run_id source chamber url method params_json fetched_at_utc http_status "
            "final_url content_type content_length etag last_modified retry_count "
            "sha256 stored_suffix filing_id result_status error_category error_detail"
        ).split()
        kw.setdefault("fetched_at_utc", utc_now_iso())
        self.conn.execute(
            f"INSERT INTO fetch_manifest ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(kw.get(c) for c in cols),
        )
        self.conn.commit()

    def last_fetch_meta(self, url: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT etag, last_modified, sha256, stored_suffix FROM fetch_manifest "
            "WHERE url=? AND result_status='ok' ORDER BY id DESC LIMIT 1",
            (url,),
        ).fetchone()

    # ── filings ─────────────────────────────────────────────────────────────
    def upsert_filing(self, filing_id: str, **fields) -> None:
        existing = self.conn.execute(
            "SELECT filing_id FROM filings WHERE filing_id=?", (filing_id,)
        ).fetchone()
        if existing:
            # Never overwrite recorded metadata with NULL: a later discovery
            # pass that lacks a field (e.g. the ZIP index has no document URL)
            # must not wipe provenance written by an earlier fetch.
            fields = {k: v for k, v in fields.items() if v is not None}
            if not fields:
                return
            fields["updated_at_utc"] = utc_now_iso()
            sets = ", ".join(f"{k}=?" for k in fields)
            self.conn.execute(
                f"UPDATE filings SET {sets} WHERE filing_id=?",
                (*fields.values(), filing_id),
            )
        else:
            fields["updated_at_utc"] = utc_now_iso()
            cols = ["filing_id", *fields.keys()]
            self.conn.execute(
                f"INSERT INTO filings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                (filing_id, *fields.values()),
            )
        self.conn.commit()

    def set_filing_state(self, filing_id: str, state: str, detail: str | None = None) -> None:
        self.conn.execute(
            "UPDATE filings SET state=?, state_detail=?, updated_at_utc=? WHERE filing_id=?",
            (state, detail, utc_now_iso(), filing_id),
        )
        self.conn.commit()

    def filings_in_state(self, states: list[str], chamber: str | None = None) -> list[sqlite3.Row]:
        q = f"SELECT * FROM filings WHERE state IN ({','.join('?' * len(states))})"
        params: list = list(states)
        if chamber:
            q += " AND chamber=?"
            params.append(chamber)
        return self.conn.execute(q + " ORDER BY filing_id", params).fetchall()

    # ── documents / transactions ────────────────────────────────────────────
    def add_document(self, **kw) -> None:
        cols = (
            "filing_id doc_role url sha256 stored_suffix content_type fetched_at_utc "
            "doc_format format_confidence format_evidence page_count"
        ).split()
        kw.setdefault("fetched_at_utc", utc_now_iso())
        self.conn.execute(
            f"INSERT OR IGNORE INTO filing_documents ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            tuple(kw.get(c) for c in cols),
        )
        self.conn.commit()

    # ── page ledger (SEN-001) ───────────────────────────────────────────────
    def record_expected_pages(self, filing_id: str, urls: list[str]) -> None:
        """Declare the page set a filing is expected to have.

        Existing 'ok' rows are preserved so a retry never discards evidence of
        pages already archived.
        """
        self._require_writable()
        with self.conn:
            for i, url in enumerate(urls):
                self.conn.execute(
                    "INSERT OR IGNORE INTO filing_page_ledger "
                    "(filing_id, page_index, url, status, last_attempt_utc) "
                    "VALUES (?,?,?,'missing',?)",
                    (filing_id, i, url, utc_now_iso()),
                )

    def record_page_result(
        self, filing_id: str, url: str, status: str, **kw
    ) -> None:
        self._require_writable()
        self.conn.execute(
            "UPDATE filing_page_ledger SET status=?, sha256=?, stored_suffix=?, "
            "error_category=?, error_detail=?, last_attempt_utc=? "
            "WHERE filing_id=? AND url=?",
            (status, kw.get("sha256"), kw.get("stored_suffix"),
             kw.get("error_category"), kw.get("error_detail"), utc_now_iso(),
             filing_id, url),
        )
        self.conn.commit()

    def page_ledger(self, filing_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM filing_page_ledger WHERE filing_id=? ORDER BY page_index",
            (filing_id,),
        ).fetchall()

    def missing_pages(self, filing_id: str) -> list[str]:
        return [
            r["url"] for r in self.conn.execute(
                "SELECT url FROM filing_page_ledger WHERE filing_id=? AND status!='ok' "
                "ORDER BY page_index", (filing_id,)
            )
        ]

    def page_counts(self, filing_id: str) -> tuple[int, int]:
        """(expected, fetched) page counts from the ledger."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(status='ok'), 0) FROM filing_page_ledger "
            "WHERE filing_id=?", (filing_id,)
        ).fetchone()
        return int(row[0]), int(row[1])

    def documents_for(self, filing_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM filing_documents WHERE filing_id=? ORDER BY id", (filing_id,)
        ).fetchall()

    def replace_raw_transactions(self, filing_id: str, rows: list[dict]) -> None:
        """Idempotent per-filing rewrite (same parser + same doc => same IDs)."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM filing_transactions_raw WHERE filing_id=?", (filing_id,)
            )
            for r in rows:
                r.setdefault("parsed_at_utc", utc_now_iso())
                cols = (
                    "raw_transaction_id filing_id doc_sha256 page row_index parser_name "
                    "parser_version extraction_method parse_confidence raw_json "
                    "source_text_evidence parsed_at_utc"
                ).split()
                self.conn.execute(
                    f"INSERT OR REPLACE INTO filing_transactions_raw ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})",
                    tuple(r.get(c) for c in cols),
                )

    def clear_filing_diagnostics(self, filing_id: str) -> None:
        """Drop prior failure/review rows for a filing before re-parsing.

        Without this, every re-parse duplicates the accountability records and
        inflates every coverage metric derived from them.
        """
        with self.conn:
            self.conn.execute("DELETE FROM parse_failures WHERE filing_id=?", (filing_id,))
            self.conn.execute(
                "DELETE FROM review_queue WHERE filing_id=? AND review_status='pending'",
                (filing_id,),
            )

    def add_parse_failure(self, **kw) -> None:
        kw.setdefault("occurred_at_utc", utc_now_iso())
        cols = "filing_id doc_sha256 failure_type detail parser_name parser_version occurred_at_utc".split()
        self.conn.execute(
            f"INSERT INTO parse_failures ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(kw.get(c) for c in cols),
        )
        self.conn.commit()

    def add_review(self, **kw) -> None:
        kw.setdefault("created_at_utc", utc_now_iso())
        cols = (
            "filing_id raw_transaction_id chamber report_year reason source_url page "
            "raw_text proposed_json confidence severity created_at_utc"
        ).split()
        self.conn.execute(
            f"INSERT INTO review_queue ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(kw.get(c) for c in cols),
        )
        self.conn.commit()
