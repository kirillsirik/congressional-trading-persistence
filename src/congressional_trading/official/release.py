"""Release generation: audit + verified datasets with manifest and checksums."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from ..paths import REPO_ROOT
from . import PIPELINE_VERSION, SCHEMA_VERSION
from .config import OfficialConfig
from .db import StateStore, utc_now_iso


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release(
    config: OfficialConfig,
    store: StateStore,
    force: bool = False,
    run_id: str | None = None,
) -> dict:
    """Build the release. REL-002: validation runs first and a failure aborts.

    Passing ``force=True`` is only for diagnostics and is recorded in the
    manifest; it never becomes the default path.
    """
    from .validate import run_validation

    ok, validation = run_validation(config, store, run_id=run_id)
    # REL-003: the state must not move between validation and publication.
    state_digest = store.content_digest()
    if state_digest["sha256"] != validation["state_digest"]["sha256"]:
        raise RuntimeError(
            "release refused: the official state changed between validation and "
            "release (input digest mismatch)"
        )
    if not ok and not force:
        failed = [c["name"] for c in validation["checks"] if not c["passed"]]
        raise RuntimeError(
            "release refused: validation failed -> " + ", ".join(failed) +
            f". See {config.reports_dir / 'release_validation.json'}"
        )
    validation_path = config.reports_dir / "release_validation.json"
    validation_hash = _sha256_file(validation_path) if validation_path.exists() else None

    norm_dir = config.normalized_dir
    release_dir = config.release_dir
    release_dir.mkdir(parents=True, exist_ok=True)

    norm = pd.read_parquet(norm_dir / "transactions_normalized.parquet")

    # 1. audit dataset: everything observed, with status and provenance
    audit_path = release_dir / "transactions_all_observed.parquet"
    norm.to_parquet(audit_path, index=False)

    # 2. verified analytical dataset: current, plausible-dated, confident rows
    verified = norm[
        (norm["is_current"] == 1)
        & norm["transaction_date"].notna()
        & norm["transaction_date_plausible"]
        & (norm["parse_confidence"].fillna(0) >= config.confidence_review_threshold)
    ]
    verified_path = release_dir / "transactions_verified.parquet"
    verified.to_parquet(verified_path, index=False)
    verified.to_csv(release_dir / "transactions_verified.csv", index=False)

    # 3. project-compatible dataset
    compat_src = norm_dir / "transactions_project_compatible.csv"
    compat_path = release_dir / "transactions_project_compatible.csv"
    compat = pd.read_csv(compat_src)
    compat.to_csv(compat_path, index=False)

    filings_counts = pd.read_sql_query(
        "SELECT chamber, report_year, state, COUNT(*) n FROM filings "
        "WHERE report_type='ptr' GROUP BY chamber, report_year, state",
        store.conn,
    )

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True, cwd=REPO_ROOT).stdout.strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "HEAD") or "unknown"
    # A release built from a dirty tree is not reproducible from `commit` alone,
    # so say so in the manifest rather than implying the commit describes the run.
    code_dirty = bool(_git("status", "--porcelain"))

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_commit": commit,
        "code_dirty": code_dirty,
        "validation_passed": bool(ok),
        "validation_report_sha256": validation_hash,
        "validation_forced": bool(force and not ok),
        "run_id": run_id,
        # Input identity this release was built from (REL-003, freeze §9).
        "state_digest_sha256": state_digest["sha256"],
        "state_table_digests": state_digest["tables"],
        "published_date_semantics": "official_filing_date",
        "row_counts": {
            "all_observed": int(len(norm)),
            "verified": int(len(verified)),
            "project_compatible": int(len(compat)),
        },
        "filing_state_counts": filings_counts.to_dict(orient="records"),
        "files": {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in [audit_path, verified_path, compat_path]
        },
        "known_gaps": (
            "See <official data root>/reports/coverage_by_year_chamber.csv and "
            "docs/coverage_and_known_gaps.md; completeness is qualified by official "
            "retention (Senate: 6 years post-service), scan/OCR coverage, and review status."
        ),
    }
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
