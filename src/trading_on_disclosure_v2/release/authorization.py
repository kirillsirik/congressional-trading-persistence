"""Canonical-release authorization for the locked two-clock design.

A canonical run has exactly three preconditions:

1. a clean working tree on a pinned git commit;
2. a valid locked configuration (configs/top_journal_two_clock_locked.yaml);
3. an APPROVED author parse-review attestation
   (configs/PI_PARSE_REVIEW_ATTESTATION.yaml), signed by the human author.

Software must never sign the attestation; this module only validates it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from trading_on_disclosure_v2.locked_config import load_locked_config

ATTESTATION_RELPATH = Path("configs") / "PI_PARSE_REVIEW_ATTESTATION.yaml"

REQUIRED_CRITICAL_FIELDS = [
    "filer",
    "report_identifier",
    "transaction_date",
    "public_date",
    "asset_description",
    "ticker_or_security_identifier",
    "direction",
    "owner",
    "amount_bracket",
]


class AttestationError(RuntimeError):
    pass


class ReleaseAuthorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AttestationState:
    status: str  # PENDING | APPROVED
    reviewer_id: str
    date: str | None
    path: Path


def load_attestation(repo_root: Path | str) -> AttestationState:
    path = Path(repo_root) / ATTESTATION_RELPATH
    if not path.exists():
        raise AttestationError(f"attestation file missing: {path}")
    rec = yaml.safe_load(path.read_text())
    if not isinstance(rec, dict):
        raise AttestationError("attestation must be a YAML mapping")
    status = rec.get("status")
    if status not in ("PENDING", "APPROVED"):
        raise AttestationError(
            f"attestation status must be PENDING or APPROVED, got {status!r}"
        )
    scope = rec.get("review_scope") or {}
    critical = scope.get("critical_fields") or []
    if list(critical) != REQUIRED_CRITICAL_FIELDS:
        raise AttestationError(
            "attestation critical_fields do not match the required list"
        )
    if not scope.get("official_electronic_records") or not scope.get(
        "parsed_scanned_records"
    ):
        raise AttestationError(
            "attestation review_scope must cover official electronic records "
            "and parsed scanned records"
        )
    if not str(rec.get("statement") or "").strip():
        raise AttestationError("attestation statement is empty")
    if status == "APPROVED":
        blank = [
            k
            for k in ("date", "signature_or_attestation_hash", "reviewer_id")
            if not str(rec.get(k) or "").strip()
        ]
        if blank:
            raise AttestationError(
                f"attestation APPROVED but fields blank: {blank}"
            )
    return AttestationState(
        status=status,
        reviewer_id=str(rec.get("reviewer_id") or ""),
        date=str(rec.get("date") or "") or None,
        path=path,
    )


def git_state(repo_root: Path | str) -> dict:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head_sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def assert_release_authorized(
    repo_root: Path | str, *, allow_dirty_tree: bool = False
) -> dict:
    """Validate all three preconditions; return a readiness record.

    ``allow_dirty_tree`` exists ONLY for the synthetic dry-run and smoke
    namespaces; the canonical entry point never sets it.
    """
    problems: list[str] = []
    config = None
    try:
        config = load_locked_config(repo_root)
    except Exception as exc:
        problems.append(f"locked config invalid: {exc}")
    try:
        attestation = load_attestation(repo_root)
        if attestation.status != "APPROVED":
            problems.append(
                "author parse-review attestation is PENDING — the human "
                "author must sign configs/PI_PARSE_REVIEW_ATTESTATION.yaml"
            )
    except AttestationError as exc:
        attestation = None
        problems.append(f"attestation invalid: {exc}")
    git = git_state(repo_root)
    if git["dirty"] and not allow_dirty_tree:
        problems.append("working tree is dirty; canonical runs require a clean commit")
    if problems:
        raise ReleaseAuthorizationError(
            "canonical release refused: " + "; ".join(problems)
        )
    return {
        "head_sha": git["head_sha"],
        "branch": git["branch"],
        "config_sha256": config.sha256,
        "attestation_status": attestation.status,
        "attestation_reviewer": attestation.reviewer_id,
        "attestation_date": attestation.date,
    }


def readiness_report(repo_root: Path | str) -> dict:
    """Non-raising readiness summary for preflight and the lock report."""
    report: dict = {"ready": False, "checks": {}, "remaining_human_actions": []}
    try:
        config = load_locked_config(repo_root)
        report["checks"]["locked_config"] = f"valid sha256={config.sha256[:12]}"
        config_ok = True
    except Exception as exc:
        report["checks"]["locked_config"] = f"INVALID: {exc}"
        config_ok = False
    try:
        att = load_attestation(repo_root)
        report["checks"]["attestation"] = att.status
        att_ok = att.status == "APPROVED"
        if not att_ok:
            report["remaining_human_actions"].append(
                "Approve and sign configs/PI_PARSE_REVIEW_ATTESTATION.yaml"
            )
    except AttestationError as exc:
        report["checks"]["attestation"] = f"INVALID: {exc}"
        att_ok = False
    git = git_state(repo_root)
    report["checks"]["git"] = git
    clean_ok = not git["dirty"]
    if not clean_ok:
        report["checks"]["git_note"] = "commit all changes before the canonical run"
    report["ready"] = config_ok and att_ok and clean_ok
    if report["ready"]:
        report["remaining_human_actions"].append(
            "Confirm the final canonical configuration and execute "
            "`make sample`"
        )
    return report
