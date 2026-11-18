"""Numbers ledger: 100% displayed-cell provenance (R07, RPT-002, JC-023)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LedgerEntry:
    ledger_key: str
    output_id: str
    raw_value: float
    rounded_value: float
    unit: str
    rounding_rule: str
    sample: str
    evidence_tier: str
    estimator_or_test_id: str
    source_artifact_sha256: str
    code_commit: str
    release_id: str

    def validate(self) -> None:
        problems = []
        if not self.ledger_key:
            problems.append("empty ledger_key")
        if self.evidence_tier not in {"D", "H", "L", "P", "conceptual"}:
            problems.append(f"invalid evidence tier {self.evidence_tier!r}")
        if len(self.source_artifact_sha256) != 64:
            problems.append("source_artifact_sha256 is not a sha256 hex digest")
        if not self.estimator_or_test_id:
            problems.append("missing estimator/test id")
        if not math.isfinite(self.raw_value):
            problems.append("raw_value not finite")
        if problems:
            raise LedgerError(f"{self.ledger_key or '<unkeyed>'}: {problems}")


@dataclass
class NumbersLedger:
    entries: dict[str, LedgerEntry] = field(default_factory=dict)

    def add(self, entry: LedgerEntry) -> None:
        entry.validate()
        if entry.ledger_key in self.entries:
            raise LedgerError(f"duplicate ledger key {entry.ledger_key!r}")
        self.entries[entry.ledger_key] = entry

    def require(self, ledger_key: str) -> LedgerEntry:
        if ledger_key not in self.entries:
            raise LedgerError(
                f"displayed value {ledger_key!r} absent from the numbers ledger"
            )
        return self.entries[ledger_key]

    def assert_shell_covered(self, frame: pd.DataFrame) -> None:
        """Every populated *_key / ledger_key cell must resolve (RPT-002)."""
        for column in frame.columns:
            if not (column == "ledger_key" or column.endswith("_key")):
                continue
            for value in frame[column].dropna():
                if value == "":
                    continue
                self.require(str(value))

    def content_hash(self) -> str:
        payload = json.dumps(
            {k: asdict(v) for k, v in sorted(self.entries.items())}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.entries.values()])

    def write(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {k: asdict(v) for k, v in sorted(self.entries.items())}, indent=2
        )
        path.write_text(body + "\n")
        digest = hashlib.sha256(body.encode()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n"
        )
        return digest


def validate_interval(
    estimate: float,
    se: float,
    lower: float,
    upper: float,
    critical_value: float,
    tolerance: float = 1e-9,
) -> None:
    """A stored symmetric interval must equal estimate ± crit·se (RPT gate)."""
    want_lower = estimate - critical_value * se
    want_upper = estimate + critical_value * se
    if abs(lower - want_lower) > tolerance or abs(upper - want_upper) > tolerance:
        raise LedgerError(
            "stored interval does not match its estimate: "
            f"got [{lower}, {upper}], expected [{want_lower}, {want_upper}]"
        )
