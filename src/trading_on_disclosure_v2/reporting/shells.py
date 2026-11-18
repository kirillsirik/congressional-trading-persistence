"""Schema-backed table shells for every catalog output (RPT-001/RPT-009).

Shells are built from configs/top_journal_output_catalog.csv. Value cells
hold numbers-ledger keys, never literals; a missing required row or column
blocks the build rather than degrading it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CATALOG_RELPATH = Path("configs") / "top_journal_output_catalog.csv"
HYPOTHESIS_REGISTRY_RELPATH = Path("configs") / "top_journal_hypothesis_registry.csv"


class ShellError(RuntimeError):
    """Missing output, missing required column/row, or literal value cell."""


#: Columns every tabular shell must carry. ``ledger_key`` cells reference the
#: numbers ledger; ``display_label`` is prose; ``evidence_tier`` is per row.
BASE_TABLE_COLUMNS = (
    "row_id",
    "display_label",
    "estimand_label",
    "evidence_tier",
    "ledger_key",
)

#: Hand-specified extra value columns for the main-paper tables. Appendix
#: tables use BASE_TABLE_COLUMNS plus free extra_columns supplied at build.
MAIN_TABLE_EXTRA_COLUMNS: dict[str, tuple[str, ...]] = {
    "T01": ("chamber", "year", "count_ledger_key"),
    "T02": ("statistic_group",),
    "T03": (
        "clock",
        "unit",
        "estimate_key",
        "multiway_interval_key",
        "block_interval_key",
        "raw_p_key",
        "adjusted_p_key",
        "n_signals_key",
    ),
    "T04": (
        "view",
        "estimate_key",
        "interval_key",
        "flag",
        "sesoi_key",
        "equivalence_decision_key",
    ),
    "T05": (
        "gross_net",
        "estimate_key",
        "hac_interval_key",
        "bootstrap_interval_key",
        "exposure_key",
        "cost_key",
    ),
    "T06": ("interaction", "estimate_key", "interval_key", "adjusted_p_key"),
}

#: Required rows (row_id values) that must exist before an export of the
#: main tables is permitted (RPT-009).
MAIN_TABLE_REQUIRED_ROWS: dict[str, tuple[str, ...]] = {
    "T03": (
        "transaction_benchmark_delta_T20",
        "disclosure_day_response_delta_D0",
        "public_action_delta_E20",
        "legacy_signed_mu_E20",
    ),
    "T04": (
        "signal_view",
        "security_entry_event_view",
        "equal_household_view",
        "leave_one_top_household_out",
        "equivalence_delta_E20",
    ),
    "T05": ("gross_performance", "net_performance", "net_alpha_ff5_mom"),
}


@dataclass(frozen=True)
class OutputSpec:
    output_id: str
    kind: str  # table | figure | ledger | report
    path_stub: str
    destination: str
    evidence_tier: str
    section: str
    test_ids: tuple[str, ...]
    content: str
    required_status: str


@dataclass
class Shell:
    spec: OutputSpec
    frame: pd.DataFrame
    caption: str
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        required = set(BASE_TABLE_COLUMNS) | set(
            MAIN_TABLE_EXTRA_COLUMNS.get(self.spec.output_id, ())
        )
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ShellError(
                f"{self.spec.output_id}: missing required columns {missing}"
            )
        required_rows = MAIN_TABLE_REQUIRED_ROWS.get(self.spec.output_id, ())
        have = set(self.frame["row_id"].astype(str))
        missing_rows = [r for r in required_rows if r not in have]
        if missing_rows:
            raise ShellError(
                f"{self.spec.output_id}: missing required rows {missing_rows}"
            )
        for column in self.frame.columns:
            if column == "ledger_key" or column.endswith("_key"):
                bad = [
                    v
                    for v in self.frame[column].dropna()
                    if v != "" and _looks_numeric(v)
                ]
                if bad:
                    raise ShellError(
                        f"{self.spec.output_id}: literal numeric value(s) {bad} in "
                        f"key column {column!r}; values must come from the "
                        "numbers ledger (JC-023)"
                    )


def _looks_numeric(value) -> bool:
    try:
        float(str(value))
        return True
    except ValueError:
        return False


def load_catalog(repo_root: Path | str) -> dict[str, OutputSpec]:
    path = Path(repo_root) / CATALOG_RELPATH
    specs: dict[str, OutputSpec] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            spec = OutputSpec(
                output_id=row["output_id"],
                kind=row["kind"],
                path_stub=row["path_stub"],
                destination=row["destination"],
                evidence_tier=row["evidence_tier"],
                section=row["section"],
                test_ids=tuple(t for t in row["test_ids"].split(";") if t),
                content=row["content"],
                required_status=row["required_status"],
            )
            specs[spec.output_id] = spec
    if not specs:
        raise ShellError("output catalog is empty")
    return specs


def load_hypothesis_registry(repo_root: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(repo_root) / HYPOTHESIS_REGISTRY_RELPATH)


def build_shell(
    repo_root: Path | str,
    output_id: str,
    rows: list[dict] | None = None,
    extra_columns: tuple[str, ...] = (),
) -> Shell:
    specs = load_catalog(repo_root)
    if output_id not in specs:
        raise ShellError(f"unknown output id {output_id!r}")
    spec = specs[output_id]
    if spec.kind == "figure":
        raise ShellError(f"{output_id} is a figure; use reporting.figures")
    columns = list(BASE_TABLE_COLUMNS)
    columns += [
        c
        for c in MAIN_TABLE_EXTRA_COLUMNS.get(output_id, ()) + tuple(extra_columns)
        if c not in columns
    ]
    frame = pd.DataFrame(rows or [], columns=columns)
    caption = f"{output_id}: {spec.content}"
    notes = [
        f"Evidence tier(s): {spec.evidence_tier}.",
        f"Registered tests: {', '.join(spec.test_ids) if spec.test_ids else 'none'}.",
        "All values resolve from the numbers ledger; no hand-entered numbers.",
    ]
    shell = Shell(spec=spec, frame=frame, caption=caption, notes=notes)
    if rows:
        shell.validate()
    return shell


def build_all_shells(repo_root: Path | str) -> dict[str, Shell]:
    """Empty-but-valid shells for every non-figure catalog output (RPT-001)."""
    out: dict[str, Shell] = {}
    for output_id, spec in load_catalog(repo_root).items():
        if spec.kind == "figure":
            continue
        out[output_id] = build_shell(repo_root, output_id)
    return out


def assert_note_matches_estimator(
    shell: Shell, registry: pd.DataFrame, note_estimator_fragment: str
) -> None:
    """A table note naming an estimator must match the registry's estimator
    for the shell's registered tests (RPT gate: notes cannot drift)."""
    if not shell.spec.test_ids:
        return
    rows = registry[registry["test_id"].isin(shell.spec.test_ids)]
    estimators = " | ".join(rows["estimator"].astype(str)) + " | " + " | ".join(
        rows["primary_inference"].astype(str)
    )
    if note_estimator_fragment.lower() not in estimators.lower():
        raise ShellError(
            f"{shell.spec.output_id}: note claims estimator "
            f"{note_estimator_fragment!r} but registry records {estimators!r}"
        )
