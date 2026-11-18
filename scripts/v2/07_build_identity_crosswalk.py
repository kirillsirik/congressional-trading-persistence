#!/usr/bin/env python
"""TC-050: build the candidate filer identity crosswalk and the HD-05 evidence pack.

Reads the official state DB strictly read-only (SRC-005, digest-checked) and
writes, into the two-clock design namespace:

- identity_crosswalk.parquet          - one row per candidate filer_uid, classified;
- identity_filing_assignment.parquet  - filing_id -> filer_uid for TC-051;
- evidence_a_merge_candidates.csv     - HD-05(a) merge side;
- evidence_a_split_candidates.csv     - HD-05(a) split side;
- evidence_b_common_surnames.csv      - HD-05(b);
- evidence_c_transitions.csv          - HD-05(c);
- evidence_d_top_volume_households.csv- HD-05(d);
- evidence_e_service_dates.csv        - HD-05(e);
- identity_review_summary.json        - machine-readable counts;
- HD05_identity_review.md             - aggregate gate-packet report (no names).

The crosswalk is a CANDIDATE: no identity is merged, split or renamed by this
script. Official documentary evidence only; no outcomes, no vendor rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from congressional_trading.paths import get_paths  # noqa: E402
from trading_on_disclosure_v2.identity import (  # noqa: E402
    BLOCKING_REASONS,
    ClassificationParams,
    build_crosswalk,
)


def content_digest(conn: sqlite3.Connection) -> str:
    h = hashlib.sha256()
    for row in conn.execute(
        "SELECT filing_id, chamber, filer_name_raw, state_district_raw, filing_date,"
        " received_date FROM filings WHERE report_type='ptr' ORDER BY filing_id"
    ):
        h.update("|".join("" if v is None else str(v) for v in row).encode())
    return h.hexdigest()


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def render_report(summary: dict, unbuildable: list[dict]) -> str:
    ev = summary["evidence_counts"]
    units = summary["filer_units_total"]
    review = summary["filer_units_needs_human_review"]
    lines = [
        "# HD-05 filer identity review - evidence pack (TC-050)",
        "",
        f"Generated: {summary['generated_utc']}",
        f"Source: official PTR census, digest `{summary['store_digest'][:16]}`",
        f"Crosswalk status: **{summary['crosswalk_status']}** "
        f"(auto-merges applied: {summary['auto_merges_applied']}, "
        f"auto-splits applied: {summary['auto_splits_applied']})",
        "",
        "## 1. Population and classification",
        "",
        "| Quantity | Count |",
        "|---|---|",
        f"| PTR filings | {summary['filings_total']} |",
    ]
    for chamber, n in sorted(summary["filings_by_chamber"].items()):
        lines.append(f"| - {chamber} | {n} |")
    lines += [
        f"| Candidate filer units (`filer_uid`) | {units} |",
        f"| Unambiguous | {summary['filer_units_unambiguous']} "
        f"({_pct(summary['filer_units_unambiguous'], units)}) |",
        f"| Needs human review | {review} ({_pct(review, units)}) |",
        f"| PTR filings attached to a unit needing review | "
        f"{summary['filings_in_units_needing_review']} "
        f"({_pct(summary['filings_in_units_needing_review'], summary['filings_total'])}) |",
        "",
        "## 2. Blocking reason codes",
        "",
        "| Reason code | Units |",
        "|---|---|",
    ]
    for code in BLOCKING_REASONS:
        lines.append(f"| `{code}` | {summary['reason_code_counts'][code]} |")
    lines += [
        "",
        "## 3. Evidence classes",
        "",
        "| Class | Artifact | Cases |",
        "|---|---|---|",
        f"| (a) ambiguous merges | `evidence_a_merge_candidates.csv` | "
        f"{ev['a_merge_candidate_pairs']} pairs |",
        f"| (a) ambiguous splits | `evidence_a_split_candidates.csv` | "
        f"{ev['a_split_candidate_cases']} cases "
        f"({ev['a_split_dormancy_cases_with_other_official_filings_in_gap']} dormancy gaps "
        f"span other official filings by the same name key) |",
        f"| (b) common surnames | `evidence_b_common_surnames.csv` | "
        f"{ev['b_shared_surname_groups']} groups / {ev['b_units_sharing_a_surname']} units |",
        f"| (c) cross-state/chamber | `evidence_c_transitions.csv` | "
        f"{ev['c_transitions_total']} "
        f"(chamber {ev['c_transitions_chamber']}, state {ev['c_transitions_state']}, "
        f"district {ev['c_transitions_district']}) |",
        f"| (d) top-volume households | `evidence_d_top_volume_households.csv` | "
        f"top {ev['d_top_volume_units']} units |",
        f"| (e) service-date validation | `evidence_e_service_dates.csv` | "
        f"{ev['e_units_with_service_date_flags']} units flagged of {units} |",
        "",
        "## 4. Coverage limits of the official store",
        "",
    ]
    if unbuildable:
        lines.append("| Evidence element | Status | Missing input |")
        lines.append("|---|---|---|")
        for item in unbuildable:
            lines.append(f"| {item['element']} | {item['status']} | {item['missing']} |")
    else:
        lines.append("None recorded.")
    lines += [
        "",
        "## 5. What the reviewer must decide",
        "",
        "1. For every merge-candidate pair: one household or two. Surname sharing is not",
        "   evidence for merging (ID-002) and no pair was merged automatically.",
        "2. For every split candidate: whether one `filer_uid` covers two households.",
        "3. For every chamber transition: whether the two chamber-scoped units are the",
        "   same household, and from which date the continued identity is effective.",
        "4. For every state transition and overlapping district change: which official",
        "   record governs (ID-003).",
        "5. For units whose service dates cannot be validated from the store: accept the",
        "   unit with the documented limitation, or exclude it from the canonical release.",
        "",
        "No decision above may be inferred from outcomes, vendor data, or market data.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dormancy-gap-days", type=int, default=1096)
    parser.add_argument("--report-year-slack", type=int, default=1)
    parser.add_argument("--top-volume-n", type=int, default=25)
    args = parser.parse_args()

    project_root = get_paths(profile="ct_only", read_only=True).project_root
    db = args.db or project_root / "data" / "official" / "state.sqlite"
    out_dir = args.out_dir or (
        project_root / "outputs_v2" / "top_journal_two_clock_design" / "identity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    params = ClassificationParams(
        dormancy_gap_days=args.dormancy_gap_days,
        report_year_slack=args.report_year_slack,
        top_volume_n=args.top_volume_n,
    )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    digest_before = content_digest(conn)
    result = build_crosswalk(conn, params)
    digest_after = content_digest(conn)
    senate_with_state = conn.execute(
        "SELECT COUNT(*) FROM filings WHERE report_type='ptr' AND chamber='senate'"
        " AND state_district_raw IS NOT NULL AND state_district_raw != ''"
    ).fetchone()[0]
    senate_total = conn.execute(
        "SELECT COUNT(*) FROM filings WHERE report_type='ptr' AND chamber='senate'"
    ).fetchone()[0]
    conn.close()
    if digest_before != digest_after:
        raise RuntimeError("SRC-005 violation: filings digest changed during read")

    unbuildable = []
    if senate_with_state == 0 and senate_total:
        unbuildable.append(
            {
                "element": "cross-state transitions for Senate filers (HD-05 c)",
                "status": "NOT BUILDABLE",
                "missing": (
                    f"the store carries no state field for any of the {senate_total} Senate"
                    " PTR filings (state_district_raw is House-only); Senate state would have"
                    " to be added by the collection layer from the official filing header"
                ),
            }
        )
    unbuildable.append(
        {
            "element": "service dates against authoritative term start/end (HD-05 e)",
            "status": "PARTIAL",
            "missing": (
                "the store holds no term or oath-of-office dates; validation is therefore"
                " internal to the filing record (observed span, reported-year agreement,"
                " amendment ordering, dormancy) and an authoritative term roster would be"
                " needed for a full date-effective check"
            ),
        }
    )

    result.units.to_parquet(out_dir / "identity_crosswalk.parquet", index=False)
    result.assignments.to_parquet(out_dir / "identity_filing_assignment.parquet", index=False)
    result.merge_candidates.to_csv(out_dir / "evidence_a_merge_candidates.csv", index=False)
    result.split_candidates.to_csv(out_dir / "evidence_a_split_candidates.csv", index=False)
    result.surname_groups.to_csv(out_dir / "evidence_b_common_surnames.csv", index=False)
    result.transitions.to_csv(out_dir / "evidence_c_transitions.csv", index=False)
    result.top_volume.to_csv(out_dir / "evidence_d_top_volume_households.csv", index=False)
    result.service_dates.to_csv(out_dir / "evidence_e_service_dates.csv", index=False)

    summary = dict(result.summary)
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["db_path"] = str(db)
    summary["store_digest"] = digest_before
    summary["read_only_check"] = {
        "digest_before": digest_before,
        "digest_after": digest_after,
        "unchanged": True,
    }
    summary["coverage_limits"] = unbuildable
    (out_dir / "identity_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "HD05_identity_review.md").write_text(render_report(summary, unbuildable))

    print(json.dumps({k: v for k, v in summary.items() if k != "read_only_check"}, indent=2))
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
