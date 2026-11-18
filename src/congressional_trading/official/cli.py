"""CLI for the official-source pipeline.

Usage: python -m congressional_trading.official <command> [options]

Commands: discover, fetch, parse, normalize, build, coverage, validate,
backfill (discover+fetch+parse+build, resumable), update (current year),
retry-failures, release. All commands exit nonzero on failure and support
--limit for bounded runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path

from ..paths import REPO_ROOT
from . import PIPELINE_VERSION
from .cascade import parse_filing
from .config import STOCK_ACT_PTR_EFFECTIVE, OfficialConfig, load_official_config
from .coverage import write_reports
from .db import StateStore
from .http import PoliteSession
from .members import MemberResolver

logger = logging.getLogger("official")


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _setup(args, read_only: bool = False) -> tuple[OfficialConfig, StateStore, str]:
    """Resolve config and open the state store.

    REL-003: ``read_only=True`` opens the DB read-only and skips the
    ``source_runs`` insert, so inspecting the state (validate/coverage/release)
    leaves the evidentiary DB byte-for-byte unchanged. The run id is still
    generated and travels into the report artifacts instead of the DB.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    overrides = {}
    if getattr(args, "data_root", None):
        overrides["data_root"] = Path(args.data_root)
    if getattr(args, "request_interval", None):
        overrides["request_interval_seconds"] = args.request_interval
    if getattr(args, "force_refresh", False):
        overrides["force_refresh"] = True
    config = load_official_config(**overrides)
    if getattr(args, "start_year", None):
        config.start_year = args.start_year
    if getattr(args, "end_year", None) and args.end_year != "auto":
        config.end_year = int(args.end_year)
    config.ensure_dirs()
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    if read_only:
        return config, StateStore(config.db_path, read_only=True), run_id
    store = StateStore(config.db_path)
    store.start_run(run_id, " ".join(sys.argv[1:]), _code_commit(), PIPELINE_VERSION,
                    {"data_root": str(config.data_root)})
    return config, store, run_id


def _years(config: OfficialConfig) -> list[int]:
    return list(range(config.start_year, config.resolved_end_year() + 1))


def cmd_discover(args) -> int:
    config, store, run_id = _setup(args)
    session = PoliteSession(config)
    results = []
    if "house" in args.chambers:
        from .house.discovery import discover_year
        from .house.search_discovery import discover_year_via_search

        for year in _years(config):
            results.append(discover_year(config, store, session, run_id, year))
            # union with the Clerk search endpoint: the ZIP index misses some
            # PTRs entirely (2012 has zero P rows there; search has 199)
            results.append(discover_year_via_search(config, store, session, run_id, year))
    if "senate" in args.chambers:
        from .senate.discovery import discover_range
        from .senate.session import establish_session

        establish_session(config, store, session, run_id)
        start = date.fromisoformat(STOCK_ACT_PTR_EFFECTIVE)
        if config.start_year > 2012:
            start = date(config.start_year, 1, 1)
        end = None
        if config.end_year is not None:
            end = date(config.end_year + 1, 1, 1)
        results.extend(discover_range(config, store, session, run_id, start, end))
    store.finish_run(run_id)
    print(json.dumps(results, indent=2, default=str))
    return 0


def cmd_fetch(args) -> int:
    config, store, run_id = _setup(args)
    session = PoliteSession(config)
    counts: dict[str, int] = {}
    for chamber in args.chambers:
        # SEN-001: a partial page set is retryable work, and the Senate adapter
        # re-requests only the pages its ledger still shows as missing.
        pending = store.filings_in_state(
            ["discovered", "fetch_retry_pending", "fetch_partial_pending"]
            if args.retry else ["discovered"],
            chamber=chamber,
        )
        pending = [f for f in pending if f["report_type"] == "ptr"]
        if args.year:
            pending = [f for f in pending if f["report_year"] == args.year]
        if args.limit:
            pending = pending[: args.limit]
        if chamber == "senate" and pending:
            from .senate.session import establish_session

            establish_session(config, store, session, run_id)
            from .senate.fetch import fetch_filing as senate_fetch

            for f in pending:
                state = senate_fetch(config, store, session, run_id, f)
                counts[state] = counts.get(state, 0) + 1
        elif chamber == "house":
            from .house.fetch import fetch_filing as house_fetch

            for f in pending:
                state = house_fetch(config, store, session, run_id, f)
                counts[state] = counts.get(state, 0) + 1
    store.finish_run(run_id)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_parse(args) -> int:
    config, store, run_id = _setup(args)
    counts: dict[str, int] = {}
    states = ["fetched"]
    if args.reparse:
        # quarantined/OCR states are included: a parser improvement is exactly
        # the reason a previously unparseable filing may now parse, and the
        # cascade re-routes true scans straight back to quarantine
        states += [
            "fetched_and_parsed", "fetched_partial", "fetched_no_transactions",
            "quarantined", "ocr_extracted", "ocr_needs_manual_entry",
        ]
    for chamber in args.chambers:
        pending = [
            f for f in store.filings_in_state(states, chamber=chamber)
            if f["report_type"] == "ptr"
        ]
        if args.limit:
            pending = pending[: args.limit]
        for f in pending:
            state = parse_filing(config, store, f)
            counts[state] = counts.get(state, 0) + 1
    store.finish_run(run_id)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_ocr(args) -> int:
    """OCR the quarantined scanned/paper backlog."""
    from .ocr import ocr_available, ocr_engine_info
    from .ocr_stage import ocr_filing

    config, store, run_id = _setup(args)
    if not ocr_available():
        print("ERROR: no OCR engine found (install tesseract + pytesseract)", file=sys.stderr)
        return 1
    logger.info("OCR engine: %s", ocr_engine_info())
    counts: dict[str, int] = {}
    pending = [
        f for f in store.filings_in_state(["quarantined"])
        if f["report_type"] == "ptr" and (not args.chambers or f["chamber"] in args.chambers)
    ]
    if args.limit:
        pending = pending[: args.limit]
    for n, f in enumerate(pending, 1):
        state = ocr_filing(config, store, f)
        counts[state] = counts.get(state, 0) + 1
        if n % 100 == 0:
            logger.info("OCR progress %d/%d: %s", n, len(pending), counts)
    store.finish_run(run_id)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_build(args) -> int:
    """Normalize + reconcile + compat export + audit dataset."""
    from .compat import (
        build_project_compatible,
        normalize_transactions,
        reconcile,
        size_bracket_sanity,
    )

    config, store, run_id = _setup(args)
    session = PoliteSession(config)
    resolver = MemberResolver(config.data_root / "members")
    resolver.load(session=session)
    df = normalize_transactions(config, store, resolver)
    if df.empty:
        print("no transactions extracted yet; nothing to build", file=sys.stderr)
        return 1
    df_rec = reconcile(df, store)
    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    df_rec.to_parquet(config.normalized_dir / "transactions_normalized.parquet", index=False)
    df_rec.to_csv(config.normalized_dir / "transactions_normalized.csv", index=False)
    current = df_rec[df_rec["is_current"] == 1]
    current.to_parquet(config.normalized_dir / "transactions_current.parquet", index=False)
    compat = build_project_compatible(df_rec)
    problems = size_bracket_sanity(compat)
    compat.to_csv(config.normalized_dir / "transactions_project_compatible.csv", index=False)
    compat.to_parquet(
        config.normalized_dir / "transactions_project_compatible.parquet", index=False
    )
    store.finish_run(run_id)
    print(json.dumps({
        "normalized_rows": len(df_rec),
        "current_rows": int(len(current)),
        "project_compatible_rows": len(compat),
        "sanity_problems": problems,
        "output_dir": str(config.normalized_dir),
    }, indent=2))
    return 1 if problems else 0


def cmd_coverage(args) -> int:
    config, store, run_id = _setup(args, read_only=True)
    summary = write_reports(config, store, REPO_ROOT / "reports" / "official")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_validate(args) -> int:
    from .validate import run_validation

    config, store, run_id = _setup(args, read_only=True)
    ok, results = run_validation(config, store, run_id=run_id)
    print(json.dumps(results, indent=2, default=str))
    return 0 if ok else 1


def cmd_backfill(args) -> int:
    """discover -> fetch -> parse -> build -> coverage, resumable."""
    rc = cmd_discover(args)
    if rc:
        return rc
    args.retry = True
    args.year = None
    rc = cmd_fetch(args)
    if rc:
        return rc
    args.reparse = False
    rc = cmd_parse(args)
    if rc:
        return rc
    rc = cmd_build(args)
    cov = cmd_coverage(args)
    return rc or cov


def cmd_update(args) -> int:
    """Incremental: current year only."""
    from .config import current_utc_year

    args.start_year = current_utc_year()
    args.end_year = "auto"
    return cmd_backfill(args)


def cmd_release(args) -> int:
    from .release import build_release

    config, store, run_id = _setup(args, read_only=True)
    manifest = build_release(config, store, run_id=run_id)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="congressional_trading.official",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--chambers", default="house,senate",
                       type=lambda s: [c.strip() for c in s.split(",") if c.strip()])
        p.add_argument("--start-year", type=int, default=None)
        p.add_argument("--end-year", default=None, help="year or 'auto' (default: auto)")
        p.add_argument("--data-root", default=None)
        p.add_argument("--request-interval", type=float, default=None)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("discover", help="discover filings from official indexes/search")
    common(p)
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("fetch", help="fetch discovered filings")
    common(p)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--retry", action="store_true", help="include fetch_retry_pending")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("parse", help="parse fetched filings")
    common(p)
    p.add_argument("--reparse", action="store_true", help="re-parse already-parsed filings")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("ocr", help="OCR the quarantined scanned/paper backlog")
    common(p)
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("build", help="normalize, reconcile amendments, export datasets")
    common(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("coverage", help="write coverage + failure reports")
    common(p)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("validate", help="run invariants and schema contracts")
    common(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("backfill", help="discover+fetch+parse+build+coverage (resumable)")
    common(p)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--retry", action="store_true", default=True)
    p.add_argument("--reparse", action="store_true")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("update", help="incremental current-year update")
    common(p)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--retry", action="store_true", default=True)
    p.add_argument("--reparse", action="store_true")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("release", help="build release artifacts with manifest + checksums")
    common(p)
    p.set_defaults(func=cmd_release)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted; state is checkpointed — re-run to resume", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("official").exception("command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
