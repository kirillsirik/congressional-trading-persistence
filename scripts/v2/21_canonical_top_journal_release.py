#!/usr/bin/env python
"""THE canonical execution command for the locked two-clock release (§24).

`make sample` runs this with ``--confirm-canonical-run``. The run:

 1. requires a clean pinned git commit;
 2. validates the locked canonical configuration;
 3. validates the source manifests;
 4. validates the author parse-review attestation (human-signed; §2);
 5. builds the canonical signal sample (official + Capitol Trades, §3-§8);
 6. attaches returns (§9-§10);
 7. executes all registered analyses (§11-§18);
 8. generates every main and appendix output;
 9-11. builds the numerical ledger, paper macros, and claim ledger;
 12-13. builds the release manifest and SHA-256 checksums;
 14. builds the paper where manuscript source is present;
 15. fails rather than silently skipping a required stage.

Modes (no stage/table/test selection flag exists; asserted by tests):
  (default)             canonical run; also needs --confirm-canonical-run
  --preflight-only      report readiness and exit
  --synthetic-dry-run   locked pipeline on synthetic fixtures, watermarked
  --historical-smoke    locked pipeline on historical inputs, watermarked;
                        engineering counts only, no estimate reporting
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trading_on_disclosure_v2.locked_config import load_locked_config  # noqa: E402
from trading_on_disclosure_v2.pipeline.canonical_locked import (  # noqa: E402
    LockedPipelineContext,
    run_locked_pipeline,
)
from trading_on_disclosure_v2.release.authorization import (  # noqa: E402
    assert_release_authorized,
    readiness_report,
)

# Local (unsynced) copy of the official-filing store built by `python -m congressional_trading.official`;
# its SQLite state file must not live in a synced folder. Set OFFICIAL_STORE_DIR to its location.
LOCAL_STORE = Path(os.environ.get("OFFICIAL_STORE_DIR", REPO_ROOT / "data" / "raw" / "official_store"))
SMOKE_WATERMARK = "HISTORICAL_SMOKE_TEST_NOT_CONFIRMATORY"

ALLOWED_FLAGS = {
    "--preflight-only",
    "--synthetic-dry-run",
    "--historical-smoke",
    "--confirm-canonical-run",
    "--out-root",
    "--help",
    "-h",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--historical-smoke", action="store_true")
    parser.add_argument(
        "--confirm-canonical-run",
        action="store_true",
        help="explicit human confirmation required for the canonical run",
    )
    parser.add_argument("--out-root", type=Path, default=None)
    exposed = {opt for action in parser._actions for opt in action.option_strings}
    unexpected = exposed - ALLOWED_FLAGS
    if unexpected:
        raise RuntimeError(f"prohibited selective-execution flags: {unexpected}")
    return parser


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source_manifests() -> dict:
    """Step 3: pinned source inventory for the locked release."""
    from congressional_trading.paths import get_paths

    project_root = get_paths(profile="ct_only", read_only=True).project_root
    ct_dir = project_root / "data" / "raw" / "capitol_trades"
    manifest_path = ct_dir / "latest_manifest.json"
    problems: list[str] = []
    record: dict = {"problems": problems}
    if not manifest_path.exists():
        problems.append(f"missing CT manifest: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text())
        snap = ct_dir / str(manifest.get("snapshot", "")) / "canonical_trades.parquet"
        if not snap.exists():
            problems.append(f"missing CT snapshot parquet: {snap}")
        else:
            record["ct_snapshot"] = manifest.get("snapshot")
            record["ct_records"] = manifest.get("records_canonical")
            record["ct_parquet_sha256"] = _sha256(snap)
    norm = LOCAL_STORE / "normalized" / "transactions_current.parquet"
    if not norm.exists():
        problems.append(f"missing official normalized store: {norm}")
    else:
        record["official_normalized_sha256"] = _sha256(norm)
    db = LOCAL_STORE / "state.sqlite"
    if not db.exists():
        problems.append(f"missing official state db: {db}")
    else:
        record["official_state_db_bytes"] = db.stat().st_size
    record["project_root"] = str(project_root)
    return record


def load_market_frames(project_root: Path) -> dict:
    """2014+ market panel (built by scripts/v2/22_build_market_panel_2014plus.py)."""
    proc = project_root / "data" / "processed"
    panel = proc / "crsp_daily_2014plus.parquet"
    bench = proc / "benchmark_daily_2014plus.parquet"
    names_path = proc / "crsp_names_2014plus.parquet"
    for p in (panel, bench, names_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing market input {p}; run "
                "scripts/v2/22_build_market_panel_2014plus.py first"
            )
    daily = pd.read_parquet(panel)
    daily["date"] = pd.to_datetime(daily["date"])
    n_dup = int(daily.duplicated(subset=["permno", "date"]).sum())
    if n_dup:
        daily = daily.drop_duplicates(subset=["permno", "date"], keep="first")
    if "dlret" not in daily.columns:
        daily["dlret"] = float("nan")
    bench_frame = pd.read_parquet(bench)
    bench_frame["date"] = pd.to_datetime(bench_frame["date"])
    benchmark = bench_frame.set_index("date")["sprtrn"]  # S&P 500 price return (excludes dividends)
    calendar = pd.DatetimeIndex(sorted(benchmark.index.unique()))
    from congressional_trading.data.factor_loader import load_factors_daily

    factors = load_factors_daily(
        project_root / "data" / "raw" / "factors" / "factors_daily.csv"
    )
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date").rename(columns={"mom": "umd"})
    names = pd.read_parquet(names_path)
    return {
        "daily": daily,
        "benchmark": benchmark,
        "calendar": calendar,
        "factors": factors,
        "risk_free": factors["rf"],
        "crsp_names": names,
        "duplicate_market_rows_dropped": n_dup,
    }


def build_real_context(out_root: Path, mode: str, freeze_hash: str) -> LockedPipelineContext:
    from congressional_trading.paths import get_paths
    from trading_on_disclosure_v2.sources.assemble import (
        assemble_canonical_transactions,
    )

    project_root = get_paths(profile="ct_only", read_only=True).project_root
    market = load_market_frames(project_root)
    identity_path = (
        project_root
        / "outputs_v2"
        / "top_journal_two_clock_design"
        / "identity"
        / "identity_filing_assignment.parquet"
    )
    identity = pd.read_parquet(identity_path) if identity_path.exists() else None
    txns, census, assembly_report = assemble_canonical_transactions(
        LOCAL_STORE, identity_assignment=identity
    )
    ctx = LockedPipelineContext(
        repo_root=REPO_ROOT,
        out_root=out_root,
        freeze_hash=freeze_hash,
        mode=mode,
        calendar=market["calendar"],
    )
    ctx.frames["canonical_transactions"] = txns
    ctx.frames["census"] = census
    ctx.frames["crsp_names"] = market["crsp_names"]
    ctx.frames["daily"] = market["daily"]
    ctx.frames["benchmark"] = market["benchmark"]
    ctx.frames["risk_free"] = market["risk_free"]
    ctx.frames["factors"] = market["factors"]
    ctx.artifacts["assembly_report"] = assembly_report
    ctx.artifacts["market_engineering"] = {
        "duplicate_market_rows_dropped": market["duplicate_market_rows_dropped"]
    }
    if census["terminal"].eq(False).any():
        raise RuntimeError(
            f"{int((~census['terminal']).sum())} nonterminal census documents"
        )
    return ctx


def run_synthetic(out_root: Path) -> Path:
    from trading_on_disclosure_v2.pipeline.synthetic import synthetic_universe_locked

    out = out_root / "LOCKED_SYNTHETIC_DRY_RUN"
    universe = synthetic_universe_locked()
    ctx = LockedPipelineContext(
        repo_root=REPO_ROOT,
        out_root=out,
        freeze_hash="SYNTHETIC_NOT_FREEZE_locked_dry_run_v1",
        mode="SYNTHETIC_DRY_RUN",
        calendar=universe["calendar"],
        n_boot=99,
        band_draws=99,
    )
    for key in ("daily", "benchmark", "risk_free", "factors", "crsp_names"):
        ctx.frames[key] = universe[key]
    ctx.frames["canonical_transactions"] = universe["canonical_transactions"]
    run_locked_pipeline(ctx)
    return out


def run_smoke(out_root: Path) -> tuple[Path, dict]:
    out = out_root / "top_journal_two_clock_locked_historical_smoke"
    ctx = build_real_context(
        out, "HISTORICAL_SMOKE_TEST", "SYNTHETIC_NOT_FREEZE_locked_smoke_v1"
    )
    ctx.n_boot = 99
    ctx.band_draws = 99
    run_locked_pipeline(ctx)
    report = {
        "mode": SMOKE_WATERMARK,
        "assembly": ctx.artifacts["assembly_report"],
        "market_engineering": ctx.artifacts["market_engineering"],
        "frame_rows": {
            k: int(len(v))
            for k, v in ctx.frames.items()
            if isinstance(v, pd.DataFrame)
        },
        "attrition": ctx.artifacts["attrition_ledger"].to_dict("records"),
        "amendment_summary": ctx.artifacts["amendment_summary"],
        "date_proxy_diagnostics": ctx.artifacts["date_proxy_diagnostics"],
        "registry_rows_executed": int(
            (ctx.frames["result_frame"]["status"] == "executed").sum()
        ),
        "registry_rows_total": int(len(ctx.frames["result_frame"])),
    }
    (out / "MODE").write_text(SMOKE_WATERMARK + "\n")
    (out / f"smoke_engineering_report__{SMOKE_WATERMARK}.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return out, report


def build_paper_if_present() -> str:
    manuscript = REPO_ROOT / "manuscript"
    tex = sorted(manuscript.glob("*.tex")) if manuscript.exists() else []
    if not tex:
        return "no manuscript source present; paper build not required"
    main = manuscript / "main.tex"
    target = main if main.exists() else tex[0]
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", target.name],
                cwd=manuscript,
                check=True,
                capture_output=True,
            )
        return f"built {target.name}"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"manuscript present but paper build failed: {exc}") from exc


CANONICAL_LOCK_PATH = REPO_ROOT / ".canonical_run.lock"

#: process fingerprints that may never run alongside (or during) the canonical release
COMPETING_PROCESS_PATTERNS = (
    "22_build_market_panel_2014plus",
    "--historical-smoke",
    "congressional_trading.official backfill",
)


def _competing_processes() -> list[str]:
    out = subprocess.run(
        ["ps", "-axo", "pid,command"], capture_output=True, text=True
    ).stdout
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        pid = line.strip().split(" ", 1)[0]
        if pid == me:
            continue
        if any(pat in line for pat in COMPETING_PROCESS_PATTERNS):
            hits.append(line.strip())
    return hits


class ExclusiveCanonicalLock:
    """flock-based exclusive lock: exactly one canonical run, and none while
    a competing panel-build/smoke/scrape process is alive."""

    def __init__(self, path: Path = CANONICAL_LOCK_PATH) -> None:
        self.path = path
        self._fh = None

    def __enter__(self):
        competing = _competing_processes()
        if competing:
            raise RuntimeError(
                "canonical run refused: competing processes active:\n  "
                + "\n  ".join(competing)
            )
        self._fh = self.path.open("w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"canonical run refused: exclusive lock {self.path} is held "
                "by another canonical run"
            ) from exc
        self._fh.write(f"pid={os.getpid()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc_info):
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()
        self.path.unlink(missing_ok=True)
        return False


def canonical_freeze_hash(auth: dict, sources: dict) -> str:
    payload = json.dumps(
        {
            "head_sha": auth["head_sha"],
            "config_sha256": auth["config_sha256"],
            "ct_parquet_sha256": sources.get("ct_parquet_sha256"),
            "official_normalized_sha256": sources.get("official_normalized_sha256"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.synthetic_dry_run:
        out = run_synthetic(args.out_root or (REPO_ROOT / "outputs_v2_local"))
        print(json.dumps({"mode": "SYNTHETIC_DRY_RUN", "out": str(out)}, indent=2))
        return 0

    if args.historical_smoke:
        if CANONICAL_LOCK_PATH.exists():
            print(
                "REFUSED: canonical run in progress (lock present); the "
                "historical smoke may not run concurrently (§3.3)",
                file=sys.stderr,
            )
            return 4
        from congressional_trading.paths import get_paths

        project_root = get_paths(profile="ct_only", read_only=True).project_root
        out, report = run_smoke(args.out_root or (project_root / "outputs_v2"))
        printable = {
            k: v for k, v in report.items() if k not in ("attrition",)
        }
        print(json.dumps(printable, indent=2, default=str))
        return 0

    report = readiness_report(REPO_ROOT)
    try:
        sources = validate_source_manifests()
        report["checks"]["source_manifests"] = (
            "ok" if not sources["problems"] else sources["problems"]
        )
        sources_ok = not sources["problems"]
    except Exception as exc:
        report["checks"]["source_manifests"] = f"INVALID: {exc}"
        sources, sources_ok = {"problems": [str(exc)]}, False
    report["ready"] = report["ready"] and sources_ok
    print(json.dumps(report, indent=2, default=str))
    if args.preflight_only:
        return 0 if report["ready"] else 3

    if not report["ready"]:
        print("REFUSED: canonical authorization incomplete", file=sys.stderr)
        return 3
    if not args.confirm_canonical_run:
        print(
            "REFUSED: pass --confirm-canonical-run to execute the canonical "
            "release (explicit human confirmation, §24)",
            file=sys.stderr,
        )
        return 3

    with ExclusiveCanonicalLock():
        auth = assert_release_authorized(REPO_ROOT)
        config = load_locked_config(REPO_ROOT)
        freeze = canonical_freeze_hash(auth, sources)
        out_root = args.out_root or (REPO_ROOT / "outputs")
        ctx = build_real_context(out_root, "canonical", freeze)
        ctx.config = config
        ctx.n_boot = 4999
        ctx.band_draws = int(config["event_band"]["draws"])
        run_locked_pipeline(ctx)
        paper = build_paper_if_present()
        release_record = {
            "mode": "canonical",
            "freeze_hash": freeze,
            "authorization": auth,
            "sources": {k: v for k, v in sources.items() if k != "problems"},
            "paper": paper,
            "outputs": ctx.artifacts.get("output_paths"),
        }
        (out_root / "RELEASE_RECORD.json").write_text(
            json.dumps(release_record, indent=2, default=str) + "\n"
        )
        print(json.dumps(release_record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
