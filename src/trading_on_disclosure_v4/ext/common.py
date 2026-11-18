"""Shared pieces of the retrospective extensions: output locations, input manifests, the headline
persistence panels (reloaded exactly as scripts/v4/run_results.py builds them) and a log helper."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.outcomes import panel as panel_mod

REPO_ROOT = panel_mod.REPO_ROOT
OUT_DIR = REPO_ROOT / "data" / "v4" / "ext"
FIG_DIR = OUT_DIR / "figures"
RESTRICTED_EXT = REPO_ROOT / "data" / "v4" / "restricted" / "ext"
LOG_DIR = RESTRICTED_EXT / "logs"
MANIFEST = OUT_DIR / "input_manifest.json"
PLAN = "docs/v4/extension_plan_2026-10-19.md"


def log_line(path: Path | None, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path, rows: int | None = None) -> dict:
    """Manifest entry: repository-relative path when possible (Dropbox paths are given by their tail only)."""
    p = Path(path)
    try:
        shown = str(p.relative_to(REPO_ROOT))
    except ValueError:
        shown = "<dropbox>/" + "/".join(p.parts[-3:])
    out = {"path": shown, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def write_manifest(module: str, entries: list[dict], extra: dict | None = None, path: Path = MANIFEST) -> None:
    """Merge this module's input entries into the shared manifest (one key per module)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {}
    current[module] = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": PLAN, "inputs": entries, **(extra or {})}
    path.write_text(json.dumps(current, indent=2, sort_keys=True))


def load_headline_context(clocks=("T", "P"), with_placebo: bool = True) -> tuple[rs.Context, pd.DatetimeIndex]:
    """The results script's context on actual outcomes (both clocks); placebo frames stream from disk on demand."""
    calendar = panel_mod.v2_trading_calendar()
    sp.register_household_exclusion(sp.EXCL_TWO_VENDOR, evf.blind_vendor_households(2))
    meta = {c: evf.event_metadata(c) for c in clocks}
    frames = {c: evf.load_event_frame(c, calendar=calendar) for c in clocks}

    def source(clock: str, flavour: str, dup: str):
        if not with_placebo:
            return iter(())
        return evf.placebo_event_frames(clock, flavour, meta[clock], dup=dup, batches=True)

    def set_name(clock: str, dup: str) -> str:
        return evf.PLACEBO_SETS[(clock, "keepdups" if dup == "keepdups" else "default")]

    ctx = rs.Context(frames=frames, calendar=calendar, cutoffs=evf.pooled_cutoffs, placebo_source=source, placebo_set_name=set_name,
                     cache_dir=REPO_ROOT / "data" / "v4" / "restricted" / "e" / "placebo_stats")
    return ctx, calendar


def headline_panel(ctx: rs.Context, clock: str, spec: sp.Spec | None = None) -> pd.DataFrame:
    """The (winsorized) headline persistence panel of one clock with the 20-day anchor block attached."""
    spec = sp.headline_spec(clock) if spec is None else spec
    return rs.block_cluster(ctx.panel(spec), ctx.calendar)


def write_rows(rows: list[dict], path: Path) -> pd.DataFrame:
    """Tidy rows in the results.csv schema -> CSV (no household identity may be in `notes`)."""
    df = rs.results_frame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df
