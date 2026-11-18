"""Shared pieces of the closeout run: output locations, an input manifest keyed by the SHA-256 digests of
the inputs and the configuration (a cached object is reused only when its recorded key equals the key of
the current inputs and configuration, never on file existence alone), and the log helper."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.ext import common as ec

REPO_ROOT = ec.REPO_ROOT
OUT_DIR = REPO_ROOT / "data" / "v4" / "final"
RESTRICTED = REPO_ROOT / "data" / "v4" / "restricted" / "final"
LOG_DIR = RESTRICTED / "logs"
MANIFEST = OUT_DIR / "input_manifest.json"
PLAN = "docs/v4/final_closeout_plan_2026-11-06.md"
BP = 1e4
H = 20


def log_line(path: Path | None, msg: str) -> None:
    ec.log_line(path, msg)


def logger(name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"
    return lambda msg: ec.log_line(path, f"[{name}] {msg}")


def file_entry(path: Path, rows: int | None = None) -> dict:
    return ec.file_entry(path, rows=rows)


def config_key(config: dict, entries: list[dict]) -> str:
    """SHA-256 of the configuration and of the input digests: the reuse key of every cached object."""
    payload = {"config": config, "inputs": sorted((e["path"], e["sha256"]) for e in entries)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def write_manifest(module: str, entries: list[dict], config: dict | None = None, extra: dict | None = None) -> str:
    """Merge this module's entry into the shared manifest (one key per module) and return its reuse key."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    key = config_key(config or {}, entries)
    current[module] = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": PLAN, "key": key,
                       "config": config or {}, "inputs": entries, **(extra or {})}
    MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True, default=str))
    return key


def sidecar_path(obj: Path) -> Path:
    return obj.with_name(obj.name + ".key.json")


def reuse_ok(obj: Path, key: str) -> bool:
    """True when `obj` exists and its sidecar records exactly `key` (inputs and configuration unchanged)."""
    sc = sidecar_path(obj)
    if not obj.exists() or not sc.exists():
        return False
    try:
        return json.loads(sc.read_text()).get("key") == key
    except (ValueError, OSError):
        return False


def write_sidecar(obj: Path, key: str, extra: dict | None = None) -> None:
    sidecar_path(obj).write_text(json.dumps({"key": key, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}, indent=2, default=str))


def write_rows(rows: list[dict], path: Path) -> pd.DataFrame:
    """Tidy rows in the results.csv schema -> CSV under data/v4/final/ (no household identity may be in `notes`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return ec.write_rows(rows, path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
