"""Shared pieces of the Round 2 closeout: run locations, logging, hashing, the private household key, contexts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.ext import common as ec

REPO_ROOT = ec.REPO_ROOT
RUN_ID = "closeout_2026-11-10"
RUN_DIR = REPO_ROOT / "data" / "v4" / "round2" / "closeout" / RUN_ID
INPUTS = RUN_DIR / "inputs"
RESTRICTED = REPO_ROOT / "data" / "v4" / "restricted" / "round2_closeout" / RUN_ID
LOG_DIR = RESTRICTED / "logs"
WORK = REPO_ROOT / "work" / "closeout"
CROSSWALK = RESTRICTED / "private" / "household_crosswalk.csv"
COMMAND_LOG = RUN_DIR / "command_log.txt"
BP = 1e4
FROZEN_RESULTS = REPO_ROOT / "data" / "v4" / "results.csv"
FROZEN_DECOMP = REPO_ROOT / "data" / "v4" / "ext" / "decomposition.csv"


def logger(name: str):
    """Line logger writing to the restricted log folder and stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"
    return lambda msg: ec.log_line(path, f"[{name}] {msg}")


def sha256_file(path: Path) -> str:
    return ec.sha256_file(Path(path))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(path: Path) -> str:
    """Repository-relative path where possible; Dropbox paths by their tail only."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return "<dropbox>/" + "/".join(p.parts[-3:])


def file_entry(path: Path, rows: int | None = None) -> dict:
    p = Path(path)
    out = {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return True


def environment() -> dict:
    import matplotlib
    import pyarrow
    import scipy
    import statsmodels

    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__, "git_head": git_head(), "git_dirty_tracked": git_dirty()}


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Full-precision CSV (repr floats)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    return path


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def record_command(cmd: str, note: str = "") -> None:
    """Append one executed command to the run's command log."""
    COMMAND_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMAND_LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{cmd}" + (f"\t# {note}" if note else "") + "\n")


def load_context(clocks=("T", "P")):
    """The results script's context on actual outcomes (both clocks), without placebo frames."""
    return ec.load_headline_context(tuple(clocks), with_placebo=False)


def household_key_map(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Global anonymous key: 1-based rank of the household id in the sorted union over both clocks' frames.

    Monotone in the original sorted household order, so sorting any cohort by the key reproduces the order
    the original routines (groupby sort=True on the string id) use. The crosswalk is written only under
    data/v4/restricted/."""
    ids = sorted(set().union(*[set(f["household_id"].astype(str)) for f in frames.values()]))
    key = pd.Series(np.arange(1, len(ids) + 1, dtype=np.int64), index=pd.Index(ids, name="household_id"), name="household_index")
    CROSSWALK.parent.mkdir(parents=True, exist_ok=True)
    key.reset_index().to_csv(CROSSWALK, index=False, lineterminator="\n")
    return key


def frozen_value(clock: str, analysis: str, variant: str, statistic: str, column: str = "value", path: Path = FROZEN_RESULTS) -> float:
    t = pd.read_csv(path)
    m = t[(t["clock"] == clock) & (t["analysis"] == analysis) & (t["variant"] == variant) & (t["statistic"] == statistic)]
    if len(m) != 1:
        raise KeyError(f"{path.name}: {clock}/{analysis}/{variant}/{statistic} matched {len(m)} rows")
    return float(m[column].iloc[0])


def pooled_cutoffs_record(clock: str, outcome: str) -> dict:
    """The committed pooled (p1, p99) cutoffs in bp by period, for metadata."""
    return {p: {"p1_bp": lo, "p99_bp": hi} for p, (lo, hi) in evf.pooled_cutoffs(clock, outcome).items()}
