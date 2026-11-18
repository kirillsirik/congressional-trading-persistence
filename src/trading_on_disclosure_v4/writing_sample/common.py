"""Paths, logging, the anonymous household key and the canonical result-row schema of the 11 November 2026 round.

Purchase- and security-level inputs and outputs stay under data/v4/restricted/writing_sample/; aggregate and
household-index outputs go to data/v4/writing_sample/computation_2026-11-11/. Original inputs are only read.
"""

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

from trading_on_disclosure_v4.round2 import common as r2

REPO_ROOT = r2.REPO_ROOT
RUN_ID = "computation_2026-11-11"
OUT = REPO_ROOT / "data" / "v4" / "writing_sample" / RUN_ID
COMPONENTS = REPO_ROOT / "data" / "v4" / "writing_sample" / "components"
RESTRICTED = REPO_ROOT / "data" / "v4" / "restricted" / "writing_sample"
LOG_DIR = OUT / "logs"
CROSSWALK = r2.CROSSWALK  # household_id -> household_index of the Round 2 closeout (read only)
CLOSEOUT = r2.RUN_DIR
BP = 1e4
H = 20

#: canonical result-row columns: the first ten are the original writing-sample schema
RESULT_COLUMNS = ["analysis", "clock", "scale", "term", "statistic", "lambda_value", "value", "unit", "inference", "source",
                  "result_key", "sample", "horizon", "benchmark", "weighting", "controls"]


def logger(name: str):
    """Line logger to OUT/logs/<name>.log and stdout. Never pass household identifiers."""
    path = LOG_DIR / f"{name}.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{name}] {msg}"
        print(line, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")

    return log


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(*arrays) -> str:
    """Digest of the byte content of numpy arrays (used for event/cutoff identity checks)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        if a.dtype == object:
            h.update("\x1f".join(map(str, a.tolist())).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def rel(path: Path) -> str:
    return r2.rel(path)


def file_entry(path: Path, rows: int | None = None) -> dict:
    p = Path(path)
    out = {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def environment() -> dict:
    env = r2.environment()
    env["machine"] = platform.machine()
    return env


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    return path


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=r2._json_default) + "\n")
    return path


def household_key() -> pd.Series:
    """household_id -> global anonymous household_index of the Round 2 closeout (read, never rewritten)."""
    t = pd.read_csv(CROSSWALK, dtype={"household_id": str})
    key = pd.Series(t["household_index"].astype(np.int64).to_numpy(), index=pd.Index(t["household_id"].astype(str), name="household_id"))
    if key.index.duplicated().any() or key.duplicated().any():
        raise AssertionError("household crosswalk is not one-to-one")
    return key


def check_key_covers(key: pd.Series, ids) -> None:
    missing = set(map(str, ids)) - set(key.index)
    if missing:
        raise AssertionError(f"{len(missing)} household ids lack an anonymous index")


class Rows:
    """Collector of canonical result rows."""

    def __init__(self, source: str):
        self.source = source
        self.rows: list[dict] = []

    def add(self, analysis: str, statistic: str, value, unit: str, *, clock: str = "T", scale: str = "raw", term: str = "",
            inference: str = "", source: str | None = None, lambda_value=np.nan, sample: str = "", horizon: str = "20",
            benchmark: str = "", weighting: str = "", controls: str = "", result_key: str | None = None) -> None:
        if value is None:
            value = np.nan
        v = float(value) if not isinstance(value, str) else float("nan")
        key = result_key or "/".join(str(p) for p in (analysis, clock, sample, scale, benchmark, term, statistic) if p != "")
        # the legacy key (analysis, clock, scale, term, statistic, lambda) must stay unique: carry sample/benchmark/horizon in term
        full_term = ";".join(x for x in (term, f"sample={sample}" if sample else "", f"benchmark={benchmark}" if benchmark else "",
                                         f"horizon={horizon}" if str(horizon) not in ("", "20") else "") if x)
        self.rows.append({"analysis": analysis, "clock": clock, "scale": scale, "term": full_term, "statistic": statistic,
                          "lambda_value": lambda_value, "value": v, "unit": unit, "inference": inference,
                          "source": source or self.source, "result_key": key, "sample": sample, "horizon": str(horizon),
                          "benchmark": benchmark, "weighting": weighting, "controls": controls})

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=RESULT_COLUMNS)
        if df["result_key"].duplicated().any():
            dup = df.loc[df["result_key"].duplicated(), "result_key"].head(5).tolist()
            raise AssertionError(f"duplicate result keys: {dup}")
        return df

    def write_component(self, name: str) -> Path:
        return write_csv(self.frame(), COMPONENTS / f"{name}.csv")


def t_interval(est: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    if not (np.isfinite(se) and se > 0 and np.isfinite(df) and df > 0):
        return np.nan, np.nan
    c = float(stats.t.ppf(0.5 + level / 2, df))
    return est - c * se, est + c * se


def t_pvalue(est: float, se: float, df: float, null: float = 0.0) -> float:
    from scipy import stats

    if not (np.isfinite(se) and se > 0 and np.isfinite(df) and df > 0):
        return np.nan
    return float(2 * stats.t.sf(abs((est - null) / se), df))


def python_cmd() -> str:
    return f"{Path(sys.executable).name} (Python {sys.version.split()[0]})"
