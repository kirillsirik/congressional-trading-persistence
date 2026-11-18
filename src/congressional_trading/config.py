"""Config loading and methodology constants.

YAML configs in ``configs/`` control run behavior; this module supplies the
loader plus constants that define the methodology of the original collection and
event-study pipeline.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT

# ── Sample period (paper §3.1.2) ─────────────────────────────────────────────
SAMPLE_START = "2023-02-28"
SAMPLE_END = None  # open-ended: extended by each new scrape snapshot

# ── Holding periods in trading days (paper §4.1.2) ───────────────────────────
HORIZONS = [5, 10, 20, 40, 60, 127]
PRIMARY_HORIZON = 20

# ── Factor model estimation (paper §4.1.2): window [t0-130, t0-11] ───────────
ESTIMATION_WINDOW_LENGTH = 120
ESTIMATION_GAP = 10
NW_LAGS = 5

# Whether the estimated intercept (alpha) is included when computing the
# factor-model expected return over the holding window (included by default).
INCLUDE_ESTIMATED_ALPHA_DEFAULT = True

# ── Event study window (paper §4.1.3) ────────────────────────────────────────
EVENT_WINDOW_PRE = -10
EVENT_WINDOW_POST = 60

# ── Firm controls ────────────────────────────────────────────────────────────
VOLATILITY_WINDOW = 60
ANNUALIZATION_FACTOR = 252

# ── Trade-size brackets → (lower, upper); upper cap for top-coded bracket ────
SIZE_BRACKETS = {
    "1K–15K": (1_001, 15_000),
    "15K–50K": (15_001, 50_000),
    "50K–100K": (50_001, 100_000),
    "100K–250K": (100_001, 250_000),
    "250K–500K": (250_001, 500_000),
    "500K–1M": (500_001, 1_000_000),
    "1M–5M": (1_000_001, 5_000_000),
    "5M–25M": (5_000_001, 25_000_000),
    "25M–50M": (25_000_001, 50_000_000),
    "50M+": (50_000_001, 50_000_001),  # top-coded
}
TOP_CODED_CAP = 5_000_000  # cap applied to open-ended "$1,000,001+" style brackets

# ── CRSP universe filters (paper §3.1.1) ─────────────────────────────────────
VALID_SHRCD = [10, 11]          # common stocks
VALID_EXCHCD = [1, 2, 3]        # NYSE, AMEX, NASDAQ

# ── Disclosure lag bounds (paper §3.1.2) ─────────────────────────────────────
MAX_DISCLOSURE_LAG_DAYS = 365

# ── Power committees (paper §4.1.4) ──────────────────────────────────────────
POWER_COMMITTEES = [
    "Appropriations",
    "Finance",
    "Ways and Means",
    "Armed Services",
    "Commerce",
    "Energy and Commerce",
]

# ── Clustering (paper §4.1.5) ────────────────────────────────────────────────
CLUSTER_COLS = ["legislator_id", "disclosure_month"]

# ── Strategy defaults (paper Ch. 7/8) ────────────────────────────────────────
ROUND_TRIP_COST_BPS = [0, 10, 25, 50, 100]
RANDOM_STATE = 42


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; relative paths resolve against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    return cfg


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
