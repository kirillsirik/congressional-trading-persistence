"""Market inputs for the v4 forward-return cache, loaded exactly as the v2
canonical release does (scripts/v2/21_canonical_top_journal_release.py,
``load_market_frames``). Read-only on Dropbox; the only write is the local
restricted copy of the daily panel.

``v2_trading_calendar`` delegates to the shared
``trading_on_disclosure_v4.data.calendar.load_calendar`` for the default root
and applies the identical rule when an explicit root is given (tests).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from congressional_trading.data.factor_loader import load_factors_daily
from congressional_trading.paths import get_paths
from trading_on_disclosure_v2.outcomes.returns import combine_with_delisting

REPO_ROOT = Path(__file__).resolve().parents[3]
RESTRICTED_N = REPO_ROOT / "data" / "v4" / "restricted" / "n"
FRAMES = REPO_ROOT / "outputs" / "frames"

FACTOR_COLS = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")
BENCHMARK_COL = "sprtrn"  # Act I benchmark: S&P 500 return series (script 21 line 148)


def project_root() -> Path:
    """Dropbox project root, resolved as script 21 does (``.env`` honoured)."""
    return get_paths(profile="ct_only", read_only=True).project_root


def processed_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed"


def load_benchmark(root: Path | None = None) -> pd.Series:
    """Date-indexed ``sprtrn`` series (identical to v2 ``benchmark``)."""
    frame = pd.read_parquet(processed_dir(root) / "benchmark_daily_2014plus.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")[BENCHMARK_COL]


def v2_trading_calendar(root: Path | None = None) -> pd.DatetimeIndex:
    """EXACTLY the v2 canonical calendar: sorted unique benchmark dates."""
    if root is None:
        from trading_on_disclosure_v4.data.calendar import load_calendar

        return load_calendar()
    benchmark = load_benchmark(root)
    return pd.DatetimeIndex(sorted(benchmark.index.unique()))


def load_factors(root: Path | None = None) -> pd.DataFrame:
    """Date-indexed FF5+MOM+rf frame with ``mom`` renamed ``umd`` (script 21)."""
    factors = load_factors_daily(
        (root or project_root()) / "data" / "raw" / "factors" / "factors_daily.csv"
    )
    factors["date"] = pd.to_datetime(factors["date"])
    return factors.set_index("date").rename(columns={"mom": "umd"})


def load_daily_panel(root: Path | None = None) -> pd.DataFrame:
    """The v2 ``daily`` frame plus ``total_ret`` from ``combine_with_delisting``.

    Source: ``data/processed/crsp_daily_2014plus.parquet`` (content-equivalent
    to ``outputs/frames/daily.csv``); duplicate (permno, date) rows keep first
    and ``dlret`` is all-NaN, exactly as script 21 lines 137-145 do.
    """
    daily = pd.read_parquet(processed_dir(root) / "crsp_daily_2014plus.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    if daily.duplicated(subset=["permno", "date"]).any():
        daily = daily.drop_duplicates(subset=["permno", "date"], keep="first")
    if "dlret" not in daily.columns:
        daily["dlret"] = float("nan")
    return combine_with_delisting(daily)


def daily_parquet_path() -> Path:
    return RESTRICTED_N / "daily.parquet"


def build_daily_parquet(out: Path | None = None, root: Path | None = None) -> Path:
    """One-time local copy of the combined daily panel (git-ignored)."""
    out = out or daily_parquet_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    load_daily_panel(root).to_parquet(out, index=False)
    return out


def load_local_daily() -> pd.DataFrame:
    path = daily_parquet_path()
    if not path.exists():
        build_daily_parquet(path)
    return pd.read_parquet(path)


def universe_permnos(frames: Path = FRAMES) -> np.ndarray:
    """Distinct PERMNOs in signals.csv plus the matched transaction frame."""
    ids: set[int] = set()
    for name in ("signals.csv", "matched_transactions.csv"):
        col = pd.read_csv(frames / name, usecols=["permno"])["permno"].dropna()
        ids.update(int(p) for p in col.unique())
    return np.array(sorted(ids), dtype=np.int64)


def align_to_calendar(
    stock: pd.Series, calendar: pd.DatetimeIndex
) -> np.ndarray:
    """Return array on the calendar grid (NaN where the security has no row)."""
    return stock.reindex(calendar).to_numpy(dtype=float)
