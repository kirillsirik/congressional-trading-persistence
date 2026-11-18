"""Cross-source duplicate filter shared by every N consumer of D's event tables.

D's `event_duplicate_flags.parquet` marks an event `dup_primary` when it contains
>= 1 vendor line that reproduces an official transaction (plan §2, CP-3). The
flag has a different meaning per clock:

* P clock: the flagged event IS the vendor duplicate signal (a second,
  near-identical event beside the official one) -> dropped by default.
* T clock: episodes collapse every same-day line, so the flagged episode is the
  official episode that also CONTAINS the vendor line; it is not double counted
  and is KEPT by default (`drop_on_T=True` exposes the stricter variant).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.outcomes import panel

EVENTS_DIR = panel.REPO_ROOT / "data" / "v4" / "restricted" / "data"
FLAGS_PATH = EVENTS_DIR / "event_duplicate_flags.parquet"
FLAG_COL = "dup_primary"


def load_duplicate_flags(path: Path = FLAGS_PATH, flag: str = FLAG_COL) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["event_id", "clock", flag])


def filter_applies(clock: str, drop_duplicates: bool, drop_on_T: bool) -> bool:
    """Whether flagged events are removed for this clock under the two switches."""
    return bool(drop_duplicates and (clock == "P" or drop_on_T))


def dup_suffix(clock: str, drop_duplicates: bool = True, drop_on_T: bool = False) -> str:
    """Test-name suffix encoding the choice: `_nodup` (flagged events removed) or `_keepdups`."""
    return "_nodup" if filter_applies(clock, drop_duplicates, drop_on_T) else "_keepdups"


def drop_duplicate_events(
    events: pd.DataFrame, clock: str, drop_duplicates: bool = True, drop_on_T: bool = False,
    flags: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove `dup_primary` events (P by default; T only with `drop_on_T`); count-only report."""
    n_in = int(len(events))
    report = {"clock": clock, "drop_duplicates": bool(drop_duplicates), "drop_on_T": bool(drop_on_T),
              "filter_applied": filter_applies(clock, drop_duplicates, drop_on_T),
              "n_events_in": n_in, "n_flagged": 0, "n_dropped": 0, "n_events_out": n_in}
    if flags is None:
        flags = load_duplicate_flags()
    flagged = set(flags.loc[(flags["clock"] == clock) & flags[FLAG_COL], "event_id"])
    is_flagged = events["event_id"].isin(flagged)
    report["n_flagged"] = int(is_flagged.sum())
    if not report["filter_applied"]:
        return events.reset_index(drop=True), report
    out = events.loc[~is_flagged].reset_index(drop=True)
    report["n_dropped"], report["n_events_out"] = int(is_flagged.sum()), int(len(out))
    return out, report
