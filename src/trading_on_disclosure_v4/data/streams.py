"""Event-stream filters for the plan's robustness runs (counts only, no outcomes).

``none`` keeps the shared event tables as built; ``drop_dup_primary`` removes the
events that would vanish if every vendor line flagged ``dup_primary`` were dropped
(all of the event's lines flagged; on the T clock this is zero episodes by
construction, see ``duplicates.t_episode_exposure``); ``official_only`` keeps
``record_source == 'official'`` events (the mandatory robustness stream).
"""

from __future__ import annotations

import pandas as pd

DUP_FILTERS = ("none", "drop_dup_primary", "official_only")


def filter_events(events: pd.DataFrame, flags: pd.DataFrame, dup_filter: str) -> pd.DataFrame:
    """Apply one of :data:`DUP_FILTERS`; ``flags`` is the event_duplicate_flags table."""
    if dup_filter == "none":
        return events
    if dup_filter == "official_only":
        return events[events["record_source"] == "official"]
    if dup_filter == "drop_dup_primary":
        gone = set(flags.loc[flags["dup_primary_all_lines"], "event_id"])
        return events[~events["event_id"].isin(gone)]
    raise ValueError(f"unknown dup_filter {dup_filter!r}; choose from {DUP_FILTERS}")


def stream_sizes(events: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Count-only table: events per clock x direction under each filter."""
    rows = []
    for name in DUP_FILTERS:
        sub = filter_events(events, flags, name)
        cell = sub.groupby(["clock", "direction"]).size().rename("n_events").reset_index()
        cell.insert(0, "dup_filter", name)
        rows.append(cell)
    return pd.concat(rows, ignore_index=True)
