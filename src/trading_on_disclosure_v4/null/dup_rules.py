"""Duplicate-filter semantics for every null-side consumer (plan §11; CP-3).

D's `event_duplicate_flags.parquet` carries two flags per event: `dup_primary`
(>= 1 line reproduces an official transaction; "any-line") and
`dup_primary_all_lines` (EVERY line does, so the event vanishes when flagged
lines are dropped; "all-lines"). The plan fixes ALL-LINES as the default: an
event with at least one non-duplicate line is new information and stays.

Rules: 'all_lines' (default), 'any_line' (wave-1/2 behaviour; set names keep
the historic `_nodup` suffix so those sets stay reproducible), 'keepdups' (no
filter). The filter applies on the public clock only; on the T clock same-day
lines collapse into one episode, so a flagged line always shares its episode
with its official twin and `dup_primary_all_lines` is False everywhere (0
episodes removed under the default). `drop_on_T=True` exposes the strict T
variant (drops any flagged episode; only meaningful under 'any_line').
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_on_disclosure_v4.outcomes import event_filters as ef

RULE_COLUMNS = {"all_lines": "dup_primary_all_lines", "any_line": "dup_primary"}
RULES = ("all_lines", "any_line", "keepdups")
DEFAULT_RULE = "all_lines"
RULE_SUFFIX = {"all_lines": "_alllines", "any_line": "_nodup"}


def check_rule(rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"unknown duplicate rule {rule!r}; choose from {RULES}")
    return rule


def rule_from_switches(drop_duplicates: bool, rule: str = DEFAULT_RULE) -> str:
    """Legacy (drop_duplicates, rule) pair -> rule name ('keepdups' when not dropping)."""
    return "keepdups" if not drop_duplicates else check_rule(rule)


def filter_applies(clock: str, rule: str, drop_on_T: bool = False) -> bool:
    return check_rule(rule) != "keepdups" and (clock == "P" or bool(drop_on_T))


def dup_suffix(clock: str, rule: str = DEFAULT_RULE, drop_on_T: bool = False) -> str:
    """Set-name suffix: `_alllines`, `_nodup` (= any-line, historic) or `_keepdups`."""
    return RULE_SUFFIX[rule] if filter_applies(clock, rule, drop_on_T) else "_keepdups"


def load_flags(path: Path = ef.FLAGS_PATH) -> pd.DataFrame:
    """event_id, clock and both primary flag columns."""
    return pd.read_parquet(path, columns=["event_id", "clock", *RULE_COLUMNS.values()])


def filter_events(
    events: pd.DataFrame, clock: str, rule: str = DEFAULT_RULE, drop_on_T: bool = False,
    flags: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply `rule` to D's events (count-only report; both rules' flag counts recorded)."""
    rule = check_rule(rule)
    flags = load_flags() if flags is None else flags
    sub = flags.loc[flags["clock"] == clock]
    counts = {f"n_flagged_{r}": int(events["event_id"].isin(sub.loc[sub[c], "event_id"]).sum())
              for r, c in RULE_COLUMNS.items()}
    col = RULE_COLUMNS.get(rule, RULE_COLUMNS["any_line"])
    view = sub[["event_id", "clock", col]].rename(columns={col: ef.FLAG_COL})
    out, report = ef.drop_duplicate_events(events, clock, rule != "keepdups", drop_on_T, flags=view)
    report = {"rule": rule, "flag_column": None if rule == "keepdups" else col, **report, **counts}
    return out, report
