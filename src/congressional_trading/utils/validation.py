"""Cross-cutting validation guards, including anti-look-ahead checks."""

from __future__ import annotations

import re

import pandas as pd

# Column-name patterns that indicate realized post-event outcomes. Feature
# matrices for predictive models and strategy-selection rules must not
# contain any of these.
FORBIDDEN_FEATURE_PATTERNS = [
    r"^raw_return_h\d+",
    r"^benchmark_return_h\d+",
    r"^ar_.*_h\d+",
    r"^car(_|$)",
    r"^label(_|$)",
    r"_future(_|$)",
    r"^post_",
]


def assert_no_lookahead_features(feature_cols: list[str]) -> None:
    """Raise if any feature column matches a realized-outcome pattern."""
    bad = [
        c
        for c in feature_cols
        if any(re.match(p, c, flags=re.IGNORECASE) for p in FORBIDDEN_FEATURE_PATTERNS)
    ]
    if bad:
        raise ValueError(
            f"Look-ahead violation: feature columns contain realized post-event "
            f"outcomes: {bad}"
        )


def assert_chronological_split(
    train_dates: pd.Series, holdout_dates: pd.Series
) -> None:
    """Raise unless every training event strictly precedes every holdout event."""
    if len(train_dates) == 0 or len(holdout_dates) == 0:
        raise ValueError("Empty train or holdout set")
    if pd.to_datetime(train_dates).max() >= pd.to_datetime(holdout_dates).min():
        raise ValueError(
            "Chronological split violated: training sample overlaps holdout period"
        )


def require_columns(df: pd.DataFrame, cols: list[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} {('in ' + context) if context else ''}")
