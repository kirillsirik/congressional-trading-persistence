"""Outcome firewall for design-only releases.

A design-only artifact may describe sources, timing, identities, securities,
direction, format, lag, and pre-event covariates. It may never contain
post-event prices, returns, abnormal returns, outcome labels, realized
strategy performance, or inferential statistics (SIG-005 / column denial in
03_NO_SNOOPING_AND_EVIDENCE_TIERS.md section 4.2).
"""

from __future__ import annotations

import re
from collections.abc import Iterable


class FirewallViolation(RuntimeError):
    pass


# Each entry: (compiled pattern, reason). Matched case-insensitively against
# normalized column names (non-alphanumerics collapsed to "_").
_FORBIDDEN_COLUMN_PATTERNS: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in [
        (r"(^|_)(ar|car|bhar)(_|$)", "abnormal-return field"),
        (r"abnormal", "abnormal-return field"),
        (r"(^|_)ret(urn)?s?(_|$)", "return field"),
        (r"(^|_)prc(_|$)|price", "price field"),
        (r"alpha|sharpe|drawdown|pnl|profit|payoff", "strategy performance field"),
        (r"p_?val|pvalue", "inferential statistic"),
        (r"(^|_)(se|stderr|std_err|tstat|t_stat|zstat|z_stat)(_|$)", "inferential statistic"),
        (r"conf_?int|ci_(lower|upper|low|high)", "inferential statistic"),
        (r"predict|score|label_future|future_label|outcome", "future outcome label"),
        (r"volatility_post|post_event", "post-event measure"),
    ]
)

# Design fields that legitimately contain these substrings.
_ALLOWED_EXACT = frozenset(
    {
        "transaction_date_reported",
        "official_filed_date",
        "official_posted_date",
        "official_public_date_proxy",
        "follower_entry_date",
        "reporting_lag_days",
        "amount_bracket_lower",
        "amount_bracket_upper",
        "price_reported_bracket",  # official bracket text, not a market price
    }
)


def normalize(column: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()


def check_design_only_columns(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Return (column, reason) pairs violating the design-only schema."""
    violations = []
    for column in columns:
        norm = normalize(column)
        if norm in _ALLOWED_EXACT:
            continue
        for pattern, reason in _FORBIDDEN_COLUMN_PATTERNS:
            if pattern.search(norm):
                violations.append((column, reason))
                break
    return violations


def assert_design_only(columns: Iterable[str], artifact_name: str) -> None:
    violations = check_design_only_columns(columns)
    if violations:
        detail = "; ".join(f"{col!r} ({reason})" for col, reason in violations)
        raise FirewallViolation(
            f"design-only artifact {artifact_name!r} contains forbidden columns: {detail}"
        )
