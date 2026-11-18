"""Paper-language guard for generated outputs (final directive §2, §19, §22, §28).

Every generated caption, note, claim, and macro passes through
``assert_language_allowed`` before it reaches an output file. The patterns
below encode the directive's prohibitions; a violation aborts the build
rather than shipping mislabeled text.
"""

from __future__ import annotations

import re

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"independently\s+double[- ]coded", "data must be described as author-reviewed"),
    (r"independently\s+audited", "data must be described as author-reviewed"),
    (r"error[- ]free", "data must be described as author-reviewed"),
    (r"externally\s+validated\s+in\s+their\s+entirety", "author-reviewed only"),
    (r"\bprospective\b", "evidence is historical, never prospective (§19)"),
    (r"\bpristine\s+confirmation\b", "evidence is historical (§19)"),
    (r"previously\s+unseen\s+holdout", "no relabeled holdout (§19)"),
    (r"\brademacher\b", "band multipliers are Gaussian, not Rademacher (§13)"),
    (r"automatic\s+block[- ]length", "band bandwidth is fixed = horizon (§13)"),
    (r"whole[- ]date\s+resampling", "not the frozen band method (§13)"),
    (r"copy[- ]?trading", "signed payoff is not literal copy trading (§11)"),
    (
        r"observed\s+official\s+date.{0,80}(proxy|capitol\s*trades)",
        "a CT-minus-one proxy is never an observed official date (§3.1)",
    ),
    (
        r"senate.{0,60}confirmat",
        "the Senate interaction stays exploratory (§16/§28)",
    ),
    (
        r"transaction[- ]clock.{0,60}follower\s+strategy",
        "the transaction clock is not publicly implementable (§4)",
    ),
    (r"economically\s+zero", "say 'not statistically distinguishable from zero' (§11)"),
)


class LanguageGuardError(RuntimeError):
    pass


def check_language(text: str) -> list[str]:
    """Return violation messages for ``text`` (empty when clean)."""
    lowered = text.lower()
    violations = []
    for pattern, why in FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            violations.append(f"forbidden phrasing /{pattern}/: {why}")
    return violations


def assert_language_allowed(text: str, where: str) -> None:
    violations = check_language(text)
    if violations:
        raise LanguageGuardError(
            f"language guard failed in {where}: " + "; ".join(violations)
        )
