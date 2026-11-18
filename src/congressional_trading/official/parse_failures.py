"""Typed parse failures.

Every parser failure used to be recorded as the single reason
``unreconstructed_row`` at high severity, which conflated two very different
things and left 526 blocking review items that no one could triage:

* text that **looks like a transaction** but could not be reconstructed — a real
  gap in a filing whose other rows do ship, and rightly blocking;
* a wrapped **asset-description fragment** (a bond's maturity line, a trust
  interest, a page-break continuation) that contains a date but no transaction
  signature — nothing is missing, and blocking a release on it is noise.

The kind decides both the review severity and whether the filing is treated as
partially parsed, so the distinction is made once, in the parser, where the
evidence is.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A failure of one of these kinds means a filing is incomplete: it blocks the
#: release gate and marks the filing partially parsed.
BLOCKING_FAILURE_KINDS = frozenset({
    "unreconstructed_row",
    "orphan_subfield_block",
    "no_parseable_table_rows",
})


@dataclass
class ParseFailure:
    """One thing a parser could not turn into a transaction."""

    kind: str
    detail: str
    page: int | None = None

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING_FAILURE_KINDS

    @property
    def severity(self) -> str:
        return "high" if self.blocking else "low"

    def __str__(self) -> str:
        where = f"page {self.page}: " if self.page is not None else ""
        return f"{where}{self.kind}: {self.detail}"
