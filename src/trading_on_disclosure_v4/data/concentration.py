"""Household event-share concentration, using the canonical v2 formula unchanged.

``household_concentration`` (trading_on_disclosure_v2.inference.influence) defines
shares as event counts per household, HHI = sum(share^2), effective household
count = 1/HHI, and top-1/5/10 shares; the Act I ledger keys
``t1.concentration.*`` come from it.
"""

from __future__ import annotations

import pandas as pd

from trading_on_disclosure_v2.inference.influence import household_concentration
from trading_on_disclosure_v4.data.eligibility import eligibility_flags

PERIOD_SETS = {"A": ["A"], "B": ["B"], "AB": ["A", "B"]}


def concentration_row(events: pd.DataFrame, **labels) -> dict:
    """One count-only row from ``household_concentration`` (no shares vector)."""
    if len(events) == 0:
        return {**labels, "n_events": 0, "n_households": 0}
    res = household_concentration(events, household_col="household_id")
    res.pop("shares")
    return {**labels, "n_events": int(len(events)), **res}


def concentration_table(
    events: pd.DataFrame, counts: pd.DataFrame, direction: str = "purchase", dup_filter: str = "none"
) -> pd.DataFrame:
    """Concentration by clock x period set x household filter (all / R1-eligible).

    ``events``/``counts`` are expected to be already filtered to the stream named by
    ``dup_filter`` (see ``streams.filter_events``); the label is written to a column.
    """
    rows = []
    for clock in ("T", "P"):
        flags = eligibility_flags(counts, clock, direction)
        r1 = set(flags.loc[flags["R1"], "household_id"])
        base = events[(events["clock"] == clock) & (events["direction"] == direction)]
        for label, periods in PERIOD_SETS.items():
            sub = base[base["period"].isin(periods)]
            rows.append(concentration_row(sub, clock=clock, period=label, households="all"))
            rows.append(
                concentration_row(
                    sub[sub["household_id"].isin(r1)], clock=clock, period=label, households="R1"
                )
            )
    out = pd.DataFrame(rows)
    out.insert(0, "dup_filter", dup_filter)
    return out
