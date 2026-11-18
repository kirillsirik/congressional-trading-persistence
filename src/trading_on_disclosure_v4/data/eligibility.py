"""Household x period activity counts and the plan's eligibility rules (counts only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNT_KEY = ["household_id", "clock", "period", "direction"]
COUNT_COLUMNS = COUNT_KEY + [
    "n_events",
    "n_quarters",
    "n_securities",
    "first_anchor",
    "last_anchor",
]
# rule -> (min n_A, min quarters_A, min securities_A, min n_B, min quarters_B)
RULES = {
    "R0": (5, 0, 0, 5, 0),
    "R1": (10, 4, 3, 5, 2),
    "R2": (20, 4, 5, 10, 2),
}
FULL_SAMPLE_MINIMA = (20, 10, 50)
ACTIVITY_BINS = [-np.inf, 0, 4, 9, 19, 49, np.inf]
ACTIVITY_LABELS = ["0", "1-4", "5-9", "10-19", "20-49", "50+"]


def household_period_counts(events: pd.DataFrame) -> pd.DataFrame:
    """One row per household x clock x period x direction with activity counts."""
    out = (
        events.groupby(COUNT_KEY, sort=True)
        .agg(
            n_events=("event_id", "size"),
            n_quarters=("anchor_quarter", "nunique"),
            n_securities=("permno", "nunique"),
            first_anchor=("anchor_date", "min"),
            last_anchor=("anchor_date", "max"),
        )
        .reset_index()
    )
    out["n_events"] = out["n_events"].astype("int64")
    return out[COUNT_COLUMNS]


def _wide(counts: pd.DataFrame, clock: str, direction: str = "purchase") -> pd.DataFrame:
    """Household rows with n/q/s for periods A and B (zeros when absent)."""
    sub = counts[(counts["clock"] == clock) & (counts["direction"] == direction)]
    sub = sub[sub["period"].isin(["A", "B"])]
    wide = sub.pivot(index="household_id", columns="period", values=["n_events", "n_quarters", "n_securities"])
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    for p in ("A", "B"):
        for v in ("n_events", "n_quarters", "n_securities"):
            col = f"{v}_{p}"
            wide[col] = wide[col].fillna(0).astype("int64") if col in wide else 0
    return wide.reset_index()


def eligibility_flags(counts: pd.DataFrame, clock: str, direction: str = "purchase") -> pd.DataFrame:
    """Per-household R0/R1/R2 flags and full-sample (A u B) count flags for one clock."""
    wide = _wide(counts, clock, direction)
    for rule, (na, qa, sa, nb, qb) in RULES.items():
        wide[rule] = (
            (wide["n_events_A"] >= na)
            & (wide["n_quarters_A"] >= qa)
            & (wide["n_securities_A"] >= sa)
            & (wide["n_events_B"] >= nb)
            & (wide["n_quarters_B"] >= qb)
        )
    wide["n_events_AB"] = wide["n_events_A"] + wide["n_events_B"]
    for m in FULL_SAMPLE_MINIMA:
        wide[f"full_n{m}"] = wide["n_events_AB"] >= m
    wide.insert(1, "clock", clock)
    wide.insert(2, "direction", direction)
    return wide


def eligible_household_table(counts: pd.DataFrame, clocks=("T", "P"), dup_filter: str = "none") -> pd.DataFrame:
    """Count-only summary: eligible households per rule x clock (purchases only).

    ``counts`` should already be built from the stream named by ``dup_filter``.
    """
    rows = []
    for clock in clocks:
        flags = eligibility_flags(counts, clock)
        for rule in list(RULES) + [f"full_n{m}" for m in FULL_SAMPLE_MINIMA]:
            sel = flags[flags[rule]]
            rows.append(
                {
                    "clock": clock,
                    "rule": rule,
                    "n_households": int(len(sel)),
                    "n_events_A": int(sel["n_events_A"].sum()),
                    "n_events_B": int(sel["n_events_B"].sum()),
                    "median_n_A": float(sel["n_events_A"].median()) if len(sel) else np.nan,
                    "median_n_B": float(sel["n_events_B"].median()) if len(sel) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.insert(0, "dup_filter", dup_filter)
    return out


def count_quantiles(flags: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Quantiles of n_A and n_B over households eligible under ``rule``."""
    sel = flags[flags[rule]]
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows = []
    for period in ("A", "B"):
        col = f"n_events_{period}"
        vals = sel[col].quantile(qs) if len(sel) else pd.Series([np.nan] * len(qs), index=qs)
        rows.append({"rule": rule, "period": period, "n_households": int(len(sel)), "mean": float(sel[col].mean()) if len(sel) else np.nan, **{f"q{int(q*100):02d}": float(v) for q, v in vals.items()}})
    return pd.DataFrame(rows)


def continuation_rates(counts: pd.DataFrame, clock: str, direction: str = "purchase") -> pd.DataFrame:
    """Share of households active in A (by activity bin) that reach n_B >= 1, 5, 10."""
    wide = _wide(counts, clock, direction)
    wide["bin_A"] = pd.cut(wide["n_events_A"], ACTIVITY_BINS, labels=ACTIVITY_LABELS)
    grp = wide.groupby("bin_A", observed=False)
    out = grp.agg(n_households=("household_id", "size"))
    for k in (1, 5, 10):
        out[f"n_with_nB_ge{k}"] = grp["n_events_B"].apply(lambda s, k=k: int((s >= k).sum()))
        out[f"share_nB_ge{k}"] = out[f"n_with_nB_ge{k}"] / out["n_households"].replace(0, np.nan)
    out.insert(0, "clock", clock)
    return out.reset_index()


def activity_distribution(events: pd.DataFrame, direction: str = "purchase") -> pd.DataFrame:
    """Histogram of purchases per household per anchor year over A u B events, by clock; counts only.

    Restricting to periods A and B (not anchor years 2015-2025) excludes the late-December
    purge events and the incomplete 2025 windows, matching the full-sample eligibility base.
    """
    sub = events[(events["direction"] == direction) & events["period"].isin(["A", "B"])]
    per = sub.groupby(["clock", "household_id", "anchor_year"]).size().rename("n").reset_index()
    bins = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]
    labels = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    per["bin"] = pd.cut(per["n"], bins, labels=labels)
    out = per.groupby(["clock", "bin"], observed=False).size().rename("n_household_years").reset_index()
    out["share"] = out["n_household_years"] / out.groupby("clock")["n_household_years"].transform("sum")
    return out
