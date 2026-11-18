"""Vendor-household coverage: which vendor households the duplicate audit can see.

The audit flags a vendor line only when the same household has a parsed official
line with the same key, so households with no parsed official filing in the
vendor era are structurally blind to it. This module classifies households and
counts their lines and purchase events; counts only, no identities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VENDOR_ERA_START = pd.Timestamp("2023-01-01")
CLASS_OFFICIAL_ONLY = "official_only_household"
CLASS_WITH = "vendor_with_official_2023plus"
CLASS_WITHOUT = "vendor_without_official_2023plus"
DUP_SHARE_BINS = [-np.inf, 0.1, 0.5, 0.9, np.inf]
DUP_SHARE_LABELS = ["<=0.1", "(0.1,0.5]", "(0.5,0.9]", ">0.9"]
PERIOD_SETS = {"A": ["A"], "B": ["B"], "X": ["X"], "AB": ["A", "B"]}


def classify_households(matched: pd.DataFrame, cutoff: pd.Timestamp = VENDOR_ERA_START) -> pd.Series:
    """household -> class; 'with' = >= 1 official line whose public (filing) date >= cutoff."""
    vendor_hh = set(matched.loc[matched["record_source"] == "capitol_trades", "filer_uid"])
    official = matched[matched["record_source"] == "official"]
    public = pd.to_datetime(official["analysis_public_date"], errors="coerce")
    with_hh = set(official.loc[public >= cutoff, "filer_uid"])
    households = sorted(set(matched["filer_uid"].dropna()))
    cls = [
        CLASS_OFFICIAL_ONLY if h not in vendor_hh else (CLASS_WITH if h in with_hh else CLASS_WITHOUT)
        for h in households
    ]
    return pd.Series(cls, index=pd.Index(households, name="household_id"), name="household_class")


def coverage_classes(matched: pd.DataFrame, rows: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Per vendor-household class: households, vendor lines, dup_primary lines, official lines."""
    official = matched[matched["record_source"] == "official"]
    public = pd.to_datetime(official["analysis_public_date"], errors="coerce")
    off_any = official.groupby("filer_uid").size()
    off_23 = official[public >= VENDOR_ERA_START].groupby("filer_uid").size()
    per_hh = rows.groupby("filer_uid").agg(n=("transaction_uid", "size"), d=("dup_primary", "sum"))
    per_hh["share"] = per_hh["d"] / per_hh["n"]
    per_hh["bin"] = pd.cut(per_hh["share"], DUP_SHARE_BINS, labels=DUP_SHARE_LABELS)
    per_hh["cls"] = classes.reindex(per_hh.index).to_numpy()
    out = []
    for label, sub in [(CLASS_WITH, per_hh[per_hh["cls"] == CLASS_WITH]),
                       (CLASS_WITHOUT, per_hh[per_hh["cls"] == CLASS_WITHOUT]),
                       ("all_vendor_households", per_hh)]:
        idx = sub.index
        rec = {
            "household_class": label,
            "n_households": int(len(sub)),
            "n_households_no_official_line_at_all": int((~idx.isin(off_any.index)).sum()),
            "vendor_lines": int(sub["n"].sum()),
            "vendor_lines_dup_primary": int(sub["d"].sum()),
            "share_vendor_lines_dup_primary": float(sub["d"].sum() / sub["n"].sum()) if len(sub) else np.nan,
            "official_lines_2023plus": int(off_23.reindex(idx).fillna(0).sum()),
            "official_lines_total": int(off_any.reindex(idx).fillna(0).sum()),
            "top1_household_vendor_lines": int(sub["n"].max()) if len(sub) else 0,
            "top2_households_vendor_lines": int(sub["n"].nlargest(2).sum()) if len(sub) else 0,
        }
        for b in DUP_SHARE_LABELS:
            rec[f"n_households_dup_share_{b}"] = int((sub["bin"] == b).sum())
        out.append(rec)
    return pd.DataFrame(out)


def coverage_by_period(events: pd.DataFrame, classes: pd.Series, direction: str = "purchase") -> pd.DataFrame:
    """Purchase events per household class x clock x period set, with the class's share of the
    period total and the within-class top-1 / top-2 household shares (counts only)."""
    sub = events[events["direction"] == direction].copy()
    sub["household_class"] = classes.reindex(sub["household_id"]).fillna(CLASS_OFFICIAL_ONLY).to_numpy()
    out = []
    for clock in ("T", "P"):
        for label, periods in PERIOD_SETS.items():
            cell = sub[(sub["clock"] == clock) & sub["period"].isin(periods)]
            total = len(cell)
            for cls in (CLASS_OFFICIAL_ONLY, CLASS_WITH, CLASS_WITHOUT):
                part = cell[cell["household_class"] == cls]
                per_hh = part.groupby("household_id").size().sort_values(ascending=False)
                n = len(part)
                out.append(
                    {
                        "household_class": cls,
                        "clock": clock,
                        "period": label,
                        "direction": direction,
                        "n_households": int(per_hh.size),
                        "n_events": n,
                        "n_events_vendor_source": int((part["record_source"] == "capitol_trades").sum()),
                        "n_events_official_source": int((part["record_source"] == "official").sum()),
                        "period_total_events": total,
                        "share_of_period_events": float(n / total) if total else np.nan,
                        "top1_household_events": int(per_hh.iloc[0]) if n else 0,
                        "top2_households_events": int(per_hh.iloc[:2].sum()) if n else 0,
                        "top2_share_of_period_events": float(per_hh.iloc[:2].sum() / total) if total else np.nan,
                    }
                )
    return pd.DataFrame(out)
