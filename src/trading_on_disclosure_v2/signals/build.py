"""Signal collapse, event aggregation, and exact attrition reconciliation.

Raw rows are provenance only; the primary statistical unit is the unique
household-security-direction signal at first sufficient public disclosure
(contract analysis_units). Direction conflicts are retained, never netted
(SIG-002); each PERMNO x entry-date outcome appears once at the event level
(SIG-003); every stage transition reconciles exactly (ATR-001/002).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from trading_on_disclosure_v2.dates import ANALYSIS_PUBLIC_COL
from trading_on_disclosure_v2.firewall import assert_design_only
from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL

#: locked primary unit (final directive §8): one unique public directional
#: signal per filer x report x security x analysis public date x direction.
SIGNAL_KEY = (
    "filer_uid",
    "document_uid",
    "permno",
    "direction",
    ANALYSIS_PUBLIC_COL,
)

#: same filer, report, and security carrying both directions => MIXED (§8)
MIXED_KEY = ("filer_uid", "document_uid", "permno")

REQUIRED_TXN_COLUMNS = ("transaction_uid", *SIGNAL_KEY)


class AttritionError(RuntimeError):
    pass


@dataclass
class AttritionLedger:
    """Append-only stage ledger; a record may not disappear without a
    reason code (ATR-002)."""

    rows: list[dict] = field(default_factory=list)

    def record(
        self, stage: str, n_in: int, n_out: int, reasons: dict[str, int]
    ) -> None:
        negative = {k: v for k, v in reasons.items() if v < 0}
        if negative:
            raise AttritionError(
                f"stage {stage!r} carries negative attrition reason(s): {negative}"
            )
        dropped = sum(reasons.values())
        if n_in - dropped != n_out:
            raise AttritionError(
                f"stage {stage!r} does not reconcile: {n_in} in, {n_out} out, "
                f"{dropped} dropped with reasons"
            )
        self.rows.append(
            {"stage": stage, "n_in": n_in, "n_out": n_out, **reasons}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).fillna(0)


def _signal_uid(row: pd.Series, key: tuple[str, ...] = SIGNAL_KEY) -> str:
    text = "|".join(str(row[k]) for k in key)
    return "sig-" + hashlib.sha256(text.encode()).hexdigest()[:20]


def build_public_signals(
    transactions: pd.DataFrame,
    ledger: AttritionLedger | None = None,
    public_date_col: str = ANALYSIS_PUBLIC_COL,
) -> pd.DataFrame:
    """Collapse same-report repeated lots into one signal (SIG-001), keeping
    lot counts and raw parents; retain direction conflicts (SIG-002). The
    locked design keys on ``analysis_public_date``; the superseded pipeline
    passes its legacy public-date column explicitly."""
    signal_key = tuple(
        public_date_col if k == ANALYSIS_PUBLIC_COL else k for k in SIGNAL_KEY
    )
    for col in ("transaction_uid", *signal_key):
        if col not in transactions.columns:
            raise KeyError(f"canonical transactions missing column {col!r}")
    assert_design_only(transactions.columns, "canonical_transactions")

    aggs: dict = {
        "lot_count": ("transaction_uid", "size"),
        "parent_transaction_uids": (
            "transaction_uid",
            lambda s: ";".join(map(str, s)),
        ),
    }
    if "transaction_date_reported" in transactions.columns:
        aggs["txn_date_min"] = ("transaction_date_reported", "min")
        aggs["txn_date_max"] = ("transaction_date_reported", "max")
    for src, out_name in (
        ("owner", "owner_composition"),
        ("amount_bracket", "amount_brackets"),
    ):
        if src in transactions.columns:
            aggs[out_name] = (
                src,
                lambda s: ";".join(sorted(set(map(str, s.dropna())))),
            )
    for src in ("analysis_public_date_source", "analysis_public_date_is_proxy"):
        if src in transactions.columns:
            aggs[src] = (src, "first")

    grouped = (
        transactions.groupby(list(signal_key), dropna=False)
        .agg(**aggs)
        .reset_index()
    )
    grouped["signal_uid"] = grouped.apply(_signal_uid, axis=1, key=signal_key)

    # §8: both directions inside the same filer x report x security => MIXED;
    # excluded from the primary directional contrast at the sample stage.
    n_dirs_report = grouped.groupby(list(MIXED_KEY))["direction"].transform("nunique")
    grouped["signal_direction"] = grouped["direction"].where(
        n_dirs_report == 1, "MIXED"
    )
    grouped["mixed_direction"] = grouped["signal_direction"] == "MIXED"

    conflict_key = ["filer_uid", "permno", public_date_col]
    n_dirs = grouped.groupby(conflict_key)["direction"].transform("nunique")
    grouped["direction_conflict"] = n_dirs > 1

    if ledger is not None:
        collapsed = len(transactions) - len(grouped)
        ledger.record(
            "collapse_to_signals",
            n_in=len(transactions),
            n_out=len(grouped),
            reasons={"collapsed_repeated_lots": collapsed},
        )
    return grouped


def build_security_entry_events(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (permno, follower_entry_date) (SIG-003)."""
    if FOLLOWER_ENTRY_COL not in signals.columns:
        raise KeyError(f"signals missing {FOLLOWER_ENTRY_COL!r}; derive entries first")
    buys = signals[signals["direction"] == "purchase"]
    sells = signals[signals["direction"] == "sale"]
    key = ["permno", FOLLOWER_ENTRY_COL]
    events = (
        signals.groupby(key)
        .agg(signal_uids=("signal_uid", lambda s: ";".join(s)))
        .reset_index()
    )
    events = events.merge(
        buys.groupby(key)["filer_uid"].nunique().rename("purchase_household_count"),
        on=key,
        how="left",
    ).merge(
        sells.groupby(key)["filer_uid"].nunique().rename("sale_household_count"),
        on=key,
        how="left",
    )
    for col in ("purchase_household_count", "sale_household_count"):
        events[col] = events[col].fillna(0).astype(int)
    total = events["purchase_household_count"] + events["sale_household_count"]
    events["conflict_indicator"] = (
        (events["purchase_household_count"] > 0) & (events["sale_household_count"] > 0)
    )
    events["signed_consensus"] = (
        events["purchase_household_count"] - events["sale_household_count"]
    ) / total.where(total > 0, 1)

    dupes = events.duplicated(subset=key).sum()
    if dupes:
        raise AttritionError(f"{dupes} duplicate security-entry events")
    return events


def assert_design_only_release(frame: pd.DataFrame, name: str) -> None:
    """Firewall check applied before any outcome join (SIG-005)."""
    assert_design_only(frame.columns, name)
