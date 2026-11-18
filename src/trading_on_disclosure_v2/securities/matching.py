"""Date-valid PERMNO candidate generation and A/B/C/D tier assignment.

The candidates source is a caller-supplied CRSP-style name-history frame
(columns: permno, ticker, name_norm, namedt, nameendt, shrcd, exchcd), so
synthetic tests need no licensed data. Outcome fields are rejected in every
input (SEC-008); nonunique candidates without an approved review are Tier D
(SEC-002/SEC-003); the common-stock screen applies after resolution at the
transaction date (SEC-006).
"""

from __future__ import annotations

import re

import pandas as pd

from trading_on_disclosure_v2.firewall import assert_design_only

VALID_SHRCD = (10, 11)
VALID_EXCHCD = (1, 2, 3)

TXN_REQUIRED = ("transaction_uid", "transaction_date_reported")
NAMES_REQUIRED = ("permno", "ticker", "name_norm", "namedt", "nameendt")


class SecurityMatchError(RuntimeError):
    pass


def normalize_name(text: str | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    out = re.sub(r"[^A-Z0-9 ]", " ", str(text).upper())
    out = re.sub(
        r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|PLC|THE|CL A|CL B)\b",
        " ",
        out,
    )
    return re.sub(r"\s+", " ", out).strip()


def _check_inputs(transactions: pd.DataFrame, names: pd.DataFrame) -> None:
    assert_design_only(transactions.columns, "security_matching_transactions")
    for col in TXN_REQUIRED:
        if col not in transactions.columns:
            raise SecurityMatchError(f"transactions missing column {col!r}")
    for col in NAMES_REQUIRED:
        if col not in names.columns:
            raise SecurityMatchError(f"name history missing column {col!r}")


def generate_candidates(
    transactions: pd.DataFrame, names: pd.DataFrame
) -> pd.DataFrame:
    """One row per (transaction, candidate PERMNO, basis); date-valid only."""
    _check_inputs(transactions, names)
    names = names.copy()
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])

    rows = []
    for txn in transactions.itertuples(index=False):
        txn_date = pd.Timestamp(txn.transaction_date_reported)
        valid = names[(names["namedt"] <= txn_date) & (txn_date <= names["nameendt"])]
        ticker = getattr(txn, "ticker_text", None)
        if ticker is not None and not pd.isna(ticker) and str(ticker).strip():
            hits = valid[valid["ticker"] == str(ticker).strip().upper()]
            for permno in sorted(hits["permno"].unique()):
                rows.append(
                    {
                        "transaction_uid": txn.transaction_uid,
                        "permno": int(permno),
                        "match_basis": "ticker_exact_date_valid",
                    }
                )
        name_norm = normalize_name(getattr(txn, "security_name_text", None))
        if name_norm:
            hits = valid[valid["name_norm"] == name_norm]
            for permno in sorted(hits["permno"].unique()):
                rows.append(
                    {
                        "transaction_uid": txn.transaction_uid,
                        "permno": int(permno),
                        "match_basis": "name_exact_date_valid",
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["transaction_uid", "permno", "match_basis"]
        )
    out = pd.DataFrame(rows).drop_duplicates()
    counts = out.groupby(["transaction_uid", "match_basis"])["permno"].transform(
        "nunique"
    )
    out["n_basis_candidates"] = counts
    return out.reset_index(drop=True)


def assign_tiers(
    transactions: pd.DataFrame,
    candidates: pd.DataFrame,
    review_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tier per transaction. First-candidate selection is structurally
    impossible: only unique candidate sets (A/B) or approved reviews (C)
    resolve; everything else is D, excluded and counted (SEC-002/003/004).
    """
    if review_records is not None:
        for col in ("transaction_uid", "permno", "reviewer", "rationale", "review_status"):
            if col not in review_records.columns:
                raise SecurityMatchError(f"review records missing column {col!r}")
        assert_design_only(review_records.columns, "security_review_records")

    resolved = []
    by_txn = dict(tuple(candidates.groupby("transaction_uid"))) if len(candidates) else {}
    for txn_uid in transactions["transaction_uid"]:
        cand = by_txn.get(txn_uid)
        tier, permno, basis = "D", None, "no_candidate"
        if cand is not None and len(cand):
            tickers = cand[cand["match_basis"] == "ticker_exact_date_valid"]
            name_hits = cand[cand["match_basis"] == "name_exact_date_valid"]
            if tickers["permno"].nunique() == 1:
                tier, permno, basis = "A", int(tickers["permno"].iloc[0]), "unique_ticker"
            elif tickers["permno"].nunique() == 0 and name_hits["permno"].nunique() == 1:
                tier, permno, basis = "B", int(name_hits["permno"].iloc[0]), "unique_name"
            else:
                basis = "nonunique_candidates"
        if tier == "D" and review_records is not None:
            approved = review_records[
                (review_records["transaction_uid"] == txn_uid)
                & (review_records["review_status"] == "approved")
            ]
            if len(approved) == 1:
                reviewed_permno = int(approved["permno"].iloc[0])
                cand_permnos = (
                    set(cand["permno"].astype(int)) if cand is not None else set()
                )
                if reviewed_permno in cand_permnos or basis == "no_candidate":
                    tier, permno, basis = "C", reviewed_permno, "approved_human_review"
            elif len(approved) > 1:
                basis = "conflicting_reviews"
        resolved.append(
            {
                "transaction_uid": txn_uid,
                "match_tier": tier,
                "permno": permno,
                "resolution_basis": basis,
            }
        )
    return pd.DataFrame(resolved)


def apply_common_stock_screen(
    resolved: pd.DataFrame, transactions: pd.DataFrame, names: pd.DataFrame
) -> pd.DataFrame:
    """Date-specific share/exchange screen AFTER resolution (SEC-006): some
    active (date-valid) name-history row must carry an eligible share code
    AND an eligible exchange code simultaneously (rowwise, never assembled
    across different rows)."""
    names = names.copy()
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    merged = resolved.merge(
        transactions[["transaction_uid", "transaction_date_reported"]],
        on="transaction_uid",
        validate="one_to_one",
    )
    flags = []
    for row in merged.itertuples(index=False):
        ok = False
        if row.match_tier in ("A", "B", "C") and row.permno is not None:
            txn_date = pd.Timestamp(row.transaction_date_reported)
            hist = names[
                (names["permno"] == row.permno)
                & (names["namedt"] <= txn_date)
                & (txn_date <= names["nameendt"])
            ]
            if len(hist):
                # rowwise: one active row must carry both eligible codes
                ok = bool(
                    (
                        hist["shrcd"].isin(VALID_SHRCD)
                        & hist["exchcd"].isin(VALID_EXCHCD)
                    ).any()
                )
        flags.append(ok)
    out = merged.copy()
    out["common_stock_screen_pass"] = flags
    return out


def selection_diagnostics(resolved: pd.DataFrame) -> pd.DataFrame:
    """Tier composition table (SEC-007); D rows are counted, never dropped."""
    diag = (
        resolved.groupby(["match_tier", "resolution_basis"])
        .size()
        .rename("n")
        .reset_index()
    )
    diag["share"] = diag["n"] / len(resolved) if len(resolved) else 0.0
    return diag
