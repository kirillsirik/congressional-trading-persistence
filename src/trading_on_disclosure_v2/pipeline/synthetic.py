"""Synthetic universe for the canonical dry run (directive section J.1).

Everything generated here is null-effect market data plus mechanically
plausible filings; outputs are labeled SYNTHETIC_DRY_RUN by the entry point.
Deterministic under a seed string; no real data touched.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.timing import PUBLIC_DATE_COL


def _rng(seed_text: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def synthetic_universe(
    seed_text: str = "two-clock-synthetic-dry-run-v1",
    n_filers: int = 12,
    n_securities: int = 15,
    n_transactions: int = 400,
    n_days: int = 420,
) -> dict:
    rng = _rng(seed_text)
    calendar = pd.bdate_range("2024-01-02", periods=n_days)
    permnos = [90000 + i for i in range(n_securities)]

    daily = []
    for permno in permnos:
        rets = rng.standard_t(6, size=n_days) * 0.015
        daily.append(
            pd.DataFrame({"permno": permno, "date": calendar, "ret": rets, "dlret": np.nan})
        )
    daily = pd.concat(daily, ignore_index=True)
    benchmark = pd.Series(rng.standard_normal(n_days) * 0.008, index=calendar)
    benchmark_ew = benchmark + rng.standard_normal(n_days) * 0.002
    risk_free = pd.Series(np.full(n_days, 0.0002), index=calendar)
    factors = pd.DataFrame(
        {
            "mkt_rf": benchmark.to_numpy() - 0.0002,
            **{
                f: rng.standard_normal(n_days) * 0.004
                for f in ("smb", "hml", "rmw", "cma", "umd")
            },
            "rf": 0.0002,
        },
        index=calendar,
    )

    names = pd.DataFrame(
        {
            "permno": permnos,
            "ticker": [f"SYN{i}" for i in range(n_securities)],
            "name_norm": [f"SYNTHETIC ISSUER {i}" for i in range(n_securities)],
            "namedt": calendar[0],
            "nameendt": calendar[-1],
            "shrcd": 10,
            "exchcd": 1,
        }
    )

    entry_positions = rng.integers(10, n_days - 140, size=n_transactions)
    txn_dates = calendar[entry_positions - rng.integers(2, 12, size=n_transactions)]
    pub_dates = calendar[entry_positions]
    txns = pd.DataFrame(
        {
            "transaction_uid": [f"synt-{i:05d}" for i in range(n_transactions)],
            "filer_uid": rng.choice(
                [f"SYNF{j:03d}" for j in range(n_filers)], size=n_transactions
            ),
            "document_uid": [f"syndoc-{i // 3:05d}" for i in range(n_transactions)],
            "permno": rng.choice(permnos, size=n_transactions),
            "direction": rng.choice(["purchase", "sale"], size=n_transactions),
            "transaction_date_reported": txn_dates,
            PUBLIC_DATE_COL: pub_dates,
            "ticker_text": None,
            "security_name_text": None,
            "reporting_lag_days": (pub_dates - txn_dates).days,
            "document_format": rng.choice(
                ["electronic_structured", "scanned_machine_printed"],
                size=n_transactions,
            ),
        }
    )
    txns["ticker_text"] = txns["permno"].map(
        dict(zip(names["permno"], names["ticker"], strict=True))
    )
    txns["security_name_text"] = txns["permno"].map(
        dict(zip(names["permno"], names["name_norm"], strict=True))
    )

    return {
        "calendar": calendar,
        "daily": daily,
        "benchmark": benchmark,
        "benchmark_ew": benchmark_ew,
        "risk_free": risk_free,
        "factors": factors,
        "crsp_names": names,
        "canonical_transactions": txns,
    }


def synthetic_universe_locked(
    seed_text: str = "two-clock-locked-synthetic-dry-run-v1",
    vendor_only_fraction: float = 0.25,
    **kwargs,
) -> dict:
    """Locked-design universe: source-native date fields (§3), vendor-only
    rows exercising the CT-minus-one proxy, a mixed-direction report, an
    exact amendment duplicate, and non-directional rows for the attrition
    path."""
    universe = synthetic_universe(seed_text=seed_text, **kwargs)
    rng = _rng(seed_text + "|locked")
    txns = universe["canonical_transactions"].copy()

    pub = pd.to_datetime(txns[PUBLIC_DATE_COL])
    vendor = rng.random(len(txns)) < vendor_only_fraction
    txns["record_source"] = np.where(vendor, "capitol_trades", "official")
    txns["official_public_date"] = pub.where(~vendor, pd.NaT)
    # vendor publication one calendar day AFTER availability, so the CT-1
    # proxy reproduces the true date on synthetic data
    txns["capitol_trades_publication_date_raw"] = (
        pub + pd.Timedelta(days=1)
    ).where(vendor, pd.NaT)
    tx_date = pd.to_datetime(txns["transaction_date_reported"])
    txns["trade_date_official"] = tx_date.where(~vendor, pd.NaT)
    txns["trade_date_capitol_trades_raw"] = (
        tx_date - pd.Timedelta(days=1)
    ).where(vendor, pd.NaT)
    txns = txns.drop(columns=[PUBLIC_DATE_COL, "transaction_date_reported"])
    txns["owner"] = rng.choice(["self", "spouse", "joint"], size=len(txns))
    txns["amount_bracket"] = rng.choice(
        ["$1,001 - $15,000", "$15,001 - $50,000"], size=len(txns)
    )
    txns["source_native_household_id"] = "src-" + txns["filer_uid"].astype(str)
    txns["is_amendment"] = False
    txns["amendment_parent_uid"] = pd.NA
    txns["chamber"] = rng.choice(["house", "senate"], size=len(txns))
    txns["report_year"] = 2024

    # a mixed-direction report: force one document to carry both directions
    # on the same security
    doc = txns.iloc[0]
    forced = txns.index[:2]
    txns.loc[forced, "document_uid"] = doc["document_uid"]
    txns.loc[forced, "permno"] = doc["permno"]
    txns.loc[forced, "filer_uid"] = doc["filer_uid"]
    txns.loc[forced[0], "direction"] = "purchase"
    txns.loc[forced[1], "direction"] = "sale"
    for col in (
        "official_public_date",
        "capitol_trades_publication_date_raw",
        "trade_date_official",
        "trade_date_capitol_trades_raw",
        "record_source",
    ):
        txns.loc[forced[1], col] = txns.loc[forced[0], col]

    # an exact amendment duplicate of row 2 in a later amendment document
    src = txns.iloc[2]
    dup = src.copy()
    dup["transaction_uid"] = "synt-amend-dup"
    dup["document_uid"] = str(src["document_uid"]) + "-A"
    dup["is_amendment"] = True
    dup["amendment_parent_uid"] = src["document_uid"]
    if pd.notna(src["official_public_date"]):
        dup["official_public_date"] = src["official_public_date"] + pd.Timedelta(days=14)
    else:
        dup["capitol_trades_publication_date_raw"] = src[
            "capitol_trades_publication_date_raw"
        ] + pd.Timedelta(days=14)
    # two non-directional rows for the §5 exclusion path
    other = txns.iloc[3].copy()
    other["transaction_uid"] = "synt-exchange"
    other["direction"] = "other"
    txns = pd.concat(
        [txns, dup.to_frame().T, other.to_frame().T], ignore_index=True
    )
    universe["canonical_transactions"] = txns
    return universe
