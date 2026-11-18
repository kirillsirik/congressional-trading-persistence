"""Quality validation for canonical scraped Capitol Trades data.

``validate_canonical`` computes a structured quality-report dict from a
canonical DataFrame (``schemas.CANONICAL_COLUMNS``); ``report_to_markdown``
renders it as the ``scrape_quality_report.md`` written into every snapshot.
Threshold-based warnings come from the ``validation`` section of the scrape
config and are surfaced both in the report and in the run manifest.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "max_negative_lag_share": 0.01,
    "min_date_parse_rate": 0.99,
    "max_unmapped_tx_type_share": 0.01,
    "max_size_parse_failure_share": 0.01,
}


def _share(numer: int, denom: int) -> float:
    return round(numer / denom, 6) if denom else 0.0


def _value_counts(series: pd.Series, top: int | None = None) -> dict[str, int]:
    counts = series.fillna("").astype(str).replace("", "<missing>").value_counts()
    if top is not None:
        counts = counts.head(top)
    return {str(k): int(v) for k, v in counts.items()}


def _lag_summary(lag: pd.Series) -> dict[str, float | None]:
    lag = pd.to_numeric(lag, errors="coerce").dropna()
    if lag.empty:
        return {k: None for k in ("count", "mean", "min", "p25", "median", "p75", "p95", "max")}
    return {
        "count": int(lag.count()),
        "mean": round(float(lag.mean()), 2),
        "min": float(lag.min()),
        "p25": float(lag.quantile(0.25)),
        "median": float(lag.median()),
        "p75": float(lag.quantile(0.75)),
        "p95": float(lag.quantile(0.95)),
        "max": float(lag.max()),
    }


def validate_canonical(df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the scrape quality report for a canonical trades DataFrame."""
    cfg = dict(DEFAULT_THRESHOLDS)
    cfg.update((config or {}).get("validation", {}))
    n = len(df)
    warnings: list[str] = []
    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": n,
        "thresholds": {k: v for k, v in cfg.items() if not isinstance(v, list)},
    }
    if n == 0:
        report["warnings"] = ["dataset is empty"]
        return report

    # ── Date parsing ─────────────────────────────────────────────────────────
    trade_ok = df["trade_date"].notna() & (df["trade_date"].astype(str) != "")
    pub_ok = df["published_date"].notna() & (df["published_date"].astype(str) != "")
    report["date_parsing"] = {
        "trade_date_parse_rate": _share(int(trade_ok.sum()), n),
        "published_date_parse_rate": _share(int(pub_ok.sum()), n),
        "relative_published_dates": int(
            df["parser_warnings"].fillna("").str.contains("relative_date").sum()
        ),
    }
    for col, rate in (
        ("trade_date", report["date_parsing"]["trade_date_parse_rate"]),
        ("published_date", report["date_parsing"]["published_date_parse_rate"]),
    ):
        if rate < cfg["min_date_parse_rate"]:
            warnings.append(
                f"{col} parse rate {rate:.4f} below threshold {cfg['min_date_parse_rate']}"
            )

    # ── Disclosure lag ───────────────────────────────────────────────────────
    lag = pd.to_numeric(df["disclosure_lag_days"], errors="coerce")
    neg = int((lag < 0).sum())
    neg_share = _share(neg, n)
    report["disclosure_lag"] = {
        "negative_lag_count": neg,
        "negative_lag_share": neg_share,
        "lag_mismatch_flags": int(
            df["parser_warnings"].fillna("").str.contains("lag_mismatch").sum()
        ),
        "distribution": _lag_summary(lag),
    }
    if neg_share > cfg["max_negative_lag_share"]:
        warnings.append(
            f"negative disclosure-lag share {neg_share:.4f} exceeds "
            f"{cfg['max_negative_lag_share']}"
        )

    # ── Transaction-type mapping ─────────────────────────────────────────────
    unmapped = int((df["transaction_type_clean"] == "other").sum())
    unmapped_share = _share(unmapped, n)
    report["transaction_types"] = {
        "mapped_share": round(1.0 - unmapped_share, 6),
        "unmapped_count": unmapped,
        "counts": _value_counts(df["transaction_type_clean"]),
    }
    if unmapped_share > cfg["max_unmapped_tx_type_share"]:
        warnings.append(
            f"unmapped transaction-type share {unmapped_share:.4f} exceeds "
            f"{cfg['max_unmapped_tx_type_share']}"
        )

    # ── Tickers / sizes ──────────────────────────────────────────────────────
    empty_ticker = int((df["ticker_clean"].fillna("").astype(str).str.strip() == "").sum())
    report["tickers"] = {
        "empty_ticker_count": empty_ticker,
        "empty_ticker_share": _share(empty_ticker, n),
    }
    size_present = df["size_raw"].fillna("").astype(str).str.strip() != ""
    size_failed = int((size_present & df["size_lower"].isna()).sum())
    size_fail_share = _share(size_failed, int(size_present.sum()))
    report["sizes"] = {
        "size_present_count": int(size_present.sum()),
        "size_parse_failures": size_failed,
        "size_parse_failure_share": size_fail_share,
    }
    if size_fail_share > cfg["max_size_parse_failure_share"]:
        warnings.append(
            f"size-bracket parse failure share {size_fail_share:.4f} exceeds "
            f"{cfg['max_size_parse_failure_share']}"
        )

    # ── Duplicates: exact vs conflicting ─────────────────────────────────────
    dup_mask = df["stable_record_key"].duplicated(keep=False)
    dup_rows = int(dup_mask.sum())
    exact_groups = conflicting_groups = 0
    if dup_rows:
        nunique = df.loc[dup_mask].groupby("stable_record_key")["row_content_hash"].nunique()
        exact_groups = int((nunique == 1).sum())
        conflicting_groups = int((nunique > 1).sum())
    report["duplicates"] = {
        "duplicate_key_rows": dup_rows,
        "duplicate_key_groups": exact_groups + conflicting_groups,
        "exact_duplicate_groups": exact_groups,
        "conflicting_duplicate_groups": conflicting_groups,
    }
    if conflicting_groups:
        warnings.append(
            f"{conflicting_groups} stable_record_key groups have conflicting "
            "content (same identity, different row_content_hash)"
        )

    # ── Required fields (from config) ────────────────────────────────────────
    missing_required: dict[str, int] = {}
    for col in (config or {}).get("validation", {}).get("require_fields", []):
        source_col = {"transaction_type_raw": "transaction_type_raw"}.get(col, col)
        if source_col in df.columns:
            missing = int((df[source_col].fillna("").astype(str).str.strip() == "").sum())
            if missing:
                missing_required[col] = missing
                warnings.append(f"required field {col} empty in {missing} records")
    report["missing_required_fields"] = missing_required

    # ── Composition and coverage ─────────────────────────────────────────────
    pub = df["published_date"].dropna().astype(str)
    report["counts"] = {
        "by_chamber": _value_counts(df["chamber"]),
        "by_party": _value_counts(df["party"]),
        "by_transaction_type": _value_counts(df["transaction_type_clean"]),
        "by_owner": _value_counts(df["owner_clean"]),
        "by_published_month": _value_counts(pub.str[:7]),
    }
    trade = df["trade_date"].dropna().astype(str)
    report["date_ranges"] = {
        "trade_date_min": trade.min() if not trade.empty else None,
        "trade_date_max": trade.max() if not trade.empty else None,
        "published_date_min": pub.min() if not pub.empty else None,
        "published_date_max": pub.max() if not pub.empty else None,
    }

    report["warnings"] = warnings
    for w in warnings:
        logger.warning("quality: %s", w)
    return report


def _md_table(d: dict[str, Any]) -> list[str]:
    lines = ["| key | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    return lines


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render the quality-report dict as markdown for the snapshot folder."""
    lines: list[str] = [
        "# Capitol Trades scrape quality report",
        "",
        f"Generated: {report.get('generated_utc', '')}  ",
        f"Records: **{report.get('n_records', 0)}**",
        "",
    ]
    warnings = report.get("warnings", [])
    lines.append("## Threshold warnings")
    lines.append("")
    if warnings:
        lines += [f"- WARNING: {w}" for w in warnings]
    else:
        lines.append("- none")
    lines.append("")

    sections = [
        ("Date parsing", report.get("date_parsing")),
        ("Disclosure lag", None),  # custom below
        ("Transaction types", None),
        ("Tickers", report.get("tickers")),
        ("Sizes", report.get("sizes")),
        ("Duplicates", report.get("duplicates")),
        ("Date ranges", report.get("date_ranges")),
    ]
    for title, payload in sections:
        if title == "Disclosure lag" and "disclosure_lag" in report:
            lag = dict(report["disclosure_lag"])
            dist = lag.pop("distribution", {})
            lines += ["## Disclosure lag", ""] + _md_table(lag) + [""]
            lines += ["### Lag distribution (days)", ""] + _md_table(dist) + [""]
        elif title == "Transaction types" and "transaction_types" in report:
            tx = dict(report["transaction_types"])
            counts = tx.pop("counts", {})
            lines += ["## Transaction types", ""] + _md_table(tx) + [""]
            lines += ["### Counts by clean type", ""] + _md_table(counts) + [""]
        elif payload:
            lines += [f"## {title}", ""] + _md_table(payload) + [""]

    for key, title in (
        ("by_chamber", "By chamber"),
        ("by_party", "By party"),
        ("by_owner", "By owner"),
        ("by_published_month", "By published month"),
    ):
        counts = report.get("counts", {}).get(key)
        if counts:
            lines += [f"## {title}", ""] + _md_table(counts) + [""]

    missing = report.get("missing_required_fields") or {}
    if missing:
        lines += ["## Missing required fields", ""] + _md_table(missing) + [""]
    return "\n".join(lines) + "\n"
