"""Amendment parent-candidate generation from official metadata (VER-001).

Candidates are evidence hints for the human review queue and are never
auto-accepted. The filer key is the full normalized name plus chamber —
surname-only matching is forbidden (ID-002). No outcome, vendor, or legacy
matched-row field is read here.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HONORIFIC_TOKENS = frozenset(
    {"hon", "mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv", "esq"}
)

_FILING_COLUMNS = (
    "filing_id",
    "chamber",
    "filer_name_raw",
    "first_name_raw",
    "last_name_raw",
    "report_type",
    "report_year",
    "filing_date",
    "received_date",
    "is_amendment",
    "amends_filing_id",
    "source_url",
)


@dataclass(frozen=True)
class GenerationParams:
    window_days: int = 400
    year_slack: int = 1


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.split(r"[\s,./]+", text.strip().lower()):
        tok = re.sub(r"[^a-z'-]", "", tok)
        if tok and tok not in _HONORIFIC_TOKENS:
            out.append(tok)
    return out


def normalize_name_key(first: str | None, last: str | None, raw: str | None) -> str:
    """Full-name key, never surname alone (ID-002)."""
    first = (first or "").strip()
    last = (last or "").strip()
    if first and last:
        toks = _tokens(f"{first} {last}")
        return " ".join(toks)
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "," in raw:
        last_part, _, first_part = raw.partition(",")
        toks = _tokens(f"{first_part} {last_part}")
    else:
        toks = _tokens(raw)
    if len(toks) < 2:
        return ""  # a single token is surname-or-forename alone: unusable key
    return " ".join(toks)


def filings_content_digest(conn: sqlite3.Connection) -> str:
    """SRC-005 read-only check: digest of every filing's state."""
    h = hashlib.sha256()
    for fid, state in conn.execute("SELECT filing_id, state FROM filings ORDER BY filing_id"):
        h.update(f"{fid}|{state}\n".encode())
    return h.hexdigest()


def load_ptr_filings(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT {', '.join(_FILING_COLUMNS)} FROM filings WHERE report_type = 'ptr'",
        conn,
    )
    df["name_key"] = [
        normalize_name_key(first, last, raw)
        for first, last, raw in zip(
            df["first_name_raw"], df["last_name_raw"], df["filer_name_raw"], strict=True
        )
    ]
    filing = pd.to_datetime(df["filing_date"].replace("", None), errors="coerce")
    received = pd.to_datetime(df["received_date"].replace("", None), errors="coerce")
    df["best_date"] = filing.fillna(received)
    df["best_date_source"] = np.select(
        [filing.notna(), received.notna()], ["filing_date", "received_date"], default="none"
    )
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce")
    return df


def unlinked_amendments(filings: pd.DataFrame) -> pd.DataFrame:
    mask = (filings["is_amendment"] == 1) & (
        filings["amends_filing_id"].isna() | (filings["amends_filing_id"] == "")
    )
    return filings.loc[mask].reset_index(drop=True)


def generate_candidates(
    conn: sqlite3.Connection, params: GenerationParams | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (amendments, candidates); every candidate row carries the
    evidence fields the ranking used. Ranking is deterministic and advisory
    (VER-001: no first-candidate auto-selection)."""
    params = params or GenerationParams()
    filings = load_ptr_filings(conn)
    amendments = unlinked_amendments(filings)
    pool = filings.loc[filings["is_amendment"] == 0]

    rows = []
    for amend in amendments.itertuples(index=False):
        if not amend.name_key:
            continue
        cand = pool.loc[
            (pool["chamber"] == amend.chamber) & (pool["name_key"] == amend.name_key)
        ]
        if cand.empty:
            continue
        year_ok = (cand["report_year"] - amend.report_year).abs() <= params.year_slack
        cand = cand.loc[year_ok.fillna(True)]
        if cand.empty:
            continue
        gap = (pd.Timestamp(amend.best_date) - cand["best_date"]).dt.days
        date_comparable = pd.notna(amend.best_date) & cand["best_date"].notna()
        if pd.notna(amend.best_date):
            keep = ~date_comparable | ((gap >= 0) & (gap <= params.window_days))
            cand, gap, date_comparable = (
                cand.loc[keep],
                gap.loc[keep],
                date_comparable.loc[keep],
            )
        if cand.empty:
            continue
        exact = (
            cand["filer_name_raw"].fillna("").str.casefold().str.strip()
            == str(amend.filer_name_raw or "").casefold().strip()
        )
        block = pd.DataFrame(
            {
                "amendment_filing_id": amend.filing_id,
                "candidate_filing_id": cand["filing_id"].to_numpy(),
                "chamber": amend.chamber,
                "name_key": amend.name_key,
                "amendment_name_raw": amend.filer_name_raw,
                "candidate_name_raw": cand["filer_name_raw"].to_numpy(),
                "name_match_level": np.where(exact, "exact_raw", "normalized"),
                "amendment_report_year": amend.report_year,
                "candidate_report_year": cand["report_year"].to_numpy(),
                "same_report_year": (cand["report_year"] == amend.report_year).to_numpy(),
                "amendment_date": amend.best_date,
                "candidate_date": cand["best_date"].to_numpy(),
                "date_comparable": date_comparable.to_numpy(),
                "date_gap_days": gap.where(date_comparable).to_numpy(),
                "candidate_source_url": cand["source_url"].to_numpy(),
            }
        )
        rows.append(block)

    if rows:
        candidates = pd.concat(rows, ignore_index=True)
    else:
        candidates = pd.DataFrame(
            columns=[
                "amendment_filing_id",
                "candidate_filing_id",
                "chamber",
                "name_key",
                "amendment_name_raw",
                "candidate_name_raw",
                "name_match_level",
                "amendment_report_year",
                "candidate_report_year",
                "same_report_year",
                "amendment_date",
                "candidate_date",
                "date_comparable",
                "date_gap_days",
                "candidate_source_url",
            ]
        )

    candidates = candidates.sort_values(
        by=[
            "amendment_filing_id",
            "date_comparable",
            "date_gap_days",
            "name_match_level",
            "same_report_year",
            "candidate_filing_id",
        ],
        ascending=[True, False, True, True, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["candidate_rank"] = candidates.groupby("amendment_filing_id").cumcount() + 1
    return amendments, candidates
