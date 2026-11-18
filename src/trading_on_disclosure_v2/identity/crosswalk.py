"""TC-050: candidate date-effective filer identity crosswalk and HD-05 evidence.

Every identity unit is keyed by ``filer_uid`` (ID-001) and derived from official
filing metadata only — no outcomes, no vendor rows, no market data. The output is
a CANDIDATE crosswalk: ambiguous units are surfaced with reason codes for a human
reviewer and are never auto-merged or auto-split (ID-002, ID-003). The
name-normalisation rule is imported from the amendment-candidate module so a
single implementation governs both.

Five HD-05 evidence classes are produced as separate reviewable frames:

a) merge/split candidates    - :func:`find_merge_candidates`, :func:`find_split_candidates`
b) common surnames           - :func:`find_surname_groups`
c) cross-state/chamber moves - :func:`find_transitions`
d) top-volume households     - :func:`top_volume_households`
e) service-date validation   - :func:`validate_service_dates`
"""

from __future__ import annotations

import hashlib
import itertools
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from ..versions.amendment_candidates import normalize_name_key

#: Reason codes that force a unit into human review. Nothing here is resolvable
#: by the pipeline: each one is a decision the reviewer must record.
REASON_MERGE_CANDIDATE = "ID-CW-001-NAME_VARIANT_MERGE_CANDIDATE"
REASON_SPLIT_DISTRICT_OVERLAP = "ID-CW-002-SPLIT_CANDIDATE_DISTRICT_OVERLAP"
REASON_SPLIT_DORMANCY = "ID-CW-003-SPLIT_CANDIDATE_DORMANCY_GAP"
REASON_CHAMBER_TRANSITION = "ID-CW-004-CHAMBER_TRANSITION_CANDIDATE"
REASON_SERVICE_UNVERIFIABLE = "ID-CW-005-SERVICE_DATE_UNVERIFIABLE"
REASON_REPORT_YEAR_CONFLICT = "ID-CW-006-REPORT_YEAR_DATE_CONFLICT"
REASON_UNUSABLE_NAME_KEY = "ID-CW-007-UNUSABLE_NAME_KEY"
REASON_STATE_TRANSITION = "ID-CW-008-STATE_TRANSITION_CANDIDATE"
REASON_AMENDMENT_DATE_ORDER = "ID-CW-009-AMENDMENT_DATE_ORDER_CONFLICT"

BLOCKING_REASONS = (
    REASON_MERGE_CANDIDATE,
    REASON_SPLIT_DISTRICT_OVERLAP,
    REASON_SPLIT_DORMANCY,
    REASON_CHAMBER_TRANSITION,
    REASON_SERVICE_UNVERIFIABLE,
    REASON_REPORT_YEAR_CONFLICT,
    REASON_UNUSABLE_NAME_KEY,
    REASON_STATE_TRANSITION,
    REASON_AMENDMENT_DATE_ORDER,
)

#: Non-blocking observations. They are evidence the reviewer sees, not defects.
FLAG_SHARED_SURNAME = "SHARED_SURNAME_NOT_MERGED"
FLAG_DISTRICT_CHANGE_DISJOINT = "DISTRICT_CHANGE_DISJOINT"
FLAG_MISSING_DATE_SOME = "EFFECTIVE_DATE_MISSING_SOME"
FLAG_TOP_VOLUME = "TOP_VOLUME_HOUSEHOLD"

STATUS_OK = "unambiguous"
STATUS_REVIEW = "needs_human_review"

_FILING_COLUMNS = (
    "filing_id",
    "chamber",
    "filer_name_raw",
    "first_name_raw",
    "last_name_raw",
    "state_district_raw",
    "report_year",
    "filing_date",
    "received_date",
    "is_amendment",
    "amends_filing_id",
)


@dataclass(frozen=True)
class ClassificationParams:
    """Thresholds for the classification rules.

    ``dormancy_gap_days`` is deliberately longer than one congressional term of
    filing silence, so that ordinary low-activity filers are not sent to review.
    """

    dormancy_gap_days: int = 1096
    report_year_slack: int = 1
    top_volume_n: int = 25


@dataclass(frozen=True)
class CrosswalkResult:
    units: pd.DataFrame
    assignments: pd.DataFrame
    merge_candidates: pd.DataFrame
    split_candidates: pd.DataFrame
    surname_groups: pd.DataFrame
    transitions: pd.DataFrame
    top_volume: pd.DataFrame
    service_dates: pd.DataFrame
    summary: dict


def make_filer_uid(chamber: str | None, name_key: str) -> str:
    """Deterministic, name-free identity id for one (chamber, full-name key).

    The digest keeps person names out of the identifier itself, so the id can be
    quoted in reports and code while names stay in the Dropbox-only evidence.
    """
    if not name_key:
        return ""
    ch = (chamber or "").strip().lower()
    digest = hashlib.sha256(f"{ch}|{name_key}".encode()).hexdigest()[:12]
    return f"FUID-{digest}"


def _unresolved_uid(chamber: str | None, filing_id: str) -> str:
    """One unit per filing when the name key is unusable: never a silent merge."""
    ch = (chamber or "").strip().lower()
    digest = hashlib.sha256(f"{ch}|{filing_id}".encode()).hexdigest()[:12]
    return f"FUID-UNRESOLVED-{digest}"


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _join(values) -> str:
    return "|".join(sorted({str(v) for v in values if v not in (None, "") and pd.notna(v)}))


def _edit_distance_le_1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    seen_diff = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if seen_diff:
            return False
        seen_diff = True
        if len(a) == len(b):
            i += 1
        j += 1
    return True


def load_ptr_filings(conn: sqlite3.Connection) -> pd.DataFrame:
    """Filing-level identity input built from official PTR metadata only."""
    df = pd.read_sql_query(
        f"SELECT {', '.join(_FILING_COLUMNS)} FROM filings WHERE report_type = 'ptr'",
        conn,
    )
    tx = pd.read_sql_query(
        """SELECT t.filing_id AS filing_id, COUNT(*) AS n_transactions
           FROM filing_transactions_raw t
           JOIN filings f ON f.filing_id = t.filing_id
           WHERE f.report_type = 'ptr'
           GROUP BY t.filing_id""",
        conn,
    )
    df = df.merge(tx, on="filing_id", how="left")
    return prepare_filings(df)


def prepare_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Derive name key, filer_uid, effective date and geography for each filing."""
    out = df.copy()
    for col in ("first_name_raw", "last_name_raw", "filer_name_raw", "state_district_raw"):
        if col not in out.columns:
            out[col] = ""
    for col in ("filing_date", "received_date", "amends_filing_id"):
        if col not in out.columns:
            out[col] = None
    if "n_transactions" not in out.columns:
        out["n_transactions"] = 0
    if "is_amendment" not in out.columns:
        out["is_amendment"] = 0
    out["n_transactions"] = pd.to_numeric(out["n_transactions"], errors="coerce").fillna(0).astype(int)
    out["is_amendment"] = pd.to_numeric(out["is_amendment"], errors="coerce").fillna(0).astype(int)

    out["name_key"] = [
        normalize_name_key(first, last, raw)
        for first, last, raw in zip(
            out["first_name_raw"], out["last_name_raw"], out["filer_name_raw"], strict=True
        )
    ]
    out["name_key_usable"] = out["name_key"].astype(bool)
    out["filer_uid"] = [
        make_filer_uid(ch, key) if key else _unresolved_uid(ch, fid)
        for ch, key, fid in zip(out["chamber"], out["name_key"], out["filing_id"], strict=True)
    ]
    out["surname"] = [k.split()[-1] if k else "" for k in out["name_key"]]
    out["first_token"] = [k.split()[0] if k else "" for k in out["name_key"]]

    filing_dt = _to_dt(out["filing_date"])
    received_dt = _to_dt(out["received_date"])
    out["effective_date"] = filing_dt.fillna(received_dt)
    out["effective_date_source"] = [
        "filing_date" if pd.notna(f) else ("received_date" if pd.notna(r) else "")
        for f, r in zip(filing_dt, received_dt, strict=True)
    ]

    sd = out["state_district_raw"].fillna("").astype(str).str.strip().str.upper()
    out["state_code"] = sd.str[:2].where(sd.str.len() >= 2, "")
    out["district_code"] = sd
    if "report_year" not in out.columns:
        out["report_year"] = pd.NA
    out["report_year"] = pd.to_numeric(out["report_year"], errors="coerce")
    return out


def build_filer_units(filings: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate identity unit (``filer_uid``)."""
    rows = []
    for uid, g in filings.groupby("filer_uid", sort=True):
        dates = g["effective_date"].dropna()
        rows.append(
            {
                "filer_uid": uid,
                "chamber": _join(g["chamber"]),
                "name_key": _join(g["name_key"]),
                "surname": _join(g["surname"]),
                "first_token": _join(g["first_token"]),
                "name_key_usable": bool(g["name_key_usable"].all()),
                "n_filings": int(len(g)),
                "n_amendments": int(g["is_amendment"].sum()),
                "n_transactions": int(g["n_transactions"].sum()),
                "first_effective_date": dates.min() if len(dates) else pd.NaT,
                "last_effective_date": dates.max() if len(dates) else pd.NaT,
                "n_missing_effective_date": int(g["effective_date"].isna().sum()),
                "states": _join(g["state_code"]),
                "districts": _join(g["district_code"]),
                "n_states": int(g.loc[g["state_code"] != "", "state_code"].nunique()),
                "n_districts": int(g.loc[g["district_code"] != "", "district_code"].nunique()),
                "first_report_year": g["report_year"].min(),
                "last_report_year": g["report_year"].max(),
            }
        )
    units = pd.DataFrame(rows)
    if units.empty:
        units = pd.DataFrame(columns=["filer_uid", "chamber", "name_key"])
    return units


def _variant_relation(key_a: str, key_b: str) -> str:
    """Name-shape relation between two full-name keys, or '' when unrelated.

    Surname equality alone is never a relation (ID-002): it returns ''.
    """
    ta, tb = key_a.split(), key_b.split()
    if not ta or not tb or ta == tb:
        return ""
    if sorted(ta) == sorted(tb):
        return "token_order_variant"
    if ta[-1] != tb[-1]:
        return ""
    if ta[0] == tb[0]:
        short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if len(short) < len(long_) and set(short).issubset(set(long_)):
            return "middle_token_variant"
        return "same_first_and_last_token_variant"
    a0, b0 = ta[0], tb[0]
    if (len(a0) == 1 and b0.startswith(a0)) or (len(b0) == 1 and a0.startswith(b0)):
        return "initial_expansion_variant"
    if len(a0) >= 4 and len(b0) >= 4 and _edit_distance_le_1(a0, b0):
        return "first_token_near_match_variant"
    return ""


def _overlap_days(a_start, a_end, b_start, b_end) -> float | None:
    if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
        return None
    lo, hi = max(a_start, b_start), min(a_end, b_end)
    return float((hi - lo).days) if hi >= lo else float((hi - lo).days)


def find_merge_candidates(units: pd.DataFrame) -> pd.DataFrame:
    """HD-05(a) merge side: pairs of units whose names may denote one household.

    Every pair is a question for the reviewer. ``auto_merged`` is structurally
    False: this function never rewrites a ``filer_uid``.
    """
    cols = [
        "uid_a",
        "uid_b",
        "relation",
        "surname",
        "name_key_a",
        "name_key_b",
        "chamber_a",
        "chamber_b",
        "same_chamber",
        "states_a",
        "states_b",
        "districts_a",
        "districts_b",
        "shared_district",
        "span_a_start",
        "span_a_end",
        "span_b_start",
        "span_b_end",
        "span_overlap_days",
        "n_filings_a",
        "n_filings_b",
        "auto_merged",
        "human_decision",
    ]
    if units.empty:
        return pd.DataFrame(columns=cols)

    # Two blocking keys: shared surname token, and the same tokens in any order
    # (a source that emitted the name reversed). Neither is a merge by itself.
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for rec in units.to_dict("records"):
        key = rec.get("name_key") or ""
        if not key:
            continue
        buckets[("surname", rec["surname"])].append(rec)
        buckets[("tokens", " ".join(sorted(key.split())))].append(rec)

    rows = []
    seen: set[tuple[str, str]] = set()
    for _, recs in sorted(buckets.items()):
        for a, b in itertools.combinations(sorted(recs, key=lambda r: r["filer_uid"]), 2):
            pair = (a["filer_uid"], b["filer_uid"])
            if pair in seen:
                continue
            relation = _variant_relation(a["name_key"], b["name_key"])
            if not relation:
                continue
            seen.add(pair)
            da = set(str(a["districts"]).split("|")) - {""}
            db = set(str(b["districts"]).split("|")) - {""}
            rows.append(
                {
                    "uid_a": a["filer_uid"],
                    "uid_b": b["filer_uid"],
                    "relation": relation,
                    "surname": a["surname"],
                    "name_key_a": a["name_key"],
                    "name_key_b": b["name_key"],
                    "chamber_a": a["chamber"],
                    "chamber_b": b["chamber"],
                    "same_chamber": a["chamber"] == b["chamber"],
                    "states_a": a["states"],
                    "states_b": b["states"],
                    "districts_a": a["districts"],
                    "districts_b": b["districts"],
                    "shared_district": _join(da & db),
                    "span_a_start": a["first_effective_date"],
                    "span_a_end": a["last_effective_date"],
                    "span_b_start": b["first_effective_date"],
                    "span_b_end": b["last_effective_date"],
                    "span_overlap_days": _overlap_days(
                        a["first_effective_date"],
                        a["last_effective_date"],
                        b["first_effective_date"],
                        b["last_effective_date"],
                    ),
                    "n_filings_a": a["n_filings"],
                    "n_filings_b": b["n_filings"],
                    "auto_merged": False,
                    "human_decision": "PENDING",
                }
            )
    return pd.DataFrame(rows, columns=cols)


def find_split_candidates(
    filings: pd.DataFrame, params: ClassificationParams | None = None
) -> pd.DataFrame:
    """HD-05(a) split side: units that may cover more than one household.

    Two official signals: House districts whose active windows interleave under
    one name, and a filing silence longer than ``dormancy_gap_days``.
    """
    params = params or ClassificationParams()
    cols = [
        "filer_uid",
        "reason",
        "detail",
        "segment_a",
        "segment_a_start",
        "segment_a_end",
        "segment_b",
        "segment_b_start",
        "segment_b_end",
        "gap_days",
        "auto_split",
        "human_decision",
    ]
    rows = []
    for uid, g in filings.groupby("filer_uid", sort=True):
        dated = g[g["effective_date"].notna()]
        districts = [d for d in dated["district_code"].unique() if d]
        if len(districts) > 1:
            spans = (
                dated[dated["district_code"] != ""]
                .groupby("district_code")["effective_date"]
                .agg(["min", "max"])
                .sort_values("min")
            )
            for i in range(len(spans) - 1):
                gap = (spans["min"].iloc[i + 1] - spans["max"].iloc[i]).days
                if gap >= 0:
                    continue
                rows.append(
                    {
                        "filer_uid": uid,
                        "reason": REASON_SPLIT_DISTRICT_OVERLAP,
                        "detail": "district active windows interleave under one name key",
                        "segment_a": spans.index[i],
                        "segment_a_start": spans["min"].iloc[i],
                        "segment_a_end": spans["max"].iloc[i],
                        "segment_b": spans.index[i + 1],
                        "segment_b_start": spans["min"].iloc[i + 1],
                        "segment_b_end": spans["max"].iloc[i + 1],
                        "gap_days": float(gap),
                        "auto_split": False,
                        "human_decision": "PENDING",
                    }
                )
        ordered = dated.sort_values("effective_date")
        dates = list(ordered["effective_date"])
        for i in range(len(dates) - 1):
            gap = (dates[i + 1] - dates[i]).days
            if gap < params.dormancy_gap_days:
                continue
            rows.append(
                {
                    "filer_uid": uid,
                    "reason": REASON_SPLIT_DORMANCY,
                    "detail": f"filing silence of {gap} days between consecutive filings",
                    "segment_a": ordered["district_code"].iloc[i],
                    "segment_a_start": dates[i],
                    "segment_a_end": dates[i],
                    "segment_b": ordered["district_code"].iloc[i + 1],
                    "segment_b_start": dates[i + 1],
                    "segment_b_end": dates[i + 1],
                    "gap_days": float(gap),
                    "auto_split": False,
                    "human_decision": "PENDING",
                }
            )
    return pd.DataFrame(rows, columns=cols)


def prepare_corroborating_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Non-PTR official filings of the same store, reduced to (name key, chamber, date).

    They carry no transactions and never enter the crosswalk. Their only use is to
    tell the reviewer whether a filing silence in the PTR record coincides with a
    silence in the whole official record.
    """
    prepared = prepare_filings(df)
    return prepared[["name_key", "chamber", "effective_date"]].dropna(subset=["effective_date"])


def annotate_gap_corroboration(
    split_candidates: pd.DataFrame,
    filings: pd.DataFrame,
    corroborating: pd.DataFrame | None,
) -> pd.DataFrame:
    """Count other official filings inside each dormancy gap.

    A positive count means the filer kept filing other official reports through the
    silence, which points at 'no reportable trades' rather than a second person. It
    is evidence for the reviewer and never resolves the case by itself.
    """
    out = split_candidates.copy()
    if out.empty:
        out["n_corroborating_official_filings_in_gap"] = pd.Series(dtype="Int64")
        return out
    if corroborating is None or corroborating.empty:
        out["n_corroborating_official_filings_in_gap"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        return out

    key_by_uid = (
        filings.groupby("filer_uid")
        .agg(name_key=("name_key", "first"), chamber=("chamber", "first"))
        .to_dict("index")
    )
    covered_chambers = set(corroborating["chamber"].unique())
    counts = []
    for _, r in out.iterrows():
        if r["reason"] != REASON_SPLIT_DORMANCY:
            counts.append(pd.NA)
            continue
        meta = key_by_uid.get(r["filer_uid"])
        # No corroborating source for the chamber means "unknown", not "zero".
        if not meta or not meta["name_key"] or meta["chamber"] not in covered_chambers:
            counts.append(pd.NA)
            continue
        same = corroborating[
            (corroborating["name_key"] == meta["name_key"])
            & (corroborating["chamber"] == meta["chamber"])
        ]
        if same.empty:
            counts.append(0)
            continue
        inside = same[
            (same["effective_date"] > r["segment_a_end"])
            & (same["effective_date"] < r["segment_b_start"])
        ]
        counts.append(int(len(inside)))
    out["n_corroborating_official_filings_in_gap"] = pd.Series(counts, dtype="Int64").values
    return out


def find_surname_groups(units: pd.DataFrame, merge_candidates: pd.DataFrame) -> pd.DataFrame:
    """HD-05(b): surnames carried by more than one distinct identity unit.

    These units stay distinct. The frame exists so the reviewer can confirm that
    surname sharing never drove a merge (ID-002); the only surname pairs that are
    even proposed for merging are those with an independent full-name relation.
    """
    cols = [
        "surname",
        "n_units",
        "filer_uids",
        "first_tokens",
        "chambers",
        "states",
        "n_units_active_concurrently",
        "n_merge_candidate_pairs",
        "auto_merged_on_surname",
    ]
    if units.empty:
        return pd.DataFrame(columns=cols)

    proposed: dict[str, int] = defaultdict(int)
    if not merge_candidates.empty:
        for surname, n in merge_candidates.groupby("surname").size().items():
            proposed[surname] = int(n)

    rows = []
    for surname, g in units[units["surname"] != ""].groupby("surname", sort=True):
        if len(g) < 2:
            continue
        recs = g.to_dict("records")
        concurrent = 0
        for a, b in itertools.combinations(recs, 2):
            ov = _overlap_days(
                a["first_effective_date"],
                a["last_effective_date"],
                b["first_effective_date"],
                b["last_effective_date"],
            )
            if ov is not None and ov >= 0:
                concurrent += 1
        rows.append(
            {
                "surname": surname,
                "n_units": int(len(g)),
                "filer_uids": _join(g["filer_uid"]),
                "first_tokens": _join(g["first_token"]),
                "chambers": _join(g["chamber"]),
                "states": _join(itertools.chain.from_iterable(
                    str(s).split("|") for s in g["states"]
                )),
                "n_units_active_concurrently": concurrent,
                "n_merge_candidate_pairs": proposed.get(surname, 0),
                "auto_merged_on_surname": False,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def find_transitions(filings: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """HD-05(c): cross-chamber and cross-state/district movement.

    Chamber transitions appear as two chamber-scoped units sharing a name key —
    whether they are one household is a human decision, not a rule.
    """
    cols = [
        "transition_type",
        "name_key",
        "filer_uid_from",
        "filer_uid_to",
        "from_label",
        "to_label",
        "from_start",
        "from_end",
        "to_start",
        "to_end",
        "gap_days",
        "spans_disjoint",
        "blocking",
        "human_decision",
    ]
    rows = []

    if not units.empty:
        for name_key, g in units[units["name_key"] != ""].groupby("name_key", sort=True):
            if g["chamber"].nunique() < 2:
                continue
            recs = sorted(
                g.to_dict("records"),
                key=lambda r: (pd.Timestamp.max if pd.isna(r["first_effective_date"])
                               else r["first_effective_date"]),
            )
            for a, b in itertools.combinations(recs, 2):
                if a["chamber"] == b["chamber"]:
                    continue
                gap = (
                    None
                    if pd.isna(a["last_effective_date"]) or pd.isna(b["first_effective_date"])
                    else float((b["first_effective_date"] - a["last_effective_date"]).days)
                )
                rows.append(
                    {
                        "transition_type": "chamber",
                        "name_key": name_key,
                        "filer_uid_from": a["filer_uid"],
                        "filer_uid_to": b["filer_uid"],
                        "from_label": a["chamber"],
                        "to_label": b["chamber"],
                        "from_start": a["first_effective_date"],
                        "from_end": a["last_effective_date"],
                        "to_start": b["first_effective_date"],
                        "to_end": b["last_effective_date"],
                        "gap_days": gap,
                        "spans_disjoint": None if gap is None else gap >= 0,
                        "blocking": True,
                        "human_decision": "PENDING",
                    }
                )

    for uid, g in filings.groupby("filer_uid", sort=True):
        dated = g[g["effective_date"].notna() & (g["district_code"] != "")]
        if dated["district_code"].nunique() < 2:
            continue
        spans = dated.groupby("district_code")["effective_date"].agg(["min", "max"]).sort_values("min")
        for i in range(len(spans) - 1):
            from_d, to_d = spans.index[i], spans.index[i + 1]
            gap = float((spans["min"].iloc[i + 1] - spans["max"].iloc[i]).days)
            state_change = from_d[:2] != to_d[:2]
            rows.append(
                {
                    "transition_type": "state" if state_change else "district",
                    "name_key": _join(g["name_key"]),
                    "filer_uid_from": uid,
                    "filer_uid_to": uid,
                    "from_label": from_d,
                    "to_label": to_d,
                    "from_start": spans["min"].iloc[i],
                    "from_end": spans["max"].iloc[i],
                    "to_start": spans["min"].iloc[i + 1],
                    "to_end": spans["max"].iloc[i + 1],
                    "gap_days": gap,
                    "spans_disjoint": gap >= 0,
                    "blocking": bool(state_change or gap < 0),
                    "human_decision": "PENDING" if (state_change or gap < 0) else "",
                }
            )
    return pd.DataFrame(rows, columns=cols)


def top_volume_households(
    units: pd.DataFrame, params: ClassificationParams | None = None
) -> pd.DataFrame:
    """HD-05(d): the units that dominate the filing and transaction census."""
    params = params or ClassificationParams()
    cols = [
        "rank_by_transactions",
        "filer_uid",
        "chamber",
        "name_key",
        "n_filings",
        "n_transactions",
        "share_of_transactions",
        "cumulative_share_of_transactions",
        "share_of_filings",
        "first_effective_date",
        "last_effective_date",
    ]
    if units.empty:
        return pd.DataFrame(columns=cols)
    total_tx = max(int(units["n_transactions"].sum()), 1)
    total_f = max(int(units["n_filings"].sum()), 1)
    top = units.sort_values(
        ["n_transactions", "n_filings", "filer_uid"], ascending=[False, False, True]
    ).head(params.top_volume_n).copy()
    top["rank_by_transactions"] = range(1, len(top) + 1)
    top["share_of_transactions"] = top["n_transactions"] / total_tx
    top["cumulative_share_of_transactions"] = top["share_of_transactions"].cumsum()
    top["share_of_filings"] = top["n_filings"] / total_f
    return top[cols].reset_index(drop=True)


def validate_service_dates(
    filings: pd.DataFrame, units: pd.DataFrame, params: ClassificationParams | None = None
) -> pd.DataFrame:
    """HD-05(e): date-effective consistency of every unit's observed service span.

    Checks are internal to the official record: presence of an effective date,
    agreement between the effective date and the reported year, ordering of
    amendments against their parent, and the longest filing silence.
    """
    params = params or ClassificationParams()
    cols = [
        "filer_uid",
        "chamber",
        "n_filings",
        "n_missing_effective_date",
        "all_dates_missing",
        "first_effective_date",
        "last_effective_date",
        "observed_span_days",
        "n_report_year_mismatch",
        "max_report_year_gap",
        "n_amendment_before_parent",
        "max_dormancy_gap_days",
        "n_states",
        "n_districts",
        "flags",
    ]
    if filings.empty:
        return pd.DataFrame(columns=cols)

    parent_date = dict(zip(filings["filing_id"], filings["effective_date"], strict=True))
    rows = []
    unit_geo = units.set_index("filer_uid") if not units.empty else None
    for uid, g in filings.groupby("filer_uid", sort=True):
        dates = g["effective_date"].dropna().sort_values()
        year_gap = (
            (g.loc[g["effective_date"].notna(), "effective_date"].dt.year
             - g.loc[g["effective_date"].notna(), "report_year"]).abs()
        )
        n_mismatch = int((year_gap > params.report_year_slack).sum())
        max_gap = float(year_gap.max()) if len(year_gap) and year_gap.notna().any() else 0.0
        amend_bad = 0
        for _, row in g[g["is_amendment"] == 1].iterrows():
            pid = row.get("amends_filing_id")
            if not pid or pd.isna(pid) or pd.isna(row["effective_date"]):
                continue
            pdate = parent_date.get(pid)
            if pdate is not None and pd.notna(pdate) and row["effective_date"] < pdate:
                amend_bad += 1
        dormancy = 0.0
        if len(dates) > 1:
            dormancy = float(dates.diff().dt.days.max())
        n_missing = int(g["effective_date"].isna().sum())
        flags = []
        if len(dates) == 0:
            flags.append(REASON_SERVICE_UNVERIFIABLE)
        elif n_missing:
            flags.append(FLAG_MISSING_DATE_SOME)
        if n_mismatch:
            flags.append(REASON_REPORT_YEAR_CONFLICT)
        if amend_bad:
            flags.append(REASON_AMENDMENT_DATE_ORDER)
        if dormancy >= params.dormancy_gap_days:
            flags.append(REASON_SPLIT_DORMANCY)
        geo = unit_geo.loc[uid] if unit_geo is not None and uid in unit_geo.index else None
        rows.append(
            {
                "filer_uid": uid,
                "chamber": _join(g["chamber"]),
                "n_filings": int(len(g)),
                "n_missing_effective_date": n_missing,
                "all_dates_missing": len(dates) == 0,
                "first_effective_date": dates.min() if len(dates) else pd.NaT,
                "last_effective_date": dates.max() if len(dates) else pd.NaT,
                "observed_span_days": (
                    float((dates.max() - dates.min()).days) if len(dates) else None
                ),
                "n_report_year_mismatch": n_mismatch,
                "max_report_year_gap": max_gap,
                "n_amendment_before_parent": amend_bad,
                "max_dormancy_gap_days": dormancy,
                "n_states": int(geo["n_states"]) if geo is not None else 0,
                "n_districts": int(geo["n_districts"]) if geo is not None else 0,
                "flags": ";".join(flags),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def classify_units(
    units: pd.DataFrame,
    merge_candidates: pd.DataFrame,
    split_candidates: pd.DataFrame,
    transitions: pd.DataFrame,
    service_dates: pd.DataFrame,
    top_volume: pd.DataFrame,
    surname_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Attach reason codes and the candidate/review status to every unit.

    A unit is ``unambiguous`` only when no blocking reason applies. Nothing in
    this function merges, splits, or renames a ``filer_uid``.
    """
    out = units.copy()
    if out.empty:
        for col in ("identity_status", "reason_codes", "flags", "human_decision",
                    "crosswalk_status"):
            out[col] = pd.Series(dtype=object)
        for col in ("n_reason_codes",):
            out[col] = pd.Series(dtype="int64")
        for col in ("auto_merged", "auto_split"):
            out[col] = pd.Series(dtype=bool)
        return out

    reasons: dict[str, set[str]] = defaultdict(set)
    flags: dict[str, set[str]] = defaultdict(set)

    for _, r in merge_candidates.iterrows():
        reasons[r["uid_a"]].add(REASON_MERGE_CANDIDATE)
        reasons[r["uid_b"]].add(REASON_MERGE_CANDIDATE)
    for _, r in split_candidates.iterrows():
        reasons[r["filer_uid"]].add(r["reason"])
    for _, r in transitions.iterrows():
        if r["transition_type"] == "chamber" and r["blocking"]:
            reasons[r["filer_uid_from"]].add(REASON_CHAMBER_TRANSITION)
            reasons[r["filer_uid_to"]].add(REASON_CHAMBER_TRANSITION)
        elif r["transition_type"] == "state":
            reasons[r["filer_uid_from"]].add(REASON_STATE_TRANSITION)
        elif r["blocking"]:
            reasons[r["filer_uid_from"]].add(REASON_SPLIT_DISTRICT_OVERLAP)
        else:
            flags[r["filer_uid_from"]].add(FLAG_DISTRICT_CHANGE_DISJOINT)
    for _, r in service_dates.iterrows():
        for code in str(r["flags"]).split(";"):
            if not code:
                continue
            if code in BLOCKING_REASONS:
                reasons[r["filer_uid"]].add(code)
            else:
                flags[r["filer_uid"]].add(code)
    for uid in top_volume.get("filer_uid", pd.Series(dtype=object)):
        flags[uid].add(FLAG_TOP_VOLUME)
    if not surname_groups.empty:
        for uids in surname_groups["filer_uids"]:
            for uid in str(uids).split("|"):
                if uid:
                    flags[uid].add(FLAG_SHARED_SURNAME)
    for uid, usable in zip(out["filer_uid"], out["name_key_usable"], strict=True):
        if not usable:
            reasons[uid].add(REASON_UNUSABLE_NAME_KEY)

    out["reason_codes"] = [";".join(sorted(reasons.get(u, ()))) for u in out["filer_uid"]]
    out["flags"] = [";".join(sorted(flags.get(u, ()))) for u in out["filer_uid"]]
    out["n_reason_codes"] = [len(reasons.get(u, ())) for u in out["filer_uid"]]
    out["identity_status"] = [
        STATUS_REVIEW if reasons.get(u) else STATUS_OK for u in out["filer_uid"]
    ]
    out["human_decision"] = [
        "PENDING" if s == STATUS_REVIEW else "" for s in out["identity_status"]
    ]
    out["crosswalk_status"] = "candidate"
    out["auto_merged"] = False
    out["auto_split"] = False
    return out


def build_crosswalk(
    conn: sqlite3.Connection, params: ClassificationParams | None = None
) -> CrosswalkResult:
    """Full TC-050 build from the official store."""
    params = params or ClassificationParams()
    filings = load_ptr_filings(conn)
    other = pd.read_sql_query(
        f"SELECT {', '.join(_FILING_COLUMNS)} FROM filings WHERE report_type != 'ptr'",
        conn,
    )
    corroborating = prepare_corroborating_filings(other) if len(other) else None
    return build_crosswalk_from_filings(filings, params, corroborating)


def build_crosswalk_from_filings(
    filings: pd.DataFrame,
    params: ClassificationParams | None = None,
    corroborating: pd.DataFrame | None = None,
) -> CrosswalkResult:
    """Same build, from an already-prepared filing frame (used by tests)."""
    params = params or ClassificationParams()
    units = build_filer_units(filings)
    merge_candidates = find_merge_candidates(units)
    split_candidates = annotate_gap_corroboration(
        find_split_candidates(filings, params), filings, corroborating
    )
    surname_groups = find_surname_groups(units, merge_candidates)
    transitions = find_transitions(filings, units)
    top_volume = top_volume_households(units, params)
    service_dates = validate_service_dates(filings, units, params)
    units = classify_units(
        units,
        merge_candidates,
        split_candidates,
        transitions,
        service_dates,
        top_volume,
        surname_groups,
    )

    status_by_uid = dict(zip(units["filer_uid"], units["identity_status"], strict=True))
    reasons_by_uid = dict(zip(units["filer_uid"], units["reason_codes"], strict=True))
    assignments = filings[
        ["filing_id", "chamber", "filer_uid", "effective_date", "effective_date_source"]
    ].copy()
    assignments["identity_status"] = [status_by_uid.get(u, "") for u in assignments["filer_uid"]]
    assignments["reason_codes"] = [reasons_by_uid.get(u, "") for u in assignments["filer_uid"]]
    assignments["crosswalk_status"] = "candidate"

    n_review = int((units["identity_status"] == STATUS_REVIEW).sum())
    reason_counts = {
        code: int(units["reason_codes"].str.contains(code, regex=False).sum())
        for code in BLOCKING_REASONS
    }
    summary = {
        "task": "TC-050",
        "gate": "HD-05",
        "crosswalk_status": "candidate",
        "params": {
            "dormancy_gap_days": params.dormancy_gap_days,
            "report_year_slack": params.report_year_slack,
            "top_volume_n": params.top_volume_n,
        },
        "filings_total": int(len(filings)),
        "filings_by_chamber": filings["chamber"].value_counts().to_dict(),
        "filer_units_total": int(len(units)),
        "filer_units_unambiguous": int(len(units)) - n_review,
        "filer_units_needs_human_review": n_review,
        "filings_in_units_needing_review": int(
            (assignments["identity_status"] == STATUS_REVIEW).sum()
        ),
        "reason_code_counts": reason_counts,
        "evidence_counts": {
            "a_merge_candidate_pairs": int(len(merge_candidates)),
            "a_split_candidate_cases": int(len(split_candidates)),
            "a_split_dormancy_cases_with_other_official_filings_in_gap": int(
                (split_candidates["n_corroborating_official_filings_in_gap"].fillna(0) > 0).sum()
            )
            if len(split_candidates)
            else 0,
            "b_shared_surname_groups": int(len(surname_groups)),
            "b_units_sharing_a_surname": int(
                sum(len(str(v).split("|")) for v in surname_groups.get("filer_uids", []))
            ),
            "c_transitions_total": int(len(transitions)),
            "c_transitions_chamber": int((transitions.get("transition_type") == "chamber").sum())
            if len(transitions)
            else 0,
            "c_transitions_state": int((transitions.get("transition_type") == "state").sum())
            if len(transitions)
            else 0,
            "c_transitions_district": int((transitions.get("transition_type") == "district").sum())
            if len(transitions)
            else 0,
            "d_top_volume_units": int(len(top_volume)),
            "e_units_with_service_date_flags": int(
                (service_dates["flags"].astype(str) != "").sum()
            )
            if len(service_dates)
            else 0,
        },
        "auto_merges_applied": 0,
        "auto_splits_applied": 0,
    }
    return CrosswalkResult(
        units=units,
        assignments=assignments,
        merge_candidates=merge_candidates,
        split_candidates=split_candidates,
        surname_groups=surname_groups,
        transitions=transitions,
        top_volume=top_volume,
        service_dates=service_dates,
        summary=summary,
    )
