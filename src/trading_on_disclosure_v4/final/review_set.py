"""The researcher's review set for the provisional reconstruction (closeout plan section 3a).

A stratified sample of the proposed recovered records (the closeout re-read, exact and near-exact routes) and of
unresolved candidates and weak pages, with everything a reviewer needs to check the record against the page
image: the page image path, the extracted date, direction, owner, asset text and bracket, the route, the
resolved ticker and PERMNO, the fields duplicate reconciliation uses, and blank verdict columns. Development
material (the pilot's development pages and the eight transcribed gold pages, and every page of their filings)
is listed and excluded from the verification strata. Restricted: data/v4/restricted/final/review/ (the sheet and
the protocol); committed: data/v4/final/review_set_counts.csv (counts per stratum). This module certifies nothing.
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.ext import history_recovery as hr
from trading_on_disclosure_v4.final import common as fc
from trading_on_disclosure_v4.final import history_reread as rr

REVIEW_DIR = fc.RESTRICTED / "review"
OUT_SHEET = REVIEW_DIR / "review_set.csv"
OUT_PAGES = REVIEW_DIR / "review_pages_zero_extraction.csv"
OUT_PROTOCOL = REVIEW_DIR / "REVIEW_PROTOCOL.md"
OUT_COUNTS = fc.OUT_DIR / "review_set_counts.csv"
PILOT_VAL = fc.REPO_ROOT / "data" / "v4" / "restricted" / "ext" / "two_households" / "pilot" / "validation"
PER_STRATUM = 8
SEED = 20261106
VERDICT_COLS = ["reviewer_date_ok", "reviewer_security_ok", "reviewer_direction_ok", "reviewer_owner_ok", "reviewer_bracket_ok", "reviewer_duplicate_ok", "reviewer_notes"]

PROTOCOL = """# Review protocol for the provisional reconstruction (closeout, 6 November 2026)

This sheet is for the researcher's independent check of the records the automated extraction proposes for the two
households' 2015-2020 histories. The extraction and the identity resolution were produced and validated by an AI
assistant (eight pages); nothing in them has been verified by a person. Until that check is done the records stay
outside the paper's primary sample and the expanded-cohort estimates stay labelled provisional.

How to review one line of `review_set.csv`:

1. Open the page image in `page_image_path` (rotated to the orientation the extraction used; the raw scan is in the
   official store). Find the schedule row at `line_top`-`line_bottom` pixels, or by the asset text.
2. Date: does the printed transaction date equal `date` (ISO)? Mark `reviewer_date_ok` 1/0.
3. Security: does the printed asset name denote the security `ticker_resolved` / `permno`? Mark `reviewer_security_ok`
   1/0; if the row is an unresolved candidate (`route` empty), write the printed asset name in `reviewer_notes`.
4. Direction: does the printed mark sit under Purchase (P) or Sale (S) as `direction` says? Mark `reviewer_direction_ok`.
5. Owner and bracket: do the printed owner code and the marked amount column agree with `owner` and `ink_bracket`?
   These two fields decide whether the row is a repeat of an earlier filing's row (duplicate reconciliation), so
   mark `reviewer_owner_ok` and `reviewer_bracket_ok`, and `reviewer_duplicate_ok` when the row's status in
   `amendment_exact_duplicate` is right.
6. A row is correct for the estimate only when date, security and direction are all right (joint correctness);
   the count of jointly correct rows per stratum is what the paper would report.

Strata (`stratum`): household x filing year x route (exact / near-exact / unresolved candidate) x page quality
(weak = the page needed the closeout re-read). `review_pages_zero_extraction.csv` lists pages that yielded no
candidate at all, to be checked for missed transactions. Development material is excluded from the verification
strata and listed with `stratum = development` for reference only.

Do not edit the extracted columns; fill only the `reviewer_*` columns. Return the sheet; the paper's provisional
labels are removed only for strata that pass the check the researcher decides on, and never by the assistant.
"""


PILOT_YEARS = (2017, 2019)
PILOT_PER_YEAR = 2


def _ranked_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Filings ranked by filing date within household and filing year (ties by sequence), the ordering of
    ext.two_households.select_sample that both the section-3 trial and the pilot used."""
    f = filings.loc[filings["filing_date"].notna(), ["household", "filing_seq", "filing_date"]].copy()
    f["year"] = pd.to_datetime(f["filing_date"]).dt.year
    f = f.sort_values(["household", "year", "filing_date", "filing_seq"], kind="stable")
    f["rank"] = f.groupby(["household", "year"]).cumcount() + 1
    return f


def development_filings(filings: pd.DataFrame) -> set[tuple[str, int]]:
    """(household, filing_seq) of every filing used as development material: the pilot's filings (the first two
    filings by filing date of 2017 and of 2019 per household, pilot filing index 1..4) and the section-3 trial
    filings named in pilot/validation/development_pages.csv ("sample year Y index k" = k-th filing of year Y).
    The development pages and the eight transcribed gold pages (sample_index.csv) lie inside these filings,
    which are excluded whole from the review pool."""
    ranked = _ranked_filings(filings)
    wanted: set[tuple[str, int, int]] = set()
    for name in ("development_pages.csv", "sample_index.csv"):
        path = PILOT_VAL / name
        if not path.exists():
            continue
        idx = pd.read_csv(path)
        for r in idx.itertuples(index=False):
            if pd.notna(r.filing_index):
                i = int(r.filing_index)
                if not 1 <= i <= len(PILOT_YEARS) * PILOT_PER_YEAR:
                    raise ValueError(f"{name}: pilot filing_index {i} outside 1..{len(PILOT_YEARS) * PILOT_PER_YEAR}")
                wanted.add((r.household, PILOT_YEARS[(i - 1) // PILOT_PER_YEAR], (i - 1) % PILOT_PER_YEAR + 1))
            else:
                m = re.search(r"sample year (\d{4}) index (\d+)", str(getattr(r, "why", "")))
                if not m:
                    raise ValueError(f"{name}: row without a pilot filing index or a 'sample year Y index k' note: {r}")
                wanted.add((r.household, int(m.group(1)), int(m.group(2))))
    keys = set()
    for hh, year, rank in sorted(wanted):
        hit = ranked[(ranked["household"] == hh) & (ranked["year"] == year) & (ranked["rank"] == rank)]
        if len(hit) != 1:
            raise ValueError(f"development filing not found in the history inventory: {hh} {year} rank {rank}")
        keys.add((hh, int(hit["filing_seq"].iloc[0])))
    return keys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="the researcher's review set for the provisional reconstruction")
    ap.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    args = ap.parse_args(argv)
    log = fc.logger("review_set")
    t0 = time.time()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rt = pd.read_parquet(rr.HISTORY_DIR / "recovered_transactions.parquet")
    rows = pd.read_parquet(rr.ROWS_PARQUET)
    pages = pd.read_csv(rr.PAGE_CSV)
    filings = pd.read_parquet(rr.FILINGS_PARQUET)
    # page image path and line position from the re-read rows (by household, filing_seq, page, raw_line_text)
    key = ["household", "filing_seq", "page", "raw_line_text"]
    img = rows[key + ["page_image_path"]].drop_duplicates(key)
    d = rt.merge(img, on=key, how="left", suffixes=("", "_rows"))
    if "page_image_path_rows" in d.columns:
        d["page_image_path"] = d["page_image_path"].fillna(d["page_image_path_rows"])
    rec = hr.reconcile(d)
    d = d.merge(rec[["household", "filing_seq", "page", "raw_line_text", "amendment_exact_duplicate", "owner_std"]], on=key, how="left")
    d["filing_year"] = pd.to_datetime(d["filing_date"]).dt.year
    weak = pages.set_index(["household", "filing_seq", "page"])["weak"]
    d["page_weak"] = [bool(weak.get(k, False)) for k in zip(d["household"], d["filing_seq"], d["page"], strict=True)]
    # development material: the pilot's filings (first two of 2017 and of 2019 per household), excluded whole
    dev_keys = development_filings(filings)
    d["development"] = [k in dev_keys for k in zip(d["household"], d["filing_seq"], strict=True)]
    log(f"development filings excluded from the review pool: {len(dev_keys)} ({sorted(dev_keys)})")
    d["route_class"] = np.where(d["route"].isin(hr.ROUTES["exact"]), "exact", np.where(d["route"] == "name, near-exact", "near_exact", np.where(d["candidate"], "unresolved_candidate", "not_candidate")))
    pool = d.loc[d["route_class"] != "not_candidate"].copy()
    pool["stratum"] = np.where(pool["development"], "development", pool["household"] + "|" + pool["filing_year"].astype(str) + "|" + pool["route_class"] + "|" + np.where(pool["page_weak"], "weak_page", "ordinary_page"))
    rng = np.random.default_rng(SEED)
    picks = []
    for stratum, g in pool.groupby("stratum", sort=True):
        if stratum == "development":
            continue
        n = min(args.per_stratum, len(g))
        picks.append(g.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    sample = pd.concat(picks, ignore_index=True) if picks else pool.iloc[:0]
    cols = ["stratum", "household", "filing_seq", "filing_year", "filing_date", "page", "page_weak", "page_image_path", "raw_line_text", "date", "direction", "owner",
            "owner_std", "asset_text", "ink_bracket", "ink_type", "mark_type", "mark_bracket", "mean_word_confidence", "route", "route_class", "ticker_resolved",
            "permno", "resolution", "amendment_exact_duplicate", "candidate"]
    cols = [c for c in cols if c in sample.columns]
    sheet = sample[cols].copy()
    for c in VERDICT_COLS:
        sheet[c] = ""
    sheet = sheet.sort_values(["household", "filing_seq", "page"], kind="stable").reset_index(drop=True)
    sheet.insert(0, "review_id", np.arange(1, len(sheet) + 1))
    sheet.to_csv(OUT_SHEET, index=False, lineterminator="\n")
    zero = pages.loc[~pages["readable_after"]].copy()
    zero = zero.merge(filings[["household", "filing_seq", "filing_date"]], on=["household", "filing_seq"], how="left")
    zero = zero[["household", "filing_seq", "filing_date", "filing_year", "page", "angle_before", "angle_after", "date_tokens_by_angle", "n_rows_after", "n_dates_after"]]
    zero["page_image_hint"] = "data/v4/restricted/final/history/pages/" + zero["household"] + "_s" + zero["filing_seq"].map(lambda k: f"{k:03d}") + "/p-" + zero["page"].astype(str) + ".rot<angle_after>.png (or the October render)"
    zero["reviewer_transactions_present"] = ""
    zero["reviewer_notes"] = ""
    zero.to_csv(OUT_PAGES, index=False, lineterminator="\n")
    OUT_PROTOCOL.write_text(PROTOCOL)
    counts = pool.groupby(["household", "filing_year", "route_class", "page_weak", "development"]).size().rename("n_pool").reset_index()
    sampled = sample.groupby(["household", "filing_year", "route_class", "page_weak"]).size().rename("n_sampled").reset_index() if len(sample) else pd.DataFrame(columns=["household", "filing_year", "route_class", "page_weak", "n_sampled"])
    counts = counts.merge(sampled, on=["household", "filing_year", "route_class", "page_weak"], how="left")
    counts["n_sampled"] = counts["n_sampled"].fillna(0).astype(int)
    counts.loc[counts["development"].astype(bool), "n_sampled"] = 0  # development strata are never sampled
    counts["pages_zero_extraction"] = np.nan
    zc = zero.groupby(["household", "filing_year"]).size()
    fc.write_csv(counts, OUT_COUNTS)
    fc.write_csv(zc.rename("pages_unreadable").reset_index(), OUT_COUNTS.with_name("review_pages_unreadable_counts.csv"))
    fc.write_manifest("review_set", [fc.file_entry(rr.HISTORY_DIR / "recovered_transactions.parquet", rows=len(rt)), fc.file_entry(rr.ROWS_PARQUET, rows=len(rows)), fc.file_entry(rr.PAGE_CSV)],
                      config={"per_stratum": args.per_stratum, "seed": SEED}, extra={"seconds": round(time.time() - t0, 1), "n_review_rows": int(len(sheet)), "n_unreadable_pages": int(len(zero)),
                                                                                      "n_development_rows_excluded": int(pool["development"].sum())})
    log(f"review sheet {len(sheet)} rows over {counts['n_sampled'].gt(0).sum()} strata; {len(zero)} unreadable pages listed; {int(pool['development'].sum())} development rows excluded; done in {time.time() - t0:.0f} s")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
