"""Coverage repair of the two households' scanned histories (closeout plan section 3b).

Diagnosis (page renders inspected before the plan): the pages the October extraction read poorly were rotated
to the wrong right angle by the orientation detector (180 or 0 degrees where the typed table needs 90), not laid
out differently. Repair, with the pilot's page procedure unchanged: every page of the 2015-2020 filings whose
chosen read holds fewer than MIN_DATES rows with a parsed date is re-read at all four right angles and the read
with the most date tokens is kept; the inventory is extended to the filings dated 2021, read the same way
(the detector's angle first, the other three when it yields fewer than MIN_DATES date tokens), whose rows dated
in 2020 belong to the earlier study period. Row-level output is restricted (data/v4/restricted/final/history/);
the committed inventory holds counts by household, year, filing and page. Nothing here enters an estimate.

Run: PYTHONPATH=src python -m trading_on_disclosure_v4.final.history_reread [--workers 8]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.ext import history_extraction as hx
from trading_on_disclosure_v4.ext import two_households as th
from trading_on_disclosure_v4.ext import two_households_pilot as tp
from trading_on_disclosure_v4.final import common as fc

HISTORY_DIR = fc.RESTRICTED / "history"
PAGES_DIR = HISTORY_DIR / "pages"
OCR_DIR = HISTORY_DIR / "ocr"
ROWS_PARQUET = HISTORY_DIR / "extracted_rows.parquet"
FILINGS_PARQUET = HISTORY_DIR / "filings.parquet"
PAGES_PARQUET = HISTORY_DIR / "pages_reread.parquet"
INVENTORY_CSV = fc.OUT_DIR / "history_coverage_inventory.csv"
PAGE_CSV = fc.OUT_DIR / "history_coverage_pages.csv"
MIN_DATES = 3
ANGLES = (0, 90, 180, 270)
EXTENDED_PERIOD = (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
FORCED = {"angle": None}


def _forced_orientation(img):
    return FORCED["angle"], "forced"


def _worker_init() -> None:
    hx._worker_init()
    tp.PAGES_DIR = PAGES_DIR
    tp.OCR_DIR = OCR_DIR
    tp.detect_orientation = _forced_orientation


def read_page(args: tuple[str, str, int, str, tuple[int, ...], int]) -> dict:
    """The pilot procedure at each angle in turn: the first angle is kept when it yields at least `min_dates`
    date tokens, otherwise the angle with the most date tokens (ties to the earlier angle)."""
    png, label, page_no, filing_key, angles, min_dates = args
    reads = []
    for angle in angles:
        FORCED["angle"] = int(angle)
        res = tp.process_page((png, label, page_no, filing_key))
        res["page"]["angle_tried"] = int(angle)
        reads.append(res)
        if len(reads) == 1 and res["page"]["date_tokens"] >= min_dates:
            break
    best = max(reads, key=lambda r: (r["page"]["date_tokens"], -reads.index(r)))
    best["page"]["angles_tried"] = "|".join(str(r["page"]["angle_tried"]) for r in reads)
    best["page"]["date_tokens_by_angle"] = "|".join(str(r["page"]["date_tokens"]) for r in reads)
    best["page"]["n_angles_tried"] = len(reads)
    return best


def extended_filings(cand: pd.DataFrame, label: str, seq_offset: int) -> pd.DataFrame:
    """The household's 2021 filings in filing-date order, numbered after the 2015-2020 sequence."""
    f = cand[(cand["fdate"] >= EXTENDED_PERIOD[0]) & (cand["fdate"] <= EXTENDED_PERIOD[1])].sort_values(["fdate", "filing_id"], kind="stable").copy()
    f["household"] = label
    f["filing_seq"] = np.arange(seq_offset + 1, seq_offset + len(f) + 1)
    f["amends_seq"] = pd.array([pd.NA] * len(f), dtype="Int64")
    f["is_amendment"] = f["is_amendment"].fillna(0).astype(int).astype(bool) if "is_amendment" in f.columns else False
    f["filing_key"] = f["household"] + "_s" + f["filing_seq"].map(lambda k: f"{k:03d}")
    return f.reset_index(drop=True)


def weak_pages(rows: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Every page of the 2015-2020 filings with its original chosen angle and the number of rows with a parsed date;
    weak = fewer than MIN_DATES such rows (pages without any row count as zero)."""
    counts = rows.groupby(["household", "filing_seq", "page"]).agg(n_rows=("date", "size"), n_dates=("date", lambda s: int(s.notna().sum())),
                                                                   orientation=("orientation", "first")).reset_index()
    pages = []
    for r in filings.itertuples(index=False):
        for p in range(1, int(r.pages) + 1):
            pages.append({"household": r.household, "filing_seq": int(r.filing_seq), "page": p})
    pg = pd.DataFrame(pages).merge(counts, on=["household", "filing_seq", "page"], how="left")
    pg["n_rows"] = pg["n_rows"].fillna(0).astype(int)
    pg["n_dates"] = pg["n_dates"].fillna(0).astype(int)
    pg["weak"] = pg["n_dates"] < MIN_DATES
    return pg


def inventory(filings: pd.DataFrame, pages: pd.DataFrame, rows_old: pd.DataFrame, rows_new: pd.DataFrame) -> pd.DataFrame:
    """Counts by household x filing year, plus per-household and total rows."""
    out = []
    filings = filings.copy()
    filings["year"] = pd.to_datetime(filings["filing_date"]).dt.year
    groups = [((h, int(y)), f) for (h, y), f in filings.groupby(["household", "year"])]
    groups += [((h, "all"), f) for h, f in filings.groupby("household")] + [(("total", "all"), filings)]
    for (h, y), f in groups:
        keys = set(zip(f["household"], f["filing_seq"], strict=True))
        pg = pages[[k in keys for k in zip(pages["household"], pages["filing_seq"], strict=True)]]
        ro = rows_old[[k in keys for k in zip(rows_old["household"], rows_old["filing_seq"], strict=True)]]
        rn = rows_new[[k in keys for k in zip(rows_new["household"], rows_new["filing_seq"], strict=True)]]
        both = rn["date"].notna() & rn["ink_type"].notna() & rn["ink_bracket"].notna()
        yr = pd.to_datetime(rn["date"], errors="coerce").dt.year
        out.append({"household": h, "year": y, "filings": int(len(f)), "pages": int(pg["page"].size),
                    "pages_weak_before": int(pg["weak"].sum()), "pages_reread": int(pg["reread"].sum()),
                    "pages_readable_before": int((pg["n_dates"] >= MIN_DATES).sum()), "pages_readable_after": int((pg["n_dates_after"] >= MIN_DATES).sum()),
                    "pages_recovered_by_reread": int((pg["weak"] & (pg["n_dates_after"] >= MIN_DATES)).sum()),
                    "pages_unreadable_after": int((pg["n_dates_after"] < MIN_DATES).sum()),
                    "rows_before": int(len(ro)), "rows_after": int(len(rn)), "rows_date_parsed_after": int(rn["date"].notna().sum()),
                    "rows_ink_type_after": int(rn["ink_type"].notna().sum()), "rows_ink_bracket_after": int(rn["ink_bracket"].notna().sum()),
                    "rows_date_type_bracket_after": int(both.sum()), "rows_dated_2015_2020": int(((yr >= 2015) & (yr <= 2020)).sum()),
                    "rows_dated_2021_or_later": int((yr >= 2021).sum()),
                    "mean_line_confidence_after": round(float(rn["mean_word_confidence"].mean()), 1) if len(rn) else np.nan})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    log = fc.logger("history_reread")
    t0 = time.time()
    root = vs.dropbox_project_root()
    official = root / "data" / "official"
    state_db, raw_store = official / "state.sqlite", official / "raw_store"
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    snapshot = pd.read_parquet(root / vs.VENDOR_SNAPSHOT, columns=["politician_slug", "politician_name", "chamber"])
    store = vs.load_ptr_filings(state_db).merge(th.read_primary_documents(state_db), on="filing_id", how="left")
    store = store.merge(hx.amendment_columns(state_db), on="filing_id", how="left")
    old_filings = pd.read_parquet(hx.FILINGS_PARQUET)
    old_rows = pd.read_parquet(hx.ROWS_PARQUET)
    base, ext = [], []
    for i, h in enumerate(households, start=1):
        label = f"household_{i}"
        cand = th.household_filings(h, snapshot, store)
        f0 = hx.history_filings(cand, label)
        chk = old_filings[old_filings["household"] == label].sort_values("filing_seq")
        if len(f0) != len(chk) or not (pd.to_datetime(f0["fdate"]).to_numpy() == pd.to_datetime(chk["filing_date"]).to_numpy()).all():
            raise ValueError(f"{label}: the 2015-2020 filing sequence does not reproduce the October extraction's")
        base.append(f0)
        ext.append(extended_filings(cand, label, int(f0["filing_seq"].max())))
    base, ext = pd.concat(base, ignore_index=True), pd.concat(ext, ignore_index=True)
    log(f"2015-2020 filings {base.groupby('household').size().to_dict()}; 2021 filings {ext.groupby('household').size().to_dict()}, "
        f"census pages {int(ext['page_count'].fillna(0).sum())}")
    # pages to (re-)read
    pg = weak_pages(old_rows, old_filings)
    weak = pg[pg["weak"]]
    log(f"2015-2020 pages {len(pg)}, weak (< {MIN_DATES} dated rows) {len(weak)} = {100 * len(weak) / len(pg):.1f}%")
    rendered = hx.render_all(pd.concat([base, ext], ignore_index=True), raw_store, args.workers)
    jobs = []
    base_keys = base.set_index(["household", "filing_seq"])["filing_key"]
    for r in weak.itertuples(index=False):
        key = base_keys.loc[(r.household, r.filing_seq)]
        pngs = rendered[key]
        if r.page <= len(pngs):
            jobs.append((str(pngs[r.page - 1]), r.household, int(r.page), key, ANGLES, MIN_DATES))
    n_weak_jobs = len(jobs)
    for r in ext.itertuples(index=False):
        for i, p in enumerate(rendered[r.filing_key], start=1):
            jobs.append((str(p), r.household, i, r.filing_key, ANGLES, MIN_DATES))
    log(f"{n_weak_jobs} weak pages at four angles and {len(jobs) - n_weak_jobs} pages of 2021 filings; {args.workers} workers")
    pages_out, rows_new = [], []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for k, res in enumerate(pool.map(read_page, jobs, chunksize=1), start=1):
            pages_out.append(res["page"])
            rows_new += res["rows"]
            if k % 50 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} pages, {time.time() - t1:.0f}s")
    pages_out = pd.DataFrame(pages_out)
    meta = pd.concat([base, ext], ignore_index=True).set_index("filing_key")[["household", "filing_seq", "fdate", "report_year", "is_amendment", "amends_seq"]].rename(columns={"fdate": "filing_date"})
    rows_new = pd.DataFrame(rows_new)
    if rows_new.empty:
        rows_new = pd.DataFrame(columns=hx.ROW_COLUMNS)
    else:
        rows_new = rows_new.drop(columns=["household"], errors="ignore").merge(meta, left_on="filing_key", right_index=True, how="left")
        rows_new = rows_new.rename(columns={"assigned_type": "mark_type", "assigned_bracket": "mark_bracket"})
        rows_new["date"] = pd.to_datetime(rows_new["date"], errors="coerce")
        rows_new["notification_date"] = pd.to_datetime(rows_new["notification_date"], errors="coerce")
        rows_new = rows_new.reindex(columns=hx.ROW_COLUMNS)
    # the final row set: the October rows of the strong pages, the best re-read of the weak pages, the 2021 rows
    reread_keys = set(zip(pages_out["household"], pages_out["filing_key"].map(lambda k: int(k.split("_s")[-1])), pages_out["page"], strict=True))
    old_keep = old_rows[[k not in reread_keys for k in zip(old_rows["household"], old_rows["filing_seq"], old_rows["page"], strict=True)]]
    rows_final = pd.concat([old_keep.assign(source="october_extraction"), rows_new.assign(source="closeout_reread")], ignore_index=True)
    rows_final = rows_final.sort_values(["household", "filing_seq", "page"], kind="stable").reset_index(drop=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    rows_final.to_parquet(ROWS_PARQUET, index=False)
    # filings table (2015-2021) and page table
    per_filing = pages_out.groupby("filing_key").agg(pages_read=("page", "size"))
    allf = pd.concat([base, ext], ignore_index=True)
    ftab = allf.set_index("filing_key").join(per_filing)
    ftab = ftab.rename(columns={"fdate": "filing_date"}).reset_index()
    ftab["pages"] = ftab["filing_key"].map(lambda k: len(rendered.get(k, []))).astype(int)
    ftab["read_status"] = np.where(ftab["pages"] > 0, "read", "no_pages_rendered")
    ftab["extended_inventory"] = ftab["filing_key"].isin(set(ext["filing_key"]))
    cols = [c for c in hx.FILING_COLUMNS if c in ftab.columns] + ["extended_inventory"]
    ftab[cols].to_parquet(FILINGS_PARQUET, index=False)
    pages_out["filing_seq"] = pages_out["filing_key"].map(lambda k: int(k.split("_s")[-1]))
    pages_out.to_parquet(PAGES_PARQUET, index=False)
    # page-level coverage table (counts only) over every page of 2015-2021
    pg_all = []
    for r in ftab.itertuples(index=False):
        for p in range(1, int(r.pages) + 1):
            pg_all.append({"household": r.household, "filing_seq": int(r.filing_seq), "page": p, "filing_year": int(pd.Timestamp(r.filing_date).year)})
    pg_all = pd.DataFrame(pg_all).merge(pg[["household", "filing_seq", "page", "n_rows", "n_dates", "orientation", "weak"]], on=["household", "filing_seq", "page"], how="left")
    pg_all["weak"] = pg_all["weak"].fillna(True).astype(bool)  # 2021 pages: read fresh
    re_ = pages_out.set_index(["household", "filing_seq", "page"])
    pg_all["reread"] = [k in re_.index for k in zip(pg_all["household"], pg_all["filing_seq"], pg_all["page"], strict=True)]
    after = rows_final.groupby(["household", "filing_seq", "page"]).agg(n_dates_after=("date", lambda s: int(s.notna().sum())), n_rows_after=("date", "size"))
    pg_all = pg_all.merge(after.reset_index(), on=["household", "filing_seq", "page"], how="left")
    pg_all["n_dates_after"] = pg_all["n_dates_after"].fillna(0).astype(int)
    pg_all["n_rows_after"] = pg_all["n_rows_after"].fillna(0).astype(int)
    pg_all["angle_before"] = pg_all["orientation"]
    pg_all["angle_after"] = [re_.loc[k, "orientation"] if k in re_.index else o for k, o in zip(zip(pg_all["household"], pg_all["filing_seq"], pg_all["page"], strict=True), pg_all["orientation"], strict=True)]
    pg_all["date_tokens_by_angle"] = [re_.loc[k, "date_tokens_by_angle"] if k in re_.index else "" for k in zip(pg_all["household"], pg_all["filing_seq"], pg_all["page"], strict=True)]
    pg_all["readable_after"] = pg_all["n_dates_after"] >= MIN_DATES
    fc.write_csv(pg_all[["household", "filing_year", "filing_seq", "page", "angle_before", "n_dates", "weak", "reread", "angle_after", "date_tokens_by_angle", "n_rows_after", "n_dates_after", "readable_after"]], PAGE_CSV)
    inv = inventory(ftab, pg_all, old_rows, rows_final)
    fc.write_csv(inv, INVENTORY_CSV)
    entries = [fc.file_entry(state_db), fc.file_entry(root / vs.VENDOR_SNAPSHOT), fc.file_entry(hx.ROWS_PARQUET, rows=len(old_rows)), fc.file_entry(hx.FILINGS_PARQUET, rows=len(old_filings))]
    fc.write_manifest("history_reread", entries, config={"min_dates": MIN_DATES, "angles": list(ANGLES), "extended_period": [str(d.date()) for d in EXTENDED_PERIOD],
                                                          "dpi": th.DPI, "prescribed_psms": list(tp.PRESCRIBED_PSMS), "extra_psms": list(tp.EXTRA_PSMS), "grid_run_px": tp.GRID_RUN_PX},
                      extra={"seconds": round(time.time() - t0, 1), "workers": args.workers, "n_weak_pages": int(n_weak_jobs), "n_pages_2021": int(len(jobs) - n_weak_jobs)})
    tot = inv[inv["household"] == "total"].iloc[0]
    log(f"total: pages {tot.pages}, weak before {tot.pages_weak_before}, recovered by re-read {tot.pages_recovered_by_reread}, unreadable after {tot.pages_unreadable_after}; "
        f"rows {tot.rows_before} -> {tot.rows_after}; done in {time.time() - t0:.0f}s")
    pd.set_option("display.width", 250)
    print(inv.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
