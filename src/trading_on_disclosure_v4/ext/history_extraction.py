"""Round-3 plan section 3, extraction step only: the pilot's page procedure on every periodic transaction report
of the two households dated 2015-2020. Every page is rendered at 300 dpi (renders already on disk are reused),
rotated, grid-whitened and read exactly as `ext.two_households_pilot.process_page` does (chosen-read rule, header
detection, ink-based type and bracket, dash dates, owner code; no new tuning). Row-level output is restricted
(`history/extracted_rows.parquet`, `history/filings.parquet`, keyed by a per-household filing sequence number,
never a filing id); the committed inventory holds counts only. Identity resolution, amendments, episodes and
estimates are not done here.

Run: PYTHONPATH=src python -m trading_on_disclosure_v4.ext.history_extraction [--workers 8] [--max-filings N]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from congressional_trading.official.storage import object_path
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.ext import common
from trading_on_disclosure_v4.ext import two_households as th
from trading_on_disclosure_v4.ext import two_households_pilot as tp

HISTORY_DIR = th.RESTRICTED / "history"
PAGES_DIR = HISTORY_DIR / "pages"
OCR_DIR = HISTORY_DIR / "ocr"
ROWS_PARQUET = HISTORY_DIR / "extracted_rows.parquet"
FILINGS_PARQUET = HISTORY_DIR / "filings.parquet"
INVENTORY_CSV = common.OUT_DIR / "history_extraction_inventory.csv"
LOG = common.LOG_DIR / "history_extraction.log"
PERIOD = (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31"))
ROW_COLUMNS = ["household", "filing_seq", "filing_date", "report_year", "is_amendment", "amends_seq", "page", "raw_line_text",
               "date", "date_pattern", "notification_date", "owner", "asset_text", "ticker", "ink_type", "ink_bracket",
               "mark_type", "mark_bracket", "n_marks", "mean_word_confidence", "orientation", "read", "page_image_path"]
FILING_COLUMNS = ["household", "filing_seq", "filing_date", "report_year", "is_amendment", "amends_seq", "pages", "pages_with_headers",
                  "candidate_rows", "read_status"]


def _worker_init() -> None:
    """Workers run the pilot's page procedure with its artefact directories redirected under history/."""
    tp._worker_init()
    tp.PAGES_DIR = PAGES_DIR
    tp.OCR_DIR = OCR_DIR


def amendment_columns(state_db: Path) -> pd.DataFrame:
    """is_amendment and amends_filing_id per PTR filing from the read-only store."""
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT filing_id, is_amendment, amends_filing_id FROM filings WHERE report_type='ptr'", con)
    finally:
        con.close()


def history_filings(cand: pd.DataFrame, label: str) -> pd.DataFrame:
    """The household's 2015-2020 filings in filing-date order with a sequence number and the amended filing's sequence."""
    f = cand[(cand["fdate"] >= PERIOD[0]) & (cand["fdate"] <= PERIOD[1])].sort_values(["fdate", "filing_id"], kind="stable").copy()
    f["household"] = label
    f["filing_seq"] = np.arange(1, len(f) + 1)
    seq_of = dict(zip(f["filing_id"], f["filing_seq"], strict=True))
    f["amends_seq"] = f["amends_filing_id"].map(seq_of).astype("Int64") if "amends_filing_id" in f.columns else pd.array([pd.NA] * len(f), dtype="Int64")
    f["amends_outside_set"] = f["amends_filing_id"].notna() & f["amends_seq"].isna() if "amends_filing_id" in f.columns else False
    f["is_amendment"] = f["is_amendment"].fillna(0).astype(int).astype(bool) if "is_amendment" in f.columns else False
    f["filing_key"] = f["household"] + "_s" + f["filing_seq"].map(lambda k: f"{k:03d}")
    return f.reset_index(drop=True)


def render_all(filings: pd.DataFrame, raw_store: Path, workers: int) -> dict[str, list[Path]]:
    """filing_key -> rendered page PNGs (pdftoppm 300 dpi grayscale; existing renders reused), documents in parallel."""
    def one(r) -> tuple[str, list[Path]]:
        return r.filing_key, th.render_pdf(object_path(raw_store, r.sha256, r.stored_suffix), th.page_dir(r.sha256))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, filings.itertuples(index=False)))


def inventory(filings: pd.DataFrame, pages: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Counts per household x report year, plus a total row."""
    out = []
    groups = [((h, y), filings[(filings["household"] == h) & (filings["report_year"] == y)]) for h, y in
              sorted(set(zip(filings["household"], filings["report_year"], strict=True)))] + [(("total", ""), filings)]
    for (h, y), f in groups:
        keys = set(f["filing_key"])
        p = pages[pages["filing_key"].isin(keys)]
        r = rows[rows["filing_key"].isin(keys)]
        both = r["date"].notna() & r["ink_type"].notna() & r["ink_bracket"].notna()
        out.append({"household": h, "report_year": y, "filings": int(len(f)), "filings_read": int(f["read_status"].eq("read").sum()),
                    "pages": int(len(p)), "pages_with_headers": int((p["headers_found"] > 0).sum()), "candidate_rows": int(len(r)),
                    "rows_date_parsed": int(r["date"].notna().sum()), "rows_ink_type": int(r["ink_type"].notna().sum()),
                    "rows_ink_bracket": int(r["ink_bracket"].notna().sum()), "rows_date_ink_type_bracket": int(both.sum()),
                    "mean_line_confidence": round(float(r["mean_word_confidence"].mean()), 1) if len(r) else np.nan})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--max-filings", type=int, default=0, help="smoke test: only the first N filings per household")
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        common.log_line(LOG, f"[history_extraction] {msg}")

    root = vs.dropbox_project_root()
    official = root / "data" / "official"
    state_db, raw_store = official / "state.sqlite", official / "raw_store"
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    snapshot = pd.read_parquet(root / vs.VENDOR_SNAPSHOT, columns=["politician_slug", "politician_name", "chamber"])
    store = vs.load_ptr_filings(state_db).merge(th.read_primary_documents(state_db), on="filing_id", how="left")
    store = store.merge(amendment_columns(state_db), on="filing_id", how="left")
    filings = pd.concat([history_filings(th.household_filings(h, snapshot, store), f"household_{i}") for i, h in enumerate(households, start=1)],
                        ignore_index=True)
    if args.max_filings:
        filings = filings.groupby("household").head(args.max_filings).reset_index(drop=True)
    log(f"filings dated 2015-2020: {filings.groupby('household').size().to_dict()}; amendments {int(filings['is_amendment'].sum())}, "
        f"amending a filing outside the set {int(filings['amends_outside_set'].sum())}; census pages {int(filings['page_count'].fillna(0).sum())}")
    t1 = time.time()
    rendered = render_all(filings, raw_store, args.workers)
    jobs = [(str(p), r.household, i, r.filing_key) for r in filings.itertuples(index=False) for i, p in enumerate(rendered[r.filing_key], start=1)]
    log(f"rendered {len(jobs)} pages in {time.time() - t1:.0f}s; OCR with {args.workers} workers")
    pages, rows = [], []
    t2 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for k, res in enumerate(pool.map(tp.process_page, jobs, chunksize=1), start=1):
            pages.append(res["page"])
            rows += res["rows"]
            if k % 50 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} pages, {time.time() - t2:.0f}s")
    pages = pd.DataFrame(pages)
    meta = filings.set_index("filing_key")[["household", "filing_seq", "fdate", "report_year", "is_amendment", "amends_seq"]].rename(columns={"fdate": "filing_date"})
    rows = pd.DataFrame(rows)
    if rows.empty:
        rows = pd.DataFrame(columns=["filing_key", "page", "raw_line_text", "date", "date_pattern", "notification_date", "owner", "asset_text", "ticker",
                                     "ink_type", "ink_bracket", "assigned_type", "assigned_bracket", "n_marks", "mean_word_confidence", "orientation",
                                     "read", "page_image_path"])
    rows = rows.drop(columns=["household"], errors="ignore").merge(meta, left_on="filing_key", right_index=True, how="left")
    rows = rows.rename(columns={"assigned_type": "mark_type", "assigned_bracket": "mark_bracket"})
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["notification_date"] = pd.to_datetime(rows["notification_date"], errors="coerce")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = rows.reindex(columns=ROW_COLUMNS).sort_values(["household", "filing_seq", "page"], kind="stable").reset_index(drop=True)
    rows_out.to_parquet(ROWS_PARQUET, index=False)
    per_filing = pages.groupby("filing_key").agg(pages=("page", "size"), pages_with_headers=("headers_found", lambda s: int((s > 0).sum())),
                                                 candidate_rows=("candidate_rows", "sum")) if len(pages) else pd.DataFrame(columns=["pages", "pages_with_headers", "candidate_rows"])
    ftab = filings.set_index("filing_key").join(per_filing)
    for c in ("pages", "pages_with_headers", "candidate_rows"):
        ftab[c] = ftab[c].fillna(0).astype(int)
    ftab["read_status"] = np.where(ftab["pages"] > 0, "read", "no_pages_rendered")
    ftab = ftab.rename(columns={"fdate": "filing_date"}).reset_index()
    ftab[FILING_COLUMNS].to_parquet(FILINGS_PARQUET, index=False)
    inv = inventory(ftab, pages, rows)
    th.guard(inv, INVENTORY_CSV)
    tot = inv[inv["household"] == "total"].iloc[0]
    log(f"total: filings {tot.filings} (read {tot.filings_read}), pages {tot.pages} (headers {tot.pages_with_headers}), candidate rows {tot.candidate_rows}, "
        f"date parsed {tot.rows_date_parsed}, ink type {tot.rows_ink_type}, ink bracket {tot.rows_ink_bracket}, date+both {tot.rows_date_ink_type_bracket}, "
        f"mean line confidence {tot.mean_line_confidence}")
    entries = [common.file_entry(state_db), common.file_entry(root / vs.VENDOR_SNAPSHOT), common.file_entry(load.FRAMES_DIR / "matched_transactions.csv", rows=len(matched))]
    common.write_manifest("history_extraction", entries, extra={"dpi": th.DPI, "prescribed_psms": list(tp.PRESCRIBED_PSMS), "extra_psms": list(tp.EXTRA_PSMS),
                                                                 "grid_run_px": tp.GRID_RUN_PX, "workers": args.workers, "max_filings": args.max_filings,
                                                                 "seconds": round(time.time() - t0, 1)})
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
