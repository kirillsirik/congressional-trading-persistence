"""Section 3 of the extension plan: the two vendor-only households with no first-period purchases.

For each household this module tabulates its official periodic transaction reports by report year
(parse state, document format, pages), restates the identity facts already measured in
data/v4/diagnostics/vendor_only_households_official_status.csv, runs the pre-specified OCR feasibility
sample (the first three filings by filing date in each year 2015-2020, rendered at 300 dpi with pdftoppm
and read by tesseract, English, page-segmentation mode 6; a page is machine-readable when at least three
of its lines each carry a mm/dd/yyyy date, a P / S / S (partial) code and an amount bracket of the
official form) and, when at least half of a household's sampled filings have a machine-readable page,
OCRs every pre-2023 filing of that household, keeps the rows with a parenthesised ticker, a date and a
code, matches tickers to PERMNOs with the date-valid exact-ticker rule against the screened CRSP names
and recomputes the headline persistence estimate once with the recovered first-period purchases added.
Identities stay in memory; page images and OCR text go to data/v4/restricted/ext/two_households/ only.

Run: PYTHONPATH=src python -m trading_on_disclosure_v4.ext.two_households [--workers N] [--skip-recovery]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from congressional_trading.official.hybrid import identity_key
from congressional_trading.official.storage import object_path
from trading_on_disclosure_v4.data import coverage as cv
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.data.calendar import HORIZON, first_session_on_or_after, window_end
from trading_on_disclosure_v4.data.events import assign_period
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.estimators import results as rs
from trading_on_disclosure_v4.estimators import specs as sp
from trading_on_disclosure_v4.ext import common
from trading_on_disclosure_v4.inference import cgm
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import forward_cache as fc

SUMMARY_CSV = common.OUT_DIR / "two_households.csv"
BY_YEAR_CSV = common.OUT_DIR / "two_households_filings_by_year.csv"
SAMPLE_CSV = common.OUT_DIR / "two_households_ocr_sample.csv"
RECOVERY_CSV = common.OUT_DIR / "two_households_recovery.csv"
TAU_CSV = common.OUT_DIR / "two_households_recovery_tau.csv"
RESTRICTED = common.RESTRICTED_EXT / "two_households"
PAGES_DIR = RESTRICTED / "pages"
OCR_DIR = RESTRICTED / "ocr"
ROWS_PARQUET = RESTRICTED / "recovered_rows.parquet"
LOG = common.LOG_DIR / "two_households.log"
DIAG_CSV = vs.DIAG_DIR / vs.OUTPUT_NAME
CRSP_NAMES = load.FRAMES_DIR / "crsp_names.csv"

YEARS = tuple(range(2012, 2027))
SAMPLE_YEARS = tuple(range(2015, 2021))
SAMPLE_PER_YEAR = 3
MIN_READABLE_LINES = 3
FEASIBLE_SHARE = 0.5
DPI = 300
PSM = 6
RECOVERY_CUTOFF = cv.VENDOR_ERA_START
RECORD_SOURCE = "official_ocr_recovery"
EVENT_FRAME_COLUMNS = [*evf.META_COLUMNS, "dup_any_line", "dup_all_lines", *evf.CACHE_Y, "sigma_event"]
PROVISIONAL = "provisional OCR recovery, unverified"
STATE_GROUPS = {"parsed": vs.PARSED_STATES, "image_only": vs.IMAGE_ONLY_STATES, "ocr_review_gated": vs.OCR_REVIEW_STATES,
                "failed": vs.FAILED_STATES}
COPIED_FROM_DIAGNOSTICS = {"vendor_lines": "vendor_lines_matched_frame", "vendor_records": "vendor_records_snapshot",
                           "vendor_publish_year_min": "vendor_publish_year_min", "vendor_publish_year_max": "vendor_publish_year_max",
                           "filings_pre2023": "official_ptr_filings_pre2023", "filings_2023plus": "official_ptr_filings_2023plus",
                           "crosswalk_units_unambiguous": "crosswalk_units_unambiguous", "name_link_failed": "name_link_failed",
                           "repaired_link_records_to_unparsed_filing": "repaired_link_records_to_unparsed_filing"}

DATE_RE = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/((?:19|20)\d{2})(?!\d)")
CODE_RE = re.compile(r"(?:^|(?<=\s))(P|S(?:\s*\(\s*partial\s*\))?)(?=[\s.,;:]|$)")
_FIGURE = r"\$\s?\d{1,3}(?:[,.]?\d{3})+"
BRACKET_RE = re.compile(rf"(?:{_FIGURE}\s*[-–—]\s*{_FIGURE}|Over\s+{_FIGURE})", re.I)
TICKER_RE = re.compile(r"\(([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\)")
LOOSE_DATE_RE = re.compile(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}")
DIAG_ANGLES = (0, 90, 180, 270)


# ---------------------------------------------------------------- text rules

def line_is_transaction(line: str) -> bool:
    """A date, a purchase/sale code and an amount bracket on one line."""
    return bool(DATE_RE.search(line) and CODE_RE.search(line) and BRACKET_RE.search(line))


def count_transaction_lines(lines: list[str]) -> int:
    return sum(line_is_transaction(ln) for ln in lines)


def page_is_readable(lines: list[str], min_lines: int = MIN_READABLE_LINES) -> bool:
    return count_transaction_lines(lines) >= min_lines


def parse_date(token: tuple[str, str, str]) -> pd.Timestamp:
    """mm/dd/yyyy groups -> Timestamp (NaT when the calendar rejects it)."""
    m, d, y = (int(t) for t in token)
    try:
        return pd.Timestamp(year=y, month=m, day=d)
    except ValueError:
        return pd.NaT


def extract_rows(lines: list[str]) -> list[dict]:
    """Transaction-shaped lines -> ticker (may be None), trade date, code, direction, bracket, text."""
    rows = []
    for line in lines:
        if not line_is_transaction(line):
            continue
        code = CODE_RE.search(line).group(1)
        ticker = TICKER_RE.search(line)
        rows.append({"ticker": ticker.group(1) if ticker else None, "trade_date": parse_date(DATE_RE.search(line).groups()),
                     "code": re.sub(r"\s+", " ", code), "direction": "purchase" if code.startswith("P") else "sale",
                     "amount_bracket": BRACKET_RE.search(line).group(0), "raw_text": line[:300]})
    return rows


# ---------------------------------------------------------------- filings

def read_primary_documents(state_db: Path) -> pd.DataFrame:
    """Primary PDF per filing (sha256, stored suffix, format, page count), read-only."""
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT filing_id, sha256, stored_suffix, doc_format, page_count FROM filing_documents "
                                 "WHERE doc_role='primary_pdf'", con)
    finally:
        con.close()


def household_filings(household: str, snapshot: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """The household's PTR filings: exact name+chamber key or surname+chamber, as `vendor_only_status.household_status`."""
    vendor = snapshot[snapshot["politician_slug"] == household]
    chamber = str(vendor["chamber"].mode().iat[0]).lower() if len(vendor) else ""
    vkeys = {identity_key(n, c) for n, c in zip(vendor["politician_name"], vendor["chamber"], strict=True)}
    surnames = {k.split("|")[0].split()[-1] for k in vkeys if k}
    exact = filings[filings["identity_key"].isin(vkeys)]
    by_surname = filings[filings["surname"].isin(surnames) & (filings["chamber_norm"] == chamber)]
    return pd.concat([exact, by_surname]).drop_duplicates("filing_id").reset_index(drop=True)


def state_group(state: pd.Series) -> pd.Series:
    out = pd.Series("other", index=state.index, dtype=object)
    for name, states in STATE_GROUPS.items():
        out[state.isin(states)] = name
    return out


def filings_by_year(filings: pd.DataFrame, label: str, years=YEARS) -> pd.DataFrame:
    """Per report year: filings by parse-state group and document format, with page counts."""
    f = filings.assign(group=state_group(filings["state"]))
    rows = []
    for year in years:
        y = f[f["report_year"] == year]
        pages = pd.to_numeric(y["page_count"], errors="coerce")
        rows.append({"household": label, "report_year": year, "n_filings": len(y),
                     **{f"n_{g}": int((y["group"] == g).sum()) for g in ("parsed", "image_only", "ocr_review_gated", "failed", "other")},
                     "n_scanned_pdf": int((y["doc_format"] == "scanned_pdf").sum()), "n_text_pdf": int((y["doc_format"] == "text_pdf").sum()),
                     "pages_total": int(pages.fillna(0).sum()), "pages_min": pages.min() if len(y) else np.nan,
                     "pages_max": pages.max() if len(y) else np.nan})
    out = pd.DataFrame(rows)
    out["pages_min"] = out["pages_min"].astype("Int64")
    out["pages_max"] = out["pages_max"].astype("Int64")
    return out


def select_sample(filings: pd.DataFrame, years=SAMPLE_YEARS, per_year: int = SAMPLE_PER_YEAR) -> pd.DataFrame:
    """The first `per_year` filings by filing date (ties by filing id) in each filing-date year."""
    f = filings.loc[filings["fdate"].notna()].copy()
    f["sample_year"] = f["fdate"].dt.year
    f = f[f["sample_year"].isin(list(years))].sort_values(["sample_year", "fdate", "filing_id"], kind="stable")
    f["sample_index"] = f.groupby("sample_year").cumcount() + 1
    return f[f["sample_index"] <= per_year].reset_index(drop=True)


def activity_summary(filings: pd.DataFrame) -> dict:
    """Filing-date facts of one household (the plan's activity-or-absence question)."""
    d = filings["fdate"]
    in_a = filings[(d >= "2015-01-01") & (d <= "2020-12-31")]
    pages = pd.to_numeric(in_a["page_count"], errors="coerce").fillna(0)
    return {"filings_total": int(len(filings)), "filings_2015_2020": int(len(in_a)),
            "filings_2015_2020_image_only": int(in_a["state"].isin(vs.IMAGE_ONLY_STATES).sum()),
            "filings_2015_2020_pages": int(pages.sum()),
            "filings_pre2023_recount": int((d < RECOVERY_CUTOFF).sum()), "filings_2023plus_recount": int((d >= RECOVERY_CUTOFF).sum()),
            "first_filing_year": int(d.dt.year.min()) if d.notna().any() else np.nan,
            "last_filing_year": int(d.dt.year.max()) if d.notna().any() else np.nan,
            "years_with_filings_2015_2020": int(in_a["fdate"].dt.year.nunique()),
            "activity_evidence_2015_2020": bool(len(in_a) >= 1)}


# ---------------------------------------------------------------- rendering and OCR

def page_dir(sha256: str) -> Path:
    return PAGES_DIR / sha256[:16]


def sorted_pages(directory: Path) -> list[Path]:
    return sorted(directory.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1]))


def render_pdf(pdf: Path, out_dir: Path, dpi: int = DPI) -> list[Path]:
    """Every page as a grayscale PNG at `dpi` (pdftoppm); pages already rendered are reused."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "rendered"
    if marker.exists():
        return sorted_pages(out_dir)
    for stale in out_dir.glob("p-*.png"):
        stale.unlink()
    proc = subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(out_dir / "p")], capture_output=True, text=True, check=False)
    marker.write_text(f"dpi={dpi}\nreturncode={proc.returncode}\n")
    return sorted_pages(out_dir)


def _worker_init() -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"


def _tesseract_lines(img) -> dict:
    """tesseract (eng, --psm 6) word table -> reading-order lines, word count and mean word confidence."""
    import pytesseract

    data = pytesseract.image_to_data(img, lang="eng", config=f"--psm {PSM}", output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list] = {}
    confs = []
    for i, text in enumerate(data["text"]):
        word = str(text).strip()
        if not word:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines.setdefault(key, []).append((int(data["word_num"][i]), word))
        conf = float(data["conf"][i])
        if conf >= 0:
            confs.append(conf)
    text_lines = [" ".join(w for _, w in sorted(ws)) for _, ws in sorted(lines.items())]
    return {"lines": text_lines, "n_words": len(confs), "mean_conf": float(np.mean(confs)) if confs else float("nan"),
            "transaction_lines": count_transaction_lines(text_lines)}


def ocr_png(png: Path, out_json: Path) -> dict:
    """The plan's read of one rendered page (as rendered, no rotation); cached as JSON beside the text."""
    if out_json.exists():
        return json.loads(out_json.read_text())
    from PIL import Image

    with Image.open(png) as img:
        out = _tesseract_lines(img)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.with_suffix(".txt").write_text("\n".join(out["lines"]))
    out_json.write_text(json.dumps(out))
    return out


def orientation_png(png: Path, out_json: Path) -> dict:
    """Diagnostic only: tesseract's orientation estimate and the best of the four right-angle orientations (most date lines)."""
    if out_json.exists():
        return json.loads(out_json.read_text())
    import pytesseract
    from PIL import Image

    with Image.open(png) as img:
        try:
            osd = pytesseract.image_to_osd(img, config="--psm 0")
            rotate = int(re.search(r"Rotate: (\d+)", osd).group(1))
        except (pytesseract.TesseractError, AttributeError):
            rotate = -1
        reads = []
        for angle in DIAG_ANGLES:
            r = _tesseract_lines(img.rotate(angle, expand=True) if angle else img)
            reads.append({"angle": angle, "n_words": r["n_words"], "mean_conf": r["mean_conf"], "transaction_lines": r["transaction_lines"],
                          "date_lines": sum(bool(DATE_RE.search(ln)) for ln in r["lines"]),
                          "loose_date_lines": sum(bool(LOOSE_DATE_RE.search(ln)) for ln in r["lines"])})
    best = max(reads, key=lambda r: (r["date_lines"], r["loose_date_lines"], r["mean_conf"] if np.isfinite(r["mean_conf"]) else -1))
    out = {"osd_rotate": rotate, "best": best, "reads": reads}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    return out


def _ocr_job(args: tuple[str, str]) -> tuple[str, dict]:
    png, out_json = args
    return png, ocr_png(Path(png), Path(out_json))


def _orientation_job(args: tuple[str, str]) -> tuple[str, dict]:
    png, out_json = args
    return png, orientation_png(Path(png), Path(out_json))


def orientation_diagnostic(docs: pd.DataFrame, workers: int, log) -> dict[str, list[dict]]:
    """Diagnostic on pages already rendered for the sample: sha256 -> per-page orientation reads."""
    pages_of = {d: [str(p) for p in sorted_pages(page_dir(d))] for d in docs["sha256"].drop_duplicates()}
    jobs = [(p, str(OCR_DIR / sha[:16] / f"{Path(p).stem}.orientation.json")) for sha, pngs in pages_of.items() for p in pngs]
    log(f"orientation diagnostic on {len(jobs)} sampled pages (four orientations each) with {workers} workers")
    results: dict[str, dict] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for k, (png, res) in enumerate(pool.map(_orientation_job, jobs, chunksize=1), start=1):
            results[png] = res
            if k % 50 == 0 or k == len(jobs):
                log(f"  orientation {k}/{len(jobs)} pages, {time.time() - t0:.0f}s")
    return {sha: [results[p] for p in pngs] for sha, pngs in pages_of.items()}


def filing_orientation_stats(pages: list[dict]) -> dict:
    best = [p["best"] for p in pages]
    n_words = sum(b["n_words"] for b in best)
    conf = sum(b["mean_conf"] * b["n_words"] for b in best if b["n_words"]) / n_words if n_words else np.nan
    return {"diag_pages_osd_rotated": int(sum(p["osd_rotate"] not in (0, -1) for p in pages)),
            "diag_pages_osd_failed": int(sum(p["osd_rotate"] == -1 for p in pages)),
            "diag_pages_best_angle_nonzero": int(sum(b["angle"] != 0 for b in best)),
            "diag_best_date_lines": int(sum(b["date_lines"] for b in best)),
            "diag_best_loose_date_lines": int(sum(b["loose_date_lines"] for b in best)),
            "diag_pages_best_ge3_date_lines": int(sum(b["date_lines"] >= MIN_READABLE_LINES for b in best)),
            "diag_best_rule_lines": int(sum(b["transaction_lines"] for b in best)),
            "diag_pages_best_readable": int(sum(b["transaction_lines"] >= MIN_READABLE_LINES for b in best)),
            "diag_best_mean_word_confidence": round(float(conf), 2) if np.isfinite(conf) else np.nan}


def ocr_filings(docs: pd.DataFrame, raw_store: Path, workers: int, log) -> dict[str, list[dict]]:
    """Render and OCR every page of the given documents; returns sha256 -> per-page results in page order."""
    jobs: list[tuple[str, str]] = []
    pages_of: dict[str, list[str]] = {}
    for d in docs.drop_duplicates("sha256").itertuples(index=False):
        pdf = object_path(raw_store, d.sha256, d.stored_suffix)
        pngs = render_pdf(pdf, page_dir(d.sha256))
        pages_of[d.sha256] = [str(p) for p in pngs]
        jobs += [(str(p), str(OCR_DIR / d.sha256[:16] / f"{p.stem}.json")) for p in pngs]
    log(f"rendered {len(pages_of)} documents, {len(jobs)} pages; OCR with {workers} workers")
    results: dict[str, dict] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for k, (png, res) in enumerate(pool.map(_ocr_job, jobs, chunksize=1), start=1):
            results[png] = res
            if k % 50 == 0 or k == len(jobs):
                log(f"  OCR {k}/{len(jobs)} pages, {time.time() - t0:.0f}s")
    return {sha: [results[p] for p in pngs] for sha, pngs in pages_of.items()}


def filing_ocr_stats(pages: list[dict]) -> dict:
    n_words = sum(p["n_words"] for p in pages)
    conf = sum(p["mean_conf"] * p["n_words"] for p in pages if p["n_words"]) / n_words if n_words else np.nan
    readable = sum(page_is_readable(p["lines"]) for p in pages)
    return {"pages": len(pages), "readable_pages": int(readable), "matched_lines_total": int(sum(p["transaction_lines"] for p in pages)),
            "readable": bool(readable >= 1), "mean_word_confidence": round(float(conf), 2) if np.isfinite(conf) else np.nan}


def feasibility(sample_rows: pd.DataFrame) -> dict:
    n = int(len(sample_rows))
    readable = int(sample_rows["readable"].sum()) if n else 0
    return {"sampled_filings": n, "sampled_pages": int(sample_rows["pages"].sum()) if n else 0,
            "readable_pages": int(sample_rows["readable_pages"].sum()) if n else 0, "readable_filings": readable,
            "matched_lines_total": int(sample_rows["matched_lines_total"].sum()) if n else 0,
            "feasible": bool(n > 0 and readable / n >= FEASIBLE_SHARE)}


# ---------------------------------------------------------------- recovery

def load_crsp_names(path: Path = CRSP_NAMES) -> pd.DataFrame:
    names = pd.read_csv(path, usecols=["permno", "ticker", "namedt", "nameendt"], dtype={"ticker": str})
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    return names


def match_permno(rows: pd.DataFrame, names: pd.DataFrame) -> pd.Series:
    """Date-valid exact-ticker rule: the unique PERMNO whose name row covers the trade date; NaN otherwise."""
    cand = rows.loc[rows["ticker"].notna() & rows["trade_date"].notna(), ["ticker", "trade_date"]].reset_index()
    cand = cand.merge(names, on="ticker", how="inner")
    cand = cand[(cand["namedt"] <= cand["trade_date"]) & (cand["trade_date"] <= cand["nameendt"])]
    n = cand.groupby("index")["permno"].nunique()
    unique = cand[cand["index"].isin(n.index[n == 1])].drop_duplicates("index").set_index("index")["permno"]
    return pd.Series(rows.index.map(unique), index=rows.index, dtype="float64")


def recovered_events(rows: pd.DataFrame, calendar: pd.DatetimeIndex, cache_path: Path = fc.CACHE_PATH) -> pd.DataFrame:
    """Episodes per (household, PERMNO, trade date, direction) in the event-frame schema with cache outcomes in bp."""
    r = rows.loc[rows["permno"].notna() & rows["trade_date"].notna()].copy()
    r["permno"] = r["permno"].astype("int64")
    key = ["household_id", "permno", "trade_date", "direction"]
    ep = r.groupby(key, sort=True).agg(n_lines=("raw_text", "size")).reset_index()
    if ep.empty:
        return pd.DataFrame(columns=EVENT_FRAME_COLUMNS)
    ep["anchor_date"] = first_session_on_or_after(ep["trade_date"], calendar)
    ep = ep.loc[ep["anchor_date"].notna()].copy()
    ep["window_end_date"] = window_end(ep["anchor_date"], calendar, HORIZON)
    ep["period"] = assign_period(ep["anchor_date"], ep["window_end_date"])
    ep["anchor_year"] = ep["anchor_date"].dt.year
    joined = ep[key].astype(str).agg("|".join, axis=1)
    ep["event_id"] = "TO:" + joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])
    ep["clock"] = "T"
    ep["record_source"] = RECORD_SOURCE
    ep["dup_any_line"] = False
    ep["dup_all_lines"] = False
    ep["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ep["anchor_date"]))
    cache = pd.read_parquet(cache_path, columns=["permno", "anchor_idx", "resid_sd", *evf.CACHE_Y.values()],
                            filters=[("permno", "in", sorted(ep["permno"].unique().tolist()))])
    out = ep.merge(cache, on=["permno", "anchor_idx"], how="left")
    for y, col in evf.CACHE_Y.items():
        out[y] = out[col].astype(float) * evf.BP
    out["sigma_event"] = out["resid_sd"].astype(float) * np.sqrt(evf.SIGMA_HORIZON) * evf.BP
    return out.drop(columns=[*evf.CACHE_Y.values(), "resid_sd", "anchor_idx", "trade_date", "n_lines"])


def tau_rows(ctx: rs.Context, added: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[list[dict], dict]:
    """Headline tau2 on the record and once more with the recovered period-A purchases appended."""
    spec = sp.headline_spec("T")
    base = ctx.frames["T"]
    extra = added[base.columns] if len(added) else base.iloc[0:0]
    aug = pd.concat([base, extra], ignore_index=True)
    if not aug["event_id"].is_unique:
        raise ValueError("recovered event ids collide with the record")
    ctx2 = rs.Context(frames={"T": aug}, calendar=calendar, cutoffs=evf.pooled_cutoffs, placebo_source=lambda *a: iter(()))
    out, stats = [], {}
    for variant, c, note in (("event_share", ctx, "headline on record, recomputed in this run"), ("ocr_recovery", ctx2, PROVISIONAL)):
        panel = rs.block_cluster(c.panel(spec), calendar)
        r = cgm.tau2_i1(panel, "ES", anchor="anchor_blk")
        n_ev, n_hh = int(r["n_events_a"] + r["n_events_b"]), int(r["n_households"])
        stats[variant] = r
        notes = f"{note}; events A {r['n_events_a']}, B {r['n_events_b']}; {r['n_anchor_dates']} blocks, dof {r['dof']}"
        if variant == "ocr_recovery":
            notes += (f"; households added {n_hh - stats['event_share']['n_households']} of 2; "
                      f"recovered period-A purchases appended {len(added)}; never substituted for the headline")
        out += [rs.row("T", "tau", variant, "tau2", r["tau2_hat"], "bp2", se_block=float(r["var_wx"] * r["se_b"]), n_events=n_ev,
                       n_households=n_hh, notes=notes),
                rs.row("T", "tau", variant, "tau", r["tau_hat"], "bp", n_events=n_ev, n_households=n_hh, notes=f"signed square root; {note}"),
                rs.row("T", "tau", variant, "t_block", r["t_stat"], "t", n_events=n_ev, n_households=n_hh,
                       notes=f"household x block clustered; dof {r['dof']}; {note}")]
    out.append(rs.row("T", "tau", "ocr_recovery", "households_added", stats["ocr_recovery"]["n_households"] - stats["event_share"]["n_households"],
                      "count", notes=f"{PROVISIONAL}; households of the two that pass the eligibility rule once their recovered purchases are added"))
    out.append(rs.row("T", "tau", "ocr_recovery", "events_added_A", len(added), "count",
                      notes=f"{PROVISIONAL}; recovered PERMNO-matched period-A purchase episodes appended before the eligibility rule"))
    return out, stats


def recovery_counts(rows: pd.DataFrame, label: str) -> list[dict]:
    """Per transaction-date year: rows extracted, rows with ticker+date+code, purchases/sales and PERMNO matches."""
    out = []
    if rows.empty:
        return out
    r = rows.assign(year=pd.to_datetime(rows["trade_date"]).dt.year)
    for year, g in r.groupby("year", dropna=True):
        kept = g[g["ticker"].notna() & g["trade_date"].notna()]
        buys, sells = kept[kept["direction"] == "purchase"], kept[kept["direction"] == "sale"]
        out.append({"household": label, "status": "recovered", "year": int(year), "rows_extracted": int(len(g)),
                    "rows_with_ticker_date_code": int(len(kept)), "purchases": int(len(buys)),
                    "purchases_matched_permno": int(buys["permno"].notna().sum()), "sales": int(len(sells)),
                    "sales_matched": int(sells["permno"].notna().sum())})
    return out


# ---------------------------------------------------------------- driver

def guard(table: pd.DataFrame, path: Path) -> None:
    vs.assert_no_identifier_columns(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--skip-recovery", action="store_true", help="stop after the feasibility verdict")
    ap.add_argument("--skip-orientation-diagnostic", action="store_true", help="omit the diagnostic re-read of the sampled pages")
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        common.log_line(LOG, f"[two_households] {msg}")

    root = vs.dropbox_project_root()
    official = root / "data" / "official"
    state_db, raw_store = official / "state.sqlite", official / "raw_store"
    diag = pd.read_csv(DIAG_CSV).set_index("household")
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    snapshot = pd.read_parquet(root / vs.VENDOR_SNAPSHOT, columns=["politician_slug", "politician_name", "chamber"])
    filings = vs.load_ptr_filings(state_db).merge(read_primary_documents(state_db), on="filing_id", how="left")
    log(f"PTR filings in the census {len(filings)}; households selected {len(households)}")

    by_year, summary, sample_rows, docs_by_label, labels = [], [], [], {}, {}
    for i, household in enumerate(households, start=1):
        label = f"household_{i}"
        labels[label] = household
        cand = household_filings(household, snapshot, filings)
        act = activity_summary(cand)
        vendor_lines = int(((matched["filer_uid"] == household) & (matched["record_source"] == "capitol_trades")).sum())
        d = diag.loc[label]
        checks = {"filings_total": (act["filings_total"], int(d["official_ptr_filings_total"])),
                  "filings_pre2023": (act["filings_pre2023_recount"], int(d["official_ptr_filings_pre2023"])),
                  "filings_2023plus": (act["filings_2023plus_recount"], int(d["official_ptr_filings_2023plus"])),
                  "vendor_lines": (vendor_lines, int(d["vendor_lines_matched_frame"]))}
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if bad:
            raise ValueError(f"{label}: recount differs from the diagnostics CSV: {bad}")
        log(f"{label}: {act['filings_total']} PTR filings (diagnostics agree on total / pre-2023 / 2023+ / vendor lines); "
            f"2015-2020 filings {act['filings_2015_2020']} with {act['filings_2015_2020_pages']} pages, "
            f"image-only {act['filings_2015_2020_image_only']}; report_year == filing-date year on "
            f"{int((cand['report_year'] == cand['fdate'].dt.year).sum())} of {len(cand)}")
        by_year.append(filings_by_year(cand, label))
        sample = select_sample(cand)
        docs_by_label[label] = {"all": cand, "sample": sample}
        row = {"household": label, **{k: d[v] for k, v in COPIED_FROM_DIAGNOSTICS.items()},
               **{k: v for k, v in act.items() if not k.endswith("_recount")}}
        summary.append(row)
    guard(pd.concat(by_year, ignore_index=True), BY_YEAR_CSV)
    log(f"wrote {BY_YEAR_CSV.relative_to(common.REPO_ROOT)}")

    sample_docs = pd.concat([v["sample"].assign(household=k) for k, v in docs_by_label.items()], ignore_index=True)
    log(f"feasibility sample: {len(sample_docs)} filings, {int(sample_docs['page_count'].sum())} pages by the census page counts")
    ocr = ocr_filings(sample_docs, raw_store, args.workers, log)
    for s in sample_docs.itertuples(index=False):
        sample_rows.append({"household": s.household, "report_year": int(s.report_year), "sample_year": int(s.sample_year),
                            "sample_index": int(s.sample_index), "doc_format": s.doc_format, **filing_ocr_stats(ocr[s.sha256])})
    sample_table = pd.DataFrame(sample_rows).sort_values(["household", "sample_year", "sample_index"]).reset_index(drop=True)
    guard(sample_table, SAMPLE_CSV)
    verdict = {}
    for row in summary:
        verdict[row["household"]] = feasibility(sample_table[sample_table["household"] == row["household"]])
        row.update(verdict[row["household"]])
        log(f"{row['household']}: sampled {row['sampled_filings']} filings / {row['sampled_pages']} pages; readable pages "
            f"{row['readable_pages']}, readable filings {row['readable_filings']}, matched lines {row['matched_lines_total']}; "
            f"feasible {row['feasible']}")
    pooled = feasibility(sample_table)
    log(f"pooled over both households: readable filings {pooled['readable_filings']} of {pooled['sampled_filings']} "
        f"(rule: at least {FEASIBLE_SHARE:.0%} with a machine-readable page)")
    if not args.skip_orientation_diagnostic:
        orient = orientation_diagnostic(sample_docs, args.workers, log)
        diag = pd.DataFrame([{"household": s.household, "sample_year": int(s.sample_year), "sample_index": int(s.sample_index),
                              **filing_orientation_stats(orient[s.sha256])} for s in sample_docs.itertuples(index=False)])
        sample_table = sample_table.merge(diag, on=["household", "sample_year", "sample_index"], how="left")
        guard(sample_table, SAMPLE_CSV)
        for row in summary:
            sub = sample_table[sample_table["household"] == row["household"]]
            row.update({k: int(sub[k].sum()) for k in ("diag_pages_osd_rotated", "diag_pages_best_angle_nonzero", "diag_best_date_lines",
                                                        "diag_best_loose_date_lines", "diag_pages_best_ge3_date_lines", "diag_best_rule_lines",
                                                        "diag_pages_best_readable")})
            log(f"{row['household']} diagnostic (not the rule): OSD-rotated pages {row['diag_pages_osd_rotated']}, best-orientation date lines "
                f"{row['diag_best_date_lines']} (loose {row['diag_best_loose_date_lines']}), pages with >= 3 date lines "
                f"{row['diag_pages_best_ge3_date_lines']}, rule lines {row['diag_best_rule_lines']}, readable pages {row['diag_pages_best_readable']}")

    feasible = [lab for lab in labels if verdict[lab]["feasible"]]
    recovery_rows, tau_table = [], None
    if feasible and not args.skip_recovery:
        names = load_crsp_names()
        ctx, calendar = common.load_headline_context(clocks=("T",), with_placebo=False)
        all_rows = []
        for lab in feasible:
            pre = docs_by_label[lab]["all"]
            pre = pre[pre["fdate"] < RECOVERY_CUTOFF]
            log(f"{lab}: OCR of {len(pre)} pre-2023 filings, {int(pre['page_count'].sum())} census pages")
            res = ocr_filings(pre, raw_store, args.workers, log)
            rows = []
            for f in pre.itertuples(index=False):
                for page_no, page in enumerate(res[f.sha256], start=1):
                    for r in extract_rows(page["lines"]):
                        rows.append({"household": lab, "household_id": labels[lab], "filing_date": f.fdate, "page": page_no, **r})
            rows = pd.DataFrame(rows, columns=["household", "household_id", "filing_date", "page", "ticker", "trade_date", "code",
                                               "direction", "amount_bracket", "raw_text"])
            rows["permno"] = match_permno(rows, names)
            for row in summary:
                if row["household"] == lab:
                    row.update({"recovery_filings_ocr": int(len(pre)), "recovery_pages_ocr": int(sum(len(v) for v in res.values()))})
            recovery_rows += recovery_counts(rows, lab)
            all_rows.append(rows)
            log(f"{lab}: rows extracted {len(rows)}, with ticker {int(rows['ticker'].notna().sum())}, PERMNO matched {int(rows['permno'].notna().sum())}")
        rows = pd.concat(all_rows, ignore_index=True)
        RESTRICTED.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(ROWS_PARQUET, index=False)
        events = recovered_events(rows, calendar)
        added = events[(events["direction"] == "purchase") & (events["period"] == "A")].reset_index(drop=True)
        for row in summary:
            row["recovered_period_a_purchases"] = int((added["household_id"] == labels[row["household"]]).sum()) if len(added) else 0
        log(f"recovered episodes {len(events)}; period-A purchases appended {len(added)}, "
            f"with finite headline outcome {int(np.isfinite(added['y_h20']).sum()) if len(added) else 0}")
        rows_out, stats = tau_rows(ctx, added, calendar)
        tau_table = common.write_rows(rows_out, TAU_CSV)
        vs.assert_no_identifier_columns(tau_table)
        log(f"tau2 on record {stats['event_share']['tau2_hat']:.2f} bp2 ({stats['event_share']['n_households']} households); "
            f"with recovery {stats['ocr_recovery']['tau2_hat']:.2f} bp2 ({stats['ocr_recovery']['n_households']} households), {PROVISIONAL}")
    for lab in labels:
        if lab not in feasible:
            recovery_rows.append({"household": lab, "status": "infeasible", **{k: v for k, v in verdict[lab].items() if k != "feasible"}})
    if not feasible:
        recovery_rows = [{"household": "both", "status": "infeasible", **{k: v for k, v in pooled.items() if k != "feasible"},
                          "notes": "no recovery run; the hand-entry backlog is the only path"}]
    guard(pd.DataFrame(recovery_rows), RECOVERY_CSV)
    guard(pd.DataFrame(summary), SUMMARY_CSV)

    entries = [common.file_entry(state_db), common.file_entry(DIAG_CSV, rows=len(diag)),
               common.file_entry(load.FRAMES_DIR / "matched_transactions.csv", rows=len(matched)),
               common.file_entry(root / vs.VENDOR_SNAPSHOT), common.file_entry(CRSP_NAMES, rows=len(load_crsp_names())),
               common.file_entry(fc.CACHE_PATH), common.file_entry(fc.MANIFEST_PATH),
               common.file_entry(ef.EVENTS_DIR / "events_T.parquet"), common.file_entry(ef.FLAGS_PATH), common.file_entry(evf.POOLED_CSV)]
    engine = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=False).stdout.split("\n")[0]
    common.write_manifest("two_households", entries, extra={"ocr_engine": engine, "dpi": DPI, "psm": PSM, "workers": args.workers,
                                                             "feasible": feasible, "seconds": round(time.time() - t0, 1)})
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
