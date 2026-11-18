"""Addendum section D: extraction pilot on the two households' scans (a validation sheet, not a recovery).

For each household the first two filings by filing date (received date when missing) of 2017 and of 2019
are rendered at 300 dpi, rotated to the orientation tesseract's detector prefers (best of the four right
angles by mean word confidence when the detector fails) and read with word boxes. The prescribed reads are
psm 6 and psm 4 on the rotated page; because the printed table grid merges rows into unreadable blocks, the
same page with its long straight dark runs (the grid lines) whitened is read as well, in psm 6, 4 and 11
(sparse text), and the read with the most date tokens is kept, every read's count recorded. Rows are built
around date tokens on geometrically grouped lines: the asset text is the words left of the date (a leading
JT/SP/DC owner code and mark tokens set aside), the ticker a parenthesised 1-5 letter token in it, and the
transaction type and amount bracket come from mark tokens (X, x, ×, ✓, or a standalone P/S/E) placed by
horizontal position against the column headers found on the page (horizontal headers from the upright
read, the vertical PURCHASE / SALE / EXCHANGE and bracket labels from the raw-orientation read, mapped to
upright coordinates); without headers the fields stay unassigned. A cell ink density per row is offered to
the researcher as a second signal only. Nothing here enters an estimate.

Run: PYTHONPATH=src python -m trading_on_disclosure_v4.ext.two_households_pilot [--workers 6]
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from congressional_trading.official.storage import object_path
from trading_on_disclosure_v4.data import load
from trading_on_disclosure_v4.data import vendor_only_status as vs
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.ext import common
from trading_on_disclosure_v4.ext import two_households as th

PILOT_DIR = th.RESTRICTED / "pilot"
PAGES_DIR = PILOT_DIR / "pages"
OCR_DIR = PILOT_DIR / "ocr"
SHEET_CSV = PILOT_DIR / "pilot_validation_sheet.csv"
COUNTS_CSV = common.OUT_DIR / "two_households_pilot.csv"
PAGES_CSV = common.OUT_DIR / "two_households_pilot_pages.csv"
LOG = common.LOG_DIR / "two_households_pilot.log"
SAMPLE_YEARS = (2017, 2019)
PER_YEAR = 2
PRESCRIBED_PSMS = (6, 4)
EXTRA_PSMS = (11,)
GRID_RUN_PX = 200
DARK = 128
MARK_RE = re.compile(r"^[xX×✓]$")
CODE_RE = re.compile(r"^[PSE]$")
DATE_TOKEN_RE = re.compile(r"^(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})[.,;:]?$")
FIGURE_RE = re.compile(r"^\$?\d{1,3}(?:[,.]\d{3})+-?\$?(?:\d{1,3}(?:[,.]\d{3})+)?$")
OWNER_CODES = ("JT", "SP", "DC")
TYPE_LABELS = {"PURCHASE": "P", "SALE": "S", "EXCHANGE": "E"}
BRACKET_LETTERS = "ABCDEFGHIJ"
BRACKET_LABELS = ("$1,001 - $15,000", "$15,001 - $50,000", "$50,001 - $100,000", "$100,001 - $250,000",
                  "$250,001 - $500,000", "$500,001 - $1,000,000", "$1,000,001 - $5,000,000",
                  "$5,000,001 - $25,000,000", "$25,000,001 - $50,000,000", "Over $50,000,000")
INK_MIN = 0.02
WORD_COLUMNS = ["text", "conf", "left", "top", "width", "height"]


# ---------------------------------------------------------------- tokens

def date_token(text: str) -> tuple[pd.Timestamp, str] | None:
    """A date-shaped token -> (date, pattern such as mm/dd/yyyy, m/d/yy or mm-dd-yyyy); None otherwise."""
    m = DATE_TOKEN_RE.match(text)
    if not m:
        return None
    mo, sep, d, y = m.groups()
    year = int(y) if len(y) == 4 else 2000 + int(y)
    try:
        ts = pd.Timestamp(year=year, month=int(mo), day=int(d))
    except ValueError:
        ts = pd.NaT
    pattern = f"{'m' * len(mo)}{sep}{'d' * len(d)}{sep}{'y' * len(y)}"
    return ts, pattern


def ticker_in(asset_text: str) -> str | None:
    m = th.TICKER_RE.search(asset_text)
    return m.group(1) if m else None


def is_mark(text: str) -> bool:
    return bool(MARK_RE.match(text) or CODE_RE.match(text))


# ---------------------------------------------------------------- geometry

def rotate_box(left: int, top: int, width: int, height: int, angle: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Box of a raw-orientation word in the frame of the page rotated `angle` degrees counter-clockwise (PIL)."""
    w_img, h_img = size
    if angle == 0:
        return left, top, width, height
    if angle == 90:
        return top, w_img - (left + width), height, width
    if angle == 180:
        return w_img - (left + width), h_img - (top + height), width, height
    if angle == 270:
        return h_img - (top + height), left, height, width
    raise ValueError(f"angle {angle} is not a right angle")


def group_lines(words: pd.DataFrame, tol: float = 0.6) -> list[list[int]]:
    """Index lists of words sharing a line: vertical centres within `tol` x the median word height, left to right."""
    if words.empty:
        return []
    centre = (words["top"] + words["height"] / 2).to_numpy(dtype=float)
    med = float(np.median(words["height"])) or 1.0
    order = np.argsort(centre, kind="stable")
    lines: list[list[int]] = []
    current: list[int] = []
    current_centre = 0.0
    for i in order:
        if current and abs(centre[i] - current_centre) > tol * med:
            lines.append(current)
            current = []
        current.append(int(i))
        current_centre = float(np.mean(centre[current]))
    if current:
        lines.append(current)
    lefts = words["left"].to_numpy()
    return [sorted(line, key=lambda k: lefts[k]) for line in lines]


def _x_range(sub: pd.DataFrame) -> tuple[float, float]:
    return float(sub["left"].min()), float((sub["left"] + sub["width"]).max())


def _nearest(x: float, columns: dict[str, tuple[float, float]]) -> tuple[str | None, float]:
    best, dist = None, np.inf
    for name, (x0, x1) in columns.items():
        d = 0.0 if x0 <= x <= x1 else min(abs(x - x0), abs(x - x1))
        if d < dist:
            best, dist = name, d
    return best, dist


BRACKET_LOWER = {"1001": 0, "1000": 0, "15001": 1, "50001": 2, "100001": 3, "250001": 4, "500001": 5, "1000001": 6,
                 "5000001": 7, "25000001": 8}
BRACKET_UPPER = {"15000": 0, "50000": 1, "100000": 2, "250000": 3, "500000": 4, "1000000": 5, "5000000": 6, "25000000": 7}
BRACKET_OVER = {"50000000": 9, "1000000": 10}
BRACKET_LABELS_K = (*BRACKET_LABELS, "Over $1,000,000 (spouse or dependent child asset)")
ANCHOR_TOL = 0.3


def header_band(words: pd.DataFrame) -> float:
    """Bottom of the header area: the top of the first word that is a date token (inf when none)."""
    tops = [w.top for w in words.itertuples(index=False) if date_token(str(w.text)) is not None]
    return float(min(tops)) if tops else float("inf")


def bracket_anchors(words: pd.DataFrame, band: float, letters: bool = True) -> list[tuple[int, float, str]]:
    """(column index, x-centre, source) from the lettered column headers and the bracket labels in the header band."""
    head = words[words["top"] < band]
    anchors: list[tuple[int, float, str]] = []
    best: list[tuple[int, float]] = []
    for idx in group_lines(head) if letters else []:
        line = head.iloc[idx]
        seen: dict[int, float] = {}
        ok = True
        for w in line.itertuples(index=False):
            t = str(w.text).upper()
            if len(t) == 1 and t in BRACKET_LETTERS + "K":
                k = (BRACKET_LETTERS + "K").index(t)
                if seen and k <= max(seen):
                    ok = False
                seen.setdefault(k, w.left + w.width / 2)
        if ok and len(seen) > len(best):
            best = sorted(seen.items())
    anchors += [(k, x, "letter") for k, x in best]
    for idx in group_lines(head):
        line = head.iloc[idx]
        texts = [str(w.text) for w in line.itertuples(index=False)]
        for j, w in enumerate(line.itertuples(index=False)):
            t = str(w.text).upper().strip(".,:;|")
            first = re.sub(r"[^0-9]", "", t.split("-")[0])
            if not first:
                continue
            prev_over = j > 0 and texts[j - 1].upper().strip(".,:;") == "OVER"
            if t.startswith("OVER") and "-" not in t:
                first = re.sub(r"[^0-9]", "", t)
                prev_over = True
            k = BRACKET_OVER.get(first) if prev_over else (BRACKET_LOWER.get(first) if "-" in t or first in BRACKET_LOWER else BRACKET_UPPER.get(first))
            if k is None and first in BRACKET_UPPER and "-" not in t:
                k = BRACKET_UPPER[first]
            if k is not None:
                anchors.append((k, w.left + w.width / 2, "label"))
    return anchors


def fit_bracket_grid(anchors: list[tuple[int, float, str]]) -> tuple[dict[str, tuple[float, float]], int]:
    """Equally spaced bracket columns through the anchors: the line through the pair of anchors with the most
    inliers (within ANCHOR_TOL x pitch), refitted on them; accepted with four distinct column indices, or three
    when a bracket label is among them; ({}, n_inliers) otherwise."""
    pts = [(int(k), float(x), src) for k, x, src in anchors]
    if len(pts) < 3:
        return {}, len(pts)
    k = np.array([a[0] for a in pts], dtype=float)
    x = np.array([a[1] for a in pts], dtype=float)
    best: np.ndarray | None = None
    best_score = (0, np.inf)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if k[i] == k[j]:
                continue
            b = (x[j] - x[i]) / (k[j] - k[i])
            if b <= 0:
                continue
            a = x[i] - b * k[i]
            res = np.abs(x - (a + b * k))
            inl = res <= ANCHOR_TOL * b
            score = (int(inl.sum()), float(res[inl].sum()))
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best, best_score = inl, score
    if best is None:
        return {}, 0
    kk, xx = k[best], x[best]
    distinct = len(set(kk))
    labels = any(src == "label" for (_, _, src), ok in zip(pts, best, strict=True) if ok)
    if distinct < 3 or (distinct == 3 and not labels):
        return {}, int(best.sum())
    letter = np.array([src == "letter" for _, _, src in pts])[best]
    if len(set(kk[letter])) >= 4:
        kk, xx = kk[letter], xx[letter]
    b, a = np.polyfit(kk, xx, 1)
    if not np.isfinite(b) or b <= 0:
        return {}, int(best.sum())
    n = 11 if int(kk.max()) >= 10 else 10
    grid = a + b * np.arange(n)
    return {lab: (float(g - b / 2), float(g + b / 2)) for lab, g in zip(BRACKET_LABELS_K[:n], grid, strict=True)}, int(best.sum())


def type_columns(labels: dict[str, tuple[float, float, float, float]]) -> tuple[dict[str, tuple[float, float]], bool]:
    """P / S / E ranges from the label boxes found (left, top, width, height); missing ones inferred in the form's order."""
    if not labels:
        return {}, False
    order = "PSE"
    centres = {c: labels[c][0] + labels[c][2] / 2 for c in labels}
    known = sorted(centres, key=lambda c: order.index(c))
    if len(known) >= 2:
        gaps = [(centres[known[i + 1]] - centres[known[i]]) / (order.index(known[i + 1]) - order.index(known[i])) for i in range(len(known) - 1)]
        pitch = float(np.median(gaps))
    else:
        left, top, width, height = labels[known[0]]
        pitch = width * (1.2 if height > width else 1.6)
    if pitch <= 0:
        return {}, False
    base = known[0]
    cols = {c: centres[base] + pitch * (order.index(c) - order.index(base)) for c in order}
    for c in known:
        cols[c] = centres[c]
    return {c: (cols[c] - pitch / 2, cols[c] + pitch / 2) for c in order}, len(known) < 3


def find_headers(upright: pd.DataFrame, vertical: pd.DataFrame | None = None) -> dict:
    """Column x-ranges from the header tokens of the official form: owner / asset / date / notified / amount, the
    PURCHASE / SALE / EXCHANGE labels (horizontal in `upright`, or vertical ones from `vertical`, raw-read words
    mapped to upright coordinates) and the bracket columns fitted through the lettered headers and bracket labels."""
    out: dict = {"found": [], "type": {}, "brackets": {}}
    if upright is None or upright.empty:
        return out
    upper = upright["text"].astype(str).str.upper()
    band = header_band(upright)
    for name, pattern in (("owner", r"^(JT|SP|DC)$"), ("asset", r"^(ASSET|NAME)$"), ("amount", r"^AMOUNT$"), ("notified", r"^NOTIFIED$")):
        sub = upright[upper.str.match(pattern)]
        if name != "owner":
            sub = sub[sub["top"] < band]
            sub = sub[sub["top"] <= sub["top"].min() + sub["height"].max()] if len(sub) else sub
        if len(sub):
            out[name] = _x_range(sub)
            out["found"].append(name)
    dates = upright[upper.eq("DATE") & (upright["top"] < band)]
    if len(dates):
        dates = dates[dates["top"] <= dates["top"].min() + dates["height"].max()].sort_values("left")
        out["date"] = _x_range(dates.iloc[:1])
        out["found"].append("date")
    labels: dict[str, tuple[float, float, float, float]] = {}
    sources = [("h", upright[upright["top"] < band])] + ([("v", vertical)] if vertical is not None and len(vertical) else [])
    for tag, src in sources:
        text = src["text"].astype(str).str.upper().str.strip(".,:;|")
        for label, code in TYPE_LABELS.items():
            hit = src[text.eq(label)]
            if len(hit) and code not in labels:
                w = hit.iloc[0]
                labels[code] = (float(w.left), float(w.top), float(w.width), float(w.height))
                out["found"].append(f"type_label:{code}{tag}")
    out["type"], inferred = type_columns(labels)
    if out["type"]:
        out["found"].append("type_columns" + ("_inferred" if inferred else ""))
    anchors = bracket_anchors(upright, band)
    if vertical is not None and len(vertical):
        anchors += bracket_anchors(vertical, float("inf"), letters=False)
    out["brackets"], n_used = fit_bracket_grid(anchors)
    if out["brackets"]:
        out["found"].append(f"bracket_columns:{n_used}anchors")
    elif anchors:
        out["found"].append(f"bracket_anchors_unfit:{len(anchors)}")
    return out


def assign_marks(line: pd.DataFrame, headers: dict) -> dict:
    """Type and bracket of one row from its mark tokens and the page's column ranges."""
    out = {"assigned_type": None, "assigned_bracket": None, "n_marks": 0, "assignment_note": ""}
    notes = []
    for w in line.itertuples(index=False):
        t = str(w.text)
        if CODE_RE.match(t):
            out["n_marks"] += 1
            if out["assigned_type"] is None:
                out["assigned_type"] = t
            else:
                notes.append("second type token")
            continue
        if not MARK_RE.match(t):
            continue
        out["n_marks"] += 1
        x = w.left + w.width / 2
        candidates = {f"type:{k}": v for k, v in headers.get("type", {}).items()}
        candidates.update({f"bracket:{k}": v for k, v in headers.get("brackets", {}).items()})
        if not candidates:
            notes.append("no columns on page")
            continue
        name, dist = _nearest(x, candidates)
        kind, value = name.split(":", 1)
        if dist > 0:
            notes.append(f"mark outside columns by {dist:.0f}px")
        if kind == "type":
            if out["assigned_type"] is None:
                out["assigned_type"] = value
            else:
                notes.append("second type mark")
        elif out["assigned_bracket"] is None:
            out["assigned_bracket"] = value
        else:
            notes.append("second bracket mark")
    if not headers.get("type") and not headers.get("brackets"):
        notes.append("no headers found on page")
    elif not headers.get("type"):
        notes.append("type columns not found")
    elif not headers.get("brackets"):
        notes.append("bracket columns not found")
    out["assignment_note"] = "; ".join(dict.fromkeys(notes))
    return out


def ink_suggestion(page: np.ndarray, y0: int, y1: int, headers: dict, clip: dict | None = None) -> dict:
    """Second signal only: the type and bracket cell with the most ink in the row band (None below INK_MIN); cells are
    clipped to `clip` = {"type": (lo, hi), "brackets": (lo, hi)} so neighbouring text does not count as ink."""
    out = {"ink_type": None, "ink_bracket": None}
    band = page[max(y0, 0):max(y1, y0 + 1)]
    if band.size == 0:
        return out
    clip = clip or {}
    for key, kind in (("ink_type", "type"), ("ink_bracket", "brackets")):
        lo, hi = clip.get(kind, (-np.inf, np.inf))
        best, density = None, 0.0
        for name, (x0, x1) in headers.get(kind, {}).items():
            a, b = max(int(max(x0, lo)), 0), int(min(x1, hi))
            if b - a < 4:
                continue
            cell = band[:, a:b]
            d = float((cell < DARK).mean()) if cell.size else 0.0
            if d > density:
                best, density = name, d
        out[key] = best if density >= INK_MIN else None
    return out


def candidate_rows(words: pd.DataFrame, headers: dict, page: np.ndarray | None = None, pitch: float | None = None) -> list[dict]:
    """One record per line carrying a date token: date, pattern, owner, asset text, ticker, marks and confidence."""
    rows = []
    lines = [words.iloc[idx].reset_index(drop=True) for idx in group_lines(words)]
    date_right = []
    for line in lines:
        ends = [w.left + w.width for w in line.itertuples(index=False) if date_token(str(w.text)) is not None]
        if ends:
            date_right.append(max(ends))
    amount_left = float(np.quantile(date_right, 0.95)) + 4 if date_right else -np.inf
    for line in lines:
        hits = [(i, date_token(str(t))) for i, t in enumerate(line["text"])]
        hits = [(i, h) for i, h in hits if h is not None]
        if not hits:
            continue
        first, (date, pattern) = hits[0]
        left = line.iloc[:first]
        owner = None
        if len(left) and str(left["text"].iloc[0]).upper() in OWNER_CODES:
            owner = str(left["text"].iloc[0]).upper()
            left = left.iloc[1:]
        asset = " ".join(str(t) for t in left["text"] if not is_mark(str(t)))
        confs = pd.to_numeric(line["conf"], errors="coerce")
        rec = {"raw_line_text": " ".join(map(str, line["text"])), "date": date, "date_pattern": pattern,
               "notification_date": hits[1][1][0] if len(hits) > 1 else pd.NaT, "owner": owner, "asset_text": asset,
               "ticker": ticker_in(asset), "mean_word_confidence": round(float(confs[confs >= 0].mean()), 1) if (confs >= 0).any() else np.nan,
               "line_top": int(line["top"].min()), "line_bottom": int((line["top"] + line["height"]).max())}
        rec.update(assign_marks(line, headers))
        if page is not None:
            half = (pitch or float(line["height"].median())) / 2
            centre = (rec["line_top"] + rec["line_bottom"]) / 2
            clip = {"type": (-np.inf, float(line["left"].iloc[first]) - 4), "brackets": (amount_left, np.inf)}
            rec.update(ink_suggestion(page, int(centre - half), int(centre + half), headers, clip))
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- page processing

def strip_grid_lines(gray: np.ndarray, min_run: int = GRID_RUN_PX) -> np.ndarray:
    """Whiten dark runs of at least `min_run` pixels along rows and columns (the printed table grid)."""
    out = gray.copy()
    dark = gray < DARK
    for axis in (0, 1):
        arr = dark if axis == 0 else dark.T
        target = out if axis == 0 else out.T
        for i in range(arr.shape[0]):
            row = arr[i]
            if row.sum() < min_run:
                continue
            edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
                if e - s >= min_run:
                    target[i, s:e] = 255
    return out


def word_table(img, psm: int) -> pd.DataFrame:
    import pytesseract

    d = pytesseract.image_to_data(img, lang="eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    t = pd.DataFrame({k: d[k] for k in ("text", "conf", "left", "top", "width", "height")})
    t["text"] = t["text"].astype(str).str.strip()
    t["conf"] = pd.to_numeric(t["conf"], errors="coerce")
    return t[t["text"] != ""].reset_index(drop=True)[WORD_COLUMNS]


def detect_orientation(img) -> tuple[int, str]:
    """Counter-clockwise angle that makes the page upright: tesseract OSD, else the best of four by mean confidence."""
    import pytesseract

    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        rotate = int(re.search(r"Rotate: (\d+)", osd).group(1))
        return (360 - rotate) % 360, "osd"
    except (pytesseract.TesseractError, AttributeError, ValueError):
        pass
    best, conf = 0, -1.0
    for angle in (0, 90, 180, 270):
        t = word_table(img.rotate(angle, expand=True) if angle else img, PRESCRIBED_PSMS[0])
        c = float(t.loc[t["conf"] >= 0, "conf"].mean()) if (t["conf"] >= 0).any() else -1.0
        if c > conf:
            best, conf = angle, c
    return best, "fallback_best_of_four"


def n_dates(t: pd.DataFrame) -> int:
    return int(sum(date_token(str(x)) is not None for x in t["text"]))


def _worker_init() -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"


def process_page(args: tuple[str, str, int, str]) -> dict:
    """Orientation, every read, chosen read, headers, candidate rows and the restricted artefacts of one page."""
    png, label, page_no, filing_key = args
    from PIL import Image

    src = Path(png)
    out_pages = PAGES_DIR / filing_key
    out_ocr = OCR_DIR / filing_key
    out_pages.mkdir(parents=True, exist_ok=True)
    out_ocr.mkdir(parents=True, exist_ok=True)
    raw = Image.open(src).convert("L")
    angle, how = detect_orientation(raw)
    upright = raw.rotate(angle, expand=True) if angle else raw
    rotated_path = out_pages / f"p-{page_no}.rot{angle}.png"
    upright.save(rotated_path)
    gray = np.array(upright)
    stripped = strip_grid_lines(gray)
    stripped_img = Image.fromarray(stripped)
    stripped_img.save(out_pages / f"p-{page_no}.rot{angle}.nogrid.png")
    reads: dict[str, pd.DataFrame] = {}
    for psm in PRESCRIBED_PSMS:
        reads[f"psm{psm}"] = word_table(upright, psm)
    for psm in (*PRESCRIBED_PSMS, *EXTRA_PSMS):
        reads[f"psm{psm}_nogrid"] = word_table(stripped_img, psm)
    raw_stripped = Image.fromarray(strip_grid_lines(np.array(raw)))
    raw_reads = {"raw_psm6_mapped": word_table(raw, PRESCRIBED_PSMS[0]), "raw_psm11_nogrid_mapped": word_table(raw_stripped, EXTRA_PSMS[0])}
    counts = {name: n_dates(t) for name, t in reads.items()}
    chosen = max(reads, key=lambda k: (counts[k], -list(reads).index(k)))
    words = reads[chosen]
    mapped = []
    for name, t in raw_reads.items():
        boxes = [rotate_box(int(w.left), int(w.top), int(w.width), int(w.height), angle, raw.size) for w in t.itertuples(index=False)]
        mapped.append(t.assign(**{k: [b[i] for b in boxes] for i, k in enumerate(("left", "top", "width", "height"))}, read=name) if len(t) else t.assign(read=name))
    vertical = pd.concat(mapped, ignore_index=True)
    union = pd.concat([words, *[t for k, t in reads.items() if k != chosen]], ignore_index=True) if len(words) else words
    headers = find_headers(union, vertical[WORD_COLUMNS])
    table = pd.concat([t.assign(read=k) for k, t in reads.items()] + [vertical], ignore_index=True)
    table.to_csv(out_ocr / f"p-{page_no}.words.tsv", sep="\t", index=False)
    lines = group_lines(words)
    tops = sorted(float(words.iloc[idx]["top"].min()) for idx in lines) if lines else []
    pitch = float(np.median(np.diff(tops))) if len(tops) > 2 else None
    rows = candidate_rows(words, headers, stripped, pitch)
    confs = words.loc[words["conf"] >= 0, "conf"]
    record = {"household": label, "filing_key": filing_key, "page": page_no, "orientation": angle, "orientation_source": how,
              "read": chosen, "psm": int(re.search(r"psm(\d+)", chosen).group(1)), "grid_stripped": chosen.endswith("_nogrid"),
              "words": int(len(words)), "mean_word_confidence": round(float(confs.mean()), 1) if len(confs) else np.nan,
              "date_tokens": counts[chosen], **{f"dates_{k}": v for k, v in counts.items()},
              "headers_found": len(headers["found"]), "header_list": "|".join(headers["found"]),
              "type_columns": bool(headers["type"]), "bracket_columns": bool(headers["brackets"]), "candidate_rows": len(rows),
              "page_image_path": str(rotated_path.relative_to(common.REPO_ROOT))}
    for r in rows:
        r.update({"household": label, "filing_key": filing_key, "page": page_no, "page_image_path": record["page_image_path"],
                  "orientation": angle, "read": chosen, "psm": record["psm"], "grid_stripped": record["grid_stripped"]})
    return {"page": record, "rows": rows}


# ---------------------------------------------------------------- driver

def pilot_sample(docs_by_label: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """First two filings by filing date of 2017 and of 2019 per household, with a filing index 1..4."""
    parts = []
    for label, cand in docs_by_label.items():
        s = th.select_sample(cand, years=SAMPLE_YEARS, per_year=PER_YEAR).assign(household=label)
        s["filing_index"] = range(1, len(s) + 1)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def count_table(pages: pd.DataFrame, sheet: pd.DataFrame, resolved: pd.Series) -> pd.DataFrame:
    rows = []
    for label in [*sorted(pages["household"].unique()), "total"]:
        p = pages if label == "total" else pages[pages["household"] == label]
        s = sheet if label == "total" else sheet[sheet["household"] == label]
        res = resolved if label == "total" else resolved.loc[s.index]
        full = s["date"].notna() & s["ticker"].notna() & s["assigned_type"].notna() & s["assigned_bracket"].notna()
        dtb = s["date"].notna() & s["assigned_type"].notna() & s["assigned_bracket"].notna()
        ink = s["date"].notna() & s["ink_type"].notna() & s["ink_bracket"].notna()
        rows.append({"household": label, "filings": int(p["filing_key"].nunique()), "pages": int(len(p)),
                     "pages_with_headers": int((p["headers_found"] > 0).sum()), "pages_with_type_columns": int(p["type_columns"].sum()),
                     "pages_with_bracket_columns": int(p["bracket_columns"].sum()),
                     "pages_prescribed_read": int((~p["grid_stripped"]).sum()), "pages_grid_stripped_read": int(p["grid_stripped"].sum()),
                     "pages_psm6": int((p["psm"] == 6).sum()), "pages_psm4": int((p["psm"] == 4).sum()), "pages_psm11": int((p["psm"] == 11).sum()),
                     "date_tokens": int(p["date_tokens"].sum()), "candidate_rows": int(len(s)), "rows_date_parsed": int(s["date"].notna().sum()),
                     "rows_with_ticker": int(s["ticker"].notna().sum()), "rows_with_type": int(s["assigned_type"].notna().sum()),
                     "rows_with_bracket": int(s["assigned_bracket"].notna().sum()), "rows_fully_assigned": int(full.sum()),
                     "rows_date_type_bracket": int(dtb.sum()), "rows_ink_type": int(s["ink_type"].notna().sum()),
                     "rows_ink_bracket": int(s["ink_bracket"].notna().sum()), "rows_date_ink_type_bracket": int(ink.sum()),
                     "tickers_common_stock_on_date": int(res.notna().sum()),
                     "mean_line_confidence": round(float(s["mean_word_confidence"].mean()), 1) if len(s) else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        common.log_line(LOG, f"[two_households_pilot] {msg}")

    root = vs.dropbox_project_root()
    official = root / "data" / "official"
    state_db, raw_store = official / "state.sqlite", official / "raw_store"
    matched = load.load_matched_transactions()
    households = vs.select_vendor_only_households(matched, 2)
    if set(households) != set(evf.blind_vendor_households(2)):
        raise ValueError("household selection differs from estimators.event_frames.blind_vendor_households")
    snapshot = pd.read_parquet(root / vs.VENDOR_SNAPSHOT, columns=["politician_slug", "politician_name", "chamber"])
    filings = vs.load_ptr_filings(state_db).merge(th.read_primary_documents(state_db), on="filing_id", how="left")
    docs = {f"household_{i}": th.household_filings(h, snapshot, filings) for i, h in enumerate(households, start=1)}
    sample = pilot_sample(docs)
    log(f"pilot sample: {len(sample)} filings, {int(sample['page_count'].sum())} census pages")
    jobs = []
    for s in sample.itertuples(index=False):
        key = f"{s.household}_f{s.filing_index}"
        pngs = th.render_pdf(object_path(raw_store, s.sha256, s.stored_suffix), th.page_dir(s.sha256))
        jobs += [(str(p), s.household, i, key) for i, p in enumerate(pngs, start=1)]
    log(f"{len(jobs)} pages (rendered PNGs reused where present); {args.workers} workers")
    pages, rows = [], []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for k, res in enumerate(pool.map(process_page, jobs, chunksize=1), start=1):
            pages.append(res["page"])
            rows += res["rows"]
            if k % 10 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} pages, {time.time() - t1:.0f}s")
    pages = pd.DataFrame(pages).sort_values(["household", "filing_key", "page"]).reset_index(drop=True)
    pages["filing_index"] = pages["filing_key"].str.extract(r"_f(\d+)$", expand=False).astype(int)
    sheet_cols = ["household", "filing_index", "page", "page_image_path", "orientation", "read", "psm", "grid_stripped", "raw_line_text",
                  "date", "date_pattern", "notification_date", "owner", "ticker", "asset_text", "assigned_type", "assigned_bracket",
                  "n_marks", "assignment_note", "ink_type", "ink_bracket", "mean_word_confidence", "line_top", "line_bottom", "manual_check"]
    sheet = pd.DataFrame(rows)
    if sheet.empty:
        sheet = pd.DataFrame(columns=[c for c in sheet_cols if c != "filing_index"] + ["filing_key"])
    sheet["filing_index"] = sheet["filing_key"].str.extract(r"_f(\d+)$", expand=False).astype(int) if len(sheet) else pd.Series(dtype=int)
    sheet["manual_check"] = ""
    sheet = sheet.sort_values(["household", "filing_index", "page", "line_top"]).reset_index(drop=True) if len(sheet) else sheet
    sheet = sheet.reindex(columns=sheet_cols)
    names = th.load_crsp_names()
    trade = sheet[["ticker", "date"]].rename(columns={"date": "trade_date"})
    resolved = th.match_permno(trade, names) if len(sheet) else pd.Series(dtype="float64")
    sheet["permno_date_valid_exact_ticker"] = resolved
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET_CSV, index=False)
    log(f"validation sheet: {len(sheet)} candidate rows -> {SHEET_CSV.relative_to(common.REPO_ROOT)}")

    counts = count_table(pages, sheet, resolved)
    th.guard(counts, COUNTS_CSV)
    page_cols = ["household", "filing_index", "page", "orientation", "orientation_source", "psm", "grid_stripped", "words", "mean_word_confidence",
                 "date_tokens", *[c for c in pages.columns if c.startswith("dates_")], "headers_found", "header_list", "type_columns",
                 "bracket_columns", "candidate_rows"]
    th.guard(pages[page_cols], PAGES_CSV)
    for r in counts.itertuples(index=False):
        log(f"{r.household}: filings {r.filings}, pages {r.pages}, headers on {r.pages_with_headers}, candidate rows {r.candidate_rows}, "
            f"ticker {r.rows_with_ticker}, type {r.rows_with_type}, bracket {r.rows_with_bracket}, fully assigned {r.rows_fully_assigned}, "
            f"tickers common stock on date {r.tickers_common_stock_on_date}, mean line confidence {r.mean_line_confidence}")
    entries = [common.file_entry(state_db), common.file_entry(th.CRSP_NAMES, rows=len(names)), common.file_entry(root / vs.VENDOR_SNAPSHOT),
               common.file_entry(load.FRAMES_DIR / "matched_transactions.csv", rows=len(matched))]
    common.write_manifest("two_households_pilot", entries, extra={"dpi": th.DPI, "prescribed_psms": list(PRESCRIBED_PSMS), "extra_psms": list(EXTRA_PSMS),
                                                                   "grid_run_px": GRID_RUN_PX, "workers": args.workers, "seconds": round(time.time() - t0, 1)})
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
