"""OCR stage for scanned and paper filings.

Design constraint (ADR-004 + the project's no-fabrication rule): OCR output is
**evidence**, not a licence to invent transactions. Historical House and
Senate paper PTRs are pre-printed forms whose transaction entries are
handwritten or checkbox-marked; tesseract reliably reads the printed form
furniture but not the filer's entries. Auto-promoting that into the
transaction tables would fabricate research data.

So this stage:

1. renders every page at 300 DPI and preserves the rendering;
2. runs OCR with word-level confidences and bounding boxes, preserving both
   the text and the full word table;
3. classifies the filing as ``machine_printed`` (transaction-shaped rows
   recoverable) or ``hand_filled`` (entries are handwriting/checkboxes);
4. extracts rows ONLY from machine-printed pages, at a confidence that keeps
   them in the review queue until a human accepts them;
5. queues everything else for manual entry with the artifacts a reviewer
   needs (page image path, OCR text, source URL).

Every filing ends in an explicit terminal state either way, so coverage
accounting stays honest.
"""

from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass, field

OCR_PARSER_NAME = "house_senate_ocr"
OCR_PARSER_VERSION = "1.0"

RENDER_DPI = 300
MIN_WORD_CONFIDENCE = 70  # tesseract confidence for a word to count as legible

_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")
_AMOUNT = re.compile(r"\$\s?[\d,]{3,}")
# The printed bracket labels that appear on every blank form: their presence
# says nothing about what the filer entered.
_FORM_BOILERPLATE = re.compile(
    r"\$1,?001\s*-\s*\$15,?000|\$15,?001\s*-\s*\$50,?000|"
    r"periodic transaction report|periodic disclosure|hand delivered|"
    r"office of public records|legislative resource center",
    re.I,
)


def ocr_available() -> bool:
    """True when a local tesseract binary and pytesseract are importable."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


def ocr_engine_info() -> dict:
    if not ocr_available():
        return {"available": False}
    import subprocess

    version = subprocess.run(
        ["tesseract", "--version"], capture_output=True, text=True, check=False
    ).stdout.split("\n")[0]
    try:
        import pytesseract

        binding = pytesseract.get_tesseract_version()
    except Exception:
        binding = None
    return {
        "available": True,
        "engine": version,
        "binding_version": str(binding),
        "dpi": RENDER_DPI,
        "min_word_confidence": MIN_WORD_CONFIDENCE,
        "parser": f"{OCR_PARSER_NAME}/{OCR_PARSER_VERSION}",
    }


@dataclass
class OcrWord:
    text: str
    conf: int
    left: int
    top: int
    width: int
    height: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass
class OcrPage:
    page_no: int
    words: list[OcrWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines: dict[int, list[OcrWord]] = {}
        for w in self.words:
            lines.setdefault(round(w.top / 12), []).append(w)
        out = []
        for key in sorted(lines):
            row = sorted(lines[key], key=lambda w: w.left)
            out.append(" ".join(w.text for w in row))
        return "\n".join(out)

    def lines(self) -> list[list[OcrWord]]:
        grouped: dict[int, list[OcrWord]] = {}
        for w in self.words:
            grouped.setdefault(round(w.top / 12), []).append(w)
        return [sorted(grouped[k], key=lambda w: w.left) for k in sorted(grouped)]


@dataclass
class OcrResult:
    pages: list[OcrPage] = field(default_factory=list)
    classification: str = "unknown"  # machine_printed | hand_filled | unreadable
    evidence: str = ""
    rows: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def word_table(self) -> str:
        """TSV of every word with confidence and bounding box (audit trail)."""
        out = ["page\ttext\tconf\tleft\ttop\twidth\theight"]
        for p in self.pages:
            for w in p.words:
                out.append(
                    f"{p.page_no}\t{w.text}\t{w.conf}\t{w.left}\t{w.top}\t{w.width}\t{w.height}"
                )
        return "\n".join(out)


def _page_images(content: bytes) -> list:
    """Render a document to PIL images (PDF via pdfplumber, else direct)."""
    from PIL import Image

    if content[:4] == b"%PDF":
        import pdfplumber

        images = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                images.append(page.to_image(resolution=RENDER_DPI).original)
        return images
    return [Image.open(io.BytesIO(content))]


def ocr_document(content: bytes, max_pages: int = 40) -> OcrResult:
    """OCR a scanned document and classify how much of it is machine-readable."""
    import pytesseract

    result = OcrResult()
    try:
        images = _page_images(content)[:max_pages]
    except Exception as exc:
        result.classification = "unreadable"
        result.evidence = f"render failed: {type(exc).__name__}: {exc}"
        return result

    for i, img in enumerate(images, start=1):
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        page = OcrPage(page_no=i)
        for j, text in enumerate(data["text"]):
            if not text.strip():
                continue
            try:
                conf = int(float(data["conf"][j]))
            except (TypeError, ValueError):
                conf = -1
            page.words.append(
                OcrWord(text.strip(), conf, data["left"][j], data["top"][j],
                        data["width"][j], data["height"][j])
            )
        result.pages.append(page)

    result.rows = _extract_rows(result)
    result.classification, result.evidence = _classify(result)
    if result.classification != "machine_printed":
        # rows recovered from a hand-filled form are not trustworthy evidence
        result.rows = []
    return result


def _extract_rows(result: OcrResult) -> list[dict]:
    """Transaction-shaped lines: a date, an amount, and descriptive text.

    Boilerplate bracket labels printed on the blank form are excluded, so a
    match means the line carries filer-entered content.
    """
    rows: list[dict] = []
    for page in result.pages:
        for line_words in page.lines():
            line = " ".join(w.text for w in line_words)
            if _FORM_BOILERPLATE.search(line):
                continue
            dates = _DATE.findall(line)
            amounts = _AMOUNT.findall(line)
            if not dates or not amounts:
                continue
            legible = [w for w in line_words if w.conf >= MIN_WORD_CONFIDENCE]
            if len(legible) < max(3, len(line_words) * 0.6):
                continue
            asset = " ".join(
                w.text for w in line_words
                if not _DATE.match(w.text) and not w.text.startswith("$")
            ).strip()
            if len(asset) < 4:
                continue
            confs = [w.conf for w in line_words if w.conf >= 0]
            # the scanned forms print the bracket as two figures; keep both so
            # the range is not truncated to its lower bound
            figures = [a.replace(" ", "") for a in amounts]
            amount_raw = (
                f"{figures[0]} - {figures[1]}" if len(figures) >= 2 else figures[0]
            )
            rows.append({
                "asset_description_raw": asset[:200],
                "transaction_date_raw": dates[0],
                "notification_date_raw": dates[1] if len(dates) > 1 else None,
                "amount_raw": amount_raw,
                "owner_raw": None,
                "transaction_type_raw": None,
                "page": page.page_no,
                "extraction": "ocr",
                # deliberately below the review threshold: OCR rows must be
                # human-accepted before they count as verified
                "confidence": round(min(0.65, (sum(confs) / len(confs)) / 100), 3),
                "raw_text": line[:500],
                "ocr_bbox": str([w.bbox for w in line_words[:12]]),
            })
    return rows


def _classify(result: OcrResult) -> tuple[str, str]:
    all_words = [w for p in result.pages for w in p.words]
    if not all_words:
        return "unreadable", "no words recognized"
    legible = [w for w in all_words if w.conf >= MIN_WORD_CONFIDENCE]
    ratio = len(legible) / len(all_words)
    if result.rows:
        return (
            "machine_printed",
            f"{len(result.rows)} transaction-shaped lines; "
            f"{ratio:.0%} of {len(all_words)} words legible",
        )
    return (
        "hand_filled",
        f"no machine-readable transaction lines; {ratio:.0%} of "
        f"{len(all_words)} words legible (printed form furniture only)",
    )
