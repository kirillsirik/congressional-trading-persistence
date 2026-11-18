"""Filing-format identification: the first stage of the parsing cascade.

Classifies a fetched document before any parser runs, with recorded evidence.
An HTML error page saved with a .pdf suffix is a fetch failure, not a filing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class FormatResult:
    doc_format: str  # structured_html | text_pdf | scanned_pdf | mixed_pdf |
    #                  image | malformed | unavailable
    confidence: float
    evidence: str
    page_count: int | None = None


def classify_document(content: bytes, content_type: str = "") -> FormatResult:
    if not content:
        return FormatResult("unavailable", 1.0, "empty body")
    head = content[:1024].lstrip()
    if head.startswith(b"%PDF"):
        return _classify_pdf(content)
    lowered = head[:256].lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html") or b"<html" in lowered:
        return FormatResult("structured_html", 0.95, "html signature")
    if head[:3] == b"GIF" or head[:2] == b"\xff\xd8" or head[:8] == b"\x89PNG\r\n\x1a\n":
        return FormatResult("image", 0.95, "image signature")
    if "pdf" in content_type.lower():
        return FormatResult("malformed", 0.9, "content-type pdf but no %PDF signature")
    return FormatResult("malformed", 0.5, f"unrecognized signature {head[:8]!r}")


def _classify_pdf(content: bytes) -> FormatResult:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            n_pages = len(pdf.pages)
            sample = pdf.pages[: min(3, n_pages)]
            chars = sum(len(p.chars) for p in sample)
            images = sum(len(p.images) for p in sample)
        if chars > 100:
            fmt = "mixed_pdf" if images > 0 and chars < 500 else "text_pdf"
            return FormatResult(
                fmt, 0.9, f"{chars} chars, {images} images in first {len(sample)} pages", n_pages
            )
        if images > 0:
            return FormatResult(
                "scanned_pdf", 0.9, f"{chars} chars, {images} images (image-only)", n_pages
            )
        return FormatResult("scanned_pdf", 0.6, f"no text layer ({chars} chars)", n_pages)
    except Exception as exc:  # malformed PDFs must not crash classification
        return FormatResult("malformed", 0.8, f"pdf open failed: {type(exc).__name__}: {exc}")
