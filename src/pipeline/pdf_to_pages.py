"""Render a PDF into per-page PIL images using pypdfium2."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium
from PIL import Image


def render_pdf(pdf_path: Path, scale: float = 2.0, page_indices: list[int] | None = None) -> Iterator[tuple[int, Image.Image]]:
    doc = pdfium.PdfDocument(str(pdf_path))
    indices = page_indices if page_indices is not None else range(len(doc))
    for i in indices:
        if i < 0 or i >= len(doc):
            continue
        page = doc[i]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        yield i, pil
    doc.close()
