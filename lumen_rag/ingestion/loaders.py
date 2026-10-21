"""Document loaders: extract plain text from PDF, DOCX, and HTML files.

All loaders return a ``{"id", "text", "metadata"}`` dict ready for
``ingest_documents``. Each loader raises ``ImportError`` with a clear install
hint if its optional dependency is missing so users only pay for what they use.

Supported:
  - ``.pdf``   → requires ``pypdf``  (``pip install lumen-rag[pdf]``)
  - ``.docx``  → requires ``python-docx`` (``pip install lumen-rag[docx]``)
  - ``.html`` / ``.htm`` → stdlib ``html.parser``, no extra deps
  - ``.txt`` / ``.md``  → plain read, no extra deps
  - ``.pptx`` → requires ``docling`` (``pip install lumen-rag[docling]``)

With ``LUMEN_DOCLING=1``, PDF/DOCX/PPTX/HTML go through Docling instead, which
understands layout, reading order, and tables and exports them as Markdown —
much better chunks for multi-column PDFs and anything tabular.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import settings

# Bounds what reaches chunking/embedding after extraction. Doesn't prevent a
# memory spike *during* pypdf/python-docx's own internal decompression of a
# crafted small file (a deeper fix needs guards inside those libraries'
# parsing, not here) — this only caps the downstream cost once parsing
# finishes, matching the same cap applied to the raw-JSON /ingest path.
_MAX_EXTRACTED_CHARS = 2_000_000


def _doc(path: Path, text: str, title: str | None = None) -> dict:
    metadata = {"source": str(path)}
    if title:
        metadata["title"] = title
    return {"id": path.stem, "text": text[:_MAX_EXTRACTED_CHARS], "metadata": metadata}


def load_txt(path: Path) -> dict:
    return _doc(path, path.read_text(encoding="utf-8", errors="replace"))


def load_pdf(path: Path) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Install it with: pip install 'lumen-rag[pdf]'"
        ) from e

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    title = reader.metadata.title if reader.metadata else None
    return _doc(path, "\n\n".join(pages), title=str(title).strip() if title else None)


def load_docx(path: Path) -> dict:
    try:
        import docx
    except ImportError as e:
        raise ImportError(
            "python-docx is required to load DOCX files. "
            "Install it with: pip install 'lumen-rag[docx]'"
        ) from e

    doc = docx.Document(str(path))
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return _doc(path, text)


def load_html(path: Path) -> dict:
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link"})

        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP_TAGS:
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS and self._skip > 0:
                self._skip -= 1

        def handle_data(self, data):
            if self._skip == 0:
                stripped = data.strip()
                if stripped:
                    self._parts.append(stripped)

        def text(self) -> str:
            return " ".join(self._parts)

    extractor = _TextExtractor()
    extractor.feed(path.read_text(encoding="utf-8", errors="replace"))
    return _doc(path, extractor.text())


_DOCLING_EXTS = {".pdf", ".docx", ".pptx", ".html", ".htm"}


@lru_cache(maxsize=1)
def _docling_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise ImportError(
            "docling is required for layout-aware parsing (and for .pptx). "
            "Install it with: pip install 'lumen-rag[docling]'"
        ) from e
    # Converter construction loads layout/table models; build it once per process.
    return DocumentConverter()


def load_with_docling(path: Path) -> dict:
    result = _docling_converter().convert(str(path))
    return _doc(path, result.document.export_to_markdown())


_LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".pptx": load_with_docling,
}


def load_file(path: str | Path) -> dict:
    """Dispatch to the right loader based on file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(_LOADERS)}"
        )
    if settings.use_docling and ext in _DOCLING_EXTS:
        loader = load_with_docling
    return loader(p)


def load_files(paths: list[str | Path]) -> list[dict]:
    """Load multiple files, returning a list of document dicts."""
    return [load_file(p) for p in paths]
