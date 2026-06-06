"""Document loaders: extract plain text from PDF, DOCX, and HTML files.

All loaders return a ``{"id", "text", "metadata"}`` dict ready for
``ingest_documents``. Each loader raises ``ImportError`` with a clear install
hint if its optional dependency is missing so users only pay for what they use.

Supported:
  - ``.pdf``   → requires ``pypdf``  (``pip install lumen-rag[pdf]``)
  - ``.docx``  → requires ``python-docx`` (``pip install lumen-rag[docx]``)
  - ``.html`` / ``.htm`` → stdlib ``html.parser``, no extra deps
  - ``.txt`` / ``.md``  → plain read, no extra deps
"""
from __future__ import annotations

from pathlib import Path


def _doc(path: Path, text: str) -> dict:
    return {"id": path.stem, "text": text, "metadata": {"source": str(path)}}


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
    return _doc(path, "\n\n".join(pages))


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
        SKIP_TAGS = {"script", "style", "head", "meta", "link"}

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


_LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
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
    return loader(p)


def load_files(paths: list[str | Path]) -> list[dict]:
    """Load multiple files, returning a list of document dicts."""
    return [load_file(p) for p in paths]
