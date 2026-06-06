"""Tests for document loaders."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lumen_rag.ingestion.loaders import load_file, load_files, load_html, load_txt


# ---------------------------------------------------------------------------
# Helper: write a temp file
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Plain text / markdown
# ---------------------------------------------------------------------------

class TestTxtLoader:
    def test_loads_text_file(self, tmp_path):
        p = _write(tmp_path, "notes.txt", "Hello world.")
        doc = load_txt(p)
        assert doc["text"] == "Hello world."
        assert doc["id"] == "notes"
        assert "notes.txt" in doc["metadata"]["source"]

    def test_loads_markdown_via_dispatch(self, tmp_path):
        p = _write(tmp_path, "readme.md", "# Title\nSome content.")
        doc = load_file(p)
        assert "Title" in doc["text"]


# ---------------------------------------------------------------------------
# HTML loader
# ---------------------------------------------------------------------------

class TestHtmlLoader:
    def test_strips_tags(self, tmp_path):
        html = "<html><body><p>Hello <b>world</b>.</p></body></html>"
        p = _write(tmp_path, "page.html", html)
        doc = load_html(p)
        assert "Hello" in doc["text"]
        assert "<b>" not in doc["text"]

    def test_strips_script_and_style(self, tmp_path):
        html = textwrap.dedent("""\
            <html><head>
              <style>body { color: red; }</style>
              <script>alert('x')</script>
            </head><body><p>Content only.</p></body></html>
        """)
        p = _write(tmp_path, "page.html", html)
        doc = load_html(p)
        assert "Content only." in doc["text"]
        assert "color" not in doc["text"]
        assert "alert" not in doc["text"]

    def test_htm_extension_dispatches(self, tmp_path):
        p = _write(tmp_path, "index.htm", "<p>Hi</p>")
        doc = load_file(p)
        assert "Hi" in doc["text"]

    def test_id_is_stem(self, tmp_path):
        p = _write(tmp_path, "about-us.html", "<p>About</p>")
        doc = load_file(p)
        assert doc["id"] == "about-us"


# ---------------------------------------------------------------------------
# load_files bulk helper
# ---------------------------------------------------------------------------

class TestLoadFiles:
    def test_loads_multiple_files(self, tmp_path):
        _write(tmp_path, "a.txt", "Alpha.")
        _write(tmp_path, "b.txt", "Beta.")
        docs = load_files([tmp_path / "a.txt", tmp_path / "b.txt"])
        texts = [d["text"] for d in docs]
        assert "Alpha." in texts
        assert "Beta." in texts


# ---------------------------------------------------------------------------
# Unsupported extension
# ---------------------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    p = _write(tmp_path, "data.csv", "col1,col2\n1,2")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(p)


# ---------------------------------------------------------------------------
# PDF/DOCX: test ImportError path when the optional dep is absent
# ---------------------------------------------------------------------------

class TestPdfImportError:
    def test_raises_import_error_when_pypdf_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdf", None)  # type: ignore[arg-type]
        p = _write(tmp_path, "file.pdf", "%PDF-1.4 fake")
        with pytest.raises(ImportError, match="pypdf"):
            from lumen_rag.ingestion.loaders import load_pdf
            load_pdf(p)


class TestDocxImportError:
    def test_raises_import_error_when_docx_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "docx", None)  # type: ignore[arg-type]
        p = _write(tmp_path, "file.docx", b"PK fake docx".decode())
        with pytest.raises(ImportError, match="python-docx"):
            from lumen_rag.ingestion.loaders import load_docx
            load_docx(p)
