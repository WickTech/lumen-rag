"""Contextual chunk headers: tell each chunk which document and section it is from.

A chunk cut from the middle of a long document often never names its subject:
"Annual billing gives two months free." could belong to any plan. Anthropic's
Contextual Retrieval prepends a short context to every chunk before embedding it
and before building the BM25 index, and reports 35% fewer failed retrievals from
that alone (49% with contextual BM25 too).

Anthropic generates the context with an LLM. This module builds it offline and
deterministically from the document's own structure: its title, and the path of
headings the chunk sits under, e.g.

    Document: Artificial Intelligence Risk Management Framework (AI RMF 1.0).
    Section: 3. AI Risks and Trustworthiness > 3.4 Accountable and Transparent.

The header is used for search only; the chunk text that answers and citations
show stays unchanged (it is stored in ``Chunk.metadata["context"]``).
"""
from __future__ import annotations

import re
from pathlib import Path

_MARKDOWN = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
# "3.4 Accountable and Transparent", "5. AI RMF Core": numbered headings in PDFs.
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,})$")
_APPENDIX = re.compile(r"^\s*(Appendix [A-Z])[:.]?\s+([A-Z][^\n]{2,})$")
_BARE_APPENDIX = re.compile(r"^\s*(Appendix [A-Z])[:.]?\s*$")  # title on the next line
MAX_HEADING_WORDS = 10  # longer numbered lines are list items, not headings
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    # Same whitespace collapsing as the chunker, so chunk offsets line up.
    return _WS.sub(" ", text).strip()


def _heading(line: str) -> tuple[int, str] | None:
    """(level, heading text) when ``line`` is a markdown or numbered heading."""
    m = _MARKDOWN.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _APPENDIX.match(line)
    if m and len(line.split()) <= MAX_HEADING_WORDS and not re.search(r"\s\d+$", line.rstrip()):
        return 1, f"{m.group(1)}: {m.group(2).strip()}"
    m = _NUMBERED.match(line)
    # Not a heading: long lines and lines ending like a sentence are list items, and a
    # trailing number is a table-of-contents page reference.
    if (
        m
        and len(line.split()) <= MAX_HEADING_WORDS
        and not line.rstrip().endswith(("-", ",", ";", ".", ":"))
        and not re.search(r"\s\d+$", line.rstrip())
        and ". " not in m.group(2)
    ):
        number = m.group(1)
        return number.count(".") + 1, f"{number}{'.' if '.' not in number else ''} {m.group(2).strip()}"
    return None


def document_title(doc: dict) -> str:
    """The document's title: metadata, else its first markdown heading, else its id or file name."""
    title = (doc.get("metadata") or {}).get("title")
    if title:
        return str(title).strip()
    for line in doc.get("text", "").splitlines():
        m = _MARKDOWN.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
        if line.strip():
            break  # only a heading on the first non-empty line counts as the title
    name = str(doc.get("id") or Path(str((doc.get("metadata") or {}).get("source", ""))).stem)
    return name.replace("-", " ").replace("_", " ").strip()


def _sections(text: str) -> list[tuple[int, str]]:
    """(offset in normalized text, heading path) for every heading, in order.

    A heading whose text occurs more than once only counts at its last occurrence:
    the earlier ones are a table of contents, which would otherwise label the
    front matter with the last entry in it.
    """
    lines = text.splitlines()
    # Offset of each line in the normalized text (non-empty lines joined by one space),
    # computed from its position: searching for the heading text would hit its
    # table-of-contents twin first.
    offsets, total = [], 0
    for line in lines:
        piece = _normalize(line)
        start = total + 1 if (piece and total) else total
        offsets.append(start)
        if piece:
            total = start + len(piece)
    parsed_lines = []
    for i, line in enumerate(lines):
        m = _BARE_APPENDIX.match(line)
        if m:
            following = next((nxt.strip() for nxt in lines[i + 1 : i + 4] if nxt.strip()), "")
            parsed = (1, f"{m.group(1)}: {following}") if 0 < len(following.split()) <= 12 else None
            parsed_lines.append((line, parsed))
        else:
            parsed_lines.append((line, _heading(line)))
    last_seen = {h[1]: i for i, (_, h) in enumerate(parsed_lines) if h is not None}
    stack: list[tuple[int, str]] = []
    found: list[tuple[int, str]] = []
    for i, (_line, parsed) in enumerate(parsed_lines):
        if parsed is None or last_seen[parsed[1]] != i:
            continue
        level, heading = parsed
        pos = offsets[i]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        found.append((pos, " > ".join(h for _, h in stack)))
    return found


def chunk_headers(doc: dict, chunks: list[str]) -> list[str]:
    """One header per chunk of ``doc`` (chunks as produced by ``chunk_text``, in order)."""
    title = document_title(doc)
    text = doc.get("text", "")
    sections = _sections(text)
    normalized = _normalize(text)
    headers = []
    cursor = 0
    for chunk in chunks:
        pos = normalized.find(chunk[:80], cursor)
        if pos < 0:
            pos = cursor
        cursor = pos + 1
        # A chunk can straddle a heading; label it by the section its middle is in.
        middle = pos + len(chunk) // 2
        path = ""
        for offset, heading_path in sections:
            if offset > middle:
                break
            path = heading_path
        if path.startswith(f"{title} > "):
            path = path[len(title) + 3 :]  # the title heading is already named
        header = f"Document: {title}."
        if path and path != title:
            header += f" Section: {path}."
        headers.append(header)
    return headers


def contextual_text(text: str, metadata: dict) -> str:
    """What gets embedded and BM25-indexed: the chunk's header, then the chunk."""
    context = metadata.get("context")
    return f"{context}\n{text}" if context else text
