"""Rule-based query decomposition for compound (multi-hop) questions.

"How much can I claim per day on a trip abroad, and how soon must I submit the
claim?" needs two documents, but a single embedding of the whole question tends
to land near one of them. Splitting it at the conjunction before the second
question word gives one sub-query per hop; the retriever searches each and fuses
that into the whole question's results: each sub-question adds its best document
not already retrieved (see ``Retriever._decomposed``).

Deliberately offline and deterministic: no model call, so it costs microseconds,
runs in CI and never invents a sub-question. It only finds *explicit* compound
questions; an implicit hop ("How much bonus do I earn if checkout is down?" needs
the severity table and the stipend) is left as one query.
"""
from __future__ import annotations

import re

_WH = r"(?:how|what|when|who|whom|which|where|why|does|do|did|is|are|was|were|can|could|must|should|will|would|has|have)"
# Split after "?", before a question word that follows a comma or a conjunction, and at ";".
_SPLIT = re.compile(rf"\?\s+|,\s*(?:and|but|or)?\s*(?={_WH}\b)|\s+(?:and|but|or)\s+(?={_WH}\b)|;\s*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
MIN_WORDS = 3  # shorter fragments ("and how?") carry no search signal


def split_question(question: str) -> list[str]:
    """Sub-questions of a compound question, or ``[question]`` when it is not compound."""
    parts = [p.strip(" ,?") for p in _SPLIT.split(question)]
    parts = [p for p in parts if len(_WORD.findall(p.lower())) >= MIN_WORDS]
    return parts if len(parts) > 1 else [question]
