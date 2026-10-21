"""Extractive reader: does a retrieved passage actually answer the question?

Retrieval confidence measures how *similar* the question is to the corpus. That
cannot catch a near miss, where the right document is retrieved but does not
contain the fact asked for ("Which brand of security key do we issue?" against
an MFA policy that never names one). A reader trained on SQuAD 2.0, which
includes unanswerable questions, reads the question and passage together and
scores the best answer span against a "no answer" option:

    answerability = best span score - no-answer score   (start + end logits)

Positive means the passage contains an answer, negative means it does not. The
engine refuses below ``LUMEN_READER_THRESHOLD``. With ``LUMEN_READER_ANSWERS=1``
the offline answer is the sentence holding the extracted span instead of the
top passages.

Runs an ONNX export on onnxruntime with a ``tokenizers`` tokenizer, so no torch.
Default model: ``onnx-community/tinyroberta-squad2-ONNX`` (deepset's
tinyroberta-squad2, int8, ~80 MB, CC-BY-4.0). Enable with ``LUMEN_READER``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import settings
from .store import ScoredChunk

MAX_LENGTH = 384  # tokens per question + passage window
STRIDE = 128  # overlap between windows of a long passage
MAX_ANSWER_TOKENS = 30


@dataclass
class Reading:
    # Best span score minus the no-answer score, over every passage read.
    answerability: float
    # The extracted answer span and the sentence around it (None when there are no passages).
    span: str | None = None
    sentence: str | None = None
    # Index into the chunks passed to ``read`` of the passage holding the span.
    chunk_index: int | None = None


class ExtractiveReader:
    def __init__(self, model_id: str, onnx_file: str = "onnx/model_quantized.onnx") -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "onnxruntime, tokenizers and huggingface_hub are required for LUMEN_READER. "
                "Install them with: pip install 'lumen-rag[reader]'"
            ) from e

        self.model_id = model_id
        self._tok = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self._tok.no_padding()
        self._tok.enable_truncation(max_length=MAX_LENGTH, stride=STRIDE, strategy="only_second")
        self._pad_id = self._tok.token_to_id("<pad>") or 0
        self._session = ort.InferenceSession(
            hf_hub_download(model_id, onnx_file), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def read(self, question: str, chunks: list[ScoredChunk]) -> Reading:
        if not chunks:
            return Reading(answerability=float("-inf"))

        # Every passage becomes one or more overlapping windows; read them in one batch.
        windows, owners = [], []
        for idx, sc in enumerate(chunks):
            enc = self._tok.encode(question, sc.chunk.text)
            for w in [enc, *enc.overflowing]:
                windows.append(w)
                owners.append(idx)

        width = max(len(w.ids) for w in windows)
        ids = np.full((len(windows), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(windows), width), dtype=np.int64)
        for row, w in enumerate(windows):
            ids[row, : len(w.ids)] = w.ids
            mask[row, : len(w.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        starts, ends = self._session.run(None, feed)

        best = Reading(answerability=float("-inf"))
        for row, w in enumerate(windows):
            score, i, j = _best_span(starts[row], ends[row], w.sequence_ids)
            if i is None:
                continue
            margin = float(score - (starts[row][0] + ends[row][0]))  # index 0 is <s>/[CLS]: no answer
            if margin > best.answerability:
                text = chunks[owners[row]].chunk.text
                lo, hi = w.offsets[i][0], w.offsets[j][1]
                best = Reading(margin, text[lo:hi].strip(), _sentence(text, lo, hi), owners[row])
        return best


def _best_span(start: np.ndarray, end: np.ndarray, sequence_ids: list) -> tuple[float, int | None, int | None]:
    ctx = np.array([s == 1 for s in sequence_ids])
    n = len(sequence_ids)
    start, end = start[:n], end[:n]
    if not ctx.any():
        return float("-inf"), None, None
    scores = start[:, None] + end[None, :]
    i_idx, j_idx = np.indices(scores.shape)
    valid = ctx[:, None] & ctx[None, :] & (j_idx >= i_idx) & (j_idx - i_idx < MAX_ANSWER_TOKENS)
    scores = np.where(valid, scores, -np.inf)
    i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return float(scores[i, j]), int(i), int(j)


_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")


def _sentence(text: str, lo: int, hi: int) -> str:
    """The sentence (or line) of ``text`` containing the span ``text[lo:hi]``."""
    start = 0
    for m in _SENTENCE_END.finditer(text, 0, lo):
        start = m.end()
    m = _SENTENCE_END.search(text, hi)
    stop = m.end() if m else len(text)
    return text[start:stop].strip().lstrip("#").strip()


@lru_cache(maxsize=2)
def _load(model_id: str, onnx_file: str) -> ExtractiveReader:
    return ExtractiveReader(model_id, onnx_file)


def get_reader() -> ExtractiveReader | None:
    """The configured reader, or None when ``LUMEN_READER`` is unset."""
    if not settings.reader_model:
        return None
    return _load(settings.reader_model, settings.reader_file)
