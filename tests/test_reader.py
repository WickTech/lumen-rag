"""Extractive answerability reader: span search, sentence extraction, engine gating."""
import os

import numpy as np
import pytest

from lumen_rag import config, llm
from lumen_rag import engine as engine_mod
from lumen_rag.engine import RagEngine
from lumen_rag.reader import Reading, _best_span, _sentence

DOCS = [
    {"id": "mfa", "text": "Each employee registers two keys. Text message codes are not accepted."},
    {"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
]


def test_best_span_stays_in_context_and_ordered():
    # sequence ids: <s> q q </s> </s> p p p </s>
    seq = [None, 0, 0, None, None, 1, 1, 1, None]
    start = np.array([5, 9, 9, 0, 0, 1, 4, 0, 0], dtype=np.float32)
    end = np.array([5, 9, 9, 0, 0, 0, 1, 6, 0], dtype=np.float32)
    score, i, j = _best_span(start, end, seq)
    assert (i, j) == (6, 7)  # question tokens 1-2 score higher but are not context
    assert score == 10


def test_best_span_without_context():
    assert _best_span(np.zeros(3), np.zeros(3), [None, 0, None])[1] is None


def test_sentence_around_span():
    text = "# Policy\nFirst rule here. The H200 lifts 35 kilograms to 2.4 metres. Last one."
    lo = text.index("35")
    assert _sentence(text, lo, lo + 2) == "The H200 lifts 35 kilograms to 2.4 metres."


class FakeReader:
    model_id = "fake"

    def __init__(self, answerability):
        self.answerability = answerability

    def read(self, question, chunks):
        return Reading(self.answerability, "two", "Each employee registers two keys.", 0)


@pytest.fixture
def eng():
    e = RagEngine()
    e.add_documents(DOCS, chunk_size=50, overlap=0)
    return e


def test_reader_refuses_below_threshold(eng, monkeypatch):
    monkeypatch.setattr(engine_mod, "get_reader", lambda: FakeReader(-5.0))
    monkeypatch.setattr(llm, "answer", lambda *a, **k: pytest.fail("model must not be called"))
    result = eng.query("which brand of key do we issue")
    assert result.abstained and result.answerability == -5.0
    assert result.confidence is None  # similarity gate is off


def test_reader_gates_but_keeps_passage_answers_by_default(eng, monkeypatch):
    monkeypatch.setattr(engine_mod, "get_reader", lambda: FakeReader(4.0))
    result = eng.query("how many keys does each employee register", mode="bm25")
    assert not result.abstained and result.answerability == 4.0
    assert "Text message codes are not accepted." in result.text


def test_reader_answers_with_the_span_sentence(eng, monkeypatch):
    monkeypatch.setattr(config.settings, "reader_answers", True)
    monkeypatch.setattr(engine_mod, "get_reader", lambda: FakeReader(4.0))
    result = eng.query("how many keys does each employee register", mode="bm25")
    assert not result.abstained and result.answerability == 4.0
    assert result.text == "(offline extractive answer) Each employee registers two keys. [1]"


def test_similarity_gate_runs_before_the_reader(eng, monkeypatch):
    monkeypatch.setattr(config.settings, "abstain_threshold", 0.99)
    monkeypatch.setattr(engine_mod, "get_reader", lambda: pytest.fail("reader must not run"))
    assert eng.query("capital of peru").abstained


def test_no_reader_keeps_old_answers(eng):
    result = eng.query("annual paid leave")
    assert result.answerability is None
    assert result.text.startswith("(offline extractive answer) ")


@pytest.mark.skipif(not os.getenv("LUMEN_TEST_READER"), reason="downloads a model; set LUMEN_TEST_READER=1")
def test_real_reader_separates_near_miss():
    pytest.importorskip("onnxruntime")
    from lumen_rag.reader import ExtractiveReader
    from lumen_rag.store import Chunk, ScoredChunk

    reader = ExtractiveReader("onnx-community/tinyroberta-squad2-ONNX")
    text = (
        "All employees must use multi-factor authentication. Hardware security keys are mandatory "
        "for engineers with production access. Each employee registers two keys."
    )
    chunks = [ScoredChunk(Chunk(id="c", doc_id="mfa", text=text, metadata={}), 1.0)]
    yes = reader.read("How many keys does each employee register?", chunks)
    no = reader.read("Which brand of hardware security key does the company issue?", chunks)
    assert yes.span == "two" and yes.answerability > 0 > no.answerability
