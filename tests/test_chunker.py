from lumen_rag.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("One sentence. Two sentence.", chunk_size=120)
    assert len(chunks) == 1


def test_respects_chunk_size():
    text = ". ".join(f"word{i} sentence here" for i in range(100)) + "."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack since we pack whole sentences
        assert len(c.split()) <= 30 + 5


def test_overlap_carries_context():
    text = ". ".join(f"alpha{i}" for i in range(60)) + "."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    # consecutive chunks should share trailing/leading tokens
    first_tail = chunks[0].split()[-5:]
    second_head = chunks[1].split()[:5]
    assert first_tail == second_head


def test_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)
