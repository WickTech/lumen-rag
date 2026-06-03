from lumen_rag.eval import metrics


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5
    assert metrics.recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert metrics.recall_at_k([], ["a"], k=3) == 0.0


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "b", "c"], ["a"], k=3) == 1 / 3
    assert metrics.precision_at_k(["a"], ["a"], k=1) == 1.0


def test_hit_rate():
    assert metrics.hit_rate(["x", "a"], ["a"], k=2) == 1.0
    assert metrics.hit_rate(["x", "y"], ["a"], k=2) == 0.0


def test_mrr():
    assert metrics.mrr(["x", "a", "b"], ["a"]) == 0.5
    assert metrics.mrr(["a"], ["a"]) == 1.0
    assert metrics.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_vs_worse():
    perfect = metrics.ndcg_at_k(["a", "b"], ["a", "b"], k=2)
    worse = metrics.ndcg_at_k(["x", "a"], ["a"], k=2)
    assert perfect == 1.0
    assert 0 < worse < 1.0
