"""PR eval report: the comparison gate and the markdown it produces."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_report.py"
ROOT = SCRIPT.parent.parent


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("eval_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(fingerprint="abc", scale=1.0):
    def overall(mrr):
        return {"recall@k": mrr, "precision@k": 0.2, "mrr": mrr, "ndcg@k": mrr, "hit_rate": mrr}

    def cats(mrr):
        return {"direct": {"n": 5, "mrr": mrr}, "paraphrase": {"n": 5, "mrr": mrr / 2}}

    return {
        "k": 5,
        "n_cases": 10,
        "n_unanswerable": 2,
        "data_fingerprint": fingerprint,
        "modes": {m: {"overall": overall(base * scale), "by_category": cats(base * scale)}
                  for m, base in (("vector", 0.7), ("bm25", 0.9), ("hybrid", 0.8))},
    }


def test_identical_runs_pass(report):
    base = _result()
    markdown, regressions, comparable = report.compare(base, copy.deepcopy(base))
    assert regressions == []
    assert comparable is True
    assert markdown.startswith(report.MARKER)
    assert "No metric dropped" in markdown


def test_drop_beyond_tolerance_is_a_regression(report):
    base = _result()
    head = _result(scale=0.9)  # every metric falls by about 7-9 points
    markdown, regressions, comparable = report.compare(base, head, max_drop=0.01)
    assert comparable is True
    assert len(regressions) == 9  # 3 modes x 3 tracked metrics
    assert "dropped by more than" in markdown
    assert "🔻" in markdown


def test_drop_within_tolerance_passes(report):
    base = _result()
    head = copy.deepcopy(base)
    head["modes"]["bm25"]["overall"]["mrr"] -= 0.005
    _, regressions, _ = report.compare(base, head, max_drop=0.01)
    assert regressions == []


def test_improvement_is_not_a_regression(report):
    base = _result()
    head = _result(scale=1.05)
    markdown, regressions, _ = report.compare(base, head)
    assert regressions == []
    assert "▲" in markdown


def test_changed_eval_data_never_blocks(report):
    base = _result(fingerprint="old")
    head = _result(fingerprint="new", scale=0.5)  # looks terrible, but the data differs
    markdown, regressions, comparable = report.compare(base, head)
    assert comparable is False
    assert regressions == []
    assert "not comparable" in markdown


def test_missing_baseline_reports_head_only(report):
    markdown, regressions, comparable = report.compare(None, _result())
    assert comparable is False
    assert regressions == []
    assert "No baseline available" in markdown
    assert "| Base |" not in markdown


def test_cli_exit_codes(report, tmp_path):
    head_path = tmp_path / "head.json"
    base_path = tmp_path / "base.json"
    out = tmp_path / "report.md"
    base_path.write_text(json.dumps(_result()))

    head_path.write_text(json.dumps(_result()))
    assert report.main(["compare", "--base", str(base_path), "--head", str(head_path), "--out", str(out)]) == 0
    assert out.read_text().startswith(report.MARKER)

    head_path.write_text(json.dumps(_result(scale=0.8)))
    assert report.main(["compare", "--base", str(base_path), "--head", str(head_path)]) == 1

    # A baseline file that does not exist is treated as "no baseline", not an error.
    missing = tmp_path / "nope.json"
    assert report.main(["compare", "--base", str(missing), "--head", str(head_path)]) == 0


def test_run_measures_this_repo(report):
    result = report.run_benchmark(ROOT)
    assert set(result["modes"]) == {"vector", "bm25", "hybrid"}
    assert result["n_cases"] >= 150
    assert result["n_unanswerable"] >= 20
    assert result["modes"]["bm25"]["overall"]["mrr"] > result["modes"]["vector"]["overall"]["mrr"]
    assert set(result["modes"]["hybrid"]["by_category"]) >= {"direct", "paraphrase", "multihop", "disambiguation"}
    # Measured against itself there must be no regression.
    _, regressions, comparable = report.compare(result, result)
    assert comparable and regressions == []


def test_run_refuses_a_root_that_is_not_the_imported_package(report, tmp_path):
    (tmp_path / "data" / "eval_v2" / "docs").mkdir(parents=True)
    (tmp_path / "data" / "eval_v2" / "eval.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="not under --root"):
        report.run_benchmark(tmp_path)
