.PHONY: install test lint demo serve

install:
	pip install -e ".[dev,openai]"

test:
	pytest -q

lint:
	ruff check lumen_rag tests

demo:
	lumen ingest data/docs
	lumen ask "How many approvals does a billing change need?"
	lumen eval data/eval.jsonl --k 3

serve:
	lumen serve
