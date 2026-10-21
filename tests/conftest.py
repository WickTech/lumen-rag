"""Make the test suite hermetic: force the offline (no-API) code paths so tests
are deterministic and never hit the network, regardless of the developer's
ambient environment. The Settings object reads env at import time, so we patch
the resolved settings as well.
"""
import os

import pytest

from lumen_rag import config

# Also at import time, not only in the fixture: some test modules import
# lumen_rag.api.app, which builds an engine at import, during collection and
# before any fixture runs. A developer's .env key would otherwise pick the
# OpenAI embedder there.
os.environ.pop("OPENAI_API_KEY", None)
config.settings.openai_api_key = ""
config.settings.embedder = "auto"
config.settings.abstain_threshold = None
config.settings.reader_model = ""
config.settings.reader_answers = False
config.settings.contextual = False
config.settings.decompose = False


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config.settings, "openai_api_key", "", raising=False)
    # Optional heavy backends stay off unless a test opts in explicitly.
    monkeypatch.setattr(config.settings, "reranker_model", "", raising=False)
    monkeypatch.setattr(config.settings, "use_docling", False, raising=False)
    monkeypatch.setattr(config.settings, "embedder", "auto", raising=False)
    monkeypatch.setattr(config.settings, "abstain_threshold", None, raising=False)
    monkeypatch.setattr(config.settings, "reader_model", "", raising=False)
    monkeypatch.setattr(config.settings, "reader_answers", False, raising=False)
    monkeypatch.setattr(config.settings, "contextual", False, raising=False)
    monkeypatch.setattr(config.settings, "decompose", False, raising=False)
