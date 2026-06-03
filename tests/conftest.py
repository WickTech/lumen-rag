"""Make the test suite hermetic: force the offline (no-API) code paths so tests
are deterministic and never hit the network, regardless of the developer's
ambient environment. The Settings object reads env at import time, so we patch
the resolved settings as well.
"""
import pytest

from lumen_rag import config


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config.settings, "openai_api_key", "", raising=False)
