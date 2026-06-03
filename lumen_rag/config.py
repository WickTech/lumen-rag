"""Central configuration, loaded from environment with sensible offline defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    index_dir: str = os.getenv("LUMEN_INDEX_DIR", ".lumen_index")

    @property
    def offline(self) -> bool:
        """True when no API key is configured; engine falls back to local embeddings."""
        return not self.openai_api_key


settings = Settings()
