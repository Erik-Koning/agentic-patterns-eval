"""Run configuration: paths, model roles, budgets. Values come from environment variables so
Inspect tasks, build scripts and tests share one source of truth.

Model IDs are deliberately unset by default: they are chosen at readiness E3 and
recorded in PROVENANCE.md. `APE_EMBEDDINGS=fake` swaps in the offline hashed
bag-of-words embedder used by dry runs and tests.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Inspect loads .env for evals; standalone build CLIs (APG authoring, LightRAG extraction) need it too.
# Never overrides variables already set in the environment.
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Config:
    worlds_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_WORLDS", ROOT / "worlds")))
    indices_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_INDICES", ROOT / "indices")))
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_CACHE", ROOT / "cache")))
    embeddings_backend: str = field(default_factory=lambda: os.environ.get("APE_EMBEDDINGS", "openai"))
    embedding_model: str = field(default_factory=lambda: os.environ.get("APE_EMBEDDING_MODEL", "UNSET-see-E3"))
    context_budget_tokens: int = field(default_factory=lambda: int(os.environ.get("APE_CONTEXT_BUDGET", "2000")))
    max_turns: int = field(default_factory=lambda: int(os.environ.get("APE_MAX_TURNS", "12")))

    def world_path(self, world_id: str) -> Path:
        split = world_id.split("-")[-2]
        return self.worlds_dir / split / f"{world_id}.json"

    @property
    def ledger_path(self) -> Path:
        return self.cache_dir / "ledger.jsonl"


def embedding_cache(cfg: Config):
    """The shared embedding cache for this config (fake backend never touches the network)."""
    from .llm.embeddings import EmbeddingCache
    from .llm.ledger import Ledger

    if cfg.embeddings_backend == "fake":
        from .llm.fake import FakeEmbeddingsClient

        return EmbeddingCache(cfg.cache_dir / "emb-fake.sqlite", "fake-bow", client=FakeEmbeddingsClient())
    return EmbeddingCache(cfg.cache_dir / "embeddings.sqlite", cfg.embedding_model, ledger=Ledger(cfg.ledger_path))
