"""Run configuration: paths, model roles, budgets. Values come from environment variables so
Inspect tasks, build scripts and tests share one source of truth.

Model IDs are deliberately unset by default: they are chosen at readiness E3 and
recorded in PROVENANCE.md. `APE_EMBEDDINGS=fake` swaps in the offline hashed
bag-of-words embedder used by dry runs and tests.

Context budgets are per arm family, so tuning one system never moves another when all arms run
in one process (the gate's test phase):

    APE_S3S_BUDGET   S3s flat retrieval: tokens packed per compile
    APE_APG_BUDGET   APG arms (APG-q/-s, APGo-q, S5o): compose's maxPromptTokens, and the graph default
    APE_LGR_BUDGET   LightRAG arms: the base of `lgr.adapter.query_params` (entity/relation/total caps)

Each defaults to APE_CONTEXT_BUDGET, then 2000. APE_CONTEXT_BUDGET itself remains S7's fallback
target (after config/s7_targets.json and APE_S7_TARGET).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Inspect loads .env for evals; standalone build CLIs (APG authoring, LightRAG extraction) need it too.
# Never overrides variables already set in the environment.
load_dotenv(ROOT / ".env", override=False)

DEFAULT_CONTEXT_BUDGET = 2000


def budget_env(*names: str, default: int = DEFAULT_CONTEXT_BUDGET) -> int:
    """The first of `names` set (non-empty) in the environment, as an int; else `default`."""
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return int(raw)
    return default


@dataclass(frozen=True)
class Config:
    worlds_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_WORLDS", ROOT / "worlds")))
    indices_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_INDICES", ROOT / "indices")))
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("APE_CACHE", ROOT / "cache")))
    embeddings_backend: str = field(default_factory=lambda: os.environ.get("APE_EMBEDDINGS", "openai"))
    embedding_model: str = field(default_factory=lambda: os.environ.get("APE_EMBEDDING_MODEL", "UNSET-see-E3"))
    # Shared default; arms read their own knob below (see the module docstring). S7 falls back to this one.
    context_budget_tokens: int = field(default_factory=lambda: budget_env("APE_CONTEXT_BUDGET"))
    s3s_budget_tokens: int = field(default_factory=lambda: budget_env("APE_S3S_BUDGET", "APE_CONTEXT_BUDGET"))
    apg_budget_tokens: int = field(default_factory=lambda: budget_env("APE_APG_BUDGET", "APE_CONTEXT_BUDGET"))
    lgr_budget_tokens: int = field(default_factory=lambda: budget_env("APE_LGR_BUDGET", "APE_CONTEXT_BUDGET"))
    max_turns: int = field(default_factory=lambda: int(os.environ.get("APE_MAX_TURNS", "12")))

    def world_path(self, world_id: str) -> Path:
        split = world_id.split("-")[-2]
        return self.worlds_dir / split / f"{world_id}.json"

    @property
    def ledger_path(self) -> Path:
        return self.cache_dir / "ledger.jsonl"


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


@dataclass(frozen=True)
class BuildConcurrency:
    """Build-time concurrency (FIX_PLAN FX-4), from the environment. Query time never uses these.

    - `llm` (APE_BUILD_LLM_CONCURRENCY, 16): build-model calls in flight per world: LightRAG's
      `llm_model_max_async` and the APG authoring semaphore.
    - `parallel_insert` (APE_BUILD_PARALLEL_INSERT, 8): LightRAG `max_parallel_insert`, i.e. documents
      (one per shared chunk, D-003) extracted at once.
    - `embed` (APE_BUILD_EMBED_CONCURRENCY, 16): LightRAG `embedding_func_max_async`.
    - `workers` (APE_BUILD_WORKERS, 4): worlds built at once by `ape.artifacts`, one process each.

    Peak in-flight build calls are about `workers * llm` (64 by default).
    """

    llm: int = 16
    parallel_insert: int = 8
    embed: int = 16
    workers: int = 4

    @classmethod
    def from_env(cls) -> BuildConcurrency:
        return cls(
            llm=_positive_int_env("APE_BUILD_LLM_CONCURRENCY", cls.llm),
            parallel_insert=_positive_int_env("APE_BUILD_PARALLEL_INSERT", cls.parallel_insert),
            embed=_positive_int_env("APE_BUILD_EMBED_CONCURRENCY", cls.embed),
            workers=_positive_int_env("APE_BUILD_WORKERS", cls.workers),
        )


def embedding_cache(cfg: Config):
    """The shared embedding cache for this config (fake backend never touches the network)."""
    from .llm.embeddings import EmbeddingCache
    from .llm.ledger import Ledger

    if cfg.embeddings_backend == "fake":
        from .llm.fake import FakeEmbeddingsClient

        return EmbeddingCache(cfg.cache_dir / "emb-fake.sqlite", "fake-bow", client=FakeEmbeddingsClient())
    return EmbeddingCache(cfg.cache_dir / "embeddings.sqlite", cfg.embedding_model, ledger=Ledger(cfg.ledger_path))
