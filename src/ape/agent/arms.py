"""Arm registry: builds (and caches) one delivery arm per (arm, world, event loop).

Loop identity is part of the key because LightRAG binds its async locks to the
loop that created them.
"""

import asyncio
from functools import lru_cache
from pathlib import Path

from ..config import ROOT, Config, embedding_cache
from ..kb.baselines import FlatHybrid, Monolith, OracleContext, RandomUnits
from ..kb.context import DeliveryArm
from ..worlds.render import chunk_world
from ..worlds.spec import World

BASELINE_ARMS = ("S1", "S3s", "S6", "S7")
KG_ARM_ENV = "APE_KG_ARM"
DEFAULT_KG_ARM = "APG-s"
_cache: dict[tuple, DeliveryArm] = {}
_locks: dict[tuple, asyncio.Lock] = {}


def load_world(world_id: str) -> World:
    return _load_path(str(Config().world_path(world_id)))


@lru_cache(maxsize=64)
def _load_path(path: str) -> World:
    return World.load(path)


def s7_targets_path() -> Path:
    import os

    return Path(os.environ.get("APE_S7_TARGETS") or ROOT / "config" / "s7_targets.json")


def _s7_targets_stamp() -> tuple:
    """The targets file's identity, for S7's cache key: a rewritten file (a re-pilot) must not reuse an S7 arm."""
    path = s7_targets_path()
    st = path.stat() if path.is_file() else None
    return (str(path), st.st_mtime_ns if st else None, st.st_size if st else None)


def s7_target(cfg: Config, world: World) -> int:
    """Per-cell S7 size, frozen from APG*'s realized median on the pilot: the targets file is
    APE_S7_TARGETS (the gate orchestrator points it at its run's file), else `config/s7_targets.json`.

    Falls back to APE_S7_TARGET, then the context budget (dry runs only); RandomUnits caps it at
    half the corpus either way.
    """
    import json
    import os

    path = s7_targets_path()
    cell = f"{world.family}-{world.level}"
    if path.exists():
        targets = json.loads(path.read_text())
        if cell in targets:
            return int(targets[cell])
    return int(os.environ.get("APE_S7_TARGET", cfg.context_budget_tokens))


def s7_per_step() -> bool:
    """Whether S7 recompiles every step (APE_S7_PER_STEP=1): the gate orchestrator sets it when APG* is per-step,
    so the placebo mirrors APG*'s delivery schedule (D-024)."""
    import os

    return os.environ.get("APE_S7_PER_STEP", "").strip() in ("1", "true", "yes")


def kg_arm_name() -> str:
    """The arm the main study's KG arm S5 (and the KG workers of M1k/M2) runs as: APE_KG_ARM, which the study runner
    sets from the gate's verdict (APG's selected arm if GO, LightRAG's otherwise; BUILD_PLAN B5), else APG-s. Its
    knobs come from the environment like any APG/LightRAG arm's."""
    import os

    name = os.environ.get(KG_ARM_ENV, "").strip() or DEFAULT_KG_ARM
    if not name.startswith(("APG", "LGR")):
        raise ValueError(f"{KG_ARM_ENV}={name!r}: the KG arm must be an APG or LightRAG arm")
    return name


async def _build(arm: str, world: World, cfg: Config) -> DeliveryArm:
    if arm == "S5":
        return await _build(kg_arm_name(), world, cfg)
    chunks = chunk_world(world)
    if arm == "S1":
        return Monolith(world, chunks)
    if arm == "S3s":
        emb = embedding_cache(cfg)
        await emb.embed([c.text for c in chunks], context={"world": world.id, "system": "flat"})
        return FlatHybrid(chunks, emb, cfg.s3s_budget_tokens)
    if arm == "S6":
        return OracleContext(world)
    if arm == "S7":
        return RandomUnits(chunks, s7_target(cfg, world), per_step=s7_per_step())
    if arm.startswith("APG") or arm == "S5o":
        from ..apg.arm import build_apg_arm

        return await build_apg_arm(arm, world, cfg)
    if arm.startswith("LGR"):
        from ..lgr.adapter import build_lgr_arm

        return await build_lgr_arm(arm, world, cfg)
    raise ValueError(f"unknown arm {arm}")


def config_fingerprint() -> tuple:
    """Every APE_* setting in force. Part of the arm cache key, so a tuning run never reuses an arm
    built under a different configuration."""
    import os

    return tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("APE_")))


def arm_provider(arm: str, cfg: Config | None = None):
    """The arm for a world, built once per configuration. Without `cfg`, the `Config` (budget knobs included)
    is read when the arm is built, inside the run, so it always matches `config_fingerprint()` in the key."""

    async def provide(world: World) -> DeliveryArm:
        c = cfg or Config()
        key = (arm, world.id, str(c.worlds_dir), str(c.cache_dir), config_fingerprint(), id(asyncio.get_running_loop()))
        if arm == "S7":
            key += _s7_targets_stamp()
        lock = _locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in _cache:
                _cache[key] = await _build(arm, world, c)
        return _cache[key]

    return provide
