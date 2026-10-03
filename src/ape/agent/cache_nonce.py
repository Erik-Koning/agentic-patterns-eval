"""The per-run cache nonce (ORCHESTRATOR_BRIEF_v2 §6.2 and §9; BUILD_PLAN B12).

**Why.** Provider prompt caches key on the exact leading tokens of a request. Two arms whose agents open with the same
long prefix (S1 and M1 on F7-1000 both start with the base prompt and the whole monolith; S8k3's attempts are S1's
loop) would read each other's cache, and so would the same arm in two runs, or in the micro-pilot and then the test.
The later one would look cheaper on cache-adjusted $ and faster on wall-clock for reasons that have nothing to do with
the arm. With a nonce, every agent prompt starts with one short neutral line that is different per run and per arm, so
no cached prefix crosses them.

**What gets the line.** `with_nonce` puts `Run reference: <nonce>` (12 hex digits) and a blank line in front of:
- every single-agent system prompt (`kb_react.kb_agent`) and every multi-agent role's (`multi.core.Delivery.system`:
  planner, orchestrator, workers, specialists, council members, chair, aggregator; S8k3's attempts are `kb_agent`);
- the F8 session's system prompt (`agent.session`), which every arm's views, the forked probes, CM-native's
  compaction input and O-state's view all start from, and the session team's workers (`multi.session_team`);
- management calls that carry no system prompt of their own (`SessionContext.cm_generate`: CM-sum's summaries,
  CM-reset's handoff notes, CM-todo's extractions) get it as a system message of its own.
Not prefixed: the `kg` role's calls (APG classify, LightRAG keyword extraction). Their prompts are built inside the KG
arms the gate tuned, and their cache-read tokens are metered per role (`mas_accounting`, `role_usage`), so any
cross-arm hit there (S5, M1k and M2 classifying the same task query on the same world) is visible, not hidden.

**Where the nonce comes from.** APE_CACHE_NONCE is a seed. A task builder (`tasks.main.main_study`,
`tasks.study_g.f8_session`) reads it when the task is created and derives the task's nonce from the seed and the
task's arm (main study: arm, delivery and exposure, since Study B runs one arm under several deliveries in one eval
set): `task_nonce`. The nonce goes into every sample's metadata (`cache_nonce`, which the brief
requires each sample to record) and the task's metadata (`cache_nonce`, `cache_nonce_seed`); the solvers read it from
the sample's metadata, so a task carries its nonce whatever the environment is later. `ape.run_study` sets the seed per
eval set: the study, the run id and its creation time, the phase and the group's directory (a plan cell, an
environment group and its configuration), and per tuning candidate. So caching works within one arm's eval set (its
tasks, worlds and epochs, and every agent of one sample share it, as a deployment's would) but never across runs,
phases, cells, groups, candidates or arms. The smoke sets a fresh seed per invocation.

**When the seed is unset** (the gate never sets it, nor do tests and ad-hoc runs), no task carries a nonce and every
prompt is byte-identical to a build without this module.

**Trade-offs.**
- Epochs of one arm share its nonce, so a later epoch can read the cache an earlier epoch of the same task wrote
  (same system prompt and task): its cache-adjusted $ is a repeat-traffic figure, not a first-seen one. Cache-hit
  claims should use first epochs (or the uncached meters, which the nonce does not touch).
- Arms that run concurrently in one eval set (an env group holds every untuned arm of a cell) no longer share a
  prefix; a deployment that ran them side by side would. The study compares arms, not a shared fleet.
- The line costs about 10 input tokens per call (cached within the arm like the rest of the prefix) and changes the
  prompt every arm sees by one neutral line, identically for all arms. It is part of what B0, the token caps and the
  window W measure.
- A provider that renders tool definitions ahead of the system message could still share a tool-definition prefix
  longer than its minimum cacheable length (OpenAI: 1,024 tokens) across arms with the same tools; the cache-read
  tokens each call reports make any such residual measurable.
"""

import hashlib
import os
from collections.abc import Iterable, Mapping
from typing import Any

NONCE_ENV = "APE_CACHE_NONCE"  # the seed; unset = no nonce anywhere
METADATA_KEY = "cache_nonce"  # sample and task metadata: the nonce this task's prompts carry
SEED_KEY = "cache_nonce_seed"  # task metadata: the seed it was derived from
LINE = "Run reference: {nonce}"
LENGTH = 12


def seed() -> str | None:
    """APE_CACHE_NONCE as set (stripped), or None."""
    raw = os.environ.get(NONCE_ENV, "").strip()
    return raw or None


def derive(seed_value: str, *parts: object) -> str:
    """A short, neutral nonce from a seed and the parts that must not share a cache (arm, delivery, exposure)."""
    key = "\x1f".join(str(p) for p in (seed_value, *parts))
    return hashlib.sha256(key.encode()).hexdigest()[:LENGTH]


def task_nonce(*parts: object) -> str | None:
    """The nonce of a task created now: derived from APE_CACHE_NONCE and `parts`; None when the seed is unset."""
    s = seed()
    return derive(s, *parts) if s else None


def line(nonce: str) -> str:
    return LINE.format(nonce=nonce)


def with_nonce(text: str, nonce: str | None) -> str:
    """`text` with the nonce line in front; `text` itself (byte for byte) without a nonce."""
    return f"{line(nonce)}\n\n{text}" if nonce else text


def of(metadata: Mapping[str, Any] | None) -> str | None:
    """The nonce a sample carries (its metadata's `cache_nonce`), or None."""
    value = (metadata or {}).get(METADATA_KEY)
    return str(value) if value else None


def stamp(samples: Iterable[Any], nonce: str | None) -> None:
    """Record the task's nonce in every sample's metadata (nothing without one)."""
    if not nonce:
        return
    for s in samples:
        s.metadata = {**(s.metadata or {}), METADATA_KEY: nonce}


def task_metadata(nonce: str | None) -> dict:
    """Task metadata entries for the nonce (none without one, so task metadata is unchanged)."""
    return {METADATA_KEY: nonce, SEED_KEY: seed()} if nonce else {}


def starts_with_nonce(text: str | None, nonce: str) -> bool:
    return (text or "").startswith(line(nonce))
