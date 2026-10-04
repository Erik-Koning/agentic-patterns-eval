"""Tuning knobs of the multi-agent arms (BUILD_PLAN B3): environment variables, read when an arm's solver is built.

| Knob | Arms | What | Values (default) |
|---|---|---|---|
| `APE_MAS_<ARM>_PROMPT` | M1 (read by S9, M1s, M1k, M2), M7 | the role notes | a variant in `prompts.VARIANTS` (`default`: B2's notes) |
| `APE_MAS_<ARM>_CLIP` | M1 (read by M1s, M1k, M2), M7 | tokens of a worker's result or a member's rationale as another agent reads it | 100-8,000 (2,000) |

**Each tuned arm has its own knobs**, so one arm's tuning candidates never move another arm, and the study runner runs
each tuned arm under exactly its own selection's knobs (`run_study.env_group`, D-042). **The orchestrator chain shares
M1's knobs** (BUILD_REVIEW A-1, D-047): S9, M1s, M1k and M2 read `APE_MAS_M1_*` and run under M1's selection. Each
step of the chain is pre-registered as one switch (S9 -> M1s: ISO; M1s -> M1: CONC; M1 -> M1k: DEL; M1k -> M2: SPEC),
and M1's confirmatory contrast is M1 - S9, so the five must run one protocol variant: with their own variants, a step
could also differ in 2-7 lines of role text, and the tune cannot tell variants apart. S9 runs the variant's planning
note (`RoleNotes.s9`, the same style as its orchestrator note) and has no clip (no agent of its reads another's text).
M2's notes still differ from M1k's by the specialization text only (the roster and the specialty sentence, which no
variant changes). There are no `APE_MAS_S9_*`, `APE_MAS_M1K_*` or `APE_MAS_M2_*` knobs: setting one raises. S8k3
has none: its attempts are S1's, and its aggregator note stays fixed. The structural parameters (3 workers, 2 critique
rounds, council k = 3; brief §4.2) are fixed a priori and are not knobs.

Every `APE_MAS_*` variable is checked whenever a multi-agent solver is built: a misspelt name or an unknown value raises
there, at task creation, before any model call. The task records them (`tasks.gate.ARM_KNOB_PREFIXES`), so they are in
every log's metadata, the arm cache key (`arms.config_fingerprint`) and each phase's fingerprint. `candidate_problems`
is the same check on a tuning candidate's env, for the study grid's checks (`tuning.study_grid_problems`).
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from . import prompts as P
from .core import TEXT_MAX_TOKENS

PREFIX = "APE_MAS_"
KNOB_ARM = {"S9": "M1", "M1": "M1", "M1s": "M1", "M1k": "M1", "M2": "M1", "M7": "M7"}  # arm -> its knobs' name part
NO_CLIP = ("S9",)  # reads M1's prompt variant; it has one agent, so no text is handed on to clip
NO_KNOBS = ("S8k3",)
CLIP_ARMS = ("M1", "M7")  # where one agent reads another's text (M1's clip is the whole orchestrator chain's)
CLIP_RANGE = (100, 8000)
KNOBS = {f"{PREFIX}{a}_PROMPT": (a, "prompt") for a in dict.fromkeys(KNOB_ARM.values())} | {f"{PREFIX}{a}_CLIP": (a, "clip") for a in CLIP_ARMS}


@dataclass(frozen=True)
class Knobs:
    """One arm's resolved knobs."""

    prompt: str = P.DEFAULT_VARIANT
    clip: int = TEXT_MAX_TOKENS

    @property
    def notes(self) -> P.RoleNotes:
        return P.VARIANTS[self.prompt]


def arm_knobs(arm: str) -> list[str]:
    """The variables `arm` reads (none for S8k3)."""
    part = KNOB_ARM.get(arm)
    return [name for name, (a, _) in KNOBS.items() if a == part]


def _value_problem(name: str, raw: str) -> str | None:
    if KNOBS[name][1] == "prompt":
        return None if raw in P.VARIANTS else f"{name}={raw!r} is not a prompt variant ({', '.join(P.VARIANTS)})"
    lo, hi = CLIP_RANGE
    try:
        ok = lo <= int(raw) <= hi
    except ValueError:
        ok = False
    return None if ok else f"{name}={raw!r} is not a whole number of tokens from {lo} to {hi}"


def env_problems(environ: Mapping[str, str]) -> list[str]:
    """What is wrong with the `APE_MAS_*` variables in `environ`: a name that is no knob, or a value a knob does not
    take. A variable set to the empty string counts as unset."""
    problems = []
    for name, raw in sorted(environ.items()):
        if not name.startswith(PREFIX) or not str(raw).strip():
            continue
        if name not in KNOBS:
            problems.append(f"{name} is not a multi-agent knob ({', '.join(KNOBS)})")
        elif problem := _value_problem(name, str(raw).strip()):
            problems.append(problem)
    return problems


def resolve(arm: str, environ: Mapping[str, str] = os.environ) -> Knobs:
    """`arm`'s knobs from `environ`, defaults where unset. Raises ValueError on any problem with an `APE_MAS_*`
    variable, whichever arm it belongs to, so a bad one is found by the first multi-agent solver built."""
    if problems := env_problems(environ):
        raise ValueError("multi-agent knobs: " + "; ".join(problems))
    values = {KNOBS[name][1]: str(environ[name]).strip() for name in arm_knobs(arm) if str(environ.get(name, "")).strip()}
    return Knobs(prompt=values.get("prompt", P.DEFAULT_VARIANT), clip=int(values.get("clip", TEXT_MAX_TOKENS)))


def candidate_problems(system: str, env: Mapping[str, str]) -> list[str]:
    """A study-grid candidate's env, for the system (plan arm) it tunes: a multi-agent arm's candidates set only that
    arm's own knobs, with values they take; no other system sets an `APE_MAS_*` knob."""
    env = {k: str(v) for k, v in env.items()}
    problems = env_problems(env)
    mas = {k for k in env if k.startswith(PREFIX)}
    if system in KNOB_ARM or system in NO_KNOBS:
        if other := sorted(set(env) - set(arm_knobs(system))):
            own = ", ".join(arm_knobs(system)) or "none"
            problems.append(f"{system} candidates set only {system}'s knobs ({own}), not {', '.join(other)}")
    elif mas:
        problems.append(f"{system} is not a multi-agent arm, so it sets no {PREFIX}* knob ({', '.join(sorted(mas))})")
    return problems
