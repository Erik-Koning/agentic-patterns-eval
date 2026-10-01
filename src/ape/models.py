"""Model roles and reasoning effort per run profile (DECISIONS D-015, FIX_PLAN FX-1).

`config/models.yaml` is the one place a run's models are chosen. Effort is set on the
Model objects themselves, not on the eval. Inspect merges the eval-wide generate config
only into the active (agent) model, so a role model (kg, judge) keeps exactly the config
it was built with. Per-call configs (APG classify's and LightRAG's response schemas, the
judge's temperature and seed) merge over that base, so they keep the role's effort.

Inspect records the agent's config in `eval.model_generate_config` and each role's in
`eval.model_roles[role].config`, so every Inspect role's effort is in the eval log.
Build calls run outside Inspect: their effort goes in the ledger context and the index manifest.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

import yaml
from inspect_ai.model import GenerateConfig, Model, get_model

from .config import ROOT

MODELS_PATH = ROOT / "config" / "models.yaml"
DEFAULT_PROFILE = "gate"
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
INSPECT_ROLES = ("agent", "kg", "judge")
REQUIRED_ROLES = ("agent", "kg", "build", "embeddings")
ROLES = (*INSPECT_ROLES, "build", "build_fallback", "embeddings")
_FIELDS = {"model", "reasoning_effort", "max_tokens", "temperature", "top_p", "seed"}


@dataclass(frozen=True)
class RoleSpec:
    """Sampling fields (temperature, top_p, seed) exist for models that accept them, e.g. the anchor
    paper's gpt-4o-mini. Gate profiles never set them, since GPT-6 reasoning models may reject them."""

    model: str
    reasoning_effort: Effort | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def generate_config(self) -> GenerateConfig:
        return GenerateConfig(
            reasoning_effort=self.reasoning_effort, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, seed=self.seed
        )


@dataclass(frozen=True)
class Profile:
    name: str
    roles: dict[str, RoleSpec]

    def role(self, name: str) -> RoleSpec:
        if name not in self.roles:
            raise ValueError(f"model profile {self.name!r} has no role {name!r}; it defines {sorted(self.roles)}")
        return self.roles[name]

    def summary(self) -> dict[str, dict]:
        """role -> {model, reasoning_effort, max_tokens}, for reports and manifests."""
        return {r: asdict(s) for r, s in self.roles.items()}


def _effort(value: Any, where: str) -> Effort | None:
    if value is not None and value not in get_args(Effort):
        raise ValueError(f"{where}: reasoning_effort {value!r} is not one of {get_args(Effort)}")
    return cast(Effort | None, value)


def load_profile(name: str | None = None, path: Path = MODELS_PATH) -> Profile:
    """The named profile (default: $APE_MODEL_PROFILE, else "gate"), validated so a typo fails before any spend."""
    name = name or os.environ.get("APE_MODEL_PROFILE") or DEFAULT_PROFILE
    profiles = (yaml.safe_load(path.read_text()) or {}).get("profiles") or {}
    if name not in profiles:
        raise ValueError(f"unknown model profile {name!r} in {path}; known: {sorted(profiles)}")
    raw = profiles[name] or {}
    if unknown := sorted(set(raw) - set(ROLES)):
        raise ValueError(f"model profile {name!r}: unknown role(s) {unknown}; roles are {list(ROLES)}")
    if missing := [r for r in REQUIRED_ROLES if r not in raw]:
        raise ValueError(f"model profile {name!r}: missing role(s) {missing}")
    roles = {}
    for role, spec in raw.items():
        where = f"model profile {name!r} role {role!r}"
        if not isinstance(spec, dict) or not spec.get("model"):
            raise ValueError(f"{where}: needs a `model`")
        if extra := sorted(set(spec) - _FIELDS):
            raise ValueError(f"{where}: unknown field(s) {extra}; fields are {sorted(_FIELDS)}")
        max_tokens = spec.get("max_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError(f"{where}: max_tokens must be a positive integer, got {max_tokens!r}")
        roles[role] = RoleSpec(
            str(spec["model"]), _effort(spec.get("reasoning_effort"), where), max_tokens, spec.get("temperature"), spec.get("top_p"), spec.get("seed")
        )
    return Profile(name, roles)


def _inspect_model(profile: Profile, role: str, model: str | None, model_args: dict) -> Model:
    if role not in INSPECT_ROLES:
        raise ValueError(f"{role!r} is not an Inspect role (one of {INSPECT_ROLES}); use build_settings() or embedding_model()")
    spec = profile.role(role)
    # memoize=False: agent and kg share a model name, and memoized instances would collapse
    # onto one object, mixing their configs and their per-role usage.
    return get_model(model or spec.model, config=spec.generate_config(), memoize=False, **model_args)


def agent_model(profile: Profile | None = None, *, model: str | None = None, **model_args: Any) -> Model:
    """The eval's main model with the profile's agent effort. `model` swaps the model and keeps the
    effort (CLI overrides, and `mockllm` in tests); `model_args` go to `get_model`."""
    return _inspect_model(profile or load_profile(), "agent", model, model_args)


def role_models(profile: Profile | None = None, roles: tuple[str, ...] = ("kg",), *, model: str | None = None, **model_args: Any) -> dict[str, Model]:
    """`model_roles` for `eval()`: each role's model built with its own effort (see the module docstring)."""
    p = profile or load_profile()
    return {r: _inspect_model(p, r, model, model_args) for r in roles}


def build_settings(profile: Profile | None = None, fallback: bool | None = None) -> tuple[str, Effort | None]:
    """(model, reasoning_effort) for `BuildLlm` (APG authoring, LightRAG extraction).

    `fallback` (or APE_BUILD_FALLBACK=1) selects D-017's fallback builder. APE_BUILD_MODEL still
    wins, for scripts that predate profiles. Its effort is the profile's only when it names one of
    the profile's build models, since another model (e.g. the anchor paper's gpt-4o-mini) may not
    accept one. APE_BUILD_EFFORT overrides the effort either way.
    """
    p = profile or load_profile()
    if fallback is None:
        fallback = os.environ.get("APE_BUILD_FALLBACK") == "1"
    spec = p.role("build_fallback" if fallback else "build")
    model = os.environ.get("APE_BUILD_MODEL") or spec.model
    known = {p.roles[r].model: p.roles[r].reasoning_effort for r in ("build_fallback", "build") if r in p.roles}
    known[spec.model] = spec.reasoning_effort
    effort = os.environ.get("APE_BUILD_EFFORT") or known.get(model)
    return model, _effort(effort, "APE_BUILD_EFFORT")


def embedding_model(profile: Profile | None = None) -> str:
    return (profile or load_profile()).role("embeddings").model
