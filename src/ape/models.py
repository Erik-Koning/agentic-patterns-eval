"""Model roles and reasoning effort per run profile (DECISIONS D-015, FIX_PLAN FX-1).

`config/models.yaml` is the one place a run's models are chosen. Effort is set on the
Model objects themselves, not on the eval. Inspect merges the eval-wide generate config
only into the active (agent) model, so a role model (kg, judge) keeps exactly the config
it was built with. Per-call configs (APG classify's and LightRAG's response schemas, the
judge's temperature and seed) merge over that base, so they keep the role's effort.

Inspect records the agent's config in `eval.model_generate_config` and each role's in
`eval.model_roles[role].config`, so every Inspect role's effort is in the eval log.
Build calls run outside Inspect: their effort goes in the ledger context and the index manifest.

Prices (FX-2) come from one table, `config/model_costs.yaml`: every eval call takes
`**eval_cost_kwargs()` so Inspect fills `ModelUsage.total_cost` for the agent and every role,
and `ape.analysis.cost.load_prices()` prices the ledger (build and embedding calls) from the
same file. `preflight()` checks, before any spend, that every model a profile calls is priced.

Concurrency (FX-3) is per profile too: an optional `concurrency:` mapping beside the roles. Its
`max_connections` goes on every Inspect model's GenerateConfig (for the same reason as effort: the
eval-wide config reaches only the agent), and `ape.runner.run_evals` passes `max_samples` and
`max_tasks` to `eval_set`.
"""

import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

import yaml
from inspect_ai.model import GenerateConfig, Model, ModelCost, ModelInfo, get_model, get_model_info, set_model_info

from .config import ROOT

MODELS_PATH = ROOT / "config" / "models.yaml"
COSTS_PATH = ROOT / "config" / "model_costs.yaml"
ENV_PATH = ROOT / ".env"
COST_FIELDS = tuple(ModelCost.model_fields)  # input, output, input_cache_write, input_cache_read
DEFAULT_PROFILE = "gate"
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
INSPECT_ROLES = ("agent", "kg", "judge", "probe")  # probe: optional (Study G F8 state probes; unset = the agent's model)
REQUIRED_ROLES = ("agent", "kg", "build", "embeddings")
ROLES = (*INSPECT_ROLES, "build", "build_fallback", "embeddings")
_FIELDS = {"model", "reasoning_effort", "max_tokens", "temperature", "top_p", "seed"}
CONCURRENCY_KEY = "concurrency"


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
class Concurrency:
    """Run concurrency for a profile (FX-3), conservative until FX-8's concurrency probe tunes it.

    - `max_connections`: concurrent API calls per model, set on every Inspect model's GenerateConfig
      (the agent's and each role's). `null` leaves it to Inspect's adaptive controller.
    - `max_samples`: samples in flight per task (`eval_set(max_samples=...)`).
    - `max_tasks`: tasks in flight at once (`eval_set(max_tasks=...)`).
    """

    max_connections: int | None = 16
    max_samples: int | None = 32
    max_tasks: int | None = 2


@dataclass(frozen=True)
class Profile:
    name: str
    roles: dict[str, RoleSpec]
    concurrency: Concurrency = Concurrency()

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
    raw = dict(profiles[name] or {})
    concurrency = _concurrency(raw.pop(CONCURRENCY_KEY, None), f"model profile {name!r} {CONCURRENCY_KEY}")
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
    return Profile(name, roles, concurrency)


def _concurrency(raw: Any, where: str) -> Concurrency:
    if raw is None:
        return Concurrency()
    fields = set(Concurrency.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) - fields:
        raise ValueError(f"{where}: expected a mapping with fields {sorted(fields)}, got {raw!r}")
    for k, v in raw.items():
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
            raise ValueError(f"{where}: {k} must be a positive integer or null, got {v!r}")
    return Concurrency(**raw)


def _inspect_model(profile: Profile, role: str, model: str | None, model_args: dict) -> Model:
    if role not in INSPECT_ROLES:
        raise ValueError(f"{role!r} is not an Inspect role (one of {INSPECT_ROLES}); use build_settings() or embedding_model()")
    spec = profile.role(role)
    config = spec.generate_config().merge(GenerateConfig(max_connections=profile.concurrency.max_connections))
    # memoize=False: agent and kg share a model name, and memoized instances would collapse
    # onto one object, mixing their configs and their per-role usage.
    return get_model(model or spec.model, config=config, memoize=False, **model_args)


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


# --- Prices and preflight (FX-2) ---------------------------------------------------------------


class SnapshotWarning(UserWarning):
    """A live preflight's model-snapshot check found nothing to refuse, but something to look at (`ape.snapshots`)."""


class PreflightError(RuntimeError):
    """A run would spend money it cannot account for, or cannot start; the message lists every problem."""


def load_costs(path: Path = COSTS_PATH) -> dict[str, dict]:
    """The price table as Inspect reads it: {"provider/model": {input, output, input_cache_write, input_cache_read}},
    USD per 1M tokens. The ledger's view of the same file (bare model names) is `ape.analysis.cost.load_prices`."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise ValueError(f"{path}: expected a mapping of model -> {{{', '.join(COST_FIELDS)}}}")
    return raw


def eval_cost_kwargs(path: Path = COSTS_PATH) -> dict[str, str]:
    """`eval()` kwargs that price every Inspect call (agent and roles) from the one price table.

    Inspect's cost config only re-prices models in its own database and raises on any other entry,
    such as the embedding model (never called through Inspect) or `mockllm` priced in a test. Those
    are registered first, so the whole file applies. Models Inspect knows keep their database info.
    """
    for name, entry in load_costs(path).items():
        if get_model_info(name) is None:
            set_model_info(name, ModelInfo(cost=ModelCost(**entry)))
    return {"model_cost_config": str(path)}


def _is_price(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and v >= 0


def _called_models(p: Profile, overrides: Mapping[str, str | None]) -> list[tuple[str, str, bool]]:
    """(label, model, is_inspect_role) for every model a run with this profile may call: the profile's
    roles, CLI overrides, and the APE_BUILD_MODEL / APE_EMBEDDING_MODEL environment overrides."""
    called = [(role, spec.model, role in INSPECT_ROLES) for role, spec in p.roles.items()]
    called += [(f"{role} (override)", m, role in INSPECT_ROLES) for role, m in overrides.items() if m]
    for var, role in (("APE_BUILD_MODEL", "build"), ("APE_EMBEDDING_MODEL", "embeddings")):
        if m := os.environ.get(var):
            called.append((f"{role} ({var})", m, False))
    return called


def _api_key_present(env_path: Path) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if env_path.is_file():
        from dotenv import dotenv_values

        return bool(dotenv_values(env_path).get("OPENAI_API_KEY"))
    return False


def preflight(
    profile: Profile | str | None = None,
    live: bool = False,
    *,
    overrides: Mapping[str, str | None] | None = None,
    costs_path: Path = COSTS_PATH,
    env_path: Path = ENV_PATH,
    probe_path: Path | None = None,
    provenance_path: Path | None = None,
) -> list[str]:
    """Problems that should stop a run before it spends money; an empty list means go.

    - the price file parses, and every entry has numeric `COST_FIELDS` (Inspect needs all four);
    - every Inspect role model (agent, kg, judge) is priced under its full "provider/model" key;
    - the build, build_fallback and embedding models are priced under their bare names, as the ledger
      looks them up (`load_prices`);
    - `overrides` (role -> model, from CLI flags) and the build/embedding env overrides are checked the same way;
    - live runs only: OPENAI_API_KEY is set in the environment or in `env_path` (never printed);
    - live runs only: every model the run calls is still served by the snapshot PROVENANCE.md pins for it, as the
      latest readiness probe saw it (`ape.snapshots.snapshot_status`). Warnings (nothing pinned yet, a pin not
      re-checked, an old probe) are issued as `SnapshotWarning`.
    `mockllm` models are exempt from the price requirement unless the file prices them.
    """
    try:
        p = profile if isinstance(profile, Profile) else load_profile(profile)
    except (OSError, ValueError, yaml.YAMLError) as e:
        return [f"model profile: {e}"]
    problems: list[str] = []
    try:
        costs = load_costs(costs_path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        costs = None
        problems.append(f"price file {costs_path} does not parse: {e}")
    if costs is not None:
        for name, entry in costs.items():
            if bad := [f for f in COST_FIELDS if not _is_price(entry.get(f))]:
                problems.append(f"price for {name!r} in {costs_path}: {bad} must be non-negative numbers")
        bare = {k.split("/", 1)[-1]: v for k, v in costs.items()}  # as ape.analysis.cost.load_prices keys them
        for label, model, inspect_role in _called_models(p, overrides or {}):
            key, table = (model, costs) if inspect_role else (model.split("/", 1)[-1], bare)
            if key in table or model.startswith("mockllm/"):
                continue
            where = 'its full "provider/model" key' if inspect_role else "its bare model name (ledger pricing)"
            problems.append(f"profile {p.name!r} role {label}: no price for {key!r} in {costs_path} (under {where})")
    if live and not _api_key_present(env_path):
        problems.append(f"OPENAI_API_KEY is not set in the environment or in {env_path}")
    if live:
        from . import snapshots

        fallback = os.environ.get("APE_BUILD_FALLBACK") == "1"  # the D-017 fallback builder is called only then
        called = {
            snapshots.alias(m)
            for label, m, _ in _called_models(p, overrides or {})
            if not m.startswith("mockllm/") and (label != "build_fallback" or fallback)
        }
        status = snapshots.snapshot_status(probe_path or snapshots.PROBE_PATH, provenance_path or snapshots.PROVENANCE_PATH, aliases=called)
        problems += status["problems"]
        for w in status["warnings"]:
            warnings.warn(w, SnapshotWarning, stacklevel=2)
    return problems


def require_preflight(profile: Profile | str | None = None, live: bool = False, **kwargs: Any) -> None:
    """`preflight()`, raising `PreflightError` that lists every problem."""
    if problems := preflight(profile, live, **kwargs):
        name = profile.name if isinstance(profile, Profile) else profile or os.environ.get("APE_MODEL_PROFILE") or DEFAULT_PROFILE
        raise PreflightError(f"preflight failed for model profile {name!r}:\n" + "\n".join(f"  - {x}" for x in problems))
