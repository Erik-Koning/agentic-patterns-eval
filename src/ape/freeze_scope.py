"""What a study's phases and freeze hash of the program-wide config files: the study's resolved slice, not the file.

config/run_plan.yaml, config/models.yaml and config/model_costs.yaml serve every study (the gate, the main study,
Study G). If a frozen run hashed them whole, any later edit to another study's cells, profiles or prices (a pilot
recalibration, a tuning grid's arms, an allocation) would break it: its build-test and test refuse a changed frozen
file. So each study fingerprints and freezes only the part that shapes its own runs:

- **run_plan.yaml**: the study's section (`studies.<study>`: profile and every phase's cells); `budget` narrowed to
  what the study's runs read (`total_usd` and `sample_cost_limit`, plus the study's own allocation where its guard
  enforces one, i.e. not the gate's; another study's allocation, or the program's contingency, is never an input);
  the right-sizing cuts that edit one of the study's cells; and, for Study G, the `study_g` block (window W,
  threshold T_abs).
- **models.yaml**: the profiles the study's cells name, its default profile, and those it chooses at run time
  (`RUNTIME_PROFILES`: the gate's PC1 anchor falls back to `anchor_luna`), each as resolved (YAML merge keys applied).
- **model_costs.yaml**: the price entries of every model those profiles (and the cells' `models:` overrides) call,
  matched by bare model name as `ape.models.preflight` matches them.

A `ConfigSlice` is the file's path (so it reads, shows and compares as that path) that hashes as its slice: phase
inputs and freeze records carry `slice` (the study) and the whole file's hash as `file_sha256`, for information only,
never enforced. A study's own files (its tuning grid, its pre-registration, its run outputs) are hashed whole.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SLICED = ("run_plan.yaml", "models.yaml", "model_costs.yaml")
RUNTIME_PROFILES = {"gate": ("anchor", "anchor_luna")}  # chosen at run time (`run_gate.anchor_profile_name`)
BUDGET_KEYS = ("total_usd", "sample_cost_limit")  # program-level budget fields every study's guard reads
UNALLOCATED = ("gate",)  # its guard checks the program total only (`run_gate.spend`), never budget.allocations


class ConfigSlice(Path):
    """A shared config file as one study sees it: the path, hashed as `config_slice(path, study)`."""

    def __init__(self, *args: Any, study: str) -> None:
        super().__init__(*args)
        self.study = study

    def with_segments(self, *pathsegments: Any) -> Path:
        return Path(*pathsegments)  # a derived path (parent, name, ...) is a plain path

    def __reduce__(self) -> tuple:
        return (_slice, (str(self), self.study))

    def content(self) -> Any:
        return config_slice(Path(self), self.study)

    def sha256(self) -> str | None:
        content = self.content()
        return None if content is None else hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def _slice(path: str, study: str) -> ConfigSlice:
    return ConfigSlice(path, study=study)


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.is_file() else None


def _bare(model: str) -> str:
    return str(model).split("/", 1)[-1]


def study_cells(raw_plan: dict, study: str) -> list[dict]:
    phases = (((raw_plan.get("studies") or {}).get(study) or {}).get("phases")) or {}
    return [c for cells in phases.values() for c in cells or []]


def plan_slice(raw_plan: dict, study: str) -> dict:
    """The study's part of run_plan.yaml (module docstring)."""
    budget = raw_plan.get("budget") or {}
    ids = {c.get("id") for c in study_cells(raw_plan, study)}
    allocation = {} if study in UNALLOCATED else {"allocation": (budget.get("allocations") or {}).get(study)}
    out = {
        "budget": {k: budget.get(k) for k in BUDGET_KEYS} | allocation,
        "studies": {study: (raw_plan.get("studies") or {}).get(study)},
        "cuts": [c for c in raw_plan.get("cuts") or [] if any(e.get("cell") in ids for e in c.get("set") or [])],
    }
    if study == "study_g":
        out["study_g"] = raw_plan.get("study_g")
    return out


def study_profiles(raw_plan: dict, study: str) -> list[str]:
    """Every profile the study's runs may use: its default, every cell's, and the ones it picks at run time."""
    default = ((raw_plan.get("studies") or {}).get(study) or {}).get("profile") or "gate"
    names = [default, *(c.get("profile") for c in study_cells(raw_plan, study) if c.get("profile")), *RUNTIME_PROFILES.get(study, ())]
    return list(dict.fromkeys(names))


def study_models(raw_plan: dict, raw_models: dict, study: str) -> list[str]:
    """Every model the study's profiles and its cells' `models:` overrides call."""
    profiles = (raw_models or {}).get("profiles") or {}
    models = [spec.get("model") for name in study_profiles(raw_plan, study) for spec in (profiles.get(name) or {}).values() if isinstance(spec, dict)]
    models += [m for c in study_cells(raw_plan, study) for m in (c.get("models") or {}).values()]
    return list(dict.fromkeys(str(m) for m in models if m))


def config_slice(path: Path, study: str) -> Any:
    """The study's slice of a shared config file (`SLICED`; the module docstring), None when a file it needs is missing.
    The other config files are read from the same directory."""
    name, folder = Path(path).name, Path(path).parent
    raw_plan = _load(folder / "run_plan.yaml")
    if name == "run_plan.yaml":
        return None if raw_plan is None else plan_slice(raw_plan, study)
    raw_models = _load(folder / "models.yaml")
    if raw_plan is None or raw_models is None:
        return None
    if name == "models.yaml":
        profiles = raw_models.get("profiles") or {}
        return {p: profiles.get(p) for p in study_profiles(raw_plan, study)}
    if name == "model_costs.yaml":
        raw_costs = _load(Path(path))
        if raw_costs is None:
            return None
        wanted = {_bare(m) for m in study_models(raw_plan, raw_models, study)}
        return {k: v for k, v in sorted(raw_costs.items()) if _bare(k) in wanted}
    raise ValueError(f"{path}: not a sliced config file ({', '.join(SLICED)})")


def config_input(path: Path, study: str) -> Path:
    """`path` as a study's input: its slice for a shared config file (`SLICED`), else the plain path."""
    return ConfigSlice(path, study=study) if Path(path).name in SLICED else Path(path)
