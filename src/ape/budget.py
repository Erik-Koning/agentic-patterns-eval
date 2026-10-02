"""Cost model for the whole program: gate, main study and Study G (FIX_PLAN FX-5).

    uv run python -m ape.budget [--scenario conservative|expected] [--study gate|main|study_g] [--detail] [--uncut]
    uv run python -m ape.budget calibrate LOGS... [--out config/budget_calibration_measured.yaml]
    uv run python -m ape.budget remaining --budget 5000 LOGS... [--ledger cache/ledger.jsonl]
    uv run python -m ape.budget spend [--registry PATH] [--json]     # program-wide spend so far (ape.spend)

Inputs, all data:
- `config/run_plan.yaml`: what runs, as study -> phase -> cell (arms, task cells, tasks, epochs, model profile,
  effort, delivery), the budget allocation, and the right-sizing cuts.
- `config/budget_assumptions.yaml`: per-call token priors, calls per sample, delivered context per arm, build and
  anchor call sizes, Study G view sizes, and the two prompt-cache scenarios.
- `config/model_costs.yaml`: prices, read with `ape.models.load_costs` (the table Inspect and the ledger use).
- `config/budget_calibration_measured.yaml` (optional): per-call tokens and calls per sample measured from real
  Inspect logs by `calibrate`. Preferred over the priors wherever (arm, agent model, effort, cell, delivery) match.

Each plan cell expands into line items (arm x task cell x delivery), and each line item into call groups (agent,
kg, embeddings, cm, probe, judge, build), priced per sample as
    calls x (input tokens x input rate + output tokens x output rate),
where the input rate blends the input price and the cached-input price by the cached share of input:
- `conservative`: cached input at the full input price (today's price table). The budget gate uses this one:
  the CLI exits 1 when the conservative total exceeds `budget.total_usd`.
- `expected`: a stated cached share per cache class at `cached_price_ratio` x input, an assumption until
  readiness E5 confirms GPT-6 cached pricing (or `cached_price: table` once the table has it).

The orchestrator (FX-6) calls `projected_cost(...)` for a phase and `program_remaining(...)` for what is left of the
program budget, and refuses the phase with `require_affordable(...)` when the projection exceeds it. Spend is
read per sample from Inspect's sample summaries, each sample once by uuid, so retried tasks and killed runs count
correctly (`log_samples`); `program_spend` sums every log dir and ledger in the program's spend registry
(`ape.spend`). `sample_cost_limit` is the per-sample runaway guard the runner passes to Inspect.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT
from .models import COSTS_PATH, MODELS_PATH, Profile, load_costs, load_profile

PLAN_PATH = ROOT / "config" / "run_plan.yaml"
ASSUMPTIONS_PATH = ROOT / "config" / "budget_assumptions.yaml"
MEASURED_PATH = ROOT / "config" / "budget_calibration_measured.yaml"
SCENARIOS = ("conservative", "expected")
KINDS = ("agent", "session", "build", "anchor", "fixed")
STUDY_ORDER = ("gate", "main", "study_g")


class BudgetError(RuntimeError):
    """The plan or its inputs are inconsistent, or a phase would spend more than is left."""


# --- Plan ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanCell:
    id: str
    study: str
    phase: str
    spec: dict  # the cell's mapping, with the study's default profile filled in

    @property
    def kind(self) -> str:
        return self.spec.get("kind", "agent")

    @property
    def enabled(self) -> bool:
        return bool(self.spec.get("enabled", True))


@dataclass(frozen=True)
class Plan:
    cells: tuple[PlanCell, ...]
    budget: dict
    study_g: dict
    cuts: tuple[dict, ...]

    def cell(self, cell_id: str) -> PlanCell:
        for c in self.cells:
            if c.id == cell_id:
                return c
        raise BudgetError(f"no plan cell {cell_id!r}")

    @property
    def applied_cuts(self) -> list[str]:
        return [c["id"] for c in self.cuts if c.get("applied")]


_FIELD_DEFAULTS = {"enabled": True}


def _field(cell: PlanCell, name: str) -> Any:
    return cell.spec.get(name, _FIELD_DEFAULTS.get(name))


def _with_edits(plan: Plan, cut: dict, apply: bool) -> Plan:
    """The plan with `cut`'s edits set to `to` (apply) or back to `from` (revert), and its flag updated."""
    edits: dict[str, dict] = defaultdict(dict)
    for e in cut["set"]:
        edits[e["cell"]][e["field"]] = e["to"] if apply else e["from"]
    known = {c.id for c in plan.cells}
    if missing := sorted(set(edits) - known):
        raise BudgetError(f"cut {cut['id']} edits unknown cell(s) {missing}")
    cells = tuple(replace(c, spec={**c.spec, **edits[c.id]}) if c.id in edits else c for c in plan.cells)
    cuts = tuple({**c, "applied": apply} if c["id"] == cut["id"] else c for c in plan.cuts)
    return replace(plan, cells=cells, cuts=cuts)


def apply_cut(plan: Plan, cut_id: str) -> Plan:
    return _with_edits(plan, _cut(plan, cut_id), True)


def revert_cut(plan: Plan, cut_id: str) -> Plan:
    return _with_edits(plan, _cut(plan, cut_id), False)


def _cut(plan: Plan, cut_id: str) -> dict:
    for c in plan.cuts:
        if c["id"] == cut_id:
            return c
    raise BudgetError(f"no cut {cut_id!r}")


def uncut(plan: Plan) -> Plan:
    """The plan before any right-sizing cut."""
    for cut_id in reversed(plan.applied_cuts):
        plan = revert_cut(plan, cut_id)
    return plan


def load_plan(path: Path = PLAN_PATH, *, uncut_plan: bool = False) -> Plan:
    """The run plan, validated: unique cell ids, known kinds, and every cut consistent with the cells
    (an applied cut's fields hold its `to` values, an unapplied cut's its `from` values)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cells: list[PlanCell] = []
    for study, sdef in (raw.get("studies") or {}).items():
        for phase, specs in (sdef.get("phases") or {}).items():
            for spec in specs or []:
                if "id" not in spec:
                    raise BudgetError(f"{path}: a cell in {study}/{phase} has no id")
                full = {"profile": sdef.get("profile", "gate"), **spec}
                if full.get("kind", "agent") not in KINDS:
                    raise BudgetError(f"{path}: cell {spec['id']}: kind {full['kind']!r} is not one of {KINDS}")
                cells.append(PlanCell(spec["id"], study, phase, full))
    if dup := sorted(k for k, n in Counter(c.id for c in cells).items() if n > 1):
        raise BudgetError(f"{path}: duplicate cell id(s) {dup}")
    plan = Plan(tuple(cells), raw.get("budget") or {}, raw.get("study_g") or {}, tuple(raw.get("cuts") or ()))
    for cut in plan.cuts:
        for e in cut["set"]:
            want = e["to"] if cut.get("applied") else e["from"]
            have = _field(plan.cell(e["cell"]), e["field"])
            if have != want:
                state = "applied" if cut.get("applied") else "not applied"
                raise BudgetError(f"{path}: cut {cut['id']} is {state}, so {e['cell']}.{e['field']} should be {want!r}, not {have!r}")
    return uncut(plan) if uncut_plan else plan


# --- Assumptions, prices, measurements -----------------------------------------------------------------


def load_assumptions(path: Path = ASSUMPTIONS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


@dataclass(frozen=True)
class Prices:
    """`ape.models.load_costs` with bare-name lookup for build and embedding models (as the ledger keys them)."""

    table: dict[str, dict]

    @classmethod
    def load(cls, path: Path = COSTS_PATH) -> Prices:
        return cls(load_costs(path))

    def entry(self, model: str) -> dict:
        if model in self.table:
            return self.table[model]
        bare = model.split("/", 1)[-1]
        for name, entry in self.table.items():
            if name.split("/", 1)[-1] == bare:
                return entry
        raise BudgetError(f"no price for {model!r} in the price table; add it to config/model_costs.yaml")


def load_measured(path: Path | None = MEASURED_PATH) -> list[dict]:
    if path is None or not Path(path).is_file():
        return []
    return list((yaml.safe_load(Path(path).read_text()) or {}).get("entries") or [])


# --- Expansion: plan cells -> line items -> call groups ------------------------------------------------


@dataclass(frozen=True)
class CallGroup:
    """Calls of one role per sample, at a mean input and output size per call."""

    role: str
    model: str
    calls: float
    input: float
    output: float
    cache: str  # cache class for the `expected` scenario
    cached_fraction: float | None = None  # measured; replaces the class's assumed share


@dataclass(frozen=True)
class LineItem:
    cell: str
    study: str
    phase: str
    kind: str
    tier: str
    arm: str
    task_cell: str
    delivery: str
    samples: float
    groups: tuple[CallGroup, ...]
    measured: bool = False
    fixed_usd: float = 0.0


def _short(model: str) -> str:
    return model.split("/", 1)[-1].removeprefix("gpt-6-")


def _tier(model: str, effort: str | None) -> str:
    return f"{_short(model)}-{effort}" if effort else _short(model)


def resolve_arm(table: dict, name: str) -> dict:
    """An arm's spec after following `like:` links; `_chain` lists the names followed (for measured lookups)."""
    chain, cur, specs = [], name, []
    while cur is not None:
        if cur in chain:
            raise BudgetError(f"arm {name!r}: `like` cycle through {chain}")
        if cur not in table:
            raise BudgetError(f"unknown arm {cur!r} (config/budget_assumptions.yaml `arms`)")
        chain.append(cur)
        specs.append(table[cur])
        cur = table[cur].get("like")
    merged: dict = {"multiplier": 1.0, "kg": None, "embed": False}
    for spec in reversed(specs):
        merged.update({k: v for k, v in spec.items() if k != "like"})
    merged["_chain"] = chain
    return merged


def _arm_counts(arms: Any) -> dict[str, int]:
    """`arms` as a list (one each) or a mapping arm -> number of candidates (tuning)."""
    if isinstance(arms, dict):
        return {str(k): int(v) for k, v in arms.items()}
    return dict(Counter(str(a) for a in arms))


def _n_tasks(cell: PlanCell) -> int:
    s = cell.spec
    if "n_tasks" in s:
        return int(s["n_tasks"])
    if "worlds" in s and "tasks_per_world" in s:
        return int(s["worlds"]) * int(s["tasks_per_world"])
    raise BudgetError(f"cell {cell.id}: needs n_tasks, or worlds and tasks_per_world")


def _output(A: dict, effort: str | None) -> float:
    table = A["output_tokens_per_call"]
    key = effort or "default"
    if key not in table:
        raise BudgetError(f"no output_tokens_per_call for effort {key!r} in config/budget_assumptions.yaml")
    return float(table[key])


def _calls_per_sample(A: dict, task_cell: str, delivery: str) -> float:
    table = A["calls_per_sample"]
    for key in (task_cell, task_cell.split("-", 1)[0]):
        if key in table and delivery in table[key]:
            return float(table[key][delivery])
    raise BudgetError(f"no calls_per_sample for {task_cell} / {delivery} in config/budget_assumptions.yaml")


class _Roles:
    """Role -> (model, effort) for a cell: its profile, then `models:` overrides, then the `effort:` override."""

    def __init__(self, models_path: Path):
        self.models_path = models_path
        self._profiles: dict[str, Profile] = {}

    def __call__(self, cell: PlanCell) -> dict[str, tuple[str, str | None]]:
        name = cell.spec["profile"]
        if name not in self._profiles:
            self._profiles[name] = load_profile(name, self.models_path)
        roles = {r: (s.model, s.reasoning_effort) for r, s in self._profiles[name].roles.items()}
        for role, model in (cell.spec.get("models") or {}).items():
            roles[role] = (str(model), roles.get(role, (None, None))[1])
        if "effort" in cell.spec:
            roles["agent"] = (roles["agent"][0], cell.spec["effort"])
        return _RoleMap(cell.id, roles)


class _RoleMap(dict):
    """role -> (model, effort); a missing role names the cell instead of raising a bare KeyError."""

    def __init__(self, cell_id: str, roles: dict):
        super().__init__(roles)
        self.cell_id = cell_id

    def __missing__(self, role: str) -> tuple[str, str | None]:
        raise BudgetError(f"cell {self.cell_id}: its profile has no {role!r} role (add it, or a `models:` override)")


class _Measured:
    def __init__(self, entries: Sequence[dict]):
        self.by_key = {(e["arm"], e["model"], e.get("effort"), e["cell"], e.get("delivery", "push")): e for e in entries}

    def find(self, arms: Sequence[str], model: str, effort: str | None, cell: str, delivery: str) -> dict | None:
        for arm in arms:
            if hit := self.by_key.get((arm, model, effort, cell, delivery)):
                return hit
        return None

    def find_family(self, arms: Sequence[str], model: str, effort: str | None, family: str) -> dict | None:
        """Any level of `family` (Study G sessions measured at another N)."""
        for arm in arms:
            for (a, m, eff, cell, _), e in self.by_key.items():
                if (a, m, eff) == (arm, model, effort) and cell.split("-", 1)[0] == family:
                    return e
        return None


def _apply_measured(groups: list[CallGroup], entry: dict, calls_scale: float = 1.0) -> list[CallGroup]:
    roles = entry.get("roles") or {}
    out = []
    for g in groups:
        if r := roles.get(g.role):
            g = replace(g, calls=r["calls_per_sample"] * calls_scale, input=r["input_per_call"], output=r["output_per_call"], cached_fraction=r.get("cached_fraction"))
        out.append(g)
    have = {g.role for g in groups}
    for role, r in roles.items():  # a role the logs show but the priors did not expect
        if role not in have:
            out.append(CallGroup(role, r["model"], r["calls_per_sample"] * calls_scale, r["input_per_call"], r["output_per_call"], "kg", r.get("cached_fraction")))
    return out


def _agent_items(cell: PlanCell, A: dict, roles: dict, measured: _Measured) -> list[LineItem]:
    s = cell.spec
    model, effort = roles["agent"]
    out_tokens = float(s["output"]) if "output" in s else _output(A, effort)
    n_tasks, epochs = _n_tasks(cell), int(s.get("epochs", 1))
    items = []
    for arm, count in _arm_counts(s["arms"]).items():
        a = resolve_arm(A["arms"], arm)
        for task_cell in s["cells"]:
            for delivery in s.get("deliveries", ["push"]):
                k = _calls_per_sample(A, task_cell, delivery)
                ctx = s.get("context", a["context"])
                if ctx == "corpus":
                    if task_cell not in A["corpus_tokens"]:
                        raise BudgetError(f"cell {cell.id}: no corpus_tokens for {task_cell}")
                    ctx = A["corpus_tokens"][task_cell]
                ctx = float(ctx) * (A["pull_context_factor"] if delivery == "pull" else 1.0)
                base = A["base_input_per_call"] + (A.get("tool_definitions_all", {}).get(task_cell, 0) if s.get("exposure") == "all" else 0)
                history = A.get("history_per_prior_call_by_cell", {}).get(task_cell, A["history_per_prior_call"]) * (k - 1) / 2
                mult = float(a["multiplier"])
                groups = [CallGroup("agent", model, k * mult, base + ctx + history, out_tokens, a["cache"])]
                compiles = (1.0 if a["compile"] == "per_query" and delivery == "push" else k) * mult
                if a["kg"]:
                    kc = A["kg_calls"][a["kg"]]
                    groups.append(CallGroup("kg", roles["kg"][0], compiles, kc["input"], kc["output"], "kg"))
                if a["embed"]:
                    groups.append(CallGroup("embeddings", roles["embeddings"][0], compiles, A["query_embedding_tokens"], 0.0, "embeddings"))
                hit = measured.find(a["_chain"], model, effort, task_cell, delivery)
                if hit:
                    groups = _apply_measured(groups, hit)
                items.append(LineItem(cell.id, cell.study, cell.phase, "agent", _tier(model, effort), arm, task_cell, delivery, count * n_tasks * epochs, tuple(groups), bool(hit)))
    return items


def _session_items(cell: PlanCell, A: dict, roles: dict, measured: _Measured, plan: Plan) -> list[LineItem]:
    s, G = cell.spec, A["study_g"]
    model, effort = roles["agent"]
    out_tokens = _output(A, effort)
    window = float(s.get("window", plan.study_g.get("window", 32000)))
    n, sessions, epochs = int(s["N"]), int(s["sessions"]), int(s.get("epochs", 1))
    views = {"full": G["view"]["full"] * window, "managed": G["view"]["managed"] * window, "oracle": float(G["view"]["oracle_tokens"])}
    probes = sum(1 for k in G["probe_checkpoints"] if k <= n)
    items = []
    for arm, count in _arm_counts(s["arms"]).items():
        if arm not in G["arms"]:
            raise BudgetError(f"cell {cell.id}: unknown Study G arm {arm!r} (budget_assumptions study_g.arms)")
        a = G["arms"][arm]
        view, calls = views[a["view"]], n * G["calls_per_item"] * float(a.get("multiplier", 1.0))
        groups = [CallGroup("agent", model, calls, view, out_tokens, a["cache"])]
        if a.get("overhead"):
            groups.append(CallGroup("cm", model, calls * float(a["overhead"]), view, out_tokens, a["cache"]))
        if probes:
            groups.append(CallGroup("probe", model, float(probes), view, out_tokens, a["cache"]))
        task_cell = f"F8-{n}"
        hit, scale = measured.find([arm], model, effort, task_cell, "push"), 1.0
        if hit is None and (hit := measured.find_family([arm], model, effort, "F8")):
            try:
                scale = n / float(hit["cell"].split("-", 1)[1])  # calls scale with items per session
            except ValueError:
                hit = None  # an F8 level that is not an item count: keep the priors
        if hit:
            groups = _apply_measured(groups, hit, scale)
        items.append(LineItem(cell.id, cell.study, cell.phase, "session", _tier(model, effort), arm, task_cell, "push", count * sessions * epochs, tuple(groups), bool(hit)))
    return items


def _build_items(cell: PlanCell, A: dict, roles: dict) -> list[LineItem]:
    s = cell.spec
    model, effort = roles["build_fallback" if s.get("fallback") else "build"]
    systems = s.get("systems", ["apg", "lightrag"])
    calls = {"apg": ("apg_author", "build_apg"), "lightrag": ("lightrag_extract", "build_lightrag")}
    items = []
    for world_cell, count in s["worlds"].items():
        if world_cell not in A["chunks_per_world"]:
            raise BudgetError(f"cell {cell.id}: no chunks_per_world for {world_cell}")
        chunks = float(A["chunks_per_world"][world_cell])
        groups = []
        for system in systems:
            spec_key, cache = calls[system]
            b = A["build_calls"][spec_key]
            groups.append(CallGroup(f"build:{system}", model, chunks * b["calls_per_chunk"], b["input"], b["output"], cache))
        groups.append(CallGroup("embeddings", roles["embeddings"][0], chunks, A["embedding_tokens_per_chunk"], 0.0, "embeddings"))
        items.append(LineItem(cell.id, cell.study, cell.phase, "build", _tier(model, effort), "+".join(systems), world_cell, "-", float(count), tuple(groups)))
    return items


def _anchor_items(cell: PlanCell, A: dict, roles: dict) -> list[LineItem]:
    s, ac = cell.spec, A["anchor_calls"]
    model, effort = roles["agent"]
    items = []
    for mode in s["modes"]:
        groups = [CallGroup("agent", model, ac["answer"]["calls"], ac["answer"]["input"], ac["answer"]["output"], "anchor")]
        if mode != "naive":
            groups.append(CallGroup("agent", model, ac["keywords"]["calls"], ac["keywords"]["input"], ac["keywords"]["output"], "anchor"))
        groups.append(CallGroup("judge", roles["judge"][0], ac["judge"]["calls"], ac["judge"]["input"], ac["judge"]["output"], "judge"))
        if "embeddings" in ac:
            groups.append(CallGroup("embeddings", roles["embeddings"][0], ac["embeddings"]["calls"], ac["embeddings"]["input"], 0.0, "embeddings"))
        items.append(LineItem(cell.id, cell.study, cell.phase, "anchor", _tier(model, effort), f"anchor-{mode}", "graphragbench-medical", mode, float(s["n_tasks"]), tuple(groups)))
    return items


def expand(plan: Plan, assumptions: dict, measured: Sequence[dict] = (), models_path: Path = MODELS_PATH) -> list[LineItem]:
    """Every enabled cell as line items. Disabled cells (cut) contribute nothing."""
    roles_of, m = _Roles(models_path), _Measured(measured)
    items: list[LineItem] = []
    for cell in plan.cells:
        if not cell.enabled:
            continue
        if cell.kind == "fixed":
            items.append(LineItem(cell.id, cell.study, cell.phase, "fixed", "fixed", "-", "-", "-", 1.0, (), fixed_usd=float(cell.spec["usd"])))
            continue
        roles = roles_of(cell)
        if cell.kind == "agent":
            items += _agent_items(cell, assumptions, roles, m)
        elif cell.kind == "session":
            items += _session_items(cell, assumptions, roles, m, plan)
        elif cell.kind == "build":
            items += _build_items(cell, assumptions, roles)
        elif cell.kind == "anchor":
            items += _anchor_items(cell, assumptions, roles)
    return items


# --- Pricing -------------------------------------------------------------------------------------------


def group_cost_per_sample(g: CallGroup, prices: Prices, scenario: dict) -> float:
    """USD per sample for one call group under a cache scenario (see the module docstring)."""
    p = prices.entry(g.model)
    mode = scenario.get("cached_price", "input")
    if mode == "input":
        cached_price = p["input"]
    elif mode == "ratio":
        cached_price = p["input"] * float(scenario["cached_price_ratio"])
    elif mode == "table":
        cached_price = p["input_cache_read"]
    else:
        raise BudgetError(f"cache scenario: cached_price {mode!r} is not one of input, ratio, table")
    share = g.cached_fraction if g.cached_fraction is not None else float((scenario.get("cached_fraction") or {}).get(g.cache, 0.0))
    input_rate = (1.0 - share) * p["input"] + share * cached_price
    return g.calls * (g.input * input_rate + g.output * p["output"]) / 1e6


@dataclass(frozen=True)
class Estimate:
    """Priced line items. Each row: the line item's labels, samples, calls, tokens, and $ per scenario."""

    rows: tuple[dict, ...]

    def total(self, scenario: str = "conservative", **match: str) -> float:
        return sum(r[scenario] for r in self.rows if all(r[k] == v for k, v in match.items()))

    def by(self, *keys: str) -> list[dict]:
        acc: dict[tuple, dict] = {}
        for r in self.rows:
            k = tuple(r[x] for x in keys)
            a = acc.setdefault(k, {**dict(zip(keys, k, strict=True)), "samples": 0.0, "calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "measured": False, **dict.fromkeys(SCENARIOS, 0.0)})
            for f in ("samples", "calls", "input_tokens", "output_tokens", *SCENARIOS):
                a[f] += r[f]
            a["measured"] = a["measured"] or r["measured"]
        return list(acc.values())


def price(items: Iterable[LineItem], prices: Prices, assumptions: dict) -> Estimate:
    scenarios = assumptions["cache_scenarios"]
    rows = []
    for it in items:
        row = {
            "cell": it.cell, "study": it.study, "phase": it.phase, "kind": it.kind, "tier": it.tier, "arm": it.arm,
            "task_cell": it.task_cell, "delivery": it.delivery, "samples": it.samples, "measured": it.measured,
            "calls": it.samples * sum(g.calls for g in it.groups),
            "input_tokens": it.samples * sum(g.calls * g.input for g in it.groups),
            "output_tokens": it.samples * sum(g.calls * g.output for g in it.groups),
        }  # fmt: skip
        for sc in SCENARIOS:
            row[sc] = it.fixed_usd + it.samples * sum(group_cost_per_sample(g, prices, scenarios[sc]) for g in it.groups)
        rows.append(row)
    return Estimate(tuple(rows))


def estimate(
    plan: Plan | None = None,
    assumptions: dict | None = None,
    prices: Prices | None = None,
    measured: Sequence[dict] | None = None,
    *,
    models_path: Path = MODELS_PATH,
) -> Estimate:
    """Price `plan` (default: config/run_plan.yaml) with the assumptions, prices and measurements (defaults: the
    repo's files; `measured=None` reads config/budget_calibration_measured.yaml if it exists)."""
    plan = plan or load_plan()
    assumptions = assumptions or load_assumptions()
    prices = prices or Prices.load()
    measured = load_measured() if measured is None else measured
    return price(expand(plan, assumptions, measured, models_path), prices, assumptions)


def projected_cost(study: str | None = None, phase: str | None = None, cell_ids: Iterable[str] | None = None, scenario: str = "conservative", **kwargs: Any) -> float:
    """$ the plan projects for a study, a phase or a set of cells (FX-6 calls this before each phase)."""
    est = estimate(**kwargs)
    ids = set(cell_ids) if cell_ids is not None else None
    return sum(r[scenario] for r in est.rows if (study is None or r["study"] == study) and (phase is None or r["phase"] == phase) and (ids is None or r["cell"] in ids))


# --- Right-sizing --------------------------------------------------------------------------------------


def right_size(plan: Plan, limit: float | None = None, **kwargs: Any) -> tuple[Plan, list[str]]:
    """From `plan` (normally the uncut plan), apply its cuts in file order while the conservative total
    exceeds `limit` (default: budget.total_usd), stopping as soon as it fits. Returns the plan and the cuts applied."""
    limit = float(plan.budget["total_usd"]) if limit is None else limit
    applied: list[str] = []
    for cut in plan.cuts:
        if cut.get("applied"):
            continue
        if estimate(plan, **kwargs).total() <= limit:
            break
        plan = apply_cut(plan, cut["id"])
        applied.append(cut["id"])
    return plan, applied


def cut_savings(plan: Plan, **kwargs: Any) -> list[dict]:
    """Each cut's conservative saving against `plan`: what reverting an applied cut would add back, or what
    applying an unapplied one would save."""
    base = estimate(plan, **kwargs).total()
    out = []
    for cut in plan.cuts:
        other = revert_cut(plan, cut["id"]) if cut.get("applied") else apply_cut(plan, cut["id"])
        delta = estimate(other, **kwargs).total() - base
        out.append({"id": cut["id"], "applied": bool(cut.get("applied")), "what": cut["what"], "saving": delta if cut.get("applied") else -delta})
    return out


def fallback_build_delta(plan: Plan, **kwargs: Any) -> float:
    """Extra conservative $ if every gate build ran on D-017's fallback builder (build_fallback role)."""
    cells = tuple(replace(c, spec={**c.spec, "fallback": True}) if c.kind == "build" and c.study == "gate" and c.spec["profile"] == "gate" else c for c in plan.cells)
    return estimate(replace(plan, cells=cells), **kwargs).total() - estimate(plan, **kwargs).total()


# --- Calibration from logs -----------------------------------------------------------------------------


def _usage_tokens(u: Any) -> tuple[float, float, float]:
    """(total input incl. cached, output incl. reasoning, cached input) of an Inspect ModelUsage."""
    if u is None:
        return 0.0, 0.0, 0.0
    cached = float(u.input_tokens_cache_read or 0)
    return float(u.input_tokens) + cached + float(u.input_tokens_cache_write or 0), float(u.output_tokens), cached


def _sample_roles(sample: Any) -> dict[str, list[float]]:
    """role -> [calls, input, output, cached] for one sample. Calls are counted from the sample's completed model
    events (role None = the agent); tokens come from Inspect's usage: `role_usage` for each role, and
    `model_usage` minus the roles for the agent. A role without usage falls back to its events' usage."""
    events = [e for e in (sample.events or []) if e.event == "model" and not getattr(e, "pending", False) and e.error is None and e.output is not None]
    calls = Counter(e.role or "agent" for e in events)
    role_usage = dict(sample.role_usage or {})
    totals = [sum(_usage_tokens(u)[i] for u in (sample.model_usage or {}).values()) for i in range(3)]
    out: dict[str, list[float]] = {}
    for role, u in role_usage.items():
        out[role] = [float(calls.get(role, 0)), *_usage_tokens(u)]
    agent = [totals[i] - sum(v[i + 1] for v in out.values()) for i in range(3)]
    out["agent"] = [float(calls.get("agent", 0)), *agent]
    for role in set(calls) - set(out):  # events for a role with no role_usage entry
        tok = [sum(_usage_tokens(e.output.usage)[i] for e in events if (e.role or "agent") == role) for i in range(3)]
        out[role] = [float(calls[role]), *tok]
    return out


def calibrate(log_files: Iterable[str | Path], out_path: Path | None = MEASURED_PATH, *, merge: bool = True) -> dict:
    """Measured calls per sample and tokens per call, per (arm, agent model, effort, cell, delivery) and role,
    from Inspect logs; written to `out_path` (merged with entries already there unless `merge=False`).
    The cost model then prefers these over the priors in config/budget_assumptions.yaml."""
    from inspect_ai.log import read_eval_log

    acc: dict[tuple, dict] = {}
    sources = []
    for f in log_files:
        log = read_eval_log(str(f))
        sources.append(str(f))
        args, meta = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or meta.get("arm")
        if not arm:
            continue  # not an arm run (e.g. the anchor); nothing to key it by
        delivery = args.get("delivery") or meta.get("delivery") or "push"
        gen = log.eval.model_generate_config
        effort = gen.reasoning_effort if gen is not None else None
        role_models = {r: (c[0] if isinstance(c, list) else c).model for r, c in (log.eval.model_roles or {}).items()}
        for s in log.samples or []:
            md = s.metadata or {}
            family, level = md.get("family", args.get("family")), md.get("level", args.get("level"))
            key = (arm, log.eval.model, effort, f"{family}-{level}", delivery)
            entry = acc.setdefault(key, {"samples": 0, "roles": defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])})
            entry["samples"] += 1
            for role, vals in _sample_roles(s).items():
                tot = entry["roles"][role]
                for i, v in enumerate(vals):
                    tot[i] += v
            entry["role_models"] = role_models
    entries = []
    for (arm, model, effort, cell, delivery), e in acc.items():
        roles = {}
        for role, (calls, inp, out, cached) in sorted(e["roles"].items(), key=lambda kv: (kv[0] != "agent", kv[0])):
            if calls <= 0:
                continue
            roles[role] = {
                "model": model if role == "agent" else e["role_models"].get(role, model),
                "calls_per_sample": round(calls / e["samples"], 4),
                "input_per_call": round(inp / calls, 2),
                "output_per_call": round(out / calls, 2),
                "cached_fraction": round(cached / inp, 4) if inp else 0.0,
            }
        entries.append({"arm": arm, "model": model, "effort": effort, "cell": cell, "delivery": delivery, "samples": e["samples"], "roles": roles})
    if merge and out_path is not None:
        new_keys = {(e["arm"], e["model"], e["effort"], e["cell"], e["delivery"]) for e in entries}
        kept = [e for e in load_measured(out_path) if (e["arm"], e["model"], e.get("effort"), e["cell"], e.get("delivery", "push")) not in new_keys]
        entries = kept + entries
    result = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "logs": sources, "entries": entries}
    if out_path is not None:
        header = (
            "# Measured per-call tokens and calls per sample (FX-5), written by `python -m ape.budget calibrate`.\n"
            "# The cost model prefers these over config/budget_assumptions.yaml per (arm, model, effort, cell, delivery).\n"
            "# input_per_call includes cached input; output_per_call includes reasoning tokens.\n"
        )
        Path(out_path).write_text(header + yaml.safe_dump(result, sort_keys=False))
    return result


# --- Spend so far (for the orchestrator) ---------------------------------------------------------------


# Inspect spend is read per sample, from each log's sample summaries, and each sample is counted once by its uuid.
# - **Retries.** A retried task's new log copies the completed samples of the failed one, uuid and usage
#   included, and its header totals then count only part of the work. The runner keeps failed attempts' logs
#   (`retry_cleanup=False`), so their errored samples' spend is not lost; deduplicating by uuid counts every
#   sample exactly once.
# - **Killed runs.** Inspect flushes sample summaries while a run is in progress (every `log_buffer` samples,
#   10 by default), so a killed run's `started` log, whose header has no usage at all, still yields the spend of
#   every flushed sample. Unflushed and in-flight samples (at most `log_buffer` + `max_samples`) are not visible.
_LOG_SAMPLES: dict[str, tuple[tuple[int, int], dict]] = {}


def log_samples(path: str | Path) -> dict:
    """One `.eval` log's samples as {"status", "samples": [(uuid key, usd, unpriced models)]}, finished or not;
    cached on (mtime, size). An unpriced call of a mock model counts as $0."""
    from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

    p = Path(path)
    st = p.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    if (hit := _LOG_SAMPLES.get(str(p))) and hit[0] == stamp:
        return hit[1]
    samples = []
    for s in read_eval_log_sample_summaries(str(p)):
        usd, unpriced = 0.0, set()
        for model, u in (s.model_usage or {}).items():
            if u.total_cost is None:
                if not model.startswith("mockllm/") and (u.input_tokens or u.output_tokens):
                    unpriced.add(model)
            else:
                usd += u.total_cost
        samples.append((s.uuid or f"{p}#{s.id}#{s.epoch}", usd, frozenset(unpriced)))
    info = {"status": read_eval_log(str(p), header_only=True).status, "samples": samples}
    _LOG_SAMPLES[str(p)] = (stamp, info)
    return info


def logs_spend(log_files: Iterable[str | Path], seen: set[str] | None = None) -> dict:
    """Inspect $ over `log_files`, each sample once (by uuid; `seen` carries uuids across calls). Logs that cannot
    be read (e.g. mid-write) are listed, not counted; an unpriced non-mock call is an error, never $0."""
    seen = set() if seen is None else seen
    usd, samples, partial, unreadable, unpriced = 0.0, 0, 0, [], set()
    for f in log_files:
        try:
            info = log_samples(f)
        except Exception as e:  # noqa: BLE001  (a zip being rewritten, a truncated file: report, don't crash the guard)
            unreadable.append(f"{f}: {type(e).__name__}: {str(e)[:120]}")
            continue
        partial += info["status"] == "started"
        for key, cost, models in info["samples"]:
            unpriced |= models
            if key not in seen:
                seen.add(key)
                usd += cost
                samples += 1
    if unpriced:
        raise BudgetError(f"logs carry unpriced calls for {sorted(unpriced)}; run with ape.models.eval_cost_kwargs()")
    return {"inspect_usd": usd, "samples": samples, "partial_logs": partial, "unreadable": unreadable}


def ledger_spend(ledger_path: str | Path | None, costs_path: Path = COSTS_PATH) -> float:
    """$ of every entry in one build/embedding ledger, priced from the table."""
    from .analysis.cost import load_prices, price_entry
    from .llm.ledger import Ledger

    if ledger_path is None or not Path(ledger_path).is_file():
        return 0.0
    prices = load_prices(costs_path)
    try:
        return float(sum(price_entry(e, prices) for e in Ledger(ledger_path).read()))
    except KeyError as e:
        raise BudgetError(f"ledger {ledger_path}: {e}") from None


def spent(log_files: Iterable[str | Path] = (), ledger_path: Path | None = None, costs_path: Path = COSTS_PATH) -> dict:
    """$ already spent by these logs (each sample once; `logs_spend`) plus one build/embedding ledger, both priced
    from the one table. Pass every log of the tasks, failed attempts included: a sample shared with a retry's
    log is counted once."""
    logs = logs_spend(log_files)
    ledger_usd = ledger_spend(ledger_path, costs_path)
    return {"inspect_usd": logs["inspect_usd"], "ledger_usd": ledger_usd, "spent_usd": logs["inspect_usd"] + ledger_usd, "partial_logs": logs["partial_logs"], "unreadable": logs["unreadable"]}


def remaining(budget_usd: float, log_files: Iterable[str | Path] = (), ledger_path: Path | None = None, costs_path: Path = COSTS_PATH) -> dict:
    """What is left of `budget_usd` after the logs' and the ledger's spend."""
    s = spent(log_files, ledger_path, costs_path)
    return {"budget_usd": float(budget_usd), **s, "remaining_usd": float(budget_usd) - s["spent_usd"]}


def program_spend(registry: Path | None = None, costs_path: Path = COSTS_PATH) -> dict:
    """Spend over the whole program registry (`ape.spend`): every `.eval` log under every registered log dir
    (finished, failed or killed; each sample once, attributed to the first dir that logged it) plus every
    registered ledger. Missing dirs and files count $0 and are listed."""
    from . import spend as reg

    registry = registry if registry is not None else reg.registry_path(live=True)
    dirs, ledgers = reg.registered(registry)
    seen: set[str] = set()
    rows, by_study, missing, unreadable, partial = [], defaultdict(float), [], [], 0
    for d, info in dirs.items():
        path = Path(d)
        if not path.is_dir():
            missing.append(d)
            continue
        s = logs_spend(sorted(path.rglob("*.eval")), seen)
        partial += s["partial_logs"]
        unreadable += s["unreadable"]
        rows.append({"log_dir": d, "label": info["label"], "study": info["study"], "live": info["live"], "usd": s["inspect_usd"], "samples": s["samples"], "partial_logs": s["partial_logs"], "finished": info["finishes"] >= info["starts"]})
        by_study[info["study"]] += s["inspect_usd"]
    ledger_rows = []
    for f, info in ledgers.items():
        if not Path(f).is_file():
            missing.append(f)
            continue
        usd = ledger_spend(f, costs_path)
        ledger_rows.append({"path": f, "label": info["label"], "study": info["study"], "usd": usd})
        by_study[info["study"]] += usd
    inspect_usd = sum(r["usd"] for r in rows)
    ledger_usd = sum(r["usd"] for r in ledger_rows)
    return {
        "registry": str(registry) if registry is not None else None,
        "inspect_usd": inspect_usd,
        "ledger_usd": ledger_usd,
        "spent_usd": inspect_usd + ledger_usd,
        "by_study": dict(sorted(by_study.items())),
        "dirs": rows,
        "ledgers": ledger_rows,
        "partial_logs": partial,
        "unfinished_dirs": [r["log_dir"] for r in rows if not r["finished"]],
        "missing": missing,
        "unreadable": unreadable,
    }


def program_remaining(budget_usd: float, registry: Path | None = None, costs_path: Path = COSTS_PATH) -> dict:
    """What is left of `budget_usd` after the whole program's spend (`program_spend`)."""
    s = program_spend(registry, costs_path)
    return {"budget_usd": float(budget_usd), **s, "remaining_usd": float(budget_usd) - s["spent_usd"]}


def sample_cost_limit(projected_sample_usd: float | None, plan: Plan | None = None) -> float:
    """Inspect's per-sample `cost_limit`, a runaway guard: `multiple` x the cost model's conservative per-sample
    projection, at least `floor_usd`; `default_usd` when no projection is given (run_plan.yaml
    budget.sample_cost_limit)."""
    cfg = ((plan or load_plan()).budget or {}).get("sample_cost_limit") or {}
    multiple, floor, default = float(cfg.get("multiple", 20)), float(cfg.get("floor_usd", 0.5)), float(cfg.get("default_usd", 2.0))
    if projected_sample_usd is None or projected_sample_usd <= 0:
        return default
    return max(floor, multiple * float(projected_sample_usd))


def require_affordable(projected_usd: float, remaining_usd: float, what: str = "phase") -> None:
    """Refuse (BudgetError) a phase whose projected cost exceeds what is left."""
    if projected_usd > remaining_usd:
        raise BudgetError(f"{what}: projected ${projected_usd:,.2f} exceeds the remaining ${remaining_usd:,.2f}; recalibrate (`python -m ape.budget calibrate`) or cut the plan")


# --- CLI -----------------------------------------------------------------------------------------------


def _usd(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def _table(rows: list[dict], cols: list[tuple[str, str]], title: str) -> str:
    def fmt(r: dict, key: str) -> str:
        v = r.get(key, "")
        if key in SCENARIOS:
            return _usd(v)
        if key in ("input_tokens", "output_tokens"):
            return f"{v / 1e6:,.1f}"
        if key in ("samples", "calls"):
            return f"{v:,.0f}"
        if key == "measured":
            return "m" if v else ""
        return str(v)

    body = [[fmt(r, k) for k, _ in cols] for r in rows]
    heads = [h for _, h in cols]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(heads)]
    right = {i for i, (k, _) in enumerate(cols) if k in (*SCENARIOS, "samples", "calls", "input_tokens", "output_tokens")}

    def line(vals: list[str]) -> str:
        return "  ".join(v.rjust(w) if i in right else v.ljust(w) for i, (v, w) in enumerate(zip(vals, widths, strict=True)))

    return "\n".join([title, line(heads), line(["-" * w for w in widths]), *(line(b) for b in body)])


def _sort(rows: list[dict], key: str) -> list[dict]:
    order = {s: i for i, s in enumerate(STUDY_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r.get("study"), 99), r.get(key, "")) if key != "study" else order.get(r["study"], 99))


def spend_report(s: dict, budget_usd: float) -> str:
    """`python -m ape.budget spend`: program spend by study, then by registered log dir and ledger."""
    lines = [
        f"Program spend (registry {s['registry']}): ${s['spent_usd']:,.2f} of ${budget_usd:,.0f} "
        f"(Inspect ${s['inspect_usd']:,.2f}, ledgers ${s['ledger_usd']:,.2f}); ${budget_usd - s['spent_usd']:,.2f} left",
        "",
    ]
    lines.append(_table([{"study": k, "usd": f"{v:,.2f}"} for k, v in s["by_study"].items()], [("study", "study"), ("usd", "$")], "By study") if s["by_study"] else "By study: nothing registered")
    if s["dirs"]:
        rows = [{"label": r["label"], "n": f"{r['samples']:,}", "usd": f"{r['usd']:,.2f}", "state": ("live" if r["live"] else "offline") + ("" if r["finished"] else ", unfinished") + (f", {r['partial_logs']} partial log(s)" if r["partial_logs"] else ""), "log_dir": r["log_dir"]} for r in s["dirs"]]
        lines += ["", _table(rows, [("label", "label"), ("n", "samples"), ("usd", "$"), ("state", "state"), ("log_dir", "log dir")], "By log dir")]
    if s["ledgers"]:
        rows = [{"label": r["label"], "usd": f"{r['usd']:,.2f}", "path": r["path"]} for r in s["ledgers"]]
        lines += ["", _table(rows, [("label", "label"), ("usd", "$"), ("path", "ledger")], "Ledgers")]
    for key, what in (("missing", "registered but missing"), ("unreadable", "unreadable (not counted)")):
        if s[key]:
            lines += ["", f"{what}:", *(f"  {x}" for x in s[key])]
    return "\n".join(lines)


def report(plan: Plan, est: Estimate, *, scenario: str = "conservative", study: str | None = None, detail: bool = False, **kwargs: Any) -> tuple[str, bool]:
    """The printed report and whether the conservative total fits `budget.total_usd`."""
    rows = [r for r in est.rows if study is None or r["study"] == study]
    sub = Estimate(tuple(rows))
    money = [("conservative", "$ cons"), ("expected", "$ exp")]
    stats = [("samples", "samples"), ("calls", "calls"), ("input_tokens", "Mtok in"), ("output_tokens", "Mtok out")]
    out = []
    order = {c.id: i for i, c in enumerate(plan.cells)}
    by_cell = sorted(sub.by("study", "phase", "cell", "tier"), key=lambda r: order[r["cell"]])
    out.append(_table(by_cell, [("study", "study"), ("phase", "phase"), ("cell", "cell"), ("tier", "tier"), *stats, ("measured", "m"), *money], "Per cell"))
    if detail:
        out.append(_table(rows, [("cell", "cell"), ("arm", "arm"), ("task_cell", "task cell"), ("delivery", "mode"), ("tier", "tier"), *stats, ("measured", "m"), *money], "\nPer line item"))
    phase_order = {(c.study, c.phase): order[c.id] for c in reversed(plan.cells)}  # first cell of each phase
    phases = sorted(sub.by("study", "phase"), key=lambda r: phase_order[(r["study"], r["phase"])])
    out.append(_table(phases, [("study", "study"), ("phase", "phase"), *money], "\nPer phase"))
    out.append(_table(_sort(sub.by("study", "tier"), "tier"), [("study", "study"), ("tier", "tier"), *money], "\nPer study and tier"))
    out.append(_table(sorted(sub.by("tier"), key=lambda r: -r["conservative"]), [("tier", "tier"), *money], "\nPer tier"))
    out.append(_table(_sort(sub.by("study"), "study"), [("study", "study"), *money], "\nPer study"))
    if study is None:
        out.append(f"\nTotal: ${_usd(est.total('conservative'))} conservative, ${_usd(est.total('expected'))} expected")
    # Allocation check (whole program, whatever --study shows).
    budget = plan.budget
    limit = float(budget.get("total_usd", float("inf")))
    total = est.total(scenario)
    out.append(f"\nAllocation check ({scenario}):")
    for name, cap in (budget.get("allocations") or {}).items():
        have = est.total(scenario, study=name)
        out.append(f"  {name:<12} ${_usd(have):>8} of ${_usd(cap):>6}  {'ok' if have <= cap else 'OVER target'}")
    left = limit - total
    target = budget.get("contingency_usd")
    if target is not None:
        out.append(f"  {'contingency':<12} ${_usd(left):>8} left of ${_usd(limit)} (target ~${_usd(target)}){'' if left >= target else '  BELOW target'}")
    cons = est.total("conservative")
    fits = cons <= limit
    out.append(f"  total        ${_usd(total):>8} of ${_usd(limit)}  ({scenario})")
    out.append(f"  conservative total ${_usd(cons)} {'<=' if fits else '>'} ${_usd(limit)}: {'FITS' if fits else 'EXCEEDS BUDGET'}")
    if plan.cuts:
        out.append("\nRight-sizing cuts (in order; saving = conservative $ against this plan):")
        for c in cut_savings(plan, **kwargs):
            out.append(f"  {c['id']}  {'applied' if c['applied'] else 'not applied':<11}  ${_usd(c['saving']):>7}  {c['what']}")
    out.append(f"\nD-017 fallback builder (build_fallback role) for every gate build: +${_usd(fallback_build_delta(plan, **kwargs))} conservative; needs approval.")
    measured = sum(1 for r in est.rows if r["measured"])
    out.append(f"Measured calibration used for {measured} of {len(est.rows)} line items ('m'); the rest use priors.")
    return "\n".join(out), fits


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "calibrate":
        ap = argparse.ArgumentParser(prog="python -m ape.budget calibrate")
        ap.add_argument("logs", nargs="+")
        ap.add_argument("--out", type=Path, default=MEASURED_PATH)
        ap.add_argument("--replace", action="store_true", help="drop entries already in --out")
        a = ap.parse_args(argv[1:])
        res = calibrate(a.logs, a.out, merge=not a.replace)
        print(f"wrote {len(res['entries'])} measured entries to {a.out}")
        return 0
    if argv and argv[0] == "spend":
        ap = argparse.ArgumentParser(prog="python -m ape.budget spend", description="Program-wide spend so far, from the spend registry (ape.spend).")
        ap.add_argument("--registry", type=Path, default=None, help="default: $APE_SPEND_REGISTRY, else cache/spend/registry.jsonl")
        ap.add_argument("--costs", type=Path, default=COSTS_PATH)
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(argv[1:])
        s = program_spend(a.registry, a.costs)
        if a.json:
            print(json.dumps(s, indent=1, default=str))
            return 0
        print(spend_report(s, float(load_plan().budget["total_usd"])))
        return 0
    if argv and argv[0] == "remaining":
        ap = argparse.ArgumentParser(prog="python -m ape.budget remaining")
        ap.add_argument("logs", nargs="*")
        ap.add_argument("--budget", type=float, required=True)
        ap.add_argument("--ledger", type=Path, default=None)
        ap.add_argument("--costs", type=Path, default=COSTS_PATH)
        a = ap.parse_args(argv[1:])
        print(yaml.safe_dump(remaining(a.budget, a.logs, a.ledger, a.costs), sort_keys=False).strip())
        return 0
    ap = argparse.ArgumentParser(prog="python -m ape.budget", description="Program cost model (FX-5). Subcommands: calibrate, remaining.")
    ap.add_argument("--scenario", choices=SCENARIOS, default="conservative", help="scenario for the allocation check (both columns are always shown)")
    ap.add_argument("--study", choices=STUDY_ORDER, help="show only this study's rows (the budget check still covers the whole plan)")
    ap.add_argument("--detail", action="store_true", help="also print every line item (arm x task cell x mode)")
    ap.add_argument("--uncut", action="store_true", help="price the plan before the right-sizing cuts")
    ap.add_argument("--plan", type=Path, default=PLAN_PATH)
    ap.add_argument("--assumptions", type=Path, default=ASSUMPTIONS_PATH)
    ap.add_argument("--measured", type=Path, default=MEASURED_PATH)
    ap.add_argument("--costs", type=Path, default=COSTS_PATH)
    ap.add_argument("--models", type=Path, default=MODELS_PATH)
    a = ap.parse_args(argv)
    plan = load_plan(a.plan, uncut_plan=a.uncut)
    kwargs = {"assumptions": load_assumptions(a.assumptions), "prices": Prices.load(a.costs), "measured": load_measured(a.measured), "models_path": a.models}
    est = estimate(plan, **kwargs)
    cuts = ", ".join(plan.applied_cuts) or "none"
    print(f"FX-5 cost model. plan {a.plan} (cuts applied: {cuts}{', reverted by --uncut' if a.uncut else ''}); prices {a.costs}\n")
    text, fits = report(plan, est, scenario=a.scenario, study=a.study, detail=a.detail, **kwargs)
    print(text)
    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
