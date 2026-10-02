"""Inspect task for the GraphRAG-Bench Medical anchor (gate precondition PC1).

    python -m ape.anchor.graphragbench build                  # once
    python -m ape.anchor.graphragbench run --mode hybrid      # agent + judge models from config/models.yaml
    # repeat with --mode mix and --mode naive, then:
    python -m ape.anchor.graphragbench pc1 <hybrid-or-mix log> <naive log>

`run` builds the answer (agent) and judge models with their profile efforts. With bare
`inspect eval`, pass the efforts yourself (`--reasoning-effort`, and the judge as
`--model-role 'judge={model: ..., reasoning_effort: ...}'`), or they run at the model default.

Answers come from LightRAG's own `aquery_llm`, using the benchmark's system prompt and the
paper's query settings. Every LightRAG model call goes through Inspect's default model, so
each one is metered in the sample that made it: the keyword extraction and the answer in
graph modes, the answer alone in naive mode.

Scoring (D-025), both judged by the "judge" role:
- The gating score follows each published number's provenance.
  - Fact Retrieval, Complex Reasoning and Contextual Summarize use the vendored-RAGAS scorer
    (`ape.anchor.ragas`).
  - Creative Generation uses today's official prompts with a tolerant classification parse.
- Today's official scorer, parsed strictly, runs on every sample as the reported sensitivity.
  Creative Generation shares its judge calls.
- Parse failures are counted per type and scorer; PC1 is not evaluable above 5%.
"""

import asyncio
import statistics
from collections import defaultdict

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, ResponseSchema, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store
from lightrag import LightRAG, QueryParam

from ape.anchor import accuracy, ragas
from ape.anchor.accuracy import Excluded
from ape.anchor.bge import REPO, REVISION, anchor_embedder
from ape.anchor.graphragbench import ANCHOR_ID, RAGAS_TYPES, default_data_dir, index_manifest, load_medical, open_anchor_rag, slug, working_dir
from ape.config import Config
from ape.lgr.adapter import KEYWORDS_SCHEMA

MODES = ("naive", "local", "global", "hybrid", "mix")
# Examples/run_lightrag.py, verbatim. LightRAG 1.5.7 fills custom system prompts with
# {context_data, response_type, user_prompt} only. v1.2.5 filled {history} with "" when
# there was no conversation, so that substitution is made here instead. See `system_prompt_for`
# for naive mode.
SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
""".replace("{history}", "")
# The paper's App. H.2 LightRAG config is LightRAG 1.2.5's defaults: 4000-token caps on entity
# descriptions, relation descriptions and chunk text. 1.2.5 applies each cap *per side* in hybrid mode,
# local (entities first) and global (relations first), and then concatenates and de-duplicates the two
# sides (operate.py `_build_query_context` / `combine_contexts`). So one hybrid context holds up to:
# - 8000 description tokens of entities;
# - 8000 of relations;
# - 2 × 3 chunks of chunk text (3 × 1200 tokens fit a 4000 cap).
# 1.5.7 instead truncates the merged entity and relation lists once each, measured on their JSON records,
# and gives chunks what is left of `max_total_tokens` after the system prompt, query and a 200-token buffer.
# Hence:
# - entities: 8000 + ~1000 of JSON-record overhead (≈ 60 records × ~17 tokens of keys, names and types);
# - relations: 8000 + ~1500 (≈ 60 records × ~25);
# - chunks: chunk_top_k 6;
# - total: big enough that 6 × 1200 chunk tokens fit after the KG parts.
# Every value is explicit, so LightRAG's env-var defaults cannot move them (D-009).
QUERY_PARAMS = {"chunk_top_k": 6, "max_entity_tokens": 9000, "max_relation_tokens": 9500, "max_total_tokens": 26500, "enable_rerank": False}
# Sampling settings live on the model roles (config/models.yaml, profile "anchor"): the official
# judge runs at temperature 0, top_p 1, seed 42 (generation_eval.py) and answers at temperature 0.7.
# Nothing is hard-coded here, so the "anchor_luna" fallback can omit parameters GPT-6 may reject.

_rags: dict[tuple, LightRAG] = {}
_locks: dict[tuple, asyncio.Lock] = {}


def system_prompt_for(mode: str) -> str:
    """Graph modes fill the knowledge base into {context_data}; naive mode fills {content_data}.
    The paper never ran LightRAG's naive mode, so its prompt only has the former."""
    return SYSTEM_PROMPT.replace("{context_data}", "{content_data}") if mode == "naive" else SYSTEM_PROMPT


def answer_llm_func(temperature: float | None):
    """LightRAG `llm_model_func` routed through Inspect's default model.

    The model is resolved on every call. LightRAG runs each call in a copy of the caller's
    context, so the call is metered in the sample that made it.
    """

    async def llm(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
        messages = ([ChatMessageSystem(content=system_prompt)] if system_prompt else []) + [ChatMessageUser(content=prompt)]
        schema = None
        if kwargs.get("response_format") or kwargs.get("keyword_extraction"):
            schema = ResponseSchema(name="keywords", json_schema=KEYWORDS_SCHEMA, strict=True)
        config = GenerateConfig(temperature=temperature, response_schema=schema)  # temperature None: the role's own setting
        return (await get_model().generate(messages, config=config)).completion

    return llm


async def _rag(cfg: Config, temperature: float | None) -> LightRAG:
    """One query-time LightRAG per (index, temperature, event loop), because LightRAG binds its locks
    to the loop that created it. The key holds the loop itself, so a dead loop's id is never reused."""
    wd = working_dir(cfg)
    key = (str(wd), temperature, asyncio.get_running_loop())
    async with _locks.setdefault(key, asyncio.Lock()):
        if key not in _rags:
            _rags[key] = await open_anchor_rag(wd, answer_llm_func(temperature), anchor_embedder(cfg), query_time=True)
    return _rags[key]


@solver
def lightrag_answer(cfg: Config, mode: str, top_k: int, temperature: float | None) -> Solver:
    prompt = system_prompt_for(mode)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        rag = await _rag(cfg, temperature)
        result = await rag.aquery_llm(state.input_text, QueryParam(mode=mode, top_k=top_k, **QUERY_PARAMS), system_prompt=prompt)
        response = result.get("llm_response", {})
        if response.get("content") is None:  # aquery_llm swallows exceptions into this
            raise RuntimeError(f"LightRAG query failed: {result.get('message')}")
        data = result.get("data", {})
        store().set(
            "lightrag",
            {
                "status": result.get("status"),
                "llm_generated": response.get("llm_generated"),
                "counts": [len(data.get(k, [])) for k in ("entities", "relationships", "chunks")],
            },
        )
        state.output = ModelOutput.from_content(str(get_model()), response["content"])
        state.messages.append(state.output.message)
        return state

    return solve


@metric
def accuracy_by_type() -> Metric:
    """The official aggregation, mean ACC per question type with failed samples dropped, for both scorers.

    Reported in percent, to match Table 2's units:
    - `<type>`: the gating scorer;
    - `current_<type>`: today's official scorer, strict;
    - `excluded`: gating-scorer samples dropped.

    Per type, as rates in [0, 1]:
    - `matched_parse_failure_rate_<type>` and `current_parse_failure_rate_<type>`: judge replies the scorer
      could not read;
    - `matched_fix_format_rate_<type>`: samples where RAGAS needed its fix-format re-ask.
    """

    def compute(scores: list[SampleScore]) -> Value:
        per: dict[str, dict] = defaultdict(lambda: {"n": 0, "m": [], "c": [], "m_failed": 0, "c_failed": 0, "fixed": 0})
        excluded = 0
        for s in scores:
            md = s.score.metadata or {}
            r = per[s.sample_metadata["question_type"]]
            r["n"] += 1
            matched, current = md.get("matched") or {}, md.get("current") or {}
            if md.get("excluded"):
                excluded += 1
            else:
                r["m"].append(s.score.as_float())
            r["m_failed"] += bool(matched.get("parse_failed"))
            r["fixed"] += bool(matched.get("fix_format_calls"))
            if current.get("accuracy") is not None:
                r["c"].append(current["accuracy"])
            r["c_failed"] += bool(current.get("classification_parse_failed"))
        out: dict[str, float] = {"excluded": excluded}
        for t, r in per.items():
            if r["m"]:
                out[t] = 100 * statistics.fmean(r["m"])
            if r["c"]:
                out[f"current_{slug(t)}"] = 100 * statistics.fmean(r["c"])
            out[f"matched_parse_failure_rate_{slug(t)}"] = r["m_failed"] / r["n"]
            out[f"current_parse_failure_rate_{slug(t)}"] = r["c_failed"] / r["n"]
            out[f"matched_fix_format_rate_{slug(t)}"] = r["fixed"] / r["n"]
        return out

    return compute


async def score_both(question: str, answer: str, ground_truth: str, qtype: str, generate, emb) -> tuple[dict, dict]:
    """(gating, sensitivity) results for one answer. Each is a dict with `accuracy` (None when the scorer could not
    score it), `parse_failed`/`classification_parse_failed` and the scorer's parts."""

    async def official() -> dict:
        try:
            r = await accuracy.answer_correctness(question, answer, ground_truth, generate, emb.queries)
        except Excluded as e:
            return {"accuracy": None, "excluded": True, "classification_parse_failed": False, "error": str(e)}
        return {**r, "excluded": False}

    async def vendored_ragas() -> dict:
        try:
            r = await ragas.answer_correctness(question, answer, ground_truth, generate, emb.documents)
        except ragas.ScorerFailed as e:
            trace = getattr(e, "trace", [])
            return {"accuracy": None, "parse_failed": True, "error": str(e), "judge_calls": len(trace), "fix_format_calls": trace.count("StringIO")}
        return {**r, "parse_failed": False}

    if qtype in RAGAS_TYPES:
        current, matched = await asyncio.gather(official(), vendored_ragas())
        matched["scorer"] = "ragas-e6305f5"
        return matched, current
    current = await official()
    if current["excluded"]:  # an ill-formed classification is ill-formed for the tolerant parse too
        matched = {"accuracy": None, "parse_failed": True, "error": current["error"]}
    else:
        try:
            matched = {**accuracy.tolerant(current), "parse_failed": False}
        except ragas.ScorerFailed as e:
            matched = {"accuracy": None, "parse_failed": True, "error": str(e)}
    matched |= {"scorer": "official-tolerant", "judge_calls": 0, "fix_format_calls": 0}  # shares the official calls
    return matched, current


@scorer(metrics=[accuracy_by_type()])
def answer_accuracy(cfg: Config) -> Scorer:
    emb = anchor_embedder(cfg)

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        judge = get_model(role="judge", required=True)

        async def generate(prompt: str) -> str:
            return (await judge.generate([ChatMessageUser(content=prompt)])).completion

        matched, current = await score_both(state.input_text, answer, target.text, state.metadata["question_type"], generate, emb)
        current.pop("classification_raw", None)
        if matched["accuracy"] is None:
            return Score(value=0.0, answer=answer, explanation=matched.get("error"), metadata={"excluded": True, "matched": matched, "current": current})
        return Score(value=matched["accuracy"], answer=answer, metadata={"excluded": False, "matched": matched, "current": current})

    return score


@task
def graphragbench_anchor(
    mode: str = "hybrid",
    n_per_type: int = 200,
    seed: int = 0,
    data: str | None = None,
    top_k: int = 30,
    temperature: float | None = None,
) -> Task:
    """`mode` = hybrid and `top_k` = 30 follow the paper (App. H.2); the repo's run script defaults to top_k 5.
    `temperature` None uses the answer model's own setting (profile "anchor": 0.7, the paper's). `n_per_type` = 200 (all 166 Creative Generation questions) keeps per-type sampling error
    near 2 pp; at 50 the ±5 pp PC1 tolerance would fail a faithful reproduction more often than not."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cfg = Config()
    bench = load_medical(data or default_data_dir(cfg), n_per_type, seed)
    emb = anchor_embedder(cfg)
    bge_id = emb.identity.split("#")[0] if emb.identity != "fake-bow" else "fake-bow"
    samples = [
        Sample(id=q["id"], input=q["question"], target=q["answer"], metadata={"question_type": q["question_type"], "evidence": q["evidence"]})
        for q in bench.questions
    ]
    return Task(
        dataset=MemoryDataset(samples, name=f"{ANCHOR_ID}-{n_per_type}pt-s{seed}"),
        solver=lightrag_answer(cfg, mode, top_k, temperature),
        scorer=answer_accuracy(cfg),
        metadata={
            "anchor": ANCHOR_ID,
            "mode": mode,
            "query": {"top_k": top_k, **QUERY_PARAMS},
            "temperature": temperature,
            "index": index_manifest(cfg, bench),
            "question_ids_hash": bench.questions_hash(),
            "scorers": {
                "gating": {t: ("ragas-e6305f5" if t in RAGAS_TYPES else "official-fdbab59, tolerant classification parse") for t in (*RAGAS_TYPES, "Creative Generation")},
                "sensitivity": "official-fdbab59, strict classification parse",
            },
            "embeddings": {
                "model": f"{REPO}@{REVISION}" if bge_id != "fake-bow" else bge_id,
                "retrieval": emb.identity,
                "ragas_similarity": "documents: CLS, normalised, raw cosine",
                "official_similarity": "queries (bge query instruction): CLS, normalised, (cos+1)/2",
            },
        },
    )
