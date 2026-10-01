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
graph modes, the answer alone in naive mode. The scorer is the official ACC, judged by the
"judge" role.
"""

import asyncio
import statistics

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, ResponseSchema, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store
from lightrag import LightRAG, QueryParam

from ape.anchor.accuracy import Excluded, answer_correctness
from ape.anchor.graphragbench import ANCHOR_ID, default_data_dir, index_manifest, load_medical, open_anchor_rag, working_dir
from ape.config import Config, embedding_cache
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
# The paper's App. H.2 LightRAG config (3 x 4000-token context caps), mapped onto 1.5.7's QueryParam.
# Entities and relations keep their caps; chunks get the rest of a 12000-token total. Every value is
# explicit, so LightRAG's env-var defaults cannot move them.
QUERY_PARAMS = {"chunk_top_k": 20, "max_entity_tokens": 4000, "max_relation_tokens": 4000, "max_total_tokens": 12000, "enable_rerank": False}
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
            _rags[key] = await open_anchor_rag(wd, answer_llm_func(temperature), embedding_cache(cfg), query_time=True)
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
    """The official aggregation: mean ACC per question type, excluded samples dropped.

    Reported in percent, to match Table 2's units. Also reports the excluded count.
    """

    def compute(scores: list[SampleScore]) -> Value:
        by_type: dict[str, list[float]] = {}
        excluded = 0
        for s in scores:
            if (s.score.metadata or {}).get("excluded"):
                excluded += 1
            else:
                by_type.setdefault(s.sample_metadata["question_type"], []).append(s.score.as_float())
        return {**{t: 100 * statistics.fmean(v) for t, v in by_type.items()}, "excluded": excluded}

    return compute


@scorer(metrics=[accuracy_by_type()])
def answer_accuracy(cfg: Config) -> Scorer:
    emb = embedding_cache(cfg)

    async def embed(texts: list[str]) -> list[list[float]]:
        return await emb.embed(texts, context={"anchor": ANCHOR_ID, "system": "judge"})

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        if not answer.strip():  # embedding APIs reject empty input; an empty answer earns nothing
            return Score(value=0.0, answer=answer, explanation="blank answer", metadata={"excluded": False})
        judge = get_model(role="judge", required=True)

        async def generate(prompt: str) -> str:
            return (await judge.generate([ChatMessageUser(content=prompt)])).completion

        try:
            r = await answer_correctness(state.input_text, answer, target.text, generate, embed)
        except Excluded as e:
            return Score(value=0.0, answer=answer, explanation=str(e), metadata={"excluded": True})
        return Score(value=r.pop("accuracy"), answer=answer, metadata={"excluded": False, **r})

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
        },
    )
