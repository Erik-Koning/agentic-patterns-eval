"""GraphRAG-Bench Medical anchor (gate precondition PC1): data, LightRAG index, pre-registered check.

Before comparing APG against LightRAG, we must show that our LightRAG setup reproduces
LightRAG's published GraphRAG-Bench scores: arXiv 2506.05690v3 Table 2, Medical rows.
Our setup is pinned lightrag-hku, the paper's models where available, rerank off, and the
paper's embedding model, BAAI/bge-large-en-v1.5, run locally (`ape.anchor.bge`; D-009, D-025).

Scoring follows each published number's provenance (D-025):
- Fact Retrieval, Complex Reasoning and Contextual Summarize gate on the vendored-RAGAS scorer that
  produced them (`ape.anchor.ragas`).
- Creative Generation gates on today's official code with a tolerant classification parse
  (`ape.anchor.accuracy.tolerant`).
- Today's official scorer, run strictly, is reported for every type as a sensitivity.

The PC1 rule (`pc1_check`, GATE_PREREG §7):
- The macro mean over the four types must land within ±5 pp of the published macro, and each type
  within ±10 pp, when the paper's model (gpt-4o-mini) answers, judges and builds. Otherwise ±10 pp and
  ±15 pp.
- Hybrid > naive is reported, not gated: our naive mode is not the paper's RAG baseline.
- A gating-scorer parse-failure rate above 5% in any type is a harness defect. PC1 is then
  "not_evaluable", with that diagnosis, instead of passing or failing.

Data comes from github.com/GraphRAG-Bench/GraphRAG-Benchmark @ fdbab59 (HF mirror: GraphRAG-Bench/GraphRAG-Bench):
    Datasets/Corpus/medical.json               {"corpus_name": "Medical", "context": <one ~218k-token document>}
    Datasets/Questions/medical_questions.json  [{"id", "source", "question", "answer", "question_type", "evidence", "evidence_relations"}]

Indexing follows the paper (App. H.2, Examples/run_lightrag.py): each corpus document is
inserted whole, and LightRAG chunks it at 1200/100 tokens. This deliberately departs from
D-003 (one LightRAG doc per pre-chunked shared chunk, with file_path = chunk ID). LightRAG
1.5.7 drops repeated file_paths as duplicates, so per-chunk docs need per-chunk paths. It
then renders up to 75 of those paths into every entity and relation line of the query
context, which eats the 4000-token KG budgets that the paper's numbers were produced under.

    python -m ape.anchor.graphragbench build [--data DIR] [--profile P]     # APE_BUILD_MODEL=gpt-4o-mini matches the paper
    python -m ape.anchor.graphragbench run --mode hybrid [--n-per-type 200] [--profile P]
    python -m ape.anchor.graphragbench pc1 GRAPH_LOG NAIVE_LOG [--tolerance PP]

`build` and `run` take their models and reasoning efforts from `config/models.yaml`
(`ape.models`): the build role for extraction, the agent role for answers, the judge role for ACC.
Both first run `ape.models.require_preflight` (every model priced, OPENAI_API_KEY set), and `run`
prices its Inspect calls from `config/model_costs.yaml`. `run` goes through `ape.runner.run_evals`
(FX-3: sample and task retries, the 2% error budget, resume): re-running the same mode, size and
models with the same `--log-dir` reuses a finished log instead of spending again.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

from lightrag.utils import EmbeddingFunc

from lightrag.base import DocStatus

from ..config import Config
from ..lgr.build import BUILD_MARKER, _dir_hash, _prepare_dir, _read_store
from ..lgr.common import index_dir, read_manifest, workspace_name, write_manifest
from ..llm.build_client import BuildLlm
from ..llm.ledger import Ledger
from ..models import build_settings, embedding_model, require_preflight
from .bge import AnchorEmbedder, anchor_embedder

ANCHOR_ID = "graphragbench-medical"
QUESTION_TYPES = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation")  # labels as in the data
# ACC (%) from 2506.05690v3 Table 2, Medical, LightRAG. The first three columns are unchanged since v1
# (2025-06-06); Creative Generation first appears in v2.
PUBLISHED_LIGHTRAG = {"Fact Retrieval": 63.32, "Complex Reasoning": 61.32, "Contextual Summarize": 63.14, "Creative Generation": 67.91}
# The scorer each published number came from (D-025): vendored RAGAS before 2025-06-14, today's code after.
RAGAS_TYPES = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize")
PAPER_MODEL = "gpt-4o-mini"
CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS = 1200, 100  # App. H.2
# PC1 (GATE_PREREG §7, D-025): (macro, per type) tolerance in pp; parse failures above this make PC1 not evaluable.
TOLERANCE_PAPER_MODEL, TOLERANCE_OTHER = (5.0, 10.0), (10.0, 15.0)
PARSE_FAILURE_LIMIT = 0.05
# Local bge serialises every embedding call behind one lock, and LightRAG's per-call timeout counts time spent
# queued. LightRAG 1.5.7's defaults (30 s, x2 internally; 8 concurrent calls of 10 texts) would time out under a
# CPU-bound model (RELIABILITY_REVIEW 2, finding 2): one call in flight, and a generous timeout.
EMBEDDING_TIMEOUT_S = 600
EMBEDDING_MAX_ASYNC = 1


def slug(qtype: str) -> str:
    """Metric-name form of a question type ("Fact Retrieval" -> "fact_retrieval")."""
    return qtype.lower().replace(" ", "_")


@dataclass(frozen=True)
class Bench:
    questions: list[dict]
    documents: list[dict]  # {"corpus_name", "context"}

    def corpus_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.documents, sort_keys=True).encode()).hexdigest()

    def questions_hash(self) -> str:
        return hashlib.sha256("\n".join(sorted(q["id"] for q in self.questions)).encode()).hexdigest()


def default_data_dir(cfg: Config) -> Path:
    return cfg.cache_dir / "graphragbench"


def _files(path: Path) -> tuple[Path, Path]:
    """(corpus, questions) from the repo layout, a flat directory, or the questions file itself."""
    if path.is_file():
        candidates = [(path.parent.parent / "Corpus" / "medical.json", path), (path.parent / "medical.json", path)]
    else:
        candidates = [
            (path / "Datasets" / "Corpus" / "medical.json", path / "Datasets" / "Questions" / "medical_questions.json"),
            (path / "medical.json", path / "medical_questions.json"),
        ]
    for corpus, questions in candidates:
        if corpus.is_file() and questions.is_file():
            return corpus, questions
    raise FileNotFoundError(f"no medical.json + medical_questions.json at {path}")


def load_medical(path_or_dir: str | Path, n_per_type: int, seed: int) -> Bench:
    """Up to `n_per_type` questions of each type, plus the corpus documents.

    Each type's pool is sorted by id, then shuffled with a per-type seed. So the sample
    ignores file order, and a larger n extends a smaller one.
    """
    corpus_path, questions_path = _files(Path(path_or_dir))
    raw = json.loads(corpus_path.read_text())
    questions = json.loads(questions_path.read_text())
    sample = []
    for qtype in QUESTION_TYPES:
        pool = sorted((q for q in questions if q["question_type"] == qtype), key=lambda q: q["id"])
        random.Random(f"{seed}:{qtype}").shuffle(pool)
        sample += pool[:n_per_type]
    return Bench(sample, raw if isinstance(raw, list) else [raw])


def working_dir(cfg: Config) -> Path:
    return index_dir(cfg, ANCHOR_ID, "extract")


async def open_anchor_rag(wd: Path, llm_func, emb: AnchorEmbedder, query_time: bool) -> LightRAG:
    """`lgr.common.open_rag`, but with the paper's chunking in place of D-003's pre-chunked 4096-token cap, and the
    paper's retrieval embeddings: bge-large-en-v1.5, mean-pooled as LightRAG 1.2.5's `hf_embed` (`ape.anchor.bge`)."""
    wd.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(wd),
        workspace=workspace_name(ANCHOR_ID),
        llm_model_func=llm_func,
        # A closure, not the bound method: LightRAG deep-copies its config, and the model behind it holds locks.
        embedding_func=EmbeddingFunc(embedding_dim=emb.dim, max_token_size=8192, func=emb.lightrag_func()),
        embedding_func_max_async=EMBEDDING_MAX_ASYNC,
        default_embedding_timeout=EMBEDDING_TIMEOUT_S,
        enable_llm_cache=not query_time,
        enable_llm_cache_for_entity_extract=True,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKENS,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


class AnchorBuildError(RuntimeError):
    """The anchor index build left a document unprocessed or empty; no manifest was written."""


def _health(docs: dict[str, dict]) -> dict:
    healthy = bool(docs) and all(d["status"] == DocStatus.PROCESSED.value and d["entities"] > 0 for d in docs.values())
    return {"documents": docs, "healthy": healthy}


def _status_value(status) -> str | None:
    return getattr(status, "value", status)


def _entity_count(rec: dict | None) -> int:
    rec = rec or {}
    return int(rec.get("count", len(rec.get("entity_names") or [])))


async def rag_extraction_health(rag: LightRAG, names: list[str]) -> dict:
    """Per corpus document: LightRAG's processing status and entity count, read from the open instance's stores (what
    `finalize_storages` writes). Healthy when every document is PROCESSED with at least one entity. The Medical
    corpus is one document, so a single chunk whose extraction kept failing fails the whole document, and LightRAG
    records that in its doc-status store without raising."""
    docs = {}
    for n in names:
        st = await rag.doc_status.get_by_id(n) or {}
        docs[n] = {"status": _status_value(st.get("status")), "entities": _entity_count(await rag.full_entities.get_by_id(n)), "error": st.get("error_msg")}
    return _health(docs)


def extraction_health(wd: Path, names: list[str]) -> dict:
    """`rag_extraction_health` from a built index's files (for inspecting an index on disk)."""
    status = _read_store(wd, "doc_status")
    entities = _read_store(wd, "full_entities")
    return _health({n: {"status": (status.get(n) or {}).get("status"), "entities": _entity_count(entities.get(n)), "error": (status.get(n) or {}).get("error_msg")} for n in names})


async def build_index(bench: Bench, cfg: Config, llm_func, build_model: str | None, build_effort: str | None = None) -> dict:
    """LightRAG's own extraction over the whole corpus (offline, never inside Inspect).

    Starts from a cleared working directory, keeping LightRAG's LLM response cache when the previous (complete or
    interrupted) build used the same build model and effort, as `ape.lgr.build` does: a rebuild after a failure pays
    only for the chunks that were not extracted. Raises `AnchorBuildError`, and writes no manifest, unless every
    document is PROCESSED with entities (`rag_extraction_health`). As with `ape.lgr.build`, LightRAG keeps a
    workspace's stores in process memory, so a rebuild belongs in a fresh process (run_gate's anchor phase builds
    at most once per process)."""
    wd = working_dir(cfg)
    emb = anchor_embedder(cfg)
    key = {"kind": "extract", "anchor": ANCHOR_ID, "corpus_hash": bench.corpus_hash(), "embedding_model": emb.identity, "build_model": build_model, "build_effort": build_effort}
    _prepare_dir(wd, key)
    rag = await open_anchor_rag(wd, llm_func, emb, query_time=False)
    names = [d["corpus_name"] for d in bench.documents]
    try:
        await rag.ainsert([d["context"] for d in bench.documents], ids=names, file_paths=names)
        health = await rag_extraction_health(rag, names)
    finally:
        await rag.finalize_storages()
    if not health["healthy"]:
        raise AnchorBuildError(
            f"anchor index build incomplete in {wd}: "
            + "; ".join(f"{n}: status {d['status']}, {d['entities']} entities" + (f" ({str(d['error'])[:200]})" if d["error"] else "") for n, d in health["documents"].items())
            + ". No manifest was written; re-running the build re-asks only the chunks LightRAG did not extract."
        )
    manifest = {
        **key,
        "documents": len(names),
        "chunking": [CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS],
        "lightrag_version": importlib.metadata.version("lightrag-hku"),
        "extraction": health,
    }
    manifest["index_hash"] = _dir_hash(wd)
    write_manifest(wd, manifest)
    (wd / BUILD_MARKER).unlink(missing_ok=True)
    return manifest


def index_manifest(cfg: Config, bench: Bench) -> dict:
    """The anchor index's manifest, if the index can be used: same corpus and embeddings, and a healthy extraction.
    The anchor task calls this when it is created, so an unusable index stops the run before any answer is paid for."""
    wd = working_dir(cfg)
    if not (wd / "ape_manifest.json").exists():
        raise FileNotFoundError(f"{wd} missing: run `python -m ape.anchor.graphragbench build` first")
    manifest = read_manifest(wd)
    if manifest["corpus_hash"] != bench.corpus_hash():
        raise RuntimeError(f"{wd} was built from a different corpus")
    if manifest.get("embedding_model") != (want := anchor_embedder(cfg).identity):
        raise RuntimeError(f"{wd} was built with embeddings {manifest.get('embedding_model')!r}, not {want!r}: rebuild it")
    if not (manifest.get("extraction") or {}).get("healthy"):
        raise RuntimeError(
            f"{wd} records no healthy extraction (every document PROCESSED with entities): it was built before that check "
            "or the build failed. Move it aside to rebuild (the LLM response cache inside makes a rebuild cheap)."
        )
    if manifest.get("index_hash") != _dir_hash(wd):
        raise RuntimeError(f"{wd}: its files no longer match the manifest's index_hash (damaged or partly copied); move it aside to rebuild")
    return manifest


def pc1_tolerance(*models: str) -> tuple[float, float]:
    """Pre-registered (macro, per-type) tolerance in pp: (5, 10) if every model in the loop is the paper's, else (10, 15)."""
    return TOLERANCE_PAPER_MODEL if all(m.split("/")[-1].startswith(PAPER_MODEL) for m in models) else TOLERANCE_OTHER


def _macro(by_type: dict, types) -> float | None:
    values = [by_type.get(t) for t in types]
    return None if None in values else sum(values) / len(values)


def pc1_check(
    results_by_type: dict,
    published: dict,
    tolerance_pp: tuple[float, float],
    naive_by_type: dict,
    parse_failure_rate: dict | None = None,
    current_by_type: dict | None = None,
) -> dict:
    """PC1 (GATE_PREREG §7, D-025) on the gating scorer's per-type ACC (%).

    `status`:
    - "not_evaluable": a type's gating-scorer parse-failure rate is above PARSE_FAILURE_LIMIT, or unknown
      (a harness defect to fix, not a result).
    - "pass": the macro mean is within ±macro tolerance of the published macro, and every type is within
      ±type tolerance.
    - "fail": anything else.

    Reported, not gated:
    - Hybrid > naive. Our naive mode is not the paper's RAG baseline, and the paper's own LightRAG loses to
      vanilla RAG on two types.
    - Today's official scorer (`current_by_type`), run strictly.
    """
    macro_tol, type_tol = tolerance_pp
    rates = parse_failure_rate or {}
    current = current_by_type or {}
    per_type = {}
    for qtype, pub in published.items():
        ours, naive = results_by_type.get(qtype), naive_by_type.get(qtype)
        per_type[qtype] = {
            "ours": ours,
            "published": pub,
            "naive": naive,
            "current_scorer": current.get(qtype),
            "parse_failure_rate": rates.get(qtype),
            "delta_pp": None if ours is None else ours - pub,
            "within_tolerance": ours is not None and abs(ours - pub) <= type_tol,
            "beats_naive": ours is not None and naive is not None and ours > naive,
        }
    macro, naive_macro, published_macro = _macro(results_by_type, published), _macro(naive_by_type, published), _macro(published, published)
    macro_ok = macro is not None and abs(macro - published_macro) <= macro_tol
    types_ok = all(r["within_tolerance"] for r in per_type.values())
    unparsed = {t: r for t, r in ((t, rates.get(t)) for t in published) if r is None or r > PARSE_FAILURE_LIMIT}
    if unparsed:
        status = "not_evaluable"
        diagnosis = "gating-scorer parse failures above {:.0%}: {} (a harness defect: inspect the judge replies)".format(
            PARSE_FAILURE_LIMIT, ", ".join(f"{t} {'unknown' if r is None else f'{r:.1%}'}" for t, r in unparsed.items())
        )
    else:
        status, diagnosis = ("pass" if macro_ok and types_ok else "fail"), None
    return {
        "status": status,
        "pass": status == "pass",
        "diagnosis": diagnosis,
        "per_type": per_type,
        "tolerance_pp": {"macro": macro_tol, "per_type": type_tol},
        "reproduces_published": macro_ok and types_ok,
        "macro": macro,
        "published_macro": published_macro,
        "macro_within_tolerance": macro_ok,
        "types_within_tolerance": types_ok,
        "current_scorer_macro": _macro(current, published),
        "naive_macro": naive_macro,
        "beats_naive": macro is not None and naive_macro is not None and macro > naive_macro,
        "parse_failure_limit": PARSE_FAILURE_LIMIT,
    }


def _metrics(log: EvalLog) -> dict[str, float]:
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


def results_by_type(log: EvalLog) -> dict[str, float]:
    """Per-type ACC (%) of the gating scorer from an anchor eval log."""
    metrics = _metrics(log)
    return {t: metrics[t] for t in QUESTION_TYPES if t in metrics}


def metric_by_type(log: EvalLog, prefix: str) -> dict[str, float]:
    """A per-type metric family (`current`, `matched_parse_failure_rate`, ...) keyed by question type."""
    metrics = _metrics(log)
    return {t: metrics[f"{prefix}_{slug(t)}"] for t in QUESTION_TYPES if f"{prefix}_{slug(t)}" in metrics}


def _role_model(log: EvalLog, role: str) -> str:
    cfg = (log.eval.model_roles or {})[role]
    return (cfg[0] if isinstance(cfg, list) else cfg).model


def pc1_from_logs(graph: EvalLog, naive: EvalLog, tolerance_pp: tuple[float, float] | None = None) -> dict:
    """PC1 from the graph-mode and naive anchor logs: gating ACC, parse-failure rates, the strict official
    scorer as a sensitivity, and what was run (scorers, embeddings, query caps)."""
    g, n = graph.eval.metadata, naive.eval.metadata
    if (g["question_ids_hash"], g["index"]["index_hash"]) != (n["question_ids_hash"], n["index"]["index_hash"]):
        raise ValueError("the two runs used different questions or indexes")
    if tolerance_pp is None:
        tolerance_pp = pc1_tolerance(graph.eval.model, _role_model(graph, "judge"), g["index"]["build_model"] or "")
    check = pc1_check(
        results_by_type(graph),
        PUBLISHED_LIGHTRAG,
        tolerance_pp,
        results_by_type(naive),
        parse_failure_rate=metric_by_type(graph, "matched_parse_failure_rate"),
        current_by_type=metric_by_type(graph, "current"),
    )
    return {
        "mode": g["mode"],
        **check,
        "current_scorer_parse_failure_rate": metric_by_type(graph, "current_parse_failure_rate"),
        "fix_format_rate": metric_by_type(graph, "matched_fix_format_rate"),
        "naive_parse_failure_rate": metric_by_type(naive, "matched_parse_failure_rate"),
        "scorers": g.get("scorers"),
        "embeddings": g.get("embeddings"),
        "query": g.get("query"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    r = sub.add_parser("run", help="anchor eval with the profile's agent (answers) and judge models")
    r.add_argument("--mode", default="hybrid")
    r.add_argument("--n-per-type", type=int, default=200)
    r.add_argument("--log-dir", help="eval-set log dir (default: $INSPECT_LOG_DIR or ./logs); a finished identical run in it is reused")
    for s in (b, r):
        s.add_argument("--data", help="dataset dir (default: $APE_CACHE/graphragbench)")
        s.add_argument("--profile", help="config/models.yaml profile (default: $APE_MODEL_PROFILE or gate)")
    p = sub.add_parser("pc1")
    p.add_argument("graph_log")
    p.add_argument("naive_log")
    p.add_argument("--tolerance", type=float, nargs=2, metavar=("MACRO_PP", "TYPE_PP"))
    args = ap.parse_args()
    if args.cmd != "pc1":
        # Default to the paper-faithful "anchor" profile (gpt-4o-mini); "anchor_luna" if it is retired (E3).
        os.environ["APE_MODEL_PROFILE"] = args.profile or os.environ.get("APE_MODEL_PROFILE") or "anchor"
        os.environ.setdefault("APE_EMBEDDING_MODEL", embedding_model())
        require_preflight(live=True)
    cfg = Config()
    if args.cmd == "pc1":
        result = pc1_from_logs(read_eval_log(args.graph_log), read_eval_log(args.naive_log), tuple(args.tolerance) if args.tolerance else None)
    elif args.cmd == "run":
        from ..runner import log_path, run_evals
        from ..tasks.anchor_graphragbench import graphragbench_anchor

        task = graphragbench_anchor(mode=args.mode, n_per_type=args.n_per_type, data=args.data)
        log_dir = args.log_dir or os.environ.get("INSPECT_LOG_DIR") or "logs"
        # The log dir may hold other runs (other modes, older logs): allow_dirty ignores them.
        _, (log,) = run_evals(task, log_dir, roles=("judge",), live=True, log_dir_allow_dirty=True)
        result = {"status": log.status, "log": log_path(log), "by_type": results_by_type(log) if log.status == "success" else None}
    else:
        model, effort = build_settings()
        bench = load_medical(args.data or default_data_dir(cfg), n_per_type=0, seed=0)
        llm = BuildLlm(model, Ledger(cfg.ledger_path), {"anchor": ANCHOR_ID, "system": "lightrag"}, reasoning_effort=effort).lightrag_func()
        result = asyncio.run(build_index(bench, cfg, llm, model, effort))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
