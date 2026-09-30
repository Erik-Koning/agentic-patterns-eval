"""GraphRAG-Bench Medical anchor (gate precondition PC1): data, LightRAG index, pre-registered check.

Before comparing APG against LightRAG, we must show that our LightRAG setup reproduces
LightRAG's published GraphRAG-Bench scores: arXiv 2506.05690v3 Table 2, Medical rows.
Our setup is pinned lightrag-hku, OpenAI models, rerank off, and our embedding cache.

Data comes from github.com/GraphRAG-Bench/GraphRAG-Benchmark @ fdbab59 (HF mirror: GraphRAG-Bench/GraphRAG-Bench):
    Datasets/Corpus/medical.json               {"corpus_name": "Medical", "context": <one ~218k-token document>}
    Datasets/Questions/medical_questions.json  [{"id", "source", "question", "answer", "question_type", "evidence", "evidence_relations"}]

Indexing follows the paper (App. H.2, Examples/run_lightrag.py): each corpus document is
inserted whole, and LightRAG chunks it at 1200/100 tokens. This deliberately departs from
D-003 (one LightRAG doc per pre-chunked shared chunk, with file_path = chunk ID). LightRAG
1.5.7 drops repeated file_paths as duplicates, so per-chunk docs need per-chunk paths. It
then renders up to 75 of those paths into every entity and relation line of the query
context, which eats the 4000-token KG budgets that the paper's numbers were produced under.

    APE_BUILD_MODEL=gpt-4o-mini python -m ape.anchor.graphragbench build [--data DIR]
    python -m ape.anchor.graphragbench pc1 GRAPH_LOG NAIVE_LOG [--tolerance PP]
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

from ..config import Config, embedding_cache
from ..lgr.build import _dir_hash
from ..lgr.common import embedding_dim, embedding_func, index_dir, read_manifest, workspace_name, write_manifest
from ..llm.build_client import BuildLlm
from ..llm.embeddings import EmbeddingCache
from ..llm.ledger import Ledger

ANCHOR_ID = "graphragbench-medical"
QUESTION_TYPES = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation")  # labels as in the data
# ACC (%) from 2506.05690v3 Table 2, Medical, LightRAG. The first three columns are unchanged since v1
# (2025-06-06); Creative Generation first appears in v2.
PUBLISHED_LIGHTRAG = {"Fact Retrieval": 63.32, "Complex Reasoning": 61.32, "Contextual Summarize": 63.14, "Creative Generation": 67.91}
PAPER_MODEL = "gpt-4o-mini"
CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS = 1200, 100  # App. H.2


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


async def open_anchor_rag(wd: Path, llm_func, emb: EmbeddingCache, query_time: bool) -> LightRAG:
    """`lgr.common.open_rag`, but with the paper's chunking in place of D-003's pre-chunked 4096-token cap."""
    wd.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(wd),
        workspace=workspace_name(ANCHOR_ID),
        llm_model_func=llm_func,
        embedding_func=embedding_func(emb, await embedding_dim(emb)),
        enable_llm_cache=not query_time,
        enable_llm_cache_for_entity_extract=True,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKENS,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def build_index(bench: Bench, cfg: Config, llm_func, build_model: str | None) -> dict:
    """LightRAG's own extraction over the whole corpus (offline, never inside Inspect)."""
    wd = working_dir(cfg)
    emb = embedding_cache(cfg)
    rag = await open_anchor_rag(wd, llm_func, emb, query_time=False)
    names = [d["corpus_name"] for d in bench.documents]
    try:
        await rag.ainsert([d["context"] for d in bench.documents], ids=names, file_paths=names)
    finally:
        await rag.finalize_storages()
    manifest = {
        "anchor": ANCHOR_ID,
        "corpus_hash": bench.corpus_hash(),
        "documents": len(names),
        "chunking": [CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS],
        "lightrag_version": importlib.metadata.version("lightrag-hku"),
        "embedding_model": emb.model,
        "build_model": build_model,
    }
    manifest["index_hash"] = _dir_hash(wd)
    write_manifest(wd, manifest)
    return manifest


def index_manifest(cfg: Config, bench: Bench) -> dict:
    wd = working_dir(cfg)
    if not (wd / "ape_manifest.json").exists():
        raise FileNotFoundError(f"{wd} missing: run `python -m ape.anchor.graphragbench build` first")
    manifest = read_manifest(wd)
    if manifest["corpus_hash"] != bench.corpus_hash():
        raise RuntimeError(f"{wd} was built from a different corpus")
    return manifest


def pc1_tolerance(*models: str) -> float:
    """Pre-registered: ±5 pp if every model in the loop is the paper's, else ±10 pp."""
    return 5.0 if all(m.split("/")[-1].startswith(PAPER_MODEL) for m in models) else 10.0


def _macro(by_type: dict, types) -> float | None:
    values = [by_type.get(t) for t in types]
    return None if None in values else sum(values) / len(values)


def pc1_check(results_by_type: dict, published: dict, tolerance_pp: float, naive_by_type: dict) -> dict:
    """PC1 passes iff every published type is within ±tolerance and the graph mode beats naive.

    "Beats naive" is judged on the macro mean over types. Per-type wins are reported but not
    required, because the paper's own LightRAG loses to vanilla RAG on Fact Retrieval (63.32 vs
    63.72) and Contextual Summarize (63.14 vs 63.72) while winning on the macro mean (63.92 vs 61.00).
    """
    per_type = {}
    for qtype, pub in published.items():
        ours, naive = results_by_type.get(qtype), naive_by_type.get(qtype)
        per_type[qtype] = {
            "ours": ours,
            "published": pub,
            "naive": naive,
            "delta_pp": None if ours is None else ours - pub,
            "within_tolerance": ours is not None and abs(ours - pub) <= tolerance_pp,
            "beats_naive": ours is not None and naive is not None and ours > naive,
        }
    macro, naive_macro = _macro(results_by_type, published), _macro(naive_by_type, published)
    reproduces = all(r["within_tolerance"] for r in per_type.values())
    beats_naive = macro is not None and naive_macro is not None and macro > naive_macro
    return {
        "per_type": per_type,
        "tolerance_pp": tolerance_pp,
        "reproduces_published": reproduces,
        "macro": macro,
        "naive_macro": naive_macro,
        "beats_naive": beats_naive,
        "pass": reproduces and beats_naive,
    }


def results_by_type(log: EvalLog) -> dict[str, float]:
    """Per-type ACC (%) from an anchor eval log."""
    metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
    return {t: metrics[t] for t in QUESTION_TYPES if t in metrics}


def _role_model(log: EvalLog, role: str) -> str:
    cfg = (log.eval.model_roles or {})[role]
    return (cfg[0] if isinstance(cfg, list) else cfg).model


def pc1_from_logs(graph: EvalLog, naive: EvalLog, tolerance_pp: float | None = None) -> dict:
    g, n = graph.eval.metadata, naive.eval.metadata
    if (g["question_ids_hash"], g["index"]["index_hash"]) != (n["question_ids_hash"], n["index"]["index_hash"]):
        raise ValueError("the two runs used different questions or indexes")
    if tolerance_pp is None:
        tolerance_pp = pc1_tolerance(graph.eval.model, _role_model(graph, "judge"), g["index"]["build_model"] or "")
    return {"mode": g["mode"], **pc1_check(results_by_type(graph), PUBLISHED_LIGHTRAG, tolerance_pp, results_by_type(naive))}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", help="dataset dir (default: $APE_CACHE/graphragbench)")
    p = sub.add_parser("pc1")
    p.add_argument("graph_log")
    p.add_argument("naive_log")
    p.add_argument("--tolerance", type=float)
    args = ap.parse_args()
    cfg = Config()
    if args.cmd == "pc1":
        result = pc1_from_logs(read_eval_log(args.graph_log), read_eval_log(args.naive_log), args.tolerance)
    else:
        model = os.environ.get("APE_BUILD_MODEL")
        if not model:
            raise RuntimeError(f"set APE_BUILD_MODEL ({PAPER_MODEL} matches the paper)")
        bench = load_medical(args.data or default_data_dir(cfg), n_per_type=0, seed=0)
        llm = BuildLlm(model, Ledger(cfg.ledger_path), {"anchor": ANCHOR_ID, "system": "lightrag"}).lightrag_func()
        result = asyncio.run(build_index(bench, cfg, llm, model))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
