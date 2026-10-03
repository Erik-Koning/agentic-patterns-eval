"""GraphRAG-Bench "Accuracy" (ACC), replicated from the official evaluation code.

Source: github.com/GraphRAG-Bench/GraphRAG-Benchmark @ fdbab59, files
Evaluation/metrics/answer_accuracy.py and Evaluation/generation_eval.py (both last
changed in 7572f7e, 2025-09-05). ACC is RAGAS-style answer correctness:

    ACC = 0.75 * F1(TP, FP, FN over LLM-classified statements) + 0.25 * (cos(answer, reference) + 1) / 2

The prompts and few-shot examples below are the official ones, verbatim. Parsing and failure
semantics are kept too, because they move the score:
- Statement lists go through the official fallback parser.
- The classification is parsed strictly, so an unparseable reply scores F1 = 0. That is the official
  behaviour, and it can collapse a whole run (GraphRAG-Benchmark PR #56: gpt-4o-mini replies
  `Output: {...}` plus reasoning). Here it is also *flagged* (`classification_parse_failed`), so PC1
  can tell a harness defect from a low score (D-025).
- An ill-formed classification raises `Excluded`; the official loop logs that sample as failed and
  averages the rest.

In PC1 this scorer is the reported sensitivity for every question type. Creative Generation's gating
number (published 2025-09-25, with this code) re-parses the *same* classification reply tolerantly
(`tolerant`). The three older types gate on the vendored-RAGAS scorer in `ape.anchor.ragas`.

Similarity follows the official code: `aembed_query` on both texts with BAAI/bge-large-en-v1.5
through LangChain (bge's query instruction, CLS pooling, normalised; `ape.anchor.bge`'s `queries`),
then (cos + 1) / 2. Known deviation: the official json5 fallback tier is folded into json_repair,
because json5 is not installed.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import NamedTuple

import json_repair
import numpy as np
from pydantic import BaseModel, ValidationError

WEIGHTS = (0.75, 0.25)
STATEMENT_KEYS = ("statements", "answers", "items", "list", "output", "result")

Generate = Callable[[str], Awaitable[str]]
Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


class Excluded(Exception):
    """The official loop drops this sample from the per-type mean."""


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason] = []
    FP: list[StatementsWithReason] = []
    FN: list[StatementsWithReason] = []


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


# --- official prompts and examples, verbatim (answer_accuracy.py lines 34-145) ---
STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

Example Input: 
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.

Example Output:
["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.","Albert Einstein was best known for developing the theory of relativity.","Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]

Input Text:
Question:{question}
Answer: {answer}

Generated Statements:
"""

# Correctness classification prompt template
CORRECTNESS_PROMPT_TEMPLATE = """
Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

Examples:
{examples}

Current Analysis:
Question: {question}
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""

# Pre-defined examples for correctness classification
CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and what is its primary function?",
            "answer": [
                "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                "The primary function of the sun is to provide light to the solar system."
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                "This fusion process in the sun's core releases a tremendous amount of energy.",
                "The energy from the sun provides heat and light, which are essential for life on Earth.",
                "The sun's light plays a critical role in Earth's climate system.",
                "Sunlight helps to drive the weather and ocean currents."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The primary function of the sun is to provide light to the solar system.",
                    "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."
                }
            ],
            "FP": [
                {
                    "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."
                }
            ],
            "FN": [
                {
                    "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "reason": "This accurate description of the sun’s power source is not included in the answer."
                },
                {
                    "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "reason": "This process and its significance are not mentioned in the answer."
                },
                {
                    "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."
                },
                {
                    "statement": "The sun's light plays a critical role in Earth's climate system.",
                    "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer."
                },
                {
                    "statement": "Sunlight helps to drive the weather and ocean currents.",
                    "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."
                }
            ]
        }
    },
    {
        "input": {
            "question": "What is the boiling point of water?",
            "answer": [
                "The boiling point of water is 100 degrees Celsius at sea level"
            ],
            "ground_truth": [
                "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "The boiling point of water can change with altitude."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The boiling point of water is 100 degrees Celsius at sea level",
                    "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."
                }
            ],
            "FP": [],
            "FN": [
                {
                    "statement": "The boiling point of water can change with altitude.",
                    "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."
                }
            ]
        }
    }
]
# --- end verbatim ---


def _safe_json_parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(json_repair.repair_json(text))
    except Exception:
        return {}


def parse_statements(raw: str) -> list[str]:
    """Official `JSONHandler.parse_with_fallbacks` + `generate_statements` normalisation."""
    content = re.sub(r"```(?:json)?|```", "", raw).strip()
    block = re.search(r"\{[\s\S]*\}", content)
    parsed = _safe_json_parse(content) or _safe_json_parse(block.group(0) if block else content) or {}
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        for key in STATEMENT_KEYS:
            if isinstance(parsed.get(key), list):
                return [str(x) for x in parsed[key]]
        return [str(v) for v in parsed.values()]
    return [str(parsed)]


class Classification(NamedTuple):
    f1: float
    parse_failed: bool  # the reply was not JSON: upstream scores F1 = 0 without a word; we flag it


def parse_classification(raw: str) -> Classification:
    """Strict, as upstream: fenced or non-JSON replies score 0 (flagged); wrong shapes exclude the sample."""
    try:
        c = ClassificationWithReason(**json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return Classification(0.0, True)
    except ValidationError as e:
        raise Excluded(f"ill-formed TP/FP/FN classification: {e.error_count()} errors") from e
    return Classification(fbeta_score(len(c.TP), len(c.FP), len(c.FN)), False)


def parse_classification_tolerant(raw: str) -> float:
    """The same reply, parsed the way a working run read it: the first balanced JSON object (RAGAS's
    `extract_json`), then LangChain's markdown-stripping, partial-JSON parse and the official model.
    Raises `ape.anchor.ragas.ScorerFailed` when even that fails (PC1 counts it as a parse failure)."""
    from .ragas import ParseError, ScorerFailed, extract_json, pydantic_parse

    try:
        c = pydantic_parse(extract_json(raw), ClassificationWithReason)
    except ParseError as e:
        raise ScorerFailed(f"unparseable classification: {e}") from e
    return fbeta_score(len(c.TP), len(c.FP), len(c.FN))


def classification_prompt(question: str, answer_stmts: list[str], gt_stmts: list[str]) -> str:
    examples = "\n".join(f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}" for ex in CORRECTNESS_EXAMPLES)
    return CORRECTNESS_PROMPT_TEMPLATE.format(examples=examples, question=question, answer=json.dumps(answer_stmts), ground_truth=json.dumps(gt_stmts))


def semantic_similarity(a: list[float], b: list[float]) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return (cos + 1) / 2


async def answer_correctness(question: str, answer: str, ground_truth: str, generate: Generate, embed: Embed) -> dict:
    """One sample's ACC in [0, 1] plus its parts. `generate` is the judge (one user message per call); `embed`
    is the official `aembed_query` (`ape.anchor.bge` `queries`). `classification_raw` is the judge's reply,
    kept so the tolerant re-parse needs no second call (None when both statement lists were empty)."""
    # A TaskGroup, not gather: if one call raises (e.g. a truncated reply), the other is cancelled, not left running.
    try:
        async with asyncio.TaskGroup() as tg:
            a = tg.create_task(generate(STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)))
            g = tg.create_task(generate(STATEMENT_GENERATOR_PROMPT.format(question=question, answer=ground_truth)))
    except* Exception as eg:
        raise eg.exceptions[0] from None
    answer_stmts, gt_stmts = a.result(), g.result()
    answer_stmts, gt_stmts = parse_statements(answer_stmts), parse_statements(gt_stmts)
    raw = None
    if not answer_stmts and not gt_stmts:
        fc, failed = 1.0, False
    else:
        raw = await generate(classification_prompt(question, answer_stmts, gt_stmts))
        fc, failed = parse_classification(raw)
    ss = semantic_similarity(*await embed([answer, ground_truth]))
    return {
        "accuracy": WEIGHTS[0] * fc + WEIGHTS[1] * ss,
        "factual_correctness": fc,
        "semantic_similarity": ss,
        "classification_parse_failed": failed,
        "classification_raw": raw,
        "answer_statements": answer_stmts,
        "reference_statements": gt_stmts,
        "judge_calls": 2 if raw is None else 3,
    }


def tolerant(result: dict) -> dict:
    """Creative Generation's gating score from the official result: the classification reply re-parsed tolerantly
    (`parse_classification_tolerant`; raises `ScorerFailed` if even that fails), the same (cos + 1) / 2 similarity."""
    fc = 1.0 if result["classification_raw"] is None else parse_classification_tolerant(result["classification_raw"])
    return {"accuracy": WEIGHTS[0] * fc + WEIGHTS[1] * result["semantic_similarity"], "factual_correctness": fc, "semantic_similarity": result["semantic_similarity"]}
