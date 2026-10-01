"""A tiny synthetic GraphRAG-Bench Medical dataset and scripted models, for offline anchor runs.

Used by the anchor tests and by `python -m ape.run_gate anchor --offline`. Nothing here calls a
network or a real model:
- `write_dataset(root)` writes the benchmark's repo layout (Datasets/Corpus/medical.json,
  Datasets/Questions/medical_questions.json) with 5 questions per type over a small corpus.
- `fake_extractor()` stands in for the build model in LightRAG's extraction format.
- `mock_answerer` is the answer model (keywords via `mock_kg`, gold answers for Fact Retrieval
  only), and `mock_judge` the ACC judge (splits sentences, classifies statements by exact match).
  Both are `mockllm` `custom_outputs` callables.
"""

import hashlib
import json
import re
from pathlib import Path

from inspect_ai.model import ModelOutput
from lightrag.prompt import PROMPTS

from ..llm.mock_agent import mock_kg

MODEL = "mockllm/model"
TOPICS = ["basal cell carcinoma", "squamous cell carcinoma", "melanoma", "Merkel cell carcinoma", "actinic keratosis"]
TEMPLATES = {
    "Fact Retrieval": ("According to the guideline, what is {t}?", "{T} is a skin condition covered by the guideline."),
    "Complex Reasoning": ("Why does sun exposure raise the risk of {t}?", "Because UV damage accumulates in skin cells, which drives {t}."),
    "Contextual Summarize": ("Summarize how the guideline manages {t}.", "The guideline recommends biopsy, then surgery or Imiquimod for {t}."),
    "Creative Generation": ("Write a short patient letter about the guideline for {t}.", "Dear patient, {t} is treatable when found early."),
}
# Repeated so LightRAG splits it into 3 chunks, and lexically close enough to the questions that
# fake bag-of-words vectors clear LightRAG's 0.2 cosine threshold (read at import, so not settable here).
FACTS = [
    f"{t[0].upper() + t[1:]} is a skin condition covered by the guideline. Sun exposure raises the risk of {t}. "
    f"The guideline recommends biopsy, then surgery or Imiquimod for {t}."
    for t in TOPICS
]
CORPUS = " ".join(FACTS * 12)
D, DONE = PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"]


def all_questions() -> list[dict]:
    """5 questions per type, with the benchmark's field names."""
    out = []
    for qtype, (q, a) in TEMPLATES.items():
        for t in TOPICS:
            qid = "Medical-" + hashlib.sha1(f"{qtype}{t}".encode()).hexdigest()[:8]
            answer = a.format(t=t, T=t[0].upper() + t[1:])
            out.append({"id": qid, "source": "Medical", "question": q.format(t=t), "answer": answer, "question_type": qtype, "evidence": answer, "evidence_relations": answer})
    return out


GOLD = {q["question"]: q["answer"] for q in all_questions() if q["question_type"] == "Fact Retrieval"}


def write_dataset(root: str | Path, questions: list[dict] | None = None, corpus: str = CORPUS) -> Path:
    """The repo layout and field names of GraphRAG-Benchmark's Datasets/ folder, under `root`."""
    root = Path(root)
    (root / "Datasets" / "Corpus").mkdir(parents=True, exist_ok=True)
    (root / "Datasets" / "Questions").mkdir(parents=True, exist_ok=True)
    (root / "Datasets" / "Corpus" / "medical.json").write_text(json.dumps({"corpus_name": "Medical", "context": corpus}))
    (root / "Datasets" / "Questions" / "medical_questions.json").write_text(json.dumps(questions or all_questions()))
    return root


def fake_extractor(calls: list[str] | None = None):
    """Build-model stand-in in LightRAG's extraction format: capitalised terms, chained by relations.
    Every prompt is appended to `calls` when given."""

    async def extract(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **_kwargs) -> str:
        if calls is not None:
            calls.append(prompt)
        m = re.search(r"---Input Text---\n```\n(.*?)\n```", prompt, re.S)
        if not m:  # gleaning / summary calls
            return DONE
        names = sorted(set(re.findall(r"\b[A-Z][a-z]{3,}\b", m.group(1))))
        rows = [f"entity{D}{n}{D}Concept{D}{n} appears in the guideline." for n in names]
        rows += [f"relation{D}{a}{D}{b}{D}co-occurs{D}{a} appears with {b}." for a, b in zip(names, names[1:])]
        return "\n".join(rows + [DONE])

    return extract


def answer_in(statement_prompt: str) -> str:
    """The text under judgement in a statement-generation prompt (its few-shot example has an "Answer:" too)."""
    return statement_prompt.rsplit("Answer: ", 1)[1].split("\n\nGenerated Statements:")[0]


def mock_answerer(messages, tools, tool_choice, config) -> ModelOutput:
    """Default model: LightRAG keywords, then gold answers for Fact Retrieval only."""
    if config.response_schema is not None:
        return mock_kg(messages, tools, tool_choice, config)
    question = next(m.text for m in reversed(messages) if m.role == "user")
    return ModelOutput.from_content(MODEL, GOLD.get(question, "I don't know"))


def mock_judge(messages, tools, tool_choice, config) -> ModelOutput:
    """Judge that splits sentences and classifies statements by exact match."""
    prompt = messages[-1].text
    if "Current Analysis:" in prompt:
        tail = prompt.split("Current Analysis:")[-1]
        ans, gt = (json.loads(re.search(rf"{k}: (.*)\n", tail).group(1)) for k in ("Answer Statements", "Ground Truth Statements"))
        out = {"TP": [s for s in ans if s in gt], "FP": [s for s in ans if s not in gt], "FN": [s for s in gt if s not in ans]}
        return ModelOutput.from_content(MODEL, json.dumps({k: [{"statement": s, "reason": "mock"} for s in v] for k, v in out.items()}))
    return ModelOutput.from_content(MODEL, json.dumps([s for s in re.split(r"(?<=\.)\s+", answer_in(prompt)) if s]))
