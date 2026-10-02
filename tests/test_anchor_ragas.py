"""PC1's gating scorer for the three RAGAS-era types (D-025): the vendored-RAGAS `answer_correctness`.

Rendering is checked byte for byte against RAGAS's own code (fixtures in tests/fixtures/anchor, see the README
there); parsing, the fix-format re-ask, NaN-on-failure, raw cosine and the weights against RAGAS's semantics.
The real bge model is tested only with APE_TEST_BGE=1 (it is a ~1.3 GB download).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ape.anchor import ragas
from ape.anchor.bge import QUERY_INSTRUCTION, anchor_embedder, fake_embedder
from ape.config import Config

FIXTURES = Path(__file__).parent / "fixtures" / "anchor"

CASES = {
    "statement_plain": (ragas.STATEMENT_GENERATOR, ragas.StatementGeneratorInput(question="What is basal cell carcinoma?", answer="Basal cell carcinoma is a skin cancer.")),
    "statement_escapes": (
        ragas.STATEMENT_GENERATOR,
        ragas.StatementGeneratorInput(question='Why "UV"?\nExplain.', answer="It's the sun’s rays — 100% \\ unicode: é, 日本.\n\nSecond paragraph."),
    ),
    "classifier": (ragas.CORRECTNESS_CLASSIFIER, ragas.QuestionAnswerGroundTruth(question="Q?", answer=["A one.", 'A "two".'], ground_truth=["G one."])),
    "classifier_empty": (ragas.CORRECTNESS_CLASSIFIER, ragas.QuestionAnswerGroundTruth(question="Q?", answer=[], ground_truth=["G."])),
    "fix_output_format": (ragas.FIX_OUTPUT_FORMAT, ragas.OutputStringAndPrompt(output_string="Output: {bad", prompt_value="the prompt\nwith lines")),
}


@pytest.mark.parametrize("name", CASES)
def test_rendering_is_byte_identical_to_ragas_own_code(name):
    spec, data = CASES[name]
    assert ragas.to_string(spec, data) == (FIXTURES / f"ragas_{name}.txt").read_text(encoding="utf-8")


# --- parsing: RagasOutputParser semantics ---


@pytest.mark.parametrize(
    "reply",
    [
        '{"statements": ["A."]}',
        'Output: {"statements": ["A."]}\n\nReasoning: one sentence, one statement.',  # GraphRAG-Benchmark PR #56 shape
        '```json\n{"statements": ["A."]}\n```',
        '{"statements": ["A."',  # truncated: LangChain's parse_partial_json closes it
    ],
)
async def test_tolerant_replies_parse_without_a_retry(reply):
    calls = []

    async def generate(prompt):
        calls.append(prompt)
        return reply

    out = await ragas.generate_model(ragas.STATEMENT_GENERATOR, ragas.StatementGeneratorInput(question="Q", answer="A."), generate, trace := [])
    assert out.statements == ["A."] and len(calls) == 1 and trace == ["StatementGeneratorOutput"]


async def test_a_wrong_shape_gets_one_fix_format_reask():
    replies = iter(['["A."]', json.dumps({"text": '{"statements": ["A."]}'})])  # a bare list fails validation

    async def generate(prompt):
        return next(replies)

    out = await ragas.generate_model(ragas.STATEMENT_GENERATOR, ragas.StatementGeneratorInput(question="Q", answer="A."), generate, trace := [])
    assert out.statements == ["A."] and trace == ["StatementGeneratorOutput", "StringIO"]


async def test_unreadable_replies_fail_like_ragas_nan_after_nested_reasks():
    prompts = []

    async def generate(prompt):
        prompts.append(prompt)
        return "I cannot comply."

    with pytest.raises(ragas.ScorerFailed):
        await ragas.generate_model(ragas.STATEMENT_GENERATOR, ragas.StatementGeneratorInput(question="Q", answer="A."), generate, [])
    # retries_left 3: the main call, then fix-format calls nested at 2, 1 and 0 retries left.
    assert len(prompts) == 4 and all(p.startswith(ragas.FIX_OUTPUT_FORMAT.instruction) for p in prompts[1:])


async def test_fixed_reply_is_parsed_strictly():
    replies = iter(['["A."]', json.dumps({"text": "Output: not JSON"})])

    async def generate(prompt):
        return next(replies)

    with pytest.raises(ragas.ScorerFailed, match="after the fix-format retry"):
        await ragas.generate_model(ragas.STATEMENT_GENERATOR, ragas.StatementGeneratorInput(question="Q", answer="A."), generate, [])


# --- scoring ---


def _judge(statements: dict[str, list[str]], classification: dict):
    async def generate(prompt):
        data = json.loads(prompt.rsplit("input: ", 1)[1].rsplit("\nOutput: ", 1)[0])
        if "ground_truth" in data:
            return json.dumps(classification)
        return json.dumps({"statements": statements[data["answer"]]})

    return generate


def _cls(tp=(), fp=(), fn=()):
    return {k: [{"statement": s, "reason": "r"} for s in v] for k, v in (("TP", tp), ("FP", fp), ("FN", fn))}


async def _same(texts):
    return [[1.0, 0.0]] * len(texts)


async def _orthogonal(texts):
    return [[1.0, 0.0], [0.0, 1.0]]


async def test_perfect_answer_scores_one():
    r = await ragas.answer_correctness("Q?", "A.", "A.", _judge({"A.": ["A."]}, _cls(tp=["A."])), _same)
    assert r["accuracy"] == pytest.approx(1.0) and r["judge_calls"] == 3 and r["fix_format_calls"] == 0


async def test_similarity_is_raw_cosine_not_rescaled():
    # The official scorer maps cosine 0 to 0.5; RAGAS's AnswerSimilarity does not.
    r = await ragas.answer_correctness("Q?", "ans", "gt", _judge({"ans": ["a1", "a2"], "gt": ["g"] * 5}, _cls(tp=["a1"], fp=["a2"], fn=["g"] * 5)), _orthogonal)
    assert r["factual_correctness"] == pytest.approx(0.25) and r["semantic_similarity"] == pytest.approx(0.0)
    assert r["accuracy"] == pytest.approx(0.75 * 0.25)


async def test_empty_statement_lists_skip_the_classifier_and_blank_answers_embed_a_space():
    seen = []

    async def embed(texts):
        seen.extend(texts)
        return [[1.0, 0.0]] * len(texts)

    r = await ragas.answer_correctness("Q?", "", "x", _judge({"": [], "x": []}, {}), embed)
    assert r["factual_correctness"] == 1.0 and r["judge_calls"] == 2 and seen == ["x", " "]


async def test_a_failed_sample_reports_its_judge_calls():
    async def generate(prompt):
        return "nope"

    with pytest.raises(ragas.ScorerFailed) as e:
        await ragas.answer_correctness("Q?", "A.", "A.", generate, _same)
    assert len(e.value.trace) == 4


# --- the anchor's embedder ---


def test_offline_anchor_embedder_is_fake_and_never_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    monkeypatch.setenv("APE_CACHE", str(tmp_path))
    assert anchor_embedder(Config()).identity == "fake-bow"
    v = fake_embedder()._documents(["", "Basal cell carcinoma."])
    assert np.all(np.linalg.norm(v, axis=1) > 0)


@pytest.mark.skipif(not os.environ.get("APE_TEST_BGE"), reason="real bge-large-en-v1.5 (~1.3 GB): set APE_TEST_BGE=1")
def test_real_bge_poolings():
    from ape.anchor.bge import DIM, BgeOnnx

    m = BgeOnnx()
    docs = m.documents(["Basal cell carcinoma is a skin cancer.", "Melanoma is a skin cancer.", "The boiling point of water is 100 degrees."])
    assert docs.shape == (3, DIM) and np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert docs[0] @ docs[1] > docs[0] @ docs[2], "related medical sentences are closer"
    q = m.queries(["What is melanoma?"])
    assert not np.allclose(q[0], m.documents(["What is melanoma?"])[0]), "queries carry bge's instruction"
    assert QUERY_INSTRUCTION.startswith("Represent this question")
    mean = m.mean_pooled(["x " * 2000])  # truncated at 512 tokens, unnormalised
    assert mean.shape == (1, DIM) and abs(np.linalg.norm(mean[0]) - 1.0) > 1e-3
