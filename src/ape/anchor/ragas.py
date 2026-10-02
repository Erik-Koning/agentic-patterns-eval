"""RAGAS `answer_correctness` as GraphRAG-Bench scored LightRAG's three older Medical numbers (PC1, D-025).

Provenance:
- Fact Retrieval, Complex Reasoning and Contextual Summarize were published before the benchmark's own
  scorer existed: on the leaderboard by 2025-05-28 and in arXiv 2506.05690 v1 (2025-06-06).
- They were scored with a vendored copy of RAGAS's `answer_correctness`. It was uploaded to
  GraphRAG-Bench/GraphRAG-Benchmark in commit e6305f5 (2025-06-09) and removed as "old evaluation codes"
  on 2025-06-14.
- The custom scorer that replaced it (294aa33, rewritten in 1899c37 on 2025-07-21) produced none of them.

This module reproduces that scorer's *behaviour*, not just its prompt text. The instructions and few-shot
examples are byte-identical to today's official prompts (`ape.anchor.accuracy`); everything around them
differs.

- **Rendering:** `PydanticPrompt.to_string` (ragas/prompt/pydantic_prompt.py). The order is: the
  instruction, a JSON-Schema output signature, the examples as indented JSON, the input as indented JSON,
  then an "Output: " cue. One user message per call.
- **Parsing:** `RagasOutputParser.parse_output_string`.
  - `extract_json` takes the first balanced `{...}` or `[...]`, starting at a ```json marker if there is one.
  - LangChain's `PydanticOutputParser` then strips markdown, runs `parse_partial_json` and validates.
  - On failure, one `FixOutputFormat` call re-asks the judge. Its reply is parsed without `extract_json`,
    and that parse failing is final.
  - RAGAS's `evaluate` turns that into NaN, so the sample leaves the mean. Here it is `ScorerFailed`.
  - `generate` is called with `retries_left=3`. So a fix-format reply that is itself malformed gets its
    own nested fix call, at most twice.
- **Statements:** the response's statements, then the reference's (`StatementGeneratorPrompt`, the
  question included). If both lists are empty, F1 = 1 and the classifier is skipped.
- **Similarity:** `AnswerSimilarity`. Each text (`or " "`) goes through `embed_text` → `aembed_documents`:
  bge-large-en-v1.5 *document* embeddings with no query instruction. The score is their raw cosine,
  with no (cos+1)/2 rescaling.
- **Weights:** `np.average([f1, similarity], weights=[0.75, 0.25])`.
- **Judge:** RAGAS sets temperature 1e-8 (`get_temperature(n=1)`). Our judge role runs at temperature 0.

LangChain core's JSON helpers are vendored below, verbatim. They come from langchain-core 0.3.65
(utils/json.py, output_parsers/json.py and output_parsers/pydantic.py; MIT). `parse_partial_json` is
itself adapted by LangChain from open-interpreter (MIT).
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

WEIGHTS = (0.75, 0.25)
RETRIES = 3  # PydanticPrompt.generate's default `retries_left`

Generate = Callable[[str], Awaitable[str]]
Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


class ParseError(Exception):
    """LangChain's OutputParserException."""


class ScorerFailed(Exception):
    """The sample would be NaN in RAGAS's evaluate: dropped from the per-type mean, counted as a parse failure."""


# --- LangChain core 0.3.65, langchain_core/utils/json.py (verbatim) ---------------------------------------


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    return re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_chars = []
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        new_char = char
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                new_char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                # Mismatched closing character; the input is malformed.
                return None

        # Append the processed character to the new string.
        new_chars.append(new_char)

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        if escaped:  # Remoe unterminated escape character
            new_chars.pop()
        new_chars.append('"')

    # Reverse the stack to get the closing characters.
    stack.reverse()

    # Try to parse mods of string until we succeed or run out of characters.
    while new_chars:
        # Close any remaining open structures in the reverse
        # order that they were opened.
        # Attempt to parse the modified string as JSON.
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_chars.pop()

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_markdown(json_string: str, *, parser: Callable[[str], Any] = parse_partial_json) -> dict:
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        # Else, use the content within the backticks
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(json_str: str, *, parser: Callable[[str], Any] = parse_partial_json) -> dict:
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


# --- end LangChain core ---------------------------------------------------------------------------------


def pydantic_parse(text: str, model: type[BaseModel]) -> BaseModel:
    """LangChain's `PydanticOutputParser.parse`: JsonOutputParser.parse_result (strip, parse_json_markdown),
    then `model_validate`; both failures raise `ParseError` (OutputParserException)."""
    text = text.strip()
    try:
        obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid json output: {text}") from e
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        raise ParseError(f"Failed to parse {model.__name__} from completion {json.dumps(obj)}. Got: {e}") from e


def extract_json(text: str) -> str:
    """ragas/prompt/utils.py `extract_json` (verbatim logic): the first balanced JSON structure."""
    # check for markdown indicator; if present, start there
    md_json_idx = text.find("```json")
    if md_json_idx != -1:
        text = text[md_json_idx:]

    # search for json delimiter pairs
    left_bracket_idx = text.find("[")
    left_brace_idx = text.find("{")

    indices = [idx for idx in (left_bracket_idx, left_brace_idx) if idx != -1]
    start_idx = min(indices) if indices else None

    # If no delimiter found, return the original text
    if start_idx is None:
        return text

    # Identify the exterior delimiters defining JSON
    open_char = text[start_idx]
    close_char = "]" if open_char == "[" else "}"

    # Initialize a count to keep track of delimiter pairs
    count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            count += 1
        elif char == close_char:
            count -= 1

        # When count returns to zero, we've found a complete structure
        if count == 0:
            return text[start_idx : i + 1]

    return text  # In case of unbalanced JSON, return the original text


# --- the prompts' pydantic models (ragas metrics _faithfulness.py, _answer_correctness.py, prompt/base.py) -----


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    statements: list[str] = Field(description="The generated statements")


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason]
    FP: list[StatementsWithReason]
    FN: list[StatementsWithReason]


class StringIO(BaseModel):
    text: str


class OutputStringAndPrompt(BaseModel):
    output_string: str
    prompt_value: str


@dataclass(frozen=True)
class PromptSpec:
    instruction: str
    output_model: type[BaseModel]
    examples: tuple[tuple[BaseModel, BaseModel], ...] = ()


def to_string(spec: PromptSpec, data: BaseModel | None = None) -> str:
    """`PydanticPrompt.to_string` (with `_generate_output_signature` and `_generate_examples`), verbatim logic."""
    signature = (
        f"Please return the output in a JSON format that complies with the "
        f"following schema as specified in JSON Schema:\n"
        f"{json.dumps(spec.output_model.model_json_schema())}"
        "Do not use single quotes in your response but double quotes,"
        "properly escaped with a backslash."
    )
    if spec.examples:
        example_strings = [
            f"Example {idx + 1}\n" + "Input: " + inp.model_dump_json(indent=4) + "\n" + "Output: " + out.model_dump_json(indent=4)
            for idx, (inp, out) in enumerate(spec.examples)
        ]
        examples = "\n--------EXAMPLES-----------\n" + "\n\n".join(example_strings)
    else:
        examples = ""
    return (
        f"{spec.instruction}\n"
        + signature
        + "\n"
        + examples
        + "\n-----------------------------\n"
        + "\nNow perform the same with the following input\n"
        + ("input: " + data.model_dump_json(indent=4, exclude_none=True) + "\n" if data is not None else "Input: (None)\n")
        + "Output: "
    )


STATEMENT_GENERATOR = PromptSpec(
    instruction="Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.",
    output_model=StatementGeneratorOutput,
    examples=(
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        ),
    ),
)

CORRECTNESS_CLASSIFIER = PromptSpec(
    instruction="Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.",
    output_model=ClassificationWithReason,
    examples=(
        (
            QuestionAnswerGroundTruth(
                question="What powers the sun and what is its primary function?",
                answer=[
                    "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "The primary function of the sun is to provide light to the solar system.",
                ],
                ground_truth=[
                    "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "The sun's light plays a critical role in Earth's climate system.",
                    "Sunlight helps to drive the weather and ocean currents.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="The primary function of the sun is to provide light to the solar system.",
                        reason="This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
                    )
                ],
                FP=[
                    StatementsWithReason(
                        statement="The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                        reason="This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
                    )
                ],
                FN=[
                    StatementsWithReason(
                        statement="The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                        reason="This accurate description of the sun’s power source is not included in the answer.",
                    ),
                    StatementsWithReason(
                        statement="This fusion process in the sun's core releases a tremendous amount of energy.",
                        reason="This process and its significance are not mentioned in the answer.",
                    ),
                    StatementsWithReason(
                        statement="The energy from the sun provides heat and light, which are essential for life on Earth.",
                        reason="The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
                    ),
                    StatementsWithReason(
                        statement="The sun's light plays a critical role in Earth's climate system.",
                        reason="This broader impact of the sun’s light on Earth's climate system is not addressed in the answer.",
                    ),
                    StatementsWithReason(
                        statement="Sunlight helps to drive the weather and ocean currents.",
                        reason="The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
                    ),
                ],
            ),
        ),
        (
            QuestionAnswerGroundTruth(
                question="What is the boiling point of water?",
                answer=["The boiling point of water is 100 degrees Celsius at sea level"],
                ground_truth=[
                    "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                    "The boiling point of water can change with altitude.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="The boiling point of water is 100 degrees Celsius at sea level",
                        reason="This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level.",
                    )
                ],
                FP=[],
                FN=[
                    StatementsWithReason(
                        statement="The boiling point of water can change with altitude.",
                        reason="This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer.",
                    )
                ],
            ),
        ),
    ),
)

FIX_OUTPUT_FORMAT = PromptSpec(
    instruction="The output string did not satisfy the constraints given in the prompt. Fix the output string and return it.",
    output_model=StringIO,
)


async def generate_model(spec: PromptSpec, data: BaseModel, generate: Generate, trace: list[str], retries_left: int = RETRIES) -> BaseModel:
    """`PydanticPrompt.generate` with `RagasOutputParser`: one judge call, parse, at most one fix-format re-ask."""
    prompt = to_string(spec, data)
    output = await generate(prompt)
    trace.append(spec.output_model.__name__)  # one entry per judge call; fix-format calls are "StringIO"
    try:
        return pydantic_parse(extract_json(output), spec.output_model)
    except ParseError as e:
        if retries_left == 0:
            raise ScorerFailed(f"{spec.output_model.__name__}: unparseable, no retries left") from e
    fixed = await generate_model(FIX_OUTPUT_FORMAT, OutputStringAndPrompt(output_string=output, prompt_value=prompt), generate, trace, retries_left - 1)
    try:
        return pydantic_parse(fixed.text, spec.output_model)
    except ParseError as e:
        raise ScorerFailed(f"{spec.output_model.__name__}: unparseable after the fix-format retry") from e


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """ragas/metrics/utils.py `fbeta_score` (same as the official scorer's)."""
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


def raw_cosine(a: list[float], b: list[float]) -> float:
    """`AnswerSimilarity`: both vectors normalised, then their dot product (no rescaling)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))


async def answer_correctness(question: str, answer: str, ground_truth: str, generate: Generate, embed_documents: Embed) -> dict:
    """One sample's RAGAS ACC in [−0.25, 1] plus its parts.

    The floor is −0.25 because raw cosine can go negative; that never happens with bge in practice.
    Raises `ScorerFailed` where RAGAS would give NaN. In that case the trace of judge calls is on the
    exception (`.trace`), so the caller can still count them."""
    trace: list[str] = []
    try:
        statements = {}
        for key, text in (("response", answer), ("reference", ground_truth)):
            out = await generate_model(STATEMENT_GENERATOR, StatementGeneratorInput(question=question, answer=text), generate, trace)
            statements[key] = out.statements
        if not all(v == [] for v in statements.values()):
            data = QuestionAnswerGroundTruth(question=question, answer=list(statements["response"]), ground_truth=list(statements["reference"]))
            c = await generate_model(CORRECTNESS_CLASSIFIER, data, generate, trace)
            f1 = fbeta_score(len(c.TP), len(c.FP), len(c.FN))
        else:
            f1 = 1.0
    except ScorerFailed as e:
        e.trace = trace
        raise
    gt_vec, ans_vec = await embed_documents([ground_truth or " ", answer or " "])
    similarity = raw_cosine(gt_vec, ans_vec)
    return {
        "accuracy": float(np.average([f1, similarity], weights=WEIGHTS)),
        "factual_correctness": f1,
        "semantic_similarity": similarity,
        "answer_statements": statements["response"],
        "reference_statements": statements["reference"],
        "judge_calls": len(trace),
        "fix_format_calls": trace.count("StringIO"),
    }
