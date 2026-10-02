"""Build-time OpenAI chat client (APG authoring, LightRAG extraction). Runs outside Inspect,
so every call is metered in the ledger under role "build".

`reasoning_effort` comes from the model profile (`ape.models.build_settings`, D-015). It is
sent on every call and recorded in each ledger entry's context; None sends nothing, for
models that take no effort.

The lazily created OpenAI client retries a failed call up to `MAX_RETRIES` times (the SDK backs off
and honours Retry-After), so parallel builds (FX-4) ride out rate limits instead of failing a world.

**Unusable responses** (RELIABILITY_REVIEW K1/K2). A call can succeed at the API level and still return nothing
usable: a refusal (`message.refusal`), empty content, output cut at the token limit (`finish_reason="length"`), a
content filter, or text in the wrong shape. `complete` returns the content with those signals (`BuildResponse`);
the two build paths act on them:
- `json` (APG authoring): re-asks on a refusal, empty or truncated reply, and on invalid JSON asks once more with
  a repair instruction (the bad reply and the parse error); after `JSON_ATTEMPTS` calls it raises
  `BuildResponseError`.
- `lightrag_func` (LightRAG extraction and summaries): re-asks on a refusal, an empty reply, or an initial
  entity-extraction reply that has neither LightRAG's record delimiter nor its completion marker (prose instead of
  records). A truncated reply with content is returned as LightRAG's `TruncatedResponse`, which LightRAG parses
  but never caches. After `LIGHTRAG_ATTEMPTS` calls it raises `BuildResponseError`; LightRAG then marks the
  document FAILED and `ape.lgr.build` decides whether the index is usable. Nothing unusable reaches LightRAG's LLM
  cache, so a rebuild re-asks exactly those chunks.

**Metering** (K7). Every completed call is metered, including retries and unusable replies, from the API's usage.
A call cancelled while in flight (a sibling's fatal error, a LightRAG timeout) or that times out client-side may
still be billed: it is recorded with its estimated input tokens, zero output tokens and `"status"` in its context.
The output a server produced for a call we gave up on cannot be known, so it is not metered. Calls that fail with
an API error are not billed and not recorded.

`max_failed_chunks(n)` is the shared tolerance for a world build: how many of its n chunks may still be unusable
after these attempts (APE_BUILD_MAX_FAILED_SHARE, default 2%, rounded down) before the world fails.
"""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from .ledger import Ledger, LedgerEntry
from .tokens import count_tokens

MAX_RETRIES = 6
JSON_ATTEMPTS = 3
LIGHTRAG_ATTEMPTS = 3
MAX_FAILED_SHARE = 0.02
REPAIR_PROMPT = "Your reply was not valid JSON for the required schema ({error}). Reply again with only the JSON object."
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


def max_failed_chunks(n_chunks: int) -> int:
    """How many of a world's n chunks may remain unusable after retries before the world's build fails."""
    raw = os.environ.get("APE_BUILD_MAX_FAILED_SHARE", "").strip()
    share = float(raw) if raw else MAX_FAILED_SHARE
    if not 0 <= share < 1:
        raise ValueError(f"APE_BUILD_MAX_FAILED_SHARE must be in [0, 1), got {share}")
    return math.floor(share * n_chunks)


class BuildResponseError(RuntimeError):
    """A build call returned nothing usable after its attempts; `problems` lists what each attempt returned."""

    def __init__(self, what: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{what}: no usable reply after {len(problems)} attempt(s): {'; '.join(problems)}")


@dataclass
class BuildResponse:
    text: str
    finish_reason: str | None = None
    refusal: str | None = None

    @property
    def problem(self) -> str | None:
        """Why this reply is unusable as is, or None."""
        if self.refusal:
            return f"refusal ({self.refusal[:80]})"
        if self.finish_reason == "content_filter":
            return "content filter"
        if self.finish_reason == "length":
            return "truncated at the output token limit"
        if not self.text.strip():
            return "empty reply"
        return None


def strip_fences(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text


def _lightrag_markers() -> tuple[str, str]:
    from lightrag.prompt import PROMPTS

    return PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"]


class BuildLlm:
    def __init__(self, model: str, ledger: Ledger, context: dict, client: Any = None, reasoning_effort: str | None = None):
        self.model = model
        self.ledger = ledger
        self.reasoning_effort = reasoning_effort
        self.context = {**context, "reasoning_effort": reasoning_effort} if reasoning_effort else context
        self._client = client
        # The last call's served snapshot (`response.model`) and usage, for the readiness probe (E3).
        self.last_model: str | None = None
        self.last_usage: dict | None = None

    def _client_(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(max_retries=MAX_RETRIES)
        return self._client

    def _record_unfinished(self, messages: list[dict], status: str) -> None:
        """A call we stopped waiting for may still be billed: record its input (estimated), no output."""
        estimate = sum(count_tokens(str(m.get("content") or "")) for m in messages)
        self.ledger.append(LedgerEntry(role="build", model=self.model, kind="chat", input_tokens=estimate, context={**self.context, "status": status}))

    async def complete(self, messages: list[dict], **kwargs) -> BuildResponse:
        """One metered chat call; the reply's text with its refusal and finish reason."""
        if self.reasoning_effort:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        try:
            resp = await self._client_().chat.completions.create(model=self.model, messages=messages, **kwargs)
        except asyncio.CancelledError:
            self._record_unfinished(messages, "cancelled")
            raise
        except Exception as e:
            if type(e).__name__ in ("APITimeoutError", "TimeoutError"):
                self._record_unfinished(messages, "timeout")
            raise
        u = resp.usage
        self.last_model = getattr(resp, "model", None)
        self.last_usage = u.model_dump() if hasattr(u, "model_dump") else None
        self.ledger.append(
            LedgerEntry(
                role="build",
                model=self.model,
                kind="chat",
                input_tokens=u.prompt_tokens,
                output_tokens=u.completion_tokens,
                cached_input_tokens=getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
                context=self.context,
            )
        )
        choice = resp.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        return BuildResponse(
            text=getattr(message, "content", None) or "",
            finish_reason=getattr(choice, "finish_reason", None),
            refusal=refusal if isinstance(refusal, str) else None,
        )

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """The reply text as is (no validation; the readiness probe uses this)."""
        return (await self.complete(messages, **kwargs)).text

    async def json(self, system: str, user: str, name: str, schema: dict, attempts: int = JSON_ATTEMPTS) -> dict:
        """A JSON object under a strict schema, re-asked or repaired until usable (module docstring)."""
        base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        messages = base
        fmt = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        problems: list[str] = []
        for _ in range(attempts):
            r = await self.complete(messages, response_format=fmt)
            if r.problem:
                problems.append(r.problem)
                messages = base
                continue
            try:
                out = json.loads(strip_fences(r.text))
            except ValueError as e:
                problems.append(f"invalid JSON ({e})")
                messages = base + [{"role": "assistant", "content": r.text}, {"role": "user", "content": REPAIR_PROMPT.format(error=e)}]
                continue
            if not isinstance(out, dict):
                problems.append(f"JSON {type(out).__name__}, not an object")
                messages = base
                continue
            return out
        raise BuildResponseError(f"{self.model} {name}", problems)

    def lightrag_func(self):
        """Adapter to LightRAG's `llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs)`, with the
        reply checks of the module docstring."""
        from lightrag.utils import TruncatedResponse

        tuple_delim, done_marker = _lightrag_markers()

        async def llm(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(history_messages or []) + [{"role": "user", "content": prompt}]
            extra = {"response_format": kwargs["response_format"]} if kwargs.get("response_format") else {}
            # An initial entity-extraction call: its system prompt teaches the delimited record format.
            extraction = not history_messages and not extra and bool(system_prompt) and tuple_delim in system_prompt
            problems: list[str] = []
            for _ in range(LIGHTRAG_ATTEMPTS):
                r = await self.complete(messages, **extra)
                if r.finish_reason == "length" and r.text.strip() and not r.refusal:
                    return TruncatedResponse(r.text)  # parsed as far as it goes, never cached
                problem = r.problem
                if problem is None and extraction and tuple_delim not in r.text and done_marker not in r.text:
                    problem = "no extraction records (neither the record delimiter nor the completion marker)"
                if problem is None:
                    return r.text
                problems.append(problem)
            raise BuildResponseError(f"{self.model} LightRAG {'extraction' if extraction else 'call'}", problems)

        return llm
