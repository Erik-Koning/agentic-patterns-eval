"""Build-time OpenAI chat client (APG authoring, LightRAG extraction). Runs outside Inspect,
so every call is metered in the ledger under role "build".

`reasoning_effort` comes from the model profile (`ape.models.build_settings`, D-015). It is
sent on every call and recorded in each ledger entry's context; None sends nothing, for
models that take no effort.

The lazily created OpenAI client retries a failed call up to `MAX_RETRIES` times (the SDK backs off
and honours Retry-After), so parallel builds (FX-4) ride out rate limits instead of failing a world.
"""

import json
from typing import Any

from .ledger import Ledger, LedgerEntry

MAX_RETRIES = 6


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

    async def chat(self, messages: list[dict], **kwargs) -> str:
        if self.reasoning_effort:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        resp = await self._client_().chat.completions.create(model=self.model, messages=messages, **kwargs)
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
        return resp.choices[0].message.content or ""

    async def json(self, system: str, user: str, name: str, schema: dict) -> dict:
        text = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
        )
        return json.loads(text)

    def lightrag_func(self):
        """Adapter to LightRAG's `llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs)`."""

        async def llm(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(history_messages or []) + [{"role": "user", "content": prompt}]
            extra = {"response_format": kwargs["response_format"]} if kwargs.get("response_format") else {}
            return await self.chat(messages, **extra)

        return llm
