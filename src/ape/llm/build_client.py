"""Build-time OpenAI chat client (APG authoring, LightRAG extraction). Runs outside Inspect,
so every call is metered in the ledger under role "build".
"""

import json
from typing import Any

from .ledger import Ledger, LedgerEntry


class BuildLlm:
    def __init__(self, model: str, ledger: Ledger, context: dict, client: Any = None):
        self.model = model
        self.ledger = ledger
        self.context = context
        self._client = client

    def _client_(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client

    async def chat(self, messages: list[dict], **kwargs) -> str:
        resp = await self._client_().chat.completions.create(model=self.model, messages=messages, **kwargs)
        u = resp.usage
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
