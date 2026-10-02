"""A scripted stand-in for `openai.AsyncOpenAI().chat.completions.create` that returns realistic, imperfect replies
(refusals, empty or truncated content, prose instead of records), for the build-robustness tests. No network."""

import asyncio
from types import SimpleNamespace


def completion(content, refusal=None, finish_reason="stop", prompt_tokens=100, completion_tokens=40, reasoning=10):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    msg = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(model="gpt-6-luna-2026-09-01", usage=usage, choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


class ScriptedChat:
    """`script(call_index, messages, kwargs)` returns a completion (or raises). `delay` makes calls overlap."""

    def __init__(self, script, delay: float = 0.0):
        self.script = script
        self.delay = delay
        self.calls = 0
        self.completed = 0
        self.messages: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, **kwargs):
        i = self.calls
        self.calls += 1
        self.messages.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        out = self.script(i, messages, kwargs)
        self.completed += 1
        return out
