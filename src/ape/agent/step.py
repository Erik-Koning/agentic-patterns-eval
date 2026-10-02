"""Shared per-step query and tool-exposure rules. Every arm uses exactly these, so
arms differ only in how they turn a query into context.
"""

import re

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolDef

from ..kb.context import ContextResult
from ..llm.tokens import truncate_to_tokens

STEP_QUERY_MAX_TOKENS = 400


def step_query(task_prompt: str, messages: list[ChatMessage]) -> str:
    """Task prompt plus the observations returned since the last assistant turn."""
    latest: list[str] = []
    for m in reversed(messages):
        if isinstance(m, ChatMessageTool):
            latest.append(m.text)
        elif isinstance(m, ChatMessageAssistant):
            break
    obs = "\n".join(reversed(latest))
    return truncate_to_tokens(f"{task_prompt}\n{obs}".strip(), STEP_QUERY_MAX_TOKENS)


def exposed_tools(all_tools: dict[str, ToolDef], ctx: ContextResult, policy: str, always_on: list[str]) -> list[ToolDef]:
    """TE-all binds every tool. TE-retrieved binds always-on tools plus the arm's own
    tool scope when it has one (APG allowlists), else the tools named in the delivered context. The name match
    ignores case: LightRAG's extraction title-cases entity names, so a tool can arrive as "Wrong_Item_Credit"."""
    if policy == "all":
        names = list(all_tools)
    elif ctx.tools is not None:
        names = [*always_on, *ctx.tools]
    elif policy == "retrieved":
        names = [*always_on, *(n for n in all_tools if re.search(rf"\b{re.escape(n)}\b", ctx.text, flags=re.IGNORECASE))]
    else:
        raise ValueError(policy)
    return [all_tools[n] for n in dict.fromkeys(names) if n in all_tools]
