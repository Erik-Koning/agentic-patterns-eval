"""One token counter shared by every arm, so context budgets are comparable.

APG compose, LightRAG budgets, S3s packing and S7 token matching all count with
this function; realized context tokens are logged with it too.
"""

from functools import lru_cache

import tiktoken

ENCODING = "o200k_base"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    ids = _encoder().encode(text, disallowed_special=())
    return text if len(ids) <= max_tokens else _encoder().decode(ids[:max_tokens])
