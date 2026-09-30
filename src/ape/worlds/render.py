"""Shared chunker: the one chunk set that every arm and both KG builders consume.

Paragraphs are never split, so every chunk's fact IDs are exact. Chunk IDs are
stable (`{doc_id}#{index:03d}`), which is what lets LightRAG's `file_path`, APG's
`props.sourceChunkIds` and S3s results all map back to the same facts.
"""

from dataclasses import dataclass

from ..llm.tokens import count_tokens
from .spec import Document, World

DEFAULT_CHUNK_TOKENS = 200


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    fact_ids: tuple[str, ...]


def chunk_document(doc: Document, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf_text: list[str] = []
    buf_facts: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_text, buf_facts, buf_tokens
        if buf_text:
            header = f"[{doc.title}]"
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{len(chunks):03d}",
                    doc_id=doc.id,
                    text=header + "\n" + "\n\n".join(buf_text),
                    fact_ids=tuple(dict.fromkeys(buf_facts)),
                )
            )
        buf_text, buf_facts, buf_tokens = [], [], 0

    for p in doc.paragraphs:
        n = count_tokens(p.text)
        if buf_text and buf_tokens + n > max_tokens:
            flush()
        buf_text.append(p.text)
        buf_facts.extend(p.fact_ids)
        buf_tokens += n
    flush()
    return chunks


def chunk_world(world: World, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> list[Chunk]:
    return [c for doc in world.documents for c in chunk_document(doc, max_tokens)]


def corpus_text(world: World) -> str:
    """The full corpus, used verbatim by the monolith arm (S1)."""
    return "\n\n".join(doc.text for doc in world.documents)
