"""BAAI/bge-large-en-v1.5 on CPU for the GraphRAG-Bench anchor (PC1, D-025): the paper's embedding model.

GraphRAG-Bench produced its LightRAG numbers with bge-large-en-v1.5 in two places, each with its own pooling:

- **Retrieval** (LightRAG's entity, relation and chunk vectors; paper App. H.2, Examples/run_lightrag.py):
  LightRAG 1.2.5's `hf_embed` runs `transformers.AutoModel` and takes `last_hidden_state.mean(dim=1)`,
  unnormalised. `mean_pooled` does the same per text. 1.2.5 batches several texts and its mean also
  covers the padding positions; computing each text on its own (the batch-of-one case, where `hf_embed`
  has no padding) keeps the vectors independent of LightRAG's batching, which 1.5.7 does differently.
- **Answer similarity** in the judge (LangChain's `HuggingFaceBgeEmbeddings`, i.e. sentence-transformers:
  CLS pooling, then L2 normalisation, `max_seq_length` 512, newlines replaced by spaces):
  - `documents`: `embed_documents`, no instruction. The vendored-RAGAS scorer (`ape.anchor.ragas`) used this.
  - `queries`: `embed_query`, prefixed with bge's query instruction. Today's official scorer
    (Evaluation/metrics/answer_accuracy.py) calls `aembed_query` on both texts.

The model is BAAI's own ONNX export at a pinned revision, run with onnxruntime and the repo's
tokenizer.json (BertNormalizer lower-casing; truncation at 512 with [CLS]/[SEP]). The files come from the
Hugging Face cache (downloaded once, ~1.3 GB). Only the anchor uses this; the gate keeps its OpenAI embeddings.

`anchor_embedder(cfg)` returns the fake bag-of-words embedder when `APE_EMBEDDINGS=fake`, so offline runs
and tests never download the model.
"""

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REPO = "BAAI/bge-large-en-v1.5"
REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"  # 2024-02-21; the onnx/ export is BAAI's own
FILES = ("onnx/model.onnx", "tokenizer.json")
DIM = 1024
MAX_LENGTH = 512  # sentence_bert_config.json max_seq_length; hf_embed truncates at the tokenizer's 512 too
# LangChain's DEFAULT_QUERY_BGE_INSTRUCTION_EN, which HuggingFaceBgeEmbeddings prepends in embed_query.
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
BATCH = 16
CACHE_ITEMS = 50_000


class BgeOnnx:
    """bge-large-en-v1.5 (pinned) through onnxruntime. Thread-safe; the session loads on first use."""

    identity = f"{REPO}@{REVISION}"

    def __init__(self, threads: int | None = None):
        self._threads = threads
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def files(self) -> dict[str, str]:
        from huggingface_hub import hf_hub_download

        return {f: hf_hub_download(REPO, f, revision=REVISION) for f in FILES}

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        paths = self.files()
        tok = Tokenizer.from_file(paths["tokenizer.json"])
        tok.enable_truncation(MAX_LENGTH)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        if self._threads:
            opts.intra_op_num_threads = self._threads
        self._session = ort.InferenceSession(paths["onnx/model.onnx"], sess_options=opts, providers=["CPUExecutionProvider"])
        self._tokenizer = tok

    def hidden_states(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Each text's `last_hidden_state` over its own tokens ([CLS] ... [SEP], no padding): shape (tokens, 1024)."""
        with self._lock:
            self._load()
            out: list[np.ndarray] = []
            for i in range(0, len(texts), BATCH):
                enc = self._tokenizer.encode_batch(list(texts[i : i + BATCH]))
                ids = np.array([e.ids for e in enc], dtype=np.int64)
                mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
                types = np.array([e.type_ids for e in enc], dtype=np.int64)
                (hidden,) = self._session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})
                out += [h[: int(m.sum())] for h, m in zip(hidden, mask, strict=True)]
            return out

    def _pooled(self, kind: str, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if (kind, t) not in self._cache]
        if missing:
            for t, h in zip(missing, self.hidden_states(missing), strict=True):
                if kind == "mean":
                    v = h.mean(axis=0)
                else:
                    v = h[0] / np.linalg.norm(h[0])
                self._cache[(kind, t)] = v.astype(np.float32)
                while len(self._cache) > CACHE_ITEMS:
                    self._cache.popitem(last=False)
        return np.stack([self._cache[(kind, t)] for t in texts]) if texts else np.zeros((0, DIM), dtype=np.float32)

    # The three poolings the anchor needs (module docstring). Sync; the async wrappers run them in a thread.
    def mean_pooled(self, texts: Sequence[str]) -> np.ndarray:
        return self._pooled("mean", list(texts))

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._pooled("cls", [t.replace("\n", " ") for t in texts])

    def queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._pooled("cls", [QUERY_INSTRUCTION + t.replace("\n", " ") for t in texts])


@dataclass
class AnchorEmbedder:
    """Async face of an embedder for the anchor: `retrieval` for LightRAG, `documents` / `queries` for the judges."""

    identity: str
    dim: int
    _mean: object
    _documents: object
    _queries: object

    async def retrieval(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._mean, texts)

    async def documents(self, texts: list[str]) -> list[list[float]]:
        return (await asyncio.to_thread(self._documents, texts)).tolist()

    async def queries(self, texts: list[str]) -> list[list[float]]:
        return (await asyncio.to_thread(self._queries, texts)).tolist()


_BGE: BgeOnnx | None = None
_BGE_LOCK = threading.Lock()


def bge() -> BgeOnnx:
    """The process-wide bge session (one ~1.3 GB model in memory, shared by every anchor task)."""
    global _BGE
    with _BGE_LOCK:
        if _BGE is None:
            _BGE = BgeOnnx()
        return _BGE


def fake_embedder() -> AnchorEmbedder:
    """Offline stand-in: the hashed bag-of-words vectors of `ape.llm.fake` for all three poolings."""
    from ..llm.fake import DIM as FAKE_DIM, bow_vector

    def vectors(texts):
        out = np.array([bow_vector(t) for t in texts], dtype=np.float32).reshape(len(texts), FAKE_DIM)
        out[~out.any(axis=1), 0] = 1.0  # blank text: a unit vector, never zero (cosine stays defined, as with bge)
        return out

    return AnchorEmbedder("fake-bow", FAKE_DIM, vectors, vectors, lambda texts: vectors([QUERY_INSTRUCTION + t for t in texts]))


def anchor_embedder(cfg) -> AnchorEmbedder:
    """bge-large-en-v1.5 (pinned), or the fake embedder when `APE_EMBEDDINGS=fake` (offline runs and tests)."""
    if cfg.embeddings_backend == "fake":
        return fake_embedder()
    m = bge()
    return AnchorEmbedder(f"{BgeOnnx.identity}#mean-pool", DIM, m.mean_pooled, m.documents, m.queries)
