"""Map delivered context back to world-spec facts, and score evidence at fact granularity.

**One rule for every arm** (RELIABILITY_REVIEW K3). A world fact counts as *delivered* when the text the agent
sees carries it, either

1. **verbatim:** one of the fact's renderings (its canonical text, or a document paragraph written for it) appears
   in the delivered text, compared after normalisation (case, whitespace, punctuation and JSON escapes ignored); or
2. **restated:** a single segment of the delivered text (a blank-line-separated block, or one JSON record of
   LightRAG's context) contains every one of the fact's *distinguishing tokens*: its digit groups (IDs, amounts,
   deadlines, dates) and its distinguishing words, i.e. the content words that occur in at most half of the
   world's facts (stopwords excluded, plurals folded). Template words shared by most facts ("effective", "team",
   "customer") may be missing or paraphrased. A segment that is itself a verbatim rendering of some fact restates
   nothing else, and facts with fewer than three distinguishing tokens count only verbatim.

Why this rule:
- Chunk-based arms (S1, S3s, S6, S7) deliver chunk text verbatim, so for them the rule gives exactly their chunks'
  facts; `ChunkIndex` is that shortcut, and tests check the two agree on every chunk of every family.
- KG arms deliver *derived* text: LightRAG entity and relation descriptions, APG leaves written by an author model.
  Crediting every fact of every source chunk of a delivered unit over-credits (one generic entity extracted from
  60 chunks claimed all 130 facts of a world while 21 were in the text); crediting nothing for a restatement
  under-credits. The rule credits what the agent can actually read, for every arm alike.
- It is strict on purpose (conservative for derived text): a restatement missing an amount, a deadline, a code or
  a distinguishing word (the document, the approver, the team) is not usable evidence for that fact, and requiring
  all of them keeps near-miss sibling facts (another region's policy, another tool of the same domain, another
  team's event on the same date) from being credited. A paraphrased distinguishing word is a miss.

`FactMatcher(world).delivered(text)` applies the rule; `fact_matcher(world)` caches one per world version.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable

from ..worlds.render import Chunk
from ..worlds.spec import World


class ChunkIndex:
    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.id: c for c in chunks}

    def facts(self, chunk_ids: list[str]) -> list[str]:
        out: list[str] = []
        for cid in chunk_ids:
            out.extend(self.by_id[cid].fact_ids)
        return list(dict.fromkeys(out))


def evidence_pr(delivered: list[str], gold: list[str]) -> tuple[float, float]:
    """(precision, recall) of delivered fact IDs against gold fact IDs; empty delivery has precision 0."""
    d, g = set(delivered), set(gold)
    hit = len(d & g)
    precision = hit / len(d) if d else 0.0
    recall = hit / len(g) if g else 1.0
    return precision, recall


# --- the delivered-text rule ----------------------------------------------------------------------

TEMPLATE_SHARE = 0.5  # a word in more than this share of the world's facts is template, not distinguishing
MIN_SIGNATURE = 3  # fewer distinguishing tokens than this: verbatim only
STOPWORDS = frozenset(
    "a an the of to in on at by for from with and or nor but if then than so as is are be been being was were has have "
    "had do does did not no any all each every this that these those it its into onto over under within without about "
    "after before up out per via must may can will shall should would could also only such there here which who whom "
    "whose what when where while their his her they them we you our your i me my he she us least most more less".split()
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d]\w*|\d\w*", re.UNICODE)
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\([nrt\"\\/])")
_BLOCK = re.compile(r"\n\s*\n")


def _unescape(text: str) -> str:
    """Undo JSON string escapes left in context text (a JSON value printed inside a larger string)."""

    def sub(m: re.Match) -> str:
        if m.group(1):
            return chr(int(m.group(1), 16))
        return {"n": "\n", "r": "\n", "t": " "}.get(m.group(2), m.group(2))

    return _ESCAPE.sub(sub, text)


def _fold(tok: str) -> str:
    if tok[0].isdigit():
        if "." in tok:  # 95.0 and 95 are the same value
            tok = tok.rstrip("0").rstrip(".")
        return tok
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def tokens(text: str) -> list[str]:
    """Normalised tokens: case-folded, thousands separators dropped, decimals canonical, plurals folded."""
    return [_fold(t) for t in _TOKEN.findall(_THOUSANDS.sub("", _unescape(text)).casefold())]


def _flat(tokens_: Iterable[str]) -> str:
    return " " + " ".join(tokens_) + " "


def segments(text: str) -> list[str]:
    """The units a restatement must fit in: blank-line-separated blocks of plain text, and one segment per JSON
    record line (LightRAG's context lists entities, relations and chunks as JSON lines). A record's single-paragraph
    values (name, type, description, ...) form one segment; a multi-paragraph value (a chunk's content) adds one
    segment per paragraph, as plain chunk text would."""
    out: list[str] = []
    plain: list[str] = []

    def flush() -> None:
        if plain:
            out.append("\n".join(plain))
            plain.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                rec = json.loads(s)
            except ValueError:
                rec = None
            if isinstance(rec, dict):
                flush()
                single, multi = [], []
                for v in rec.values():
                    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                        continue
                    (multi if _BLOCK.search(str(v)) else single).append(str(v))
                if single:
                    out.append(" | ".join(single))
                for v in multi:
                    out.extend(b for b in _BLOCK.split(v) if b.strip())
                continue
        if s:
            plain.append(line)
        else:
            flush()
    flush()
    return out


class FactMatcher:
    """The delivered-text rule (module docstring) for one world. Built once per world version; matching is
    index-driven, so a compile pays microseconds, not a scan of every fact."""

    def __init__(self, world: World):
        self.world_id = world.id
        renderings: dict[str, list[str]] = defaultdict(list)
        for doc in world.documents:
            for p in doc.paragraphs:
                for f in p.fact_ids:
                    renderings[f].append(p.text)
        self.kb_facts = [f for f in world.facts if f in renderings]  # facts the knowledge base renders
        self._facts = list(world.facts)
        fact_tokens = {f: tokens(world.facts[f].text) for f in self._facts}
        df: dict[str, int] = defaultdict(int)
        for toks in fact_tokens.values():
            for t in set(toks):
                df[t] += 1

        # Verbatim forms, indexed by their rarest token (a form present verbatim contains all of its tokens).
        self._forms: list[tuple[str, str]] = []  # (fact, flattened form)
        self._form_anchor: dict[str, list[int]] = defaultdict(list)
        for f in self._facts:
            for text in dict.fromkeys([world.facts[f].text, *renderings.get(f, [])]):
                toks = tokens(text)
                if not toks:
                    continue
                anchor = min(toks, key=lambda t: (df.get(t, 0), t))
                self._form_anchor[anchor].append(len(self._forms))
                self._forms.append((f, _flat(toks)))

        # Restatement signatures: every distinguishing token must be present, so the rarest one is a sufficient
        # index anchor.
        template = TEMPLATE_SHARE * len(self._facts)
        self._sig: dict[str, frozenset[str]] = {}
        self._sig_anchor: dict[str, list[str]] = defaultdict(list)
        for f, toks in fact_tokens.items():
            sig = frozenset(t for t in toks if t[0].isdigit() or (t not in STOPWORDS and len(t) > 1 and df[t] <= template))
            if len(sig) < MIN_SIGNATURE:
                continue
            self._sig[f] = sig
            self._sig_anchor[min(sig, key=lambda t: (df[t], t))].append(f)

    def delivered(self, text: str) -> list[str]:
        """World facts the text carries, in world order (verbatim or restated; module docstring)."""
        if not text or not text.strip():
            return []
        found: set[str] = set()
        for seg in (tokens(s) for s in segments(text)):
            flat, st = _flat(seg), set(seg)
            verbatim = False
            for anchor in st & self._form_anchor.keys():
                for i in self._form_anchor[anchor]:
                    f, form = self._forms[i]
                    if form in flat:
                        found.add(f)
                        verbatim = True
            if verbatim:  # a rendering of a fact restates nothing else
                continue
            for anchor in st & self._sig_anchor.keys():
                for f in self._sig_anchor[anchor]:
                    if f not in found and self._sig[f] <= st:
                        found.add(f)
        return [f for f in self._facts if f in found]

    def coverage(self, texts: Iterable[str]) -> float:
        """Share of the knowledge base's facts that some text carries (an authoring/extraction diagnostic)."""
        if not self.kb_facts:
            return 1.0
        got: set[str] = set()
        for t in texts:
            got.update(self.delivered(t))
        return sum(f in got for f in self.kb_facts) / len(self.kb_facts)


_MATCHERS: dict[tuple[str, str], FactMatcher] = {}


def fact_matcher(world: World) -> FactMatcher:
    """One matcher per world version (cached; worlds are immutable once generated)."""
    key = (world.id, world.content_hash())
    if key not in _MATCHERS:
        if len(_MATCHERS) > 64:
            _MATCHERS.clear()
        _MATCHERS[key] = FactMatcher(world)
    return _MATCHERS[key]
