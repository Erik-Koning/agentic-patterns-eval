"""Map delivered context back to world-spec facts, and score evidence at fact granularity.

**One rule for every arm** (RELIABILITY_REVIEW K3). A world fact counts as *delivered* when the text the agent
sees carries it, either

1. **verbatim:** one of the fact's renderings (its canonical text, or a document paragraph written for it) appears
   in the delivered text, compared after normalisation (case, whitespace, punctuation and JSON escapes ignored); or
2. **restated:** one *window* of the delivered text contains every one of the fact's *distinguishing tokens*: its
   digit groups (IDs, amounts, deadlines, dates) and its distinguishing words, i.e. the content words that are
   not template. A word is template when it occurs in more than half of the world's facts, or in more than half of
   the facts *of the same kind* (policy, exception, tool, procedure, ...; kinds with at least `MIN_KIND_FACTS`
   facts). So an exception's own boilerplate ("amends", "instead", "as written") may be paraphrased even though
   exceptions are a minority of a world's facts. A segment that is itself a verbatim rendering of some fact
   restates nothing else, and facts with fewer than three distinguishing tokens count only verbatim.
   - **Windows.** A segment is a blank-line-separated block of plain text, or one JSON record of LightRAG's context.
     A record's description merged from several sources (LightRAG joins them with `<SEP>`) is split into its
     fragments, each kept with the record's other fields (name, type). A window is up to `window` consecutive
     sentences of one fragment or block, `window` being one more than the most sentences any of the world's fact
     renderings has. A fact's restatement fits in one; tokens pooled from unrelated fragments or distant sentences
     never jointly credit a fact.
   - **Normalisation** (both sides alike): case, punctuation and JSON escapes ignored; thousands separators dropped;
     the number words zero to twenty and the tens map to digits ("seven" = "7"); plurals and common inflections
     folded ("approve", "approves" and "approved" are one word; "amends" = "amend").

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

TEMPLATE_SHARE = 0.5  # a word in more than this share of the world's facts (or of its kind's facts) is template
MIN_KIND_FACTS = 4  # a kind's own template share is judged only with at least this many facts of that kind
MIN_SIGNATURE = 3  # fewer distinguishing tokens than this: verbatim only
GRAPH_FIELD_SEP = "<SEP>"  # lightrag.constants.GRAPH_FIELD_SEP: how LightRAG merges one unit's descriptions
NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"}
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
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _unescape(text: str) -> str:
    """Undo JSON string escapes left in context text (a JSON value printed inside a larger string)."""

    def sub(m: re.Match) -> str:
        if m.group(1):
            return chr(int(m.group(1), 16))
        return {"n": "\n", "r": "\n", "t": " "}.get(m.group(2), m.group(2))

    return _ESCAPE.sub(sub, text)


def _stem(w: str) -> str:
    """A light, symmetric inflection fold: approve/approves/approved/approving -> "approv", deny/denies/denied ->
    "deny", amend/amends/amended -> "amend", photo/photos -> "photo". Facts and delivered text are folded alike, so a
    rare collision of unrelated words only loosens a signature slightly."""
    if len(w) <= 3:
        return w
    if len(w) > 4 and w.endswith(("ies", "ied")):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us")):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w


def _fold(tok: str) -> str:
    tok = NUMBER_WORDS.get(tok, tok)
    if tok[0].isdigit():
        if "." in tok:  # 95.0 and 95 are the same value
            tok = tok.rstrip("0").rstrip(".")
        return tok
    return _stem(tok)


def tokens(text: str) -> list[str]:
    """Normalised tokens: case-folded, thousands separators dropped, decimals canonical, number words as digits,
    plurals and inflections folded."""
    return [_fold(t) for t in _TOKEN.findall(_THOUSANDS.sub("", _unescape(text)).casefold())]


def _flat(tokens_: Iterable[str]) -> str:
    return " " + " ".join(tokens_) + " "


def units(text: str) -> list[tuple[str, str]]:
    """(header, body) pairs a restatement must fit in (module docstring).

    - Plain text: one unit per blank-line-separated block (no header), and one per `<SEP>` fragment of a block.
    - A JSON record line (LightRAG's context lists entities, relations and chunks as JSON lines): its
      single-paragraph values form one unit. When the longest value (the description) was merged from several
      sources, it is split on `<SEP>`: one unit per fragment, each with the record's other values (name, type,
      keywords) as header. A multi-paragraph value (a chunk's content) adds one unit per paragraph, as plain chunk
      text would."""
    out: list[tuple[str, str]] = []
    plain: list[str] = []

    def add(header: str, body: str) -> None:
        for frag in body.split(GRAPH_FIELD_SEP):
            if frag.strip():
                out.append((header, frag))

    def flush() -> None:
        if plain:
            add("", "\n".join(plain))
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
                    body = max(single, key=len)
                    if GRAPH_FIELD_SEP in body:
                        rest = list(single)
                        rest.remove(body)
                        add(" | ".join(rest), body)
                    else:
                        add("", " | ".join(single))
                for v in multi:
                    for b in _BLOCK.split(v):
                        add("", b)
                continue
        if s:
            plain.append(line)
        else:
            flush()
    flush()
    return out


def segments(text: str) -> list[str]:
    """`units` as flat strings (header and body joined): what verbatim matching looks at."""
    return [" | ".join(x for x in (h, b) if x) for h, b in units(text)]


def _sentences(text: str) -> list[str]:
    return [x for x in _SENTENCE.split(text) if x.strip()]


def _windows(body: str, span: int) -> list[str]:
    """Runs of up to `span` consecutive sentences of `body` (all of it when it has no more sentences than that)."""
    sents = _sentences(body)
    if len(sents) <= span:
        return [body]
    return [" ".join(sents[i : i + span]) for i in range(len(sents) - span + 1)]


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
        kind = {f: world.facts[f].kind for f in self._facts}
        df: dict[str, int] = defaultdict(int)
        df_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        n_kind: dict[str, int] = defaultdict(int)
        for f, toks in fact_tokens.items():
            n_kind[kind[f]] += 1
            for t in set(toks):
                df[t] += 1
                df_kind[kind[f]][t] += 1
        # One more sentence than the longest fact rendering: the restatement window (module docstring).
        self.window = 1 + max(
            (len(_sentences(text)) for f in self._facts for text in [world.facts[f].text, *renderings.get(f, [])]),
            default=1,
        )

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

        def distinguishing(t: str, k: str) -> bool:
            if t[0].isdigit():
                return True
            if t in STOPWORDS or len(t) <= 1 or df[t] > template:
                return False
            return n_kind[k] < MIN_KIND_FACTS or df_kind[k][t] <= TEMPLATE_SHARE * n_kind[k]

        self._sig: dict[str, frozenset[str]] = {}
        self._sig_anchor: dict[str, list[str]] = defaultdict(list)
        for f, toks in fact_tokens.items():
            sig = frozenset(t for t in toks if distinguishing(t, kind[f]))
            if len(sig) < MIN_SIGNATURE:
                continue
            self._sig[f] = sig
            self._sig_anchor[min(sig, key=lambda t: (df[t], t))].append(f)

    def delivered(self, text: str) -> list[str]:
        """World facts the text carries, in world order (verbatim or restated; module docstring)."""
        if not text or not text.strip():
            return []
        found: set[str] = set()
        for header, body in units(text):
            seg = tokens(" | ".join(x for x in (header, body) if x))
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
            head = set(tokens(header))
            for window in _windows(body, self.window):
                wt = head | set(tokens(window))
                for anchor in wt & self._sig_anchor.keys():
                    for f in self._sig_anchor[anchor]:
                        if f not in found and self._sig[f] <= wt:
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
    key = (world.id, world.artifact_hash())
    if key not in _MATCHERS:
        if len(_MATCHERS) > 64:
            _MATCHERS.clear()
        _MATCHERS[key] = FactMatcher(world)
    return _MATCHERS[key]
