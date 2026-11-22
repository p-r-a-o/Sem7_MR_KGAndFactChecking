"""Rule-based claim parser: text -> (head, relation, tail). Stdlib-only, pure functions.

Pipeline: (1) link entity mentions by leftmost-longest label match, (2) score relations by lexical
overlap with the relation name plus a few hand-written cue patterns, (3) order head/tail by mention
order, swapping when a cue says the sentence is in passive/inverse form.

This is a deliberately simple baseline behind a stable interface; swap in an LLM parser by
replacing `parse_claim` (keep the return shape). It does not reuse common/verbalize templates.
"""
import re
from dataclasses import dataclass
from typing import Optional

STOPWORDS = {"a", "an", "the", "of", "in", "is", "was", "are", "to", "and", "or", "as", "by", "for",
             "on", "at", "it", "be", "has", "have", "with", "from", "that", "this"}

# relation -> ordered [(regex on non-entity text, swap_head_tail)]; first match wins.
CUES: dict[str, list[tuple[str, bool]]] = {
    "/people/person/place_of_birth": [(r"\bbirthplace\b", True), (r"\bborn\b", False)],
    "/people/person/nationality": [(r"\bcitizen|\bnationality\b|\bnational\b", False)],
    "/people/person/profession": [(r"\bworks? as\b|\bprofession\b|\boccupation\b", False)],
    "/people/person/languages": [(r"\bspeaks?\b", False)],
    "/film/film/genre": [(r"\bgenre\b|\bfilm\b|\bmovie\b", False)],
    "/film/film/country": [(r"\bproduced\b|\bmade in\b", False)],
    "/location/location/contains": [(r"\bcontains?\b", False), (r"\blocated (in|within)\b|\bpart of\b|\bwithin\b", True)],
    "/location/country/capital": [(r"\bis the capital of\b", True), (r"\bcapital\b", False)],
    "/music/artist/genre": [(r"\bplays?\b|\bmusic\b", False)],
    "_hypernym": [(r"\bbroader term\b", True), (r"\bkind of\b|\btype of\b|\bis an? \b", False)],
    "_instance_hypernym": [(r"\binstance of\b", False)],
    "_has_part": [(r"\bpart\b|\bhas\b", False)],
    "_member_meronym": [(r"\bmember of\b", True)],
    "_derivationally_related_form": [(r"\bderivationally\b|\bderived\b", False)],
    "_similar_to": [(r"\bsimilar\b", False)],
    "_also_see": [(r"\balso\b|\bsee\b", False)],
    "_verb_group": [(r"\bverb group\b", False)],
    "_synset_domain_topic_of": [(r"\bdomain\b|\btopic\b", False)],
}


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int          # exclusive token index
    entity: str       # canonical ID
    surface: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[: -len(suf)]
    return tok


class EntityIndex:
    """Normalized label tokens -> entity ID. On duplicate labels the higher-degree entity wins."""

    def __init__(self, entity_labels: dict[str, str], degrees: Optional[dict[str, int]] = None):
        degrees = degrees or {}
        self.table: dict[tuple[str, ...], str] = {}
        for ent, label in entity_labels.items():
            key = tuple(tokenize(label))
            if not key or (len(key) == 1 and key[0] in STOPWORDS):
                continue
            if key not in self.table or degrees.get(ent, 0) > degrees.get(self.table[key], 0):
                self.table[key] = ent
        self.max_len = max((len(k) for k in self.table), default=1)

    def link(self, tokens: list[str]) -> list[Span]:
        spans: list[Span] = []
        i = 0
        while i < len(tokens):
            for n in range(min(self.max_len, len(tokens) - i), 0, -1):
                key = tuple(tokens[i:i + n])
                if key in self.table and not (n == 1 and key[0] in STOPWORDS):
                    spans.append(Span(i, i + n, self.table[key], " ".join(key)))
                    i += n
                    break
            else:
                i += 1
        return spans


def _relation_phrase(relation: str) -> set[str]:
    words = relation.split("/")[-1].strip("_").split("_")
    return {_stem(w) for w in words if w and w not in STOPWORDS}


def score_relations(rest_tokens: list[str], relations: list[str]) -> list[tuple[float, bool, str]]:
    """[(score, swap, relation)] sorted best-first. Cue hit = +1.0; lexical overlap adds up to 1.0."""
    rest_text = " ".join(rest_tokens)
    rest_stems = {_stem(t) for t in rest_tokens}
    scored = []
    for rel in relations:
        phrase = _relation_phrase(rel)
        overlap = len(phrase & rest_stems) / len(phrase) if phrase else 0.0
        cue_score, swap = 0.0, False
        for pattern, do_swap in CUES.get(rel, []):
            if re.search(pattern, rest_text):
                cue_score, swap = 1.0, do_swap
                break
        scored.append((cue_score + overlap, swap, rel))
    scored.sort(key=lambda x: -x[0])
    return scored


def parse_claim(claim: str, index: EntityIndex, relations: list[str], min_score: float = 0.5) -> dict:
    """Return {"triple": [h, r, t], "linked_entities": {surface: id}} or raise ParseError."""
    tokens = tokenize(claim)
    spans = index.link(tokens)
    if len(spans) < 2:
        raise ParseError(f"Found {len(spans)} known entities (need 2) in: {claim!r}")
    first, second = spans[0], spans[1]
    covered = {i for s in (first, second) for i in range(s.start, s.end)}
    rest = [t for i, t in enumerate(tokens) if i not in covered]
    score, swap, relation = score_relations(rest, relations)[0]
    if score < min_score:
        raise ParseError(f"No relation matched the words {rest!r}")
    head, tail = (second, first) if swap else (first, second)
    return {"triple": [head.entity, relation, tail.entity],
            "linked_entities": {first.surface: first.entity, second.surface: second.entity}}
