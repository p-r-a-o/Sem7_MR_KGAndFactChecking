"""Rule-based claim parser: text -> (head, relation, tail). Stdlib-only, pure functions.

The entity mentions, the relation and the direction of a claim are mutually dependent, so they are
decided **jointly** rather than in sequence:

  1. enumerate every candidate entity span (every label that matches at some token position, all
     entity IDs sharing that label), and every candidate relation (cue-pattern hit or lexical
     overlap with the relation name);
  2. score every (span pair, relation, direction) combination with
     relation evidence + KG type-consistency + token coverage - scaffolding overlap;
  3. return the best combination.

Deciding sequentially - link the first two entities, then score the relation on the leftover words -
is what the earlier version did, and it loses in three ways that the joint search fixes:

  * **scaffolding leakage.** In "Junction is a kind of joining", the word "kind" is itself a WN18RR
    entity, so greedy leftmost-longest linking returned (junction, kind) and never reached
    "joining". A span that overlaps the matched relation cue is now penalised.
  * **relation ambiguity.** FB15k-237 has many relations sharing a last path segment
    (/people/ethnicity/people vs /people/cause_of_death/people). Lexical scoring cannot separate
    them; the type-consistency pools can, because only one of them ever takes "Scottish people"
    as a head.
  * **lemma ambiguity.** 31% of WN18RR lemmas name several synsets ("run" names 33). The text
    cannot say which, but the pools can: prefer the synset that actually occurs in that position
    of that relation.

Type consistency is read from the training graph only, so nothing about the evaluated triple leaks
in. The cue table is hand-written and deliberately does NOT import common/verbalize templates, so
evaluation is not a trivial round trip (HANDOFF.md section 6).

Swap in an LLM parser by replacing `parse_claim`, keeping its return shape.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

STOPWORDS = {"a", "an", "the", "of", "in", "is", "was", "are", "to", "and", "or", "as", "by", "for",
             "on", "at", "it", "be", "has", "have", "with", "from", "that", "this"}

# relation -> ordered [(regex on the claim text, swap_head_tail)]; first match wins.
CUES: dict[str, list[tuple[str, bool]]] = {
    "/people/person/place_of_birth": [(r"\bbirthplace\b", True), (r"\bborn\b", False)],
    "/people/person/nationality": [(r"\bcitizen|\bnationality\b|\bnational\b", False)],
    "/people/person/profession": [(r"\bworks? as\b|\bprofession\b|\boccupation\b", False)],
    "/people/person/languages": [(r"\bspeaks?\b", False)],
    "/film/film/genre": [(r"\bgenre\b|\bis an? \b(?=[a-z -]+ (film|movie)s?\b)", False)],
    "/film/film/country": [(r"\bproduced\b|\bmade in\b", False)],
    "/location/location/contains": [(r"\bcontains?\b", False), (r"\blocated (in|within)\b|\bpart of\b|\bwithin\b", True)],
    "/location/country/capital": [(r"\bis the capital of\b", True), (r"\bcapital\b", False)],
    "/music/artist/genre": [(r"\bplays?\b|\bmusic\b", False)],
    "_hypernym": [(r"\bis a broader term for\b", True), (r"\bkind of\b|\btype of\b|\bis an? \b", False)],
    "_instance_hypernym": [(r"\binstance of\b", False)],
    "_has_part": [(r"\bhas\b.*\bas a part\b|\bpart\b", False)],
    "_member_meronym": [(r"\bis a member of\b", True), (r"\bmember\b", False)],
    "_derivationally_related_form": [(r"\bderivationally\b|\bderived\b", False)],
    "_similar_to": [(r"\bsimilar\b", False)],
    "_also_see": [(r"\balso\b|\bsee\b", False)],
    "_verb_group": [(r"\bverb group\b", False)],
    "_synset_domain_topic_of": [(r"\bdomain\b|\btopic\b", False)],
}

# Score weights. Tuned on the valid split of both datasets (see eval/eval_parser.py).
W_KG_FIT = 1.5          # type consistency of (head, tail) with the relation's train pools
W_COVERAGE = 0.5        # fraction of claim tokens explained by the two entity spans
W_SCAFFOLD = 1.0        # per entity span that overlaps the matched relation cue
W_CUE_ANY = 0.4         # per entity span made of words that ANY relation cue claims as phrasing
W_SUBSPAN = 0.75        # per entity span strictly contained in another candidate span
W_CUE_DIRECTION = 1.0   # agreement with the direction the matched cue implies; must outrank a
                        # single type-pool near-miss (W_KG_FIT/2), or symmetric-looking relations
                        # such as _member_meronym come out backwards
W_PHRASE = 0.4          # per word of the relation's name matched verbatim in the leftover words,
                        # capped at 2 words: a one-word match ("...film", "...country") is far
                        # weaker evidence than a multi-word one ("film release region")
MAX_IDS_PER_LABEL = 8   # cap on synsets considered for an ambiguous lemma
MAX_SPANS_PER_START = 3

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int          # exclusive token index
    entity: str       # canonical ID
    surface: str

    @property
    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class Candidate:
    """One fully-specified reading of a claim, with the evidence that scored it."""
    head: Span
    tail: Span
    relation: str
    score: float
    relation_score: float
    kg_fit: float


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tokenize_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokens plus their (char_start, char_end), so a regex match can be mapped to token indices."""
    matches = list(_TOKEN_RE.finditer(text.lower()))
    return [m.group(0) for m in matches], [m.span() for m in matches]


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[: -len(suf)]
    return tok


class EntityIndex:
    """Normalized surface tokens -> candidate entity IDs, highest-degree first.

    Unlike a one-ID-per-label map, ambiguous lemmas keep all their synsets so the joint search can
    choose between them; `MAX_IDS_PER_LABEL` bounds the branching. `aliases` adds extra surface
    forms per entity - used for the parenthetical gloss that identifies an ambiguous WN18RR lemma
    (see common/verbalize.disambiguated_labels). The gloss form tokenizes to a longer span than the
    bare lemma, so the sub-span penalty makes the joint search prefer it when the claim carries one.
    """

    def __init__(self, entity_labels: dict[str, str], degrees: Optional[dict[str, int]] = None,
                 aliases: Optional[dict[str, str]] = None):
        self.degrees = degrees or {}
        buckets: dict[tuple[str, ...], list[str]] = {}
        surfaces = [entity_labels] + ([aliases] if aliases else [])
        for table in surfaces:
            for ent, label in table.items():
                key = tuple(tokenize(label))
                if not key or (len(key) == 1 and key[0] in STOPWORDS):
                    continue
                buckets.setdefault(key, []).append(ent)
        self.table: dict[tuple[str, ...], tuple[str, ...]] = {
            key: tuple(sorted(dict.fromkeys(ents), key=lambda e: -self.degrees.get(e, 0))
                       [:MAX_IDS_PER_LABEL])
            for key, ents in buckets.items()}
        self.max_len = max((len(k) for k in self.table), default=1)

    def candidate_spans(self, tokens: list[str]) -> list[Span]:
        """Every label match at every start position (longest first, capped per position).

        Overlapping spans are kept on purpose; `parse_claim` picks a non-overlapping pair.
        """
        spans: list[Span] = []
        for i in range(len(tokens)):
            found = 0
            for n in range(min(self.max_len, len(tokens) - i), 0, -1):
                key = tuple(tokens[i:i + n])
                if key not in self.table or (n == 1 and key[0] in STOPWORDS):
                    continue
                surface = " ".join(key)
                spans.extend(Span(i, i + n, ent, surface) for ent in self.table[key])
                found += 1
                if found >= MAX_SPANS_PER_START:
                    break
        return spans


@dataclass(frozen=True)
class ParserResources:
    """Everything the parser needs for one dataset. Build once per dataset, then reuse."""
    index: EntityIndex
    relations: tuple[str, ...]
    head_pool: dict[str, frozenset[str]] = field(default_factory=dict)
    tail_pool: dict[str, frozenset[str]] = field(default_factory=dict)
    relation_stems: dict[str, frozenset[str]] = field(default_factory=dict)
    relation_tokens: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _relation_words(relation: str) -> tuple[str, ...]:
    """Last path segment as words: '/film/film/featured_film_locations' -> ('featured','film','locations')."""
    return tuple(w for w in relation.split("/")[-1].strip("_").split("_") if w)


def _relation_phrase(relation: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in _relation_words(relation) if w not in STOPWORDS)


def phrase_occurrences(tokens: list[str], phrase: tuple[str, ...]) -> list[frozenset[int]]:
    """Token-index sets where `phrase` occurs as a contiguous run in `tokens`."""
    if not phrase or len(phrase) > len(tokens):
        return []
    return [frozenset(range(i, i + len(phrase)))
            for i in range(len(tokens) - len(phrase) + 1)
            if tuple(tokens[i:i + len(phrase)]) == phrase]


def build_resources(entity_labels: dict[str, str], relations: Iterable[str],
                    train: Iterable[tuple[str, str, str]] = (),
                    degrees: Optional[dict[str, int]] = None,
                    aliases: Optional[dict[str, str]] = None) -> ParserResources:
    """Assemble the parser's per-dataset resources, including type-consistency pools from TRAIN."""
    rels = tuple(relations)
    heads: dict[str, set[str]] = {}
    tails: dict[str, set[str]] = {}
    for h, r, t in train:
        heads.setdefault(r, set()).add(h)
        tails.setdefault(r, set()).add(t)
    return ParserResources(
        index=EntityIndex(entity_labels, degrees, aliases),
        relations=rels,
        head_pool={r: frozenset(v) for r, v in heads.items()},
        tail_pool={r: frozenset(v) for r, v in tails.items()},
        relation_stems={r: _relation_phrase(r) for r in rels},
        relation_tokens={r: _relation_words(r) for r in rels})


def _covered_tokens(pattern: str, lowered: str,
                    char_spans: list[tuple[int, int]]) -> frozenset[int]:
    """Token indices inside ANY match of `pattern`.

    Every match matters, not just the leftmost one: the `_hypernym` cue
    `\bkind of\b|\btype of\b|\bis an? \b` matches "is a" before "kind of" in
    "Portraiture is a kind of portrayal", and if only that first match counted, the word "kind" -
    itself a WN18RR entity - would escape the scaffolding penalty and be linked as the head.
    """
    return frozenset(i for m in re.finditer(pattern, lowered)
                     for i, (s, e) in enumerate(char_spans)
                     if s >= m.start() and e <= m.end())


def cue_matches(text: str, char_spans: list[tuple[int, int]],
                relations: Iterable[str]) -> tuple[dict[str, tuple[frozenset[int], bool]],
                                                   frozenset[int]]:
    """(relation -> (token indices its first matching cue consumes, swap flag), all cue tokens).

    Matched once over the whole claim so that per-pair scoring stays set arithmetic. The second
    element is the union over every cue of every relation: words that some relation uses as
    phrasing are weak entity mentions regardless of which relation ends up winning.
    """
    lowered = text.lower()
    out: dict[str, tuple[frozenset[int], bool]] = {}
    any_cue: set[int] = set()
    for rel in relations:
        matched = False
        for pattern, swap in CUES.get(rel, []):
            covered = _covered_tokens(pattern, lowered, char_spans)
            if not covered:
                continue
            any_cue |= covered
            if not matched:
                out[rel] = (covered, swap)
                matched = True
    return out, frozenset(any_cue)


def _shortlist(res: ParserResources, all_stems: set[str],
               cues: dict[str, tuple[frozenset[int], bool]]) -> list[str]:
    """Relations worth scoring at all: a cue fired, or the relation name shares a stem with the text."""
    return [r for r in res.relations
            if r in cues or (res.relation_stems.get(r, frozenset()) & all_stems)]


def rank_candidates(claim: str, res: ParserResources, limit: int = 5) -> list[Candidate]:
    """Score every (span pair, relation, direction) reading of `claim`, best first.

    Exposed separately from `parse_claim` so error analysis can see the runners-up.
    """
    tokens, char_spans = tokenize_spans(claim)
    spans = res.index.candidate_spans(tokens)
    if len({(s.start, s.end) for s in spans}) < 2:
        return []

    cues, any_cue = cue_matches(claim, char_spans, res.relations)
    # A mention truncated to part of a longer known label ("genus" out of "genus enhydra",
    # "church" out of "church building") is almost always the wrong reading.
    extents = {(s.start, s.end) for s in spans}
    subsumed = {(a, b) for (a, b) in extents
                if any(c <= a and b <= d and (c, d) != (a, b) for (c, d) in extents)}
    stems = [_stem(t) for t in tokens]
    all_stems = set(stems)
    shortlist = _shortlist(res, all_stems, cues)
    if not shortlist:
        return []

    # Where each shortlisted relation's own name appears verbatim. FB15k-237's generic claim
    # template is "The <relation name> of {h} is {t}", so this is a strong signal there; it is the
    # only thing that separates /film/film/release_date_s./.../film_release_region from
    # /film/film/genre, whose loose cue would otherwise win. Expect it to be much weaker on
    # real-world claims, which do not quote the schema - note that when reporting these numbers.
    phrase_hits = {r: phrase_occurrences(tokens, res.relation_tokens.get(r) or _relation_words(r))
                   for r in shortlist}

    n_tokens = len(tokens) or 1
    degrees = res.index.degrees
    out: list[Candidate] = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if a.end > b.start and b.end > a.start:
                continue                      # overlapping mentions
            if a.entity == b.entity:
                continue
            covered = frozenset(a.indices) | frozenset(b.indices)
            soft_penalty = (W_SUBSPAN * sum((s.start, s.end) in subsumed for s in (a, b))
                            + W_CUE_ANY * sum(bool(any_cue & frozenset(s.indices))
                                              for s in (a, b)))
            rest_stems = {stems[k] for k in range(len(tokens)) if k not in covered}
            coverage = len(covered) / n_tokens
            deg_bonus = math.log1p(degrees.get(a.entity, 0) + degrees.get(b.entity, 0)) * 1e-3
            for rel in shortlist:
                phrase = res.relation_stems.get(rel) or _relation_phrase(rel)
                overlap = len(phrase & rest_stems) / len(phrase) if phrase else 0.0
                cue_tokens, cue_swap = cues.get(rel, (frozenset(), None))
                cue_hit = bool(cue_tokens) and not (cue_tokens & covered)
                hits = phrase_hits.get(rel, ())
                phrase_len = next((len(occ) for occ in hits if not (occ & covered)), 0)
                rel_score = ((1.0 if cue_hit else 0.0) + overlap
                             + W_PHRASE * min(phrase_len, 2))
                if rel_score <= 0.0:
                    continue
                scaffold = sum(1 for s in (a, b) if cue_tokens & frozenset(s.indices))
                base = (rel_score + W_COVERAGE * coverage
                        - W_SCAFFOLD * scaffold - soft_penalty + deg_bonus)
                heads, tails = res.head_pool.get(rel, frozenset()), res.tail_pool.get(rel, frozenset())
                for head, tail, swapped in ((a, b, False), (b, a, True)):
                    kg_fit = 0.5 * ((head.entity in heads) + (tail.entity in tails))
                    direction = (W_CUE_DIRECTION if cue_swap is not None and cue_swap == swapped
                                 else 0.0)
                    out.append(Candidate(head=head, tail=tail, relation=rel,
                                         score=base + W_KG_FIT * kg_fit + direction,
                                         relation_score=rel_score, kg_fit=kg_fit))
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def parse_claim(claim: str, res: ParserResources, min_score: float = 0.5) -> dict:
    """Return {"triple": [h, r, t], "linked_entities": {surface: id}} or raise ParseError.

    `min_score` applies to the relation evidence alone, so a confident relation is still required;
    KG type-consistency only chooses between readings, it cannot invent one.
    """
    ranked = rank_candidates(claim, res)
    if not ranked:
        raise ParseError(f"No two known entities and a matching relation in: {claim!r}")
    best = ranked[0]
    if best.relation_score < min_score:
        raise ParseError(f"No relation matched the wording of: {claim!r}")
    return {"triple": [best.head.entity, best.relation, best.tail.entity],
            "linked_entities": {best.head.surface: best.head.entity,
                                best.tail.surface: best.tail.entity}}
