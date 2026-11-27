"""Claim -> query for multi-entity claims (FactKG): which entities, which relations, negated or not.

* Entities: FactKG gives every claim its entity set (`Entity_set`, including the test split); that is
  the dataset's standard setting, so when a claim text is a known FactKG claim we use it. Any other
  text goes through a longest-match linker over entity labels.
* Relations: how relevant each KG relation is to the claim text. Two signals, max-combined:
  - lexical: claim words that match words in the relation name ("birthPlace" -> birth, place);
  - learned lexicon: P(relation | word) counted over FactKG *train* claims and their gold evidence
    relation paths (scripts/build_lexicon.py). Words from the claim's own entity names are ignored.
Everything here is stdlib-only.
"""
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_WORD = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_NEGATION = re.compile(r"\b(not|never|no|nor|neither|none|nobody|nothing|without)\b|n't\b")
STOPWORDS = frozenset("""a an the of in on at to for by with from and or as is was were are be been being has have
had do does did it its this that these those he she they his her their him them i you we me my our your who whom
which what when where there here also both than then so such very just about into over after before during known
s t""".split())


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(w: str) -> str:
    """Crude suffix stripping, enough to match 'born'/'birth' poorly but 'located'/'location' well."""
    for suf in ("ations", "ation", "ings", "ing", "ers", "er", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def relation_words(rel: str) -> list[str]:
    """'birthPlace' -> ['birth', 'place']; '/people/person/place_of_birth' -> ['place', 'birth']."""
    last = rel.split("/")[-1].lstrip("~")
    parts = _CAMEL.findall(last.replace("_", " ").replace(".", " ").replace("-", " ")) or [last]
    return [p.lower() for p in parts if p.lower() not in STOPWORDS]


def content_words(text: str, entity_labels: Iterable[str] = ()) -> list[str]:
    ent = {w for lab in entity_labels for w in words(lab)}
    return [w for w in words(text) if w not in STOPWORDS and w not in ent and not w.isdigit()]


def is_negated(text: str) -> bool:
    return bool(_NEGATION.search(text.lower()))


@dataclass
class Lexicon:
    """P(relation | word) from training claims. Relations are base names (no '~')."""
    word_rel: dict[str, dict[str, int]] = field(default_factory=dict)
    word: dict[str, int] = field(default_factory=dict)
    min_count: int = 3

    @classmethod
    def build(cls, examples: Iterable[tuple[str, list[str], Iterable[str]]], min_count: int = 3) -> "Lexicon":
        """examples: (claim text, entity labels, gold relations)."""
        word_rel: dict[str, Counter] = {}
        word: Counter = Counter()
        for text, labels, rels in examples:
            ws = set(stem(w) for w in content_words(text, labels))
            rels = {r.lstrip("~") for r in rels}
            word.update(ws)
            for w in ws:
                word_rel.setdefault(w, Counter()).update(rels)
        keep = {w for w, c in word.items() if c >= min_count}
        return cls({w: dict(word_rel[w]) for w in keep}, {w: word[w] for w in keep}, min_count)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"word_rel": self.word_rel, "word": self.word, "min_count": self.min_count}),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["Lexicon"]:
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["word_rel"], d["word"], d.get("min_count", 3))

    def prob(self, w: str, rel: str) -> float:
        n = self.word.get(w)
        return self.word_rel.get(w, {}).get(rel, 0) / n if n else 0.0


@dataclass
class ClaimQuery:
    claim: str
    entities: list[str]                          # canonical IDs, claim order
    negated: bool
    stems: set[str]                              # content-word stems (entity words removed)
    linking: str                                 # "given entity set" | "label match"

    def relevance(self, rel: str, lexicon: Optional[Lexicon]) -> tuple[float, str]:
        """(score in [0, 1], why) for a relation name (inverse marker ignored)."""
        base = rel.lstrip("~")
        rws = {stem(w) for w in relation_words(base)}
        lexical = len(rws & self.stems) / len(rws) if rws else 0.0
        learned, best_w = 0.0, ""
        if lexicon is not None:
            for w in self.stems:
                p = lexicon.prob(w, base)
                if p > learned:
                    learned, best_w = p, w
        if lexical >= learned:
            return lexical, (f"claim words match relation name '{base}'" if lexical else "no match")
        return learned, f"claim word '{best_w}' -> '{base}' (P={learned:.2f} in FactKG train evidence)"


class EntityLinker:
    """Longest-match linker over entity labels, preferring high-degree entities on ties."""

    def __init__(self, labels: dict[str, str], degree: dict[str, int], max_ngram: int = 8):
        self.index: dict[str, str] = {}
        self.max_ngram = max_ngram
        for e, lab in labels.items():
            key = " ".join(words(lab))
            if key and (key not in self.index or degree.get(e, 0) > degree.get(self.index[key], 0)):
                self.index[key] = e

    def link(self, text: str) -> list[str]:
        """A match must start like a name in the original text (capitalised or a number), so ordinary
        words do not link to same-named entities ("born" -> Born, "born in Chicago" -> the song)."""
        raw = re.findall(r"[A-Za-z0-9]+", text)
        toks = [t.lower() for t in raw]
        found, i = [], 0
        while i < len(toks):
            for n in range(min(self.max_ngram, len(toks) - i), 0, -1):
                key = " ".join(toks[i: i + n])
                e = self.index.get(key)
                if e is not None and not key.isdigit() and (raw[i][0].isupper() or raw[i][0].isdigit())                         and (n > 1 or toks[i] not in STOPWORDS):
                    found.append(e)
                    i += n
                    break
            else:
                i += 1
        return list(dict.fromkeys(found))


def load_entity_sets(claim_files: Iterable[Path]) -> dict[str, list[str]]:
    """claim text -> given entity set, from data/claims/<dataset>/*.jsonl rows that carry one."""
    out: dict[str, list[str]] = {}
    for p in claim_files:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("entity_set"):
                    out[d["claim"].strip()] = d["entity_set"]
    return out


def analyse(claim: str, entity_sets: dict[str, list[str]], linker: Optional[EntityLinker],
            labels: dict[str, str]) -> ClaimQuery:
    given = entity_sets.get(claim.strip())
    if given is not None:
        ents, how = list(given), "given entity set"
    else:
        ents, how = (linker.link(claim) if linker else []), "label match"
    stems = {stem(w) for w in content_words(claim, [labels.get(e, e) for e in ents])}
    return ClaimQuery(claim, ents, is_negated(claim), stems, how)
