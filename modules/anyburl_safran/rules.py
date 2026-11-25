"""AnyBURL rule files: parsing, indexing and grounding. Pure stdlib, no I/O beyond reading the rule file.

Rule file format (AnyBURL output, documented by SAFRAN)
------------------------------------------------------
One rule per line, tab separated::

    {predicted}\t{correctlyPredicted}\t{confidence}\t{head} <= {body}

* The ``confidence`` column is the **raw** ``correctlyPredicted / predicted`` measured on the training
  set (verified against the mined FB15k-237 rules). AnyBURL's own ``Apply`` re-computes it as
  ``correctlyPredicted / (predicted + UNSEEN_NEGATIVE_EXAMPLES)`` with ``UNSEEN_NEGATIVE_EXAMPLES = 5``
  by default, so `parse_rule`/`load_rules` apply the same smoothing: without it, the ~45k rules that
  scored 1.0 off two or three predictions outrank every well-supported rule.
* ``head`` is a single atom, ``body`` is ``atom, atom, ...`` (possibly empty, for zero rules).
* Cyclic rules have head ``r(X,Y)``; acyclic rules have ``r(X,c)`` or ``r(c,Y)`` with ``c`` a constant.
* Body atoms are ordered from ``X`` towards ``Y`` and adjacent atoms share one variable, but atom
  *direction* is free: both ``p(X,A)`` and ``p(A,X)`` occur.

Assumption: an argument is a **variable** iff it matches ``[A-Z][0-9]?`` (AnyBURL uses X, Y, A, B, ...).
Entity IDs in our datasets are ``/m/...`` (FB15k-237) or digit strings (WN18RR), so they never collide.
"""
import heapq
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from common.schema import Triple

_VARIABLE = re.compile(r"^[A-Z][0-9]?$")
# relation names contain /._- but never parentheses or commas, so a flat regex is enough
_ATOM = re.compile(r"([^\s(),]+)\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")
_SEPARATOR = "<="


def is_variable(arg: str) -> bool:
    return bool(_VARIABLE.match(arg))


@dataclass(frozen=True)
class Atom:
    """One literal ``relation(subject, object)``; arguments are variable names or entity IDs."""
    relation: str
    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.relation}({self.subject},{self.object})"


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom, ...]
    confidence: float             # used for ranking: smoothed unless unseen_negative_examples=0
    predicted: int
    correctly_predicted: int
    text: str                     # original rule string, shown as Explanation.rule
    file_confidence: float = 0.0  # the raw correct/predicted value AnyBURL wrote

    @property
    def relation(self) -> str:
        return self.head.relation

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def kind(self) -> str:
        """``cyclic`` (head r(X,Y)), ``tail_constant`` (r(X,c)), ``head_constant`` (r(c,Y)), ``zero``."""
        if not self.body:
            return "zero"
        if is_variable(self.head.subject) and is_variable(self.head.object):
            return "cyclic"
        return "tail_constant" if is_variable(self.head.subject) else "head_constant"

    @property
    def stats_text(self) -> str:
        smoothing = ("" if f"{self.confidence:.3f}" == f"{self.file_confidence:.3f}"
                     else f", smoothed from {self.file_confidence:.3f}")
        return (f"confidence {self.confidence:.3f}{smoothing}; {self.correctly_predicted} of "
                f"{self.predicted} training predictions correct")


def parse_rule(line: str, unseen_negative_examples: int = 0) -> Optional[Rule]:
    """Parse one rule-file line. Returns None if the line is blank or does not match the format.

    `unseen_negative_examples` reproduces AnyBURL's apply-time confidence
    ``correctlyPredicted / (predicted + UNSEEN_NEGATIVE_EXAMPLES)``; 0 keeps the file's raw value.
    """
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 4:
        return None
    head_text, separator, body_text = parts[3].partition(_SEPARATOR)
    if not separator:
        return None
    head_atoms = _ATOM.findall(head_text)
    if len(head_atoms) != 1:
        return None
    try:
        predicted, correct, confidence = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError:
        return None
    smoothed = (correct / (predicted + unseen_negative_examples)
                if unseen_negative_examples and predicted + unseen_negative_examples else confidence)
    return Rule(head=Atom(*head_atoms[0]),
                body=tuple(Atom(*m) for m in _ATOM.findall(body_text)),
                confidence=smoothed, predicted=predicted, correctly_predicted=correct,
                text=parts[3].strip(), file_confidence=confidence)


@dataclass
class RuleSet:
    """Rules grouped by head relation, each group sorted by decreasing confidence."""
    rules: list[Rule]
    by_relation: dict[str, list[Rule]] = field(default_factory=dict)
    skipped_lines: int = 0
    dropped_by_filter: int = 0
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.by_relation:
            for rule in sorted(self.rules, key=lambda r: -r.confidence):
                self.by_relation.setdefault(rule.relation, []).append(rule)

    def for_relation(self, relation: str) -> list[Rule]:
        return self.by_relation.get(relation, [])

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in self.rules:
            counts[rule.kind] = counts.get(rule.kind, 0) + 1
        return counts

    def summary(self) -> str:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(self.counts_by_kind().items()))
        return (f"{len(self.rules)} rules over {len(self.by_relation)} relations ({kinds}); "
                f"{self.dropped_by_filter} filtered out, {self.skipped_lines} unparseable lines")


def load_rules(path: Path, min_confidence: float = 0.0, min_correct_predictions: int = 0,
               max_body_length: Optional[int] = None,
               max_rules_per_relation: Optional[int] = None,
               unseen_negative_examples: int = 5) -> RuleSet:
    """Read an AnyBURL rule snapshot, dropping rules below the given thresholds.

    A long AnyBURL run produces millions of rules (most of them low-confidence rules with a constant),
    while a query only ever applies the most confident few hundred per relation. `max_rules_per_relation`
    keeps the best ones as the file streams past, which bounds both load time and memory.
    """
    kept: list[Rule] = []
    heaps: dict[str, list[tuple[float, int, Rule]]] = {}
    tie_break = itertools.count()
    skipped = dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rule = parse_rule(line, unseen_negative_examples)
            if rule is None:
                skipped += 1
            elif (rule.confidence < min_confidence
                  or rule.correctly_predicted < min_correct_predictions
                  or (max_body_length is not None and rule.length > max_body_length)):
                dropped += 1
            elif max_rules_per_relation is None:
                kept.append(rule)
            else:
                heap = heaps.setdefault(rule.relation, [])
                entry = (rule.confidence, next(tie_break), rule)
                if len(heap) < max_rules_per_relation:
                    heapq.heappush(heap, entry)
                    continue
                if heap[0][0] < rule.confidence:
                    heapq.heapreplace(heap, entry)
                dropped += 1
    for heap in heaps.values():
        kept.extend(entry[2] for entry in heap)
    return RuleSet(rules=kept, skipped_lines=skipped, dropped_by_filter=dropped, source=str(path))


# ----------------------------------------------------------------------------- the training graph

@dataclass
class TripleIndex:
    """Adjacency over the TRAIN split: the only facts a rule is allowed to ground on."""
    out: dict[tuple[str, str], tuple[str, ...]]    # (subject, relation) -> objects
    inc: dict[tuple[str, str], tuple[str, ...]]    # (object, relation)  -> subjects
    facts: set[Triple]

    @classmethod
    def from_triples(cls, triples: Sequence[Triple]) -> "TripleIndex":
        out: dict[tuple[str, str], list[str]] = {}
        inc: dict[tuple[str, str], list[str]] = {}
        for h, r, t in triples:
            out.setdefault((h, r), []).append(t)
            inc.setdefault((t, r), []).append(h)
        return cls(out={k: tuple(v) for k, v in out.items()},
                   inc={k: tuple(v) for k, v in inc.items()},
                   facts={(h, r, t) for h, r, t in triples})

    def objects(self, subject: str, relation: str) -> tuple[str, ...]:
        return self.out.get((subject, relation), ())

    def subjects(self, obj: str, relation: str) -> tuple[str, ...]:
        return self.inc.get((obj, relation), ())

    def has(self, head: str, relation: str, tail: str) -> bool:
        return (head, relation, tail) in self.facts


@dataclass
class Budget:
    """Caps the edges a single rule may expand, so hub entities cannot blow up a query."""
    steps: int

    def spend(self) -> bool:
        self.steps -= 1
        return self.steps >= 0

    @property
    def exhausted(self) -> bool:
        return self.steps < 0


# ----------------------------------------------------------------------------- grounding

def _resolve(arg: str, bindings: dict[str, str]) -> Optional[str]:
    """Current value of an argument: itself if a constant, its binding if bound, else None."""
    return bindings.get(arg) if is_variable(arg) else arg


def _bind(atom: Atom, head: str, tail: str, bindings: dict[str, str]) -> Optional[dict[str, str]]:
    """Extend `bindings` with atom -> (head, tail); None if that contradicts an existing binding."""
    new = dict(bindings)
    for arg, value in ((atom.subject, head), (atom.object, tail)):
        if is_variable(arg):
            if new.setdefault(arg, value) != value:
                return None
        elif arg != value:
            return None
    return new


def _ground(atoms: list[tuple[int, Atom]], bindings: dict[str, str], index: TripleIndex,
            budget: Budget, forbid: Optional[Triple]) -> Iterator[tuple[dict[str, str], dict[int, Triple]]]:
    if not atoms:
        yield bindings, {}
        return
    # Evaluate a connected atom first: one whose subject or object is already known. AnyBURL bodies are
    # connected chains, so this reaches every atom when the head variable is bound. If nothing is
    # seedable (e.g. an r(c,Y) rule whose body never touches a constant) the rule simply does not fire.
    position = next((p for p, (_, a) in enumerate(atoms)
                     if _resolve(a.subject, bindings) is not None or _resolve(a.object, bindings) is not None),
                    None)
    if position is None:
        return
    slot, atom = atoms[position]
    rest = atoms[:position] + atoms[position + 1:]
    subject, obj = _resolve(atom.subject, bindings), _resolve(atom.object, bindings)
    if subject is not None and obj is not None:
        pairs: Sequence[tuple[str, str]] = [(subject, obj)] if index.has(subject, atom.relation, obj) else []
    elif subject is not None:
        pairs = [(subject, o) for o in index.objects(subject, atom.relation)]
    else:
        pairs = [(s, obj) for s in index.subjects(obj, atom.relation)]
    for head, tail in pairs:
        if not budget.spend():
            return
        fact: Triple = (head, atom.relation, tail)
        extended = None if fact == forbid else _bind(atom, head, tail, bindings)
        if extended is None:
            continue
        for final, facts in _ground(rest, extended, index, budget, forbid):
            yield final, {slot: fact, **facts}


def ground(body: Sequence[Atom], bindings: dict[str, str], index: TripleIndex, budget: Budget,
           forbid: Optional[Triple] = None, max_results: Optional[int] = None,
           ) -> Iterator[tuple[dict[str, str], tuple[Triple, ...]]]:
    """Yield (complete bindings, supporting facts in body order) for each grounding of `body`.

    `forbid` excludes one fact from being used as evidence (so a triple cannot prove itself).
    Stops early when the budget runs out or `max_results` groundings have been produced.
    """
    produced = 0
    for final, facts in _ground(list(enumerate(body)), dict(bindings), index, budget, forbid):
        yield final, tuple(facts[i] for i in range(len(body)))
        produced += 1
        if max_results is not None and produced >= max_results:
            return
