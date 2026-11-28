"""Reasoning-free baselines (stdlib-only).

They answer two questions a report needs settled before it can credit a module with "reasoning":

* `FrequencyLP` - how far does a relation-conditional popularity prior get on link prediction? It is
  AnyBURL's "zero rules" idea (predict from frequency alone) with exact, tie-aware, filtered ranks.
* `PriorCV` - can the claim-verification benchmark be solved from entity popularity alone? Negatives are
  built by swapping in entities, and if the swap shifts popularity, a classifier that never looks at
  the graph's structure separates the classes (cf. hypothesis-only baselines in NLI, Poliak et al. 2018).
  An AUC well above 0.5 here means the benchmark leaks, and a module must beat it to claim anything.
"""
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from typing import Collection, Iterable, Optional, Sequence

from common.schema import Triple


class FrequencyLP:
    """Rank entities by how often they fill the queried slot of the relation in the training split."""

    def __init__(self, train: Iterable[Triple], entities: Sequence[str]):
        self.entities = list(entities)
        self._counts: dict[tuple[str, str], Counter] = {}
        self._sorted: dict[tuple[str, str], list[int]] = {}
        for h, r, t in train:
            self._counts.setdefault((r, "tail"), Counter())[t] += 1
            self._counts.setdefault((r, "head"), Counter())[h] += 1

    def _sorted_counts(self, key: tuple[str, str]) -> list[int]:
        if key not in self._sorted:
            counts = self._counts.get(key, Counter())
            self._sorted[key] = sorted(list(counts.values()) + [0] * (len(self.entities) - len(counts)))
        return self._sorted[key]

    def rank(self, gold: str, relation: str, side: str, known_others: Collection[str] = (),
             bound: Optional[str] = None) -> float:
        """Filtered average rank of `gold` (ties share the mean rank; float, 1.0 = best).

        `side` is the side being predicted. `known_others` are other known-true answers and `bound` is the
        query's given entity; both are removed from the ranking, as in filtered evaluation.
        """
        key = (relation, side)
        counts = self._counts.get(key, Counter())
        ordered = self._sorted_counts(key)
        c = counts.get(gold, 0)
        greater = len(ordered) - bisect_right(ordered, c)
        equal = bisect_right(ordered, c) - bisect_left(ordered, c)           # includes gold itself
        skip = set(known_others) | ({bound} if bound is not None else set())
        for entity in skip:
            if entity == gold:
                continue
            ce = counts.get(entity, 0)
            if ce > c:
                greater -= 1
            elif ce == c:
                equal -= 1
        return greater + 1 + (equal - 1) / 2


class PriorCV:
    """Score a claim by popularity alone. Higher = looks more like a true triple.

    kind="relation_position": log(1+count) of the head among the relation's training heads plus the same
        for the tail among its training tails (what a type-consistent corruption disturbs most).
    kind="degree": log(1+degree) of head and tail over the whole training graph, relation ignored.
    """

    KINDS = ("relation_position", "degree")

    def __init__(self, train: Iterable[Triple], kind: str = "relation_position"):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        self.kind = kind
        self._head: Counter = Counter()
        self._tail: Counter = Counter()
        self._degree: Counter = Counter()
        for h, r, t in train:
            self._head[(r, h)] += 1
            self._tail[(r, t)] += 1
            self._degree[h] += 1
            self._degree[t] += 1

    def score(self, triple: Triple) -> float:
        h, r, t = triple
        if self.kind == "degree":
            return math.log1p(self._degree[h]) + math.log1p(self._degree[t])
        return math.log1p(self._head[(r, h)]) + math.log1p(self._tail[(r, t)])
