"""Link-prediction evaluation helpers (stdlib-only, pure functions).

`rank_metrics` in `common/metrics.py` turns ranks into MRR / Hits@k. This module supplies what goes
into it: the standard *filtered* rank (other known-true answers do not count against the gold one),
Bordes-style relation categories, and the query sampling shared by every LP evaluation.
"""
import random
from collections import defaultdict
from typing import Collection, Iterable, Optional, Sequence

from common.schema import Triple

Side = str  # "tail" = (h, r, ?), "head" = (?, r, t)


def sample_queries(triples: Sequence[Triple], n: int, seed: int = 0) -> list[Triple]:
    """The first `n` of a seeded shuffle. Same seed -> same queries, so modules are compared on identical data."""
    shuffled = list(triples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


class AnswerIndex:
    """All known-true answers per query over the given splits, for filtered ranking."""

    def __init__(self, *splits: Iterable[Triple]):
        self._tails: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._heads: dict[tuple[str, str], set[str]] = defaultdict(set)
        for split in splits:
            for h, r, t in split:
                self._tails[(h, r)].add(t)
                self._heads[(t, r)].add(h)

    def others(self, triple: Triple, side: Side) -> set[str]:
        """Known-true answers for the query behind `triple`, except the gold answer itself."""
        h, r, t = triple
        if side == "tail":
            return self._tails.get((h, r), set()) - {t}
        if side == "head":
            return self._heads.get((t, r), set()) - {h}
        raise ValueError("side must be 'tail' or 'head'")


def filtered_rank(candidates: Sequence[str], gold: str, known: Collection[str]) -> Optional[int]:
    """1-based rank of `gold` once other known-true answers are removed. None if it is not in the list."""
    rank = 0
    for cand in candidates:
        if cand == gold:
            return rank + 1
        if cand not in known:
            rank += 1
    return None


def requested_k(k: int, n_known: int, cap: int = 1000) -> int:
    """How many candidates to ask a module for so that `k` remain after filtering `n_known` answers."""
    return min(cap, k + n_known)


# ----------------------------------------------------------------------------- relation categories

def relation_categories(train: Iterable[Triple], threshold: float = 1.5) -> dict[str, str]:
    """Bordes et al. (2013): classify each relation as 1-1, 1-N, N-1 or N-N.

    tph = mean tails per head, hpt = mean heads per tail; a side is 'N' when its average reaches
    `threshold`. Names read head-to-tail: '1-N' means one head, many tails.
    """
    tails: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    heads: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for h, r, t in train:
        tails[r][h].add(t)
        heads[r][t].add(h)
    out = {}
    for r in tails:
        tph = sum(len(v) for v in tails[r].values()) / len(tails[r])
        hpt = sum(len(v) for v in heads[r].values()) / len(heads[r])
        out[r] = f"{'N' if hpt >= threshold else '1'}-{'N' if tph >= threshold else '1'}"
    return out


def reciprocal_ranks(ranks: Sequence[Optional[float]]) -> list[float]:
    """1/rank, with a miss (None) scoring 0 - the per-query values that paired tests compare."""
    return [1.0 / r if r else 0.0 for r in ranks]
