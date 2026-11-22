"""Pure (torch/DGL-free) logic for the PowerLink wrapper: path decoding, ranking, verdicts, text.

Conventions
-----------
PowerLink works on integer IDs over a graph with forward edges (relation r in [0, num_rel)) and
explicit inverse edges (relation r + num_rel). A `Hop` is one traversed edge in that graph.
Its supporting *fact* is always reported in canonical orientation (head, relation, tail).
"""
import heapq
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from common.schema import Explanation, ExplanationStep, Triple


@dataclass(frozen=True)
class Vocab:
    ent_ids: list[str]        # index -> canonical entity ID
    rel_names: list[str]      # index -> canonical relation name (forward relations only)
    labels: dict[str, str]    # canonical entity ID -> readable label

    def label(self, idx: int) -> str:
        e = self.ent_ids[idx]
        return self.labels.get(e, e)


@dataclass(frozen=True)
class Hop:
    src: int                  # entity index the edge was traversed from
    dst: int                  # entity index it leads to
    rel: int                  # [0, 2*num_rel); >= num_rel means the inverse edge
    weight: float             # explainer edge-mask weight in [0, 1]


# ----------------------------------------------------------------------------- path decoding

def pair_index(src: Sequence[int], dst: Sequence[int], rels: Sequence[int],
               weights: Sequence[float]) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """(u, v) -> [(relation, weight)] over all edges of the computation graph."""
    index: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for u, v, r, w in zip(src, dst, rels, weights):
        index.setdefault((u, v), []).append((r, w))
    return index


def decode_path(path: Iterable[tuple[int, int]], pairs: dict[tuple[int, int], list[tuple[int, float]]],
                node_ids: Sequence[int]) -> list[Hop]:
    """Turn PowerLink's local (u, v) node pairs into Hops with global entity indices.

    PowerLink paths carry no relation; between parallel edges we keep the one with the largest mask weight.
    """
    hops = []
    for u, v in path:
        options = pairs.get((u, v))
        if not options:
            continue
        rel, weight = max(options, key=lambda x: x[1])
        hops.append(Hop(node_ids[u], node_ids[v], rel, weight))
    return hops


def invert_rel(rel: int, num_rel: int) -> int:
    return rel + num_rel if rel < num_rel else rel - num_rel


def reverse_hops(hops: Sequence[Hop], num_rel: int) -> list[Hop]:
    """Walk the same path backwards (used for head queries, which are answered via inverse relations)."""
    return [Hop(h.dst, h.src, invert_rel(h.rel, num_rel), h.weight) for h in reversed(hops)]


def hop_fact(hop: Hop, num_rel: int) -> tuple[int, int, int]:
    """Canonical-orientation (head, relation, tail) indices of the fact behind a hop."""
    if hop.rel < num_rel:
        return hop.src, hop.rel, hop.dst
    return hop.dst, hop.rel - num_rel, hop.src


def is_chain(hops: Sequence[Hop]) -> bool:
    return bool(hops) and all(a.dst == b.src for a, b in zip(hops, hops[1:]))


# ----------------------------------------------------------------------------- text / explanation

def _rel_text(hop: Hop, num_rel: int, vocab: Vocab) -> str:
    name = vocab.rel_names[hop.rel % num_rel]
    return name if hop.rel < num_rel else f"{name} (inverse)"


def format_path(hops: Sequence[Hop], num_rel: int, vocab: Vocab) -> str:
    if not hops:
        return ""
    out = vocab.label(hops[0].src)
    for h in hops:
        out += f" --[{_rel_text(h, num_rel, vocab)}]--> {vocab.label(h.dst)}"
    return out


def hop_to_step(hop: Hop, num_rel: int, vocab: Vocab) -> ExplanationStep:
    h, r, t = hop_fact(hop, num_rel)
    fact: Triple = (vocab.ent_ids[h], vocab.rel_names[r], vocab.ent_ids[t])
    arrow = f"{vocab.label(hop.src)} --[{_rel_text(hop, num_rel, vocab)}]--> {vocab.label(hop.dst)}"
    return ExplanationStep(triple=fact, note=f"{arrow} (importance {hop.weight:.2f})")


def path_explanation(paths: Sequence[Sequence[Hop]], num_rel: int, vocab: Vocab, intro: str,
                     paths_shown: int = 3) -> Optional[Explanation]:
    """Explanation from ranked PowerLink paths (best first). None if there is no usable path."""
    paths = [p for p in paths if p]
    if not paths:
        return None
    best = paths[0]
    if is_chain(best):
        text = f"{intro} Strongest supporting path: {format_path(best, num_rel, vocab)}."
    else:  # PowerLink falls back to the top-weighted edges when it finds no connecting path
        text = (f"{intro} No connected path was found; the most influential edges are: "
                + "; ".join(format_path([h], num_rel, vocab) for h in best) + ".")
    others = [format_path(p, num_rel, vocab) for p in paths[1:paths_shown] if is_chain(p)]
    if others:
        text += " Other paths: " + " | ".join(others) + "."
    return Explanation(kind="path", steps=[hop_to_step(h, num_rel, vocab) for h in best], text=text,
                       confidence=sum(h.weight for h in best) / len(best))


def fallback_explanation(reason: str) -> Explanation:
    return Explanation(kind="path", steps=[], text=reason, confidence=None)


# ----------------------------------------------------------------------------- ranking / verdicts

def top_k(scores: Sequence[float], k: int, exclude: set[int]) -> list[int]:
    """Indices of the k highest scores, skipping `exclude`."""
    candidates = (i for i in range(len(scores)) if i not in exclude)
    return heapq.nlargest(k, candidates, key=scores.__getitem__)


def filtered_rank(scores: Sequence[float], target: int, known: set[int]) -> int:
    """1-based rank of `target`, ignoring other known-true answers (ties resolved optimistically)."""
    t = scores[target]
    return 1 + sum(1 for i, s in enumerate(scores) if s > t and i != target and i not in known)


def decide_verdict(rank: int, support_rank: int, refute_rank: int) -> str:
    if rank <= support_rank:
        return "SUPPORTED"
    if rank > refute_rank:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
