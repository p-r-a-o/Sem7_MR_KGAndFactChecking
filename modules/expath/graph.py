"""Integer-indexed graph for eXpath: CSR adjacency over forward + inverse edges, per-relation sparse
matrices (for rule metrics), an overlay for facts added at run time (node addition), and eXpath's
path aggregation stage (all relation paths of length <= 3 between two entities).

Relation ids live in a doubled space: r in [0, R) is a stored relation, r + R its inverse
(eXpath writes the inverse as r'; the shared KnowledgeGraph uses r + "^-1").
Only numpy/scipy are needed.
"""
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from scipy import sparse

from common.datasets.base import INVERSE_SUFFIX, KnowledgeGraph
from common.relnames import short_names

Hop = tuple[int, int, int]          # (from entity, relation in doubled space, to entity)
Path = tuple[Hop, ...]


@dataclass
class Overlay:
    """Facts added after load (e.g. retrieved from DBpedia). Kept apart from the mined statistics."""
    entity_ids: list[str] = field(default_factory=list)          # new entities, appended after the base ones
    adj: dict[int, list[tuple[int, int]]] = field(default_factory=dict)   # entity -> [(rel, nbr)] both directions
    source: dict[tuple[int, int, int], str] = field(default_factory=dict)  # forward fact -> provenance


class IndexedGraph:
    def __init__(self, ent_ids: Sequence[str], rel_names: Sequence[str], triples: np.ndarray,
                 labels: Optional[dict[str, str]] = None):
        self.ent_ids = list(ent_ids)
        self.ent2idx = {e: i for i, e in enumerate(self.ent_ids)}
        self.rel_names = list(rel_names)
        self.rel2idx = {r: i for i, r in enumerate(self.rel_names)}
        self.labels = labels or {}
        self.N, self.R = len(self.ent_ids), len(self.rel_names)
        self.triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        h, r, t = self.triples[:, 0], self.triples[:, 1], self.triples[:, 2]

        src = np.concatenate([h, t])
        rel = np.concatenate([r, r + self.R])
        dst = np.concatenate([t, h])
        order = np.lexsort((dst, rel, src))
        self.src, self.rel, self.dst = src[order], rel[order], dst[order]
        self.indptr = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.src, minlength=self.N), out=self.indptr[1:])
        self._key = self.rel * self.N + self.dst           # sorted within each entity's segment
        self._matrices: dict[int, sparse.csr_matrix] = {}
        self._rel_count = np.bincount(r, minlength=self.R)
        self.overlay = Overlay()
        self._short: Optional[dict[str, str]] = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_kg(cls, kg: KnowledgeGraph) -> "IndexedGraph":
        """Index the TRAIN split only, as every module does, so held-out facts never leak."""
        ent_ids = sorted({x for h, _, t in kg.train for x in (h, t)} | set(kg.entity_labels))
        ent2idx = {e: i for i, e in enumerate(ent_ids)}
        rel_names = list(kg.relations)
        rel2idx = {r: i for i, r in enumerate(rel_names)}
        arr = np.fromiter((v for h, r, t in kg.train for v in (ent2idx[h], rel2idx[r], ent2idx[t])),
                          dtype=np.int64, count=3 * len(kg.train)).reshape(-1, 3)
        return cls(ent_ids, rel_names, np.unique(arr, axis=0) if len(arr) else arr, kg.entity_labels)

    # ------------------------------------------------------------------ names

    @property
    def num_entities(self) -> int:
        return self.N + len(self.overlay.entity_ids)

    def entity_id(self, idx: int) -> str:
        return self.ent_ids[idx] if idx < self.N else self.overlay.entity_ids[idx - self.N]

    def label(self, idx: int) -> str:
        e = self.entity_id(idx)
        return self.labels.get(e, e)

    def inv(self, rel: int) -> int:
        return rel + self.R if rel < self.R else rel - self.R

    def rel_base(self, rel: int) -> str:
        return self.rel_names[rel % self.R]

    def rel_name(self, rel: int) -> str:
        """Name in the shared KnowledgeGraph convention (inverse = name + '^-1')."""
        return self.rel_base(rel) + ("" if rel < self.R else INVERSE_SUFFIX)

    def rel_index(self, name: str) -> Optional[int]:
        if name.endswith(INVERSE_SUFFIX):
            base = self.rel2idx.get(name[: -len(INVERSE_SUFFIX)])
            return None if base is None else base + self.R
        return self.rel2idx.get(name)

    def fact(self, hop: Hop) -> tuple[str, str, str]:
        """Canonical (head, relation, tail) IDs of the stored fact behind a hop."""
        a, r, b = hop
        if r < self.R:
            return self.entity_id(a), self.rel_names[r], self.entity_id(b)
        return self.entity_id(b), self.rel_names[r - self.R], self.entity_id(a)

    def rel_short(self, rel: int) -> str:
        """Readable name of the stored relation (common/relnames.py), e.g. 'award nominee'."""
        if getattr(self, "_short", None) is None:                   # also graphs unpickled from a cache
            self._short = short_names(self.rel_names)
        return self._short[self.rel_base(rel)]

    def hop_text(self, hop: Hop) -> str:
        """The stored fact behind a hop, written the way it is stored: 'Lautner —[award nominee]→ Swift'."""
        a, r, b = hop
        if r >= self.R:
            a, b = b, a
        return f"{self.label(a)} —[{self.rel_short(r)}]→ {self.label(b)}"

    # ------------------------------------------------------------------ adjacency

    def neighbors(self, e: int) -> tuple[np.ndarray, np.ndarray]:
        """(relations, neighbours) of every edge leaving e, inverse edges included."""
        if e < self.N:
            lo, hi = self.indptr[e], self.indptr[e + 1]
            rels, nbrs = self.rel[lo:hi], self.dst[lo:hi]
        else:
            rels = nbrs = np.empty(0, dtype=np.int64)
        extra = self.overlay.adj.get(e)
        if extra:
            rels = np.concatenate([rels, np.fromiter((r for r, _ in extra), dtype=np.int64)])
            nbrs = np.concatenate([nbrs, np.fromiter((n for _, n in extra), dtype=np.int64)])
        return rels, nbrs

    def neighbors_by_rel(self, e: int, rel: int) -> np.ndarray:
        rels, nbrs = self.neighbors(e)
        return nbrs[rels == rel]

    def degree(self, e: int) -> int:
        base = int(self.indptr[e + 1] - self.indptr[e]) if e < self.N else 0
        return base + len(self.overlay.adj.get(e, ()))

    def has_edge(self, h: int, rel: int, t: int) -> bool:
        if h < self.N and t < self.N:
            lo, hi = self.indptr[h], self.indptr[h + 1]
            key = rel * self.N + t
            i = lo + np.searchsorted(self._key[lo:hi], key)
            if i < hi and self._key[i] == key:
                return True
        return any(r == rel and n == t for r, n in self.overlay.adj.get(h, ()))

    def provenance(self, hop: Hop) -> str:
        a, r, b = hop
        fwd = (a, r, b) if r < self.R else (b, r - self.R, a)
        return self.overlay.source.get(fwd, "knowledge graph (train split)")

    def relation_count(self, rel: int) -> int:
        return int(self._rel_count[rel % self.R])

    # ------------------------------------------------------------------ sparse matrices (base graph only)

    def matrix(self, rel: int) -> sparse.csr_matrix:
        """Binary N x N adjacency of a relation (doubled space). Cached."""
        if rel not in self._matrices:
            base = rel % self.R
            mask = self.triples[:, 1] == base
            h, t = self.triples[mask, 0], self.triples[mask, 2]
            rows, cols = (h, t) if rel < self.R else (t, h)
            m = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(self.N, self.N))
            m.data[:] = 1
            self._matrices[rel] = m
        return self._matrices[rel]

    def reach(self, frontier: np.ndarray, rel: int) -> np.ndarray:
        """Entities reachable from any frontier entity in one `rel` step (base graph + overlay)."""
        base = frontier[frontier < self.N]
        out = np.unique(self.matrix(rel)[base].indices) if len(base) else np.empty(0, dtype=np.int64)
        if self.overlay.adj:
            extra = [n for e in frontier for r, n in self.overlay.adj.get(int(e), ()) if r == rel]
            if extra:
                out = np.union1d(out, np.asarray(extra, dtype=np.int64))
        return out

    # ------------------------------------------------------------------ overlay (node addition)

    def add_entity(self, entity_id: str, label: Optional[str] = None) -> int:
        idx = self.ent2idx.get(entity_id)
        if idx is None:
            idx = self.num_entities
            self.overlay.entity_ids.append(entity_id)
            self.ent2idx[entity_id] = idx
        if label and entity_id not in self.labels:
            self.labels[entity_id] = label
        return idx

    def add_fact(self, h: int, rel: int, t: int, source: str) -> bool:
        """Add a forward fact (rel < R) to the overlay. Returns False if it was already known."""
        if rel >= self.R or self.has_edge(h, rel, t):
            return False
        self.overlay.adj.setdefault(h, []).append((rel, t))
        self.overlay.adj.setdefault(t, []).append((rel + self.R, h))
        self.overlay.source[(h, rel, t)] = source
        return True


# ---------------------------------------------------------------------------- path aggregation

def _to_target(g: IndexedGraph, t: int) -> dict[int, list[int]]:
    """y -> relations r with an edge y --r--> t (read off t's inverse neighbourhood)."""
    out: dict[int, list[int]] = {}
    rels, nbrs = g.neighbors(t)
    for r, y in zip(rels.tolist(), nbrs.tolist()):
        out.setdefault(y, []).append(g.inv(r))
    return out


def enumerate_paths(g: IndexedGraph, h: int, t: int, max_len: int = 3, exclude: Optional[Hop] = None,
                    max_paths: int = 2000, hub_degree: int = 1000, max_start: int = 300,
                    rng: Optional[np.random.Generator] = None) -> list[Path]:
    """Grounded paths h -> t of length <= max_len without repeated entities (eXpath path aggregation).

    Adapted from eXpath's `find_paths_bfs` (process.py). Two changes keep it tractable on large graphs:
    it meets in the middle through t's neighbourhood instead of a plain BFS, and for length-3 paths it
    skips middle entities with more than `hub_degree` edges and explores at most `max_start` of h's
    neighbours. `exclude` is the fact being explained (forward hop), which must not explain itself.
    """
    rng = rng or np.random.default_rng(0)
    excluded = set()
    if exclude is not None:
        a, r, b = exclude
        excluded = {(a, r, b), (b, g.inv(r), a)}
    paths: list[Path] = []
    h_rels, h_nbrs = g.neighbors(h)

    for r1, x in zip(h_rels.tolist(), h_nbrs.tolist()):
        if x == t and (h, r1, t) not in excluded:
            paths.append(((h, r1, t),))
    if max_len < 2:
        return paths[:max_paths]

    to_t = _to_target(g, t)
    for r1, x in zip(h_rels.tolist(), h_nbrs.tolist()):
        if x in (h, t):
            continue
        for r2 in to_t.get(x, ()):
            paths.append(((h, r1, x), (x, r2, t)))
            if len(paths) >= max_paths:
                return paths
    if max_len < 3 or not to_t:
        return paths

    targets = np.fromiter(to_t.keys(), dtype=np.int64)
    idx = np.arange(len(h_nbrs))
    if len(idx) > max_start:
        idx = rng.choice(idx, size=max_start, replace=False)
    for i in idx.tolist():
        r1, x = int(h_rels[i]), int(h_nbrs[i])
        if x in (h, t) or g.degree(x) > hub_degree:
            continue
        x_rels, x_nbrs = g.neighbors(x)
        mask = np.isin(x_nbrs, targets) & (x_nbrs != h) & (x_nbrs != t) & (x_nbrs != x)
        for r2, y in zip(x_rels[mask].tolist(), x_nbrs[mask].tolist()):
            for r3 in to_t[y]:
                paths.append(((h, r1, x), (x, r2, y), (y, r3, t)))
                if len(paths) >= max_paths:
                    return paths
    return paths


def relation_path(path: Path) -> tuple[int, ...]:
    """eXpath 'relation path': the path with intermediate entities removed."""
    return tuple(r for _, r, _ in path)


def group_by_relation_path(paths: Iterable[Path]) -> dict[tuple[int, ...], list[Path]]:
    groups: dict[tuple[int, ...], list[Path]] = {}
    for p in paths:
        groups.setdefault(relation_path(p), []).append(p)
    return groups


def ground_body(g: IndexedGraph, start: int, body: Sequence[int], end: int) -> Optional[Path]:
    """One grounding of a relation path from start to end (used to show a rule's evidence)."""
    layers = [np.array([start], dtype=np.int64)]
    for rel in body[:-1]:
        nxt = g.reach(layers[-1], rel)
        if not len(nxt):
            return None
        layers.append(nxt)
    hops: list[Hop] = []
    current = end
    for i in range(len(body) - 1, -1, -1):
        rel = body[i]
        preds = g.neighbors_by_rel(current, g.inv(rel))           # p --rel--> current
        allowed = set(layers[i].tolist())
        on_path = {current} | {a for a, _, _ in hops}
        choice = next((p for p in preds.tolist() if p in allowed and (p not in on_path or i == 0)), None)
        if choice is None:
            return None
        hops.append((choice, rel, current))
        current = choice
    hops.reverse()
    return tuple(hops) if hops and hops[0][0] == start else None


def iter_hops(paths: Iterable[Path]) -> Iterator[Hop]:
    for p in paths:
        yield from p
