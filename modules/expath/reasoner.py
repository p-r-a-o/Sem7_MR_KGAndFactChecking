"""Rule application (link prediction) and eXpath's explanation of one triple.

`explain_triple` follows eXpath's three stages for a link (h, r, t):
  1. path aggregation   - grounded paths h -> t (length <= 3), grouped into relation paths;
  2. rule mining        - each relation path is a CP rule body for r, scored with SC/HC/supp
                          (looked up in the mined rule base, else computed on demand like process.py);
                          Uc rules come from the constants around h and t;
  3. critical facts     - every fact gets a noisy-OR of the rules it takes part in.
The *confidence* of the link is the aggregate of its rules' scores, and the *proof* is the best rule
plus one grounding of its body in the graph: both can be re-checked (see `proof.py`).

`apply_rules` runs the mined rules forward from a query entity to rank candidates for (h, r, ?).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from modules.expath.config import ReasoningConfig, RuleQuality
from modules.expath.graph import (Hop, IndexedGraph, Path, enumerate_paths, ground_body,
                                  group_by_relation_path)
from modules.expath.rules import Rule, cp_metrics, path_instances, uc_metrics


def noisy_or(scores) -> float:
    """eXpath calculateConfidence: 1 - prod(1 - s)."""
    out = 1.0
    for s in scores:
        out *= 1.0 - s
    return 1.0 - out


def aggregate(scores, how: str = "noisy-or") -> float:
    """Confidence of a link from its rules' scores. "max" and "max-plus" report the best rule."""
    scores = list(scores)
    if not scores:
        return 0.0
    return noisy_or(scores) if how == "noisy-or" else max(scores)


@dataclass
class RuleBase:
    rules: list[Rule]
    smoothing: int = 10
    cp_by_head: dict[int, list[Rule]] = field(default_factory=dict)
    cp_lookup: dict[tuple[int, tuple[int, ...]], Rule] = field(default_factory=dict)
    uc_index: dict[tuple[int, int, int], list[Rule]] = field(default_factory=dict)
    on_demand: dict[tuple, object] = field(default_factory=dict)   # Rule | None (pending) | "ineligible"

    def __post_init__(self) -> None:
        for r in self.rules:
            if r.kind == "CP":
                self.cp_by_head.setdefault(r.head, []).append(r)
                self.cp_lookup[(r.head, r.body)] = r
            else:
                self.uc_index.setdefault((r.head, r.body[0], r.body_const), []).append(r)
        for lst in self.cp_by_head.values():
            lst.sort(key=lambda r: -r.score(self.smoothing))


@dataclass
class Firing:
    rule: Rule
    score: float
    groundings: list[Path] = field(default_factory=list)   # CP: body paths; Uc: the single body fact


@dataclass
class TripleEvidence:
    head: int
    rel: int
    tail: int
    confidence: float
    firings: list[Firing]                       # best first
    critical_facts: list[tuple[Hop, float]]     # eXpath stage 3, best first
    n_paths: int


# ----------------------------------------------------------------------------- link prediction

def apply_rules(g: IndexedGraph, rb: RuleBase, src: int, rel: int, cfg: ReasoningConfig) -> dict[int, list[Firing]]:
    """Fire every mined rule with head `rel` from `src`. Returns candidate -> firing rules (no groundings yet)."""
    out: dict[int, list[Firing]] = {}
    for rule in rb.cp_by_head.get(rel, ()):
        frontier = np.array([src], dtype=np.int64)
        for step in rule.body:
            frontier = g.reach(frontier, step)
            if not len(frontier) or len(frontier) > cfg.max_frontier:
                frontier = np.empty(0, dtype=np.int64)
                break
        score = rule.score(rb.smoothing)
        for c in frontier.tolist():
            if c != src:
                out.setdefault(c, []).append(Firing(rule, score))
    rels, nbrs = g.neighbors(src)
    for b, c2 in zip(rels.tolist(), nbrs.tolist()):
        for rule in rb.uc_index.get((rel, b, c2), ()):
            if rule.target != src:
                out.setdefault(rule.target, []).append(Firing(rule, rule.score(rb.smoothing), [((src, b, c2),)]))
    return out


def rank_candidates(firings: dict[int, list[Firing]], how: str, exclude: set[int], k: int) -> list[tuple[int, float]]:
    """Top-k candidates. "max-plus" (AnyBURL's ranking) compares candidates by their best rule, then their
    second best, and so on, which stops many weak correlated rules from outvoting one strong rule."""
    cands = [c for c in firings if c not in exclude]
    if how == "max-plus":
        keyed = {c: sorted((f.score for f in firings[c]), reverse=True)[:10] for c in cands}
        order = sorted(cands, key=lambda c: ([-x for x in keyed[c]], c))
        return [(c, keyed[c][0]) for c in order[:k]]
    scored = [(c, aggregate((f.score for f in firings[c]), how)) for c in cands]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


def ground_firings(g: IndexedGraph, src: int, cand: int, firings: list[Firing], limit: int) -> list[Firing]:
    """Best `limit` firings for a candidate, each with one grounding (CP bodies are grounded here)."""
    out = []
    for f in sorted(firings, key=lambda f: (-f.score, len(f.rule.body))):
        if f.rule.kind == "CP" and not f.groundings:
            path = ground_body(g, src, f.rule.body, cand)
            if path is None:
                continue
            f = Firing(f.rule, f.score, [path])
        out.append(f)
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------- explain one triple

def _cp_rule(g: IndexedGraph, rb: RuleBase, rel: int, body: tuple[int, ...], budget: list[int],
             row_chunk: int, cfg: ReasoningConfig) -> Optional[Rule]:
    """Mined rule if there is one, else metrics computed on demand (cached).

    The budget is charged for every eligible on-demand rule a query *considers*, cached or not, so a
    query sees exactly the same rules whatever was asked before it (verdicts must not depend on history).
    Ineligible bodies (too long / too costly) are never considered."""
    rule = rb.cp_lookup.get((rel, body))
    if rule is not None:
        return rule
    key = ("CP", rel, body)
    if key not in rb.on_demand:
        eligible = (len(body) <= cfg.max_on_demand_body_len
                    and max(g.relation_count(r) for r in (rel, *body)) <= cfg.max_on_demand_nnz
                    and path_instances(g, body) <= cfg.max_on_demand_instances)
        rb.on_demand[key] = "ineligible" if not eligible else None     # None = eligible, not computed yet
    if rb.on_demand[key] == "ineligible" or budget[0] <= 0:
        return None
    budget[0] -= 1
    if rb.on_demand[key] is None:
        supp, n_body, n_head = cp_metrics(g, rel, body, row_chunk)
        rb.on_demand[key] = Rule("CP", rel, body, supp, n_body, n_head)
    return rb.on_demand[key]


def _uc_rule(g: IndexedGraph, rb: RuleBase, rel: int, target: int, b: int, c2: int,
             xs_head: set, max_const_degree: int) -> Optional[Rule]:
    for r in rb.uc_index.get((rel, b, c2), ()):
        if r.target == target:
            return r
    key = ("Uc", rel, target, b, c2)
    if key not in rb.on_demand:
        if g.degree(c2) > max_const_degree:
            return None
        supp, n_body, n_head = uc_metrics(g, rel, target, b, c2, xs_head)
        rb.on_demand[key] = Rule("Uc", rel, (b,), supp, n_body, n_head, target=target, body_const=c2)
    return rb.on_demand[key]


def explain_triple(g: IndexedGraph, rb: RuleBase, h: int, rel: int, t: int, cfg: ReasoningConfig,
                   quality: RuleQuality, row_chunk: int = 4096, max_uc_edges: int = 200,
                   budget: Optional[list[int]] = None) -> TripleEvidence:
    """eXpath stages 1-3 for (h, rel, t); `rel` may be an inverse relation (head queries).
    `budget` ([n], mutated) caps on-demand CP metric computations; pass one list to share it across
    several calls (e.g. all KGE candidates of one query). Default: cfg.max_on_demand_rules for this call."""
    fwd = (h, rel, t) if rel < g.R else (t, rel - g.R, h)
    # stage 1: path aggregation
    paths = enumerate_paths(g, h, t, cfg.max_path_len, exclude=fwd, max_paths=cfg.max_paths)
    groups = group_by_relation_path(paths)

    # stage 2: CP rules from relation paths (most-grounded first), Uc rules from h's and t's constants
    firings: list[Firing] = []
    budget = budget if budget is not None else [cfg.max_on_demand_rules]
    for body, ps in sorted(groups.items(), key=lambda kv: (-len(kv[1]), len(kv[0]))):
        if body == (rel,):
            continue
        rule = _cp_rule(g, rb, rel, body, budget, row_chunk, cfg)
        if rule is not None and rule.passes(quality) and rule.supp >= cfg.min_rule_supp:
            firings.append(Firing(rule, rule.score(rb.smoothing), ps))
    for anchor, head_rel, target in ((h, rel, t), (t, g.inv(rel), h)):
        if g.degree(target) > cfg.max_uc_const_degree:
            continue                                   # hub constant: Uc rules about it are uninformative
        xs_head = set(g.neighbors_by_rel(target, g.inv(head_rel)).tolist())
        rels, nbrs = g.neighbors(anchor)
        for b, c2 in list(zip(rels.tolist(), nbrs.tolist()))[:max_uc_edges]:
            if c2 == target:
                continue
            rule = _uc_rule(g, rb, head_rel, target, b, c2, xs_head, cfg.max_uc_const_degree)
            if rule is not None and rule.passes(quality) and rule.supp >= cfg.min_rule_supp:
                firings.append(Firing(rule, rule.score(rb.smoothing), [((anchor, b, c2),)]))
    firings = [f for f in firings if f.score > 0]
    firings.sort(key=lambda f: (-f.score, len(f.rule.body)))       # ties: the shorter proof first

    # stage 3: critical facts. eXpath weights a CP fact by how often it is the first / last hop of the
    # rule's paths, scaled by the KGE relevance of the head/tail relation (Rh, Rt). Without the KGE
    # post-training step Rh = Rt = 0, and eXpath then uses the plain average of the two proportions.
    fact_scores: dict[Hop, list[float]] = {}
    for f in firings:
        if f.rule.kind == "Uc":
            fact_scores.setdefault(f.groundings[0][0], []).append(f.score)
            continue
        n = len(f.groundings)
        firsts, lasts = {}, {}
        for p in f.groundings:
            firsts[p[0]] = firsts.get(p[0], 0) + 1
            lasts[p[-1]] = lasts.get(p[-1], 0) + 1
        for hop in set(firsts) | set(lasts):
            importance = (firsts.get(hop, 0) / n + lasts.get(hop, 0) / n) / 2
            fact_scores.setdefault(hop, []).append(importance * f.score)
    critical = sorted(((hop, noisy_or(s)) for hop, s in fact_scores.items()), key=lambda x: -x[1])

    confidence = aggregate((f.score for f in firings), cfg.aggregation)
    return TripleEvidence(h, rel, t, confidence, firings, critical, len(paths))
