"""Path-based verification of multi-entity claims (FactKG).

A FactKG claim names several entities and states facts linking them (one-hop, conjunction,
multi-hop, existence, negation). The verdict is built from checkable pieces:

  link(a, b)   - a path between two claim entities whose relations are relevant to the claim text
                 (mean relevance >= path_relevance, and no hop below hop_relevance)
                 (eXpath path aggregation, length <= 3), or, if the graph has no such path,
                 a link (a, r, b) for a claim-relevant relation r that eXpath *infers* with a rule
                 (KG completion used to prove veracity: the rule + its grounding is the proof);
  existence    - a single-entity claim: the entity has an edge of a claim-relevant relation;
  conflict     - the claim needs a --r--> b, but the graph gives a --r--> x for other x only.

SUPPORTED   all claim entities are connected by links (a spanning tree of them is the proof);
REFUTED     some entity cannot be linked and a conflict explains why;
NOT_ENOUGH_INFO otherwise. A negated claim swaps SUPPORTED and REFUTED.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from common.schema import CVResult, Explanation, ExplanationStep
from modules.expath.claims import ClaimQuery
from modules.expath.config import ReasoningConfig, RuleQuality, VerdictConfig
from modules.expath.graph import Hop, IndexedGraph, Path, enumerate_paths
from modules.expath.proof import PROOF_TAG, hop_step, path_text
from modules.expath.reasoner import RuleBase, explain_triple
from modules.expath.rules import formal_rule, metrics_readable, rule_readable

Relevance = Callable[[int], tuple[float, str]]   # doubled relation id -> (score, why)


@dataclass
class Link:
    a: int
    b: int
    score: float
    path: Path
    how: str                                 # "path" | "rule"
    rule: Optional[str] = None
    rule_metrics: Optional[str] = None


@dataclass
class ClaimEvidence:
    links: list[Link] = field(default_factory=list)
    conflicts: list[tuple[Hop, float, str]] = field(default_factory=list)   # (fact, relevance, why)
    existence: Optional[tuple[Hop, float, str]] = None
    # existence claim, no claim-relevant edge, but the entity is described: its facts (closed-world proof)
    absent: list[Hop] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)


def _best_path(g: IndexedGraph, a: int, b: int, rel_score: Relevance, max_len: int,
               max_paths: int, hop_floor: float = 0.0) -> Optional[tuple[float, Path]]:
    best = None
    for p in enumerate_paths(g, a, b, max_len, max_paths=max_paths):
        # mean relevance of the hops, but no hop may be (almost) unrelated to the claim: without the floor
        # one strong hop carried an unrelated one ("Obama --successor--> X --birthPlace--> Chicago")
        rels = [rel_score(r)[0] for _, r, _ in p]
        if min(rels) < hop_floor:
            continue
        s = sum(rels) / len(rels)
        s -= 0.01 * (len(p) - 1)                        # prefer the shorter of equally relevant paths
        if best is None or s > best[0]:
            best = (s, p)
    return best


def _relevant_relations(g: IndexedGraph, e: int, rel_score: Relevance, threshold: float) -> list[tuple[int, float, str]]:
    rels, _ = g.neighbors(e)
    out = {}
    for r in set(rels.tolist()):
        s, why = rel_score(r)
        if s >= threshold:
            out[r] = (s, why)
    return sorted(((r, s, w) for r, (s, w) in out.items()), key=lambda x: -x[1])


def is_literal(g: IndexedGraph, e: int) -> bool:
    return g.entity_id(e).startswith('"')


def find_conflicts(g: IndexedGraph, ents: list[int], comp: dict[int, int], rel_score: Relevance,
                   threshold: float, functional: Optional[Sequence[float]] = None) -> list[tuple[Hop, float, str]]:
    """Type-consistent conflicts. For claim entities a, b that could not be connected, the claim needs
    a --r--> b for some claim-relevant r. If the graph instead gives a --r--> x (x not in the claim), that
    contradicts the claim to the extent that r is functional (one value per subject in the graph) and
    that b is the kind of thing r points to (b has an incoming r edge somewhere; else half weight).
    Literal anchors are skipped. `comp` maps entity -> connected component of the claim evidence."""
    out: dict[Hop, tuple[float, str]] = {}
    for a in ents:
        if is_literal(g, a):
            continue
        rels_a = _relevant_relations(g, a, rel_score, threshold)[:4]
        for b in ents:
            if b == a or comp[a] == comp[b]:
                continue
            b_rels = set(g.neighbors(b)[0].tolist())
            for r, s, why in rels_a:
                targets = g.neighbors_by_rel(a, r).tolist()
                if b in targets:
                    continue
                weight = s * (functional[r] if functional is not None else 1.0)
                weight *= 1.0 if g.inv(r) in b_rels else 0.5
                for x in [x for x in targets if x not in ents][:2]:
                    hop = (a, r, x)
                    if hop not in out or out[hop][0] < weight:
                        out[hop] = (weight, why)
    ranked = sorted(((hop, w, why) for hop, (w, why) in out.items()), key=lambda c: -c[1])
    return [c for c in ranked if c[1] >= threshold]


def gather_evidence(g: IndexedGraph, rb: RuleBase, ents: list[int], rel_score: Relevance,
                    vcfg: VerdictConfig, rcfg: ReasoningConfig, quality: RuleQuality,
                    single: bool, rule_budget: int = 3,
                    functional: Optional[Sequence[float]] = None) -> ClaimEvidence:
    """`single`: the claim itself names one entity (existence claim). With fewer than two entities found
    in the graph otherwise, there is nothing to connect and the result stays empty (NOT_ENOUGH_INFO)."""
    ev = ClaimEvidence()
    if not single and len(ents) < 2:
        return ev
    if single and len(ents) == 1:
        e = ents[0]
        rels, nbrs = g.neighbors(e)
        best = None
        for r, x in zip(rels.tolist(), nbrs.tolist()):
            s, why = rel_score(r)
            if s >= vcfg.relation_relevance and (best is None or s > best[1]):
                best = ((e, r, x), s, why)
        ev.existence = best
        if best is None:
            ev.absent = list(zip([e] * len(rels), rels.tolist(), nbrs.tolist()))
        return ev

    # pairwise links (best relevant path), then union-find to see what is connected
    parent = {e: e for e in ents}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    candidates: list[Link] = []
    unresolved: list[tuple[int, int]] = []
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            bp = _best_path(g, a, b, rel_score, min(rcfg.max_path_len, 3), 500, vcfg.hop_relevance)
            if bp is not None and bp[0] >= vcfg.path_relevance:
                candidates.append(Link(a, b, bp[0], bp[1], "path"))
            else:
                unresolved.append((a, b))

    # KG completion: an unlinked pair (a, b) is linked if eXpath infers a --r--> b for a claim-relevant r
    budget = rcfg.max_on_demand_rules
    local_cfg = ReasoningConfig(**{**rcfg.__dict__, "max_on_demand_rules": min(budget, 5),
                                   "max_on_demand_body_len": 2, "max_paths": 500})
    tried = 0
    for a, b in unresolved:
        for anchor, other in ((a, b), (b, a)):
            for r, s, _ in _relevant_relations(g, anchor, rel_score, vcfg.relation_relevance)[:2]:
                if tried >= rule_budget:
                    break
                tried += 1
                tev = explain_triple(g, rb, anchor, r, other, local_cfg, quality)
                if tev.confidence >= vcfg.link_support and tev.firings and tev.firings[0].groundings:
                    f = tev.firings[0]
                    candidates.append(Link(anchor, other, min(s, tev.confidence), f.groundings[0], "rule",
                                           formal_rule(g, f.rule),
                                           f"{rule_readable(g, f.rule)} ({metrics_readable(g, f.rule)})"))
                    break

    for link in sorted(candidates, key=lambda l: -l.score):
        ra, rb_ = find(link.a), find(link.b)
        if ra != rb_:
            parent[ra] = rb_
            ev.links.append(link)

    roots = {find(e) for e in ents}
    if len(roots) > 1:
        main = max(roots, key=lambda r: sum(find(e) == r for e in ents))
        lonely = [e for e in ents if find(e) != main]
        ev.unlinked = [g.entity_id(e) for e in lonely]
        ev.conflicts = find_conflicts(g, ents, {e: find(e) for e in ents}, rel_score,
                                      vcfg.conflict, functional)
    return ev


def decide(g: IndexedGraph, claim: str, q: ClaimQuery, ev: ClaimEvidence, missing: list[str],
           enrichment_note: str = "", closed_world_existence: bool = True,
           min_described: int = 1, absent_confidence: float = 0.6) -> CVResult:
    """Verdict + proof from gathered evidence."""
    steps: list[ExplanationStep] = []
    rule, conf = None, None
    names = ", ".join(g.labels.get(e, e) for e in q.entities) or "none"
    head = f"Claim entities ({q.linking}): {names}."
    if missing:
        head += f" Not in the graph: {', '.join(missing)}."
    if enrichment_note:
        head += " " + enrichment_note

    if ev.existence is not None:
        hop, s, why = ev.existence
        steps.append(hop_step(g, hop, f"{PROOF_TAG} (existence)"))
        verdict, conf = "SUPPORTED", s
        body = f"The graph states {g.hop_text(hop)}; the relation is relevant to the claim ({why})."
    elif closed_world_existence and len(ev.absent) >= min_described:
        # Existence claim about an entity the graph describes, with no claim-relevant relation among
        # its facts: under the closed-world reading FactKG uses for existence claims, the claim is false.
        # The proof is the entity's actual facts, so the absence can be checked.
        verdict, conf = "REFUTED", absent_confidence
        e = ev.absent[0][0]
        rel_names = sorted({g.rel_base(r) for _, r, _ in ev.absent})
        for hop in ev.absent[:5]:
            steps.append(hop_step(g, hop, f"{PROOF_TAG} (described entity: known fact, not a claim-relevant relation)"))
        body = (f"{g.label(e)} is described by {len(ev.absent)} facts over the relations {', '.join(rel_names[:12])}"
                f"{' ...' if len(rel_names) > 12 else ''}; none is relevant to the claim, so the claimed "
                f"relation is absent (closed-world reading of an existence claim).")
    elif ev.links and not ev.unlinked and len(q.entities) > 1 and not missing:
        verdict = "SUPPORTED"
        parts = []
        for link in ev.links:
            for hop in link.path:
                note = f"{PROOF_TAG} ({'rule body grounding' if link.how == 'rule' else 'evidence path'})"
                steps.append(hop_step(g, hop, note))
            if link.how == "rule":
                rule = rule or link.rule
                parts.append(f"{g.label(link.a)} and {g.label(link.b)} are not linked directly, but the rule "
                             f"*{link.rule_metrics}* infers the link from **{path_text(g, link.path)}**")
            else:
                parts.append(f"**{path_text(g, link.path)}** (relevance to the claim {link.score:.2f})")
        conf = min(l.score for l in ev.links)
        body = "Every claim entity is connected by evidence relevant to the claim:\n" + "\n".join(f"- {p}" for p in parts)
    elif ev.conflicts:
        verdict = "REFUTED"
        for hop, s, why in ev.conflicts[:4]:
            steps.append(hop_step(g, hop, f"{PROOF_TAG} (conflicting fact, {why})"))
        for link in ev.links:
            for hop in link.path:
                steps.append(hop_step(g, hop, "evidence path (consistent part of the claim)"))
        conf = ev.conflicts[0][1]
        hop = ev.conflicts[0][0]
        body = (f"No claim-relevant path reaches {', '.join(g.labels.get(e, e) for e in ev.unlinked)}; instead the graph "
                f"states {g.hop_text(hop)}" + (f" (and {len(ev.conflicts) - 1} more conflicting facts)."
                                                if len(ev.conflicts) > 1 else "."))
    else:
        verdict = "NOT_ENOUGH_INFO"
        for link in ev.links:
            for hop in link.path:
                steps.append(hop_step(g, hop, "evidence path (partial)"))
        body = ("No claim-relevant path or rule connects all claim entities, and no conflicting fact was found."
                if len(q.entities) != 1 else
                "The entity has no edge whose relation is relevant to the claim.")

    if q.negated and verdict != "NOT_ENOUGH_INFO":
        verdict = "REFUTED" if verdict == "SUPPORTED" else "SUPPORTED"
        body += " The claim is negated, so this evidence " + (
            "contradicts it." if verdict == "REFUTED" else "supports it.")
    if not steps:
        steps.append(ExplanationStep(triple=None, note=body))
    ex = Explanation(kind="path", steps=steps, text=f"{head} {body}", rule=rule, confidence=conf)
    return CVResult(claim=claim, triple=None, verdict=verdict, confidence=conf, explanation=ex)
