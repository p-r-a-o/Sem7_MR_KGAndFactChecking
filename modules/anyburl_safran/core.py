"""Applying AnyBURL rules: firing, aggregation, ranking, verdicts and explanation text.

Pure logic (stdlib only, no HTTP, no config objects) so it is directly testable.

Aggregation
-----------
``max``      AnyBURL's default: a candidate scores as its single most confident rule, ties broken by
             the next most confident rule, and so on (lexicographic over the sorted confidence list).
``noisy-or`` SAFRAN-style combination: ``1 - prod(1 - confidence)`` over the rules that fired. This is
             plain noisy-or; real SAFRAN additionally clusters redundant rules (see the module README).
             ``dedupe=True`` removes only the cheap case: several rules firing on the same evidence.
"""
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

from common.schema import Explanation, ExplanationStep, Triple
from modules.anyburl_safran.rules import Budget, Rule, RuleSet, TripleIndex, ground, is_variable

Direction = Literal["tail", "head"]
Aggregation = Literal["max", "noisy-or"]


@dataclass(frozen=True)
class Firing:
    """One rule that derived a candidate, together with the facts it used."""
    rule: Rule
    facts: tuple[Triple, ...]


@dataclass(frozen=True)
class Candidate:
    candidate: str
    score: float
    sort_key: tuple[float, ...]
    firings: tuple[Firing, ...]          # confidence-descending

    @property
    def best(self) -> Firing:
        return self.firings[0]


@dataclass(frozen=True)
class ApplyLimits:
    """Bounds for one query. Grounding is cheap per rule but there can be thousands of rules.

    Defaults measured on 200 FB15k-237 test triples (both directions) against the 1000s snapshot,
    max aggregation. The apply-time limit is what binds, not the grounding budget:

        max_rules / saturation    MRR     Hits@10   ms/query   RSS
        5000  /  500             0.219   0.385     11         1.1GB
        10000 / 1000  (default)  0.230   0.420     22         1.6GB
        20000 / 2000             0.235   0.445     60         2.3GB
    """
    max_rules: int = 10000
    max_groundings_per_rule: int = 20
    step_budget_per_rule: int = 20000
    saturation_candidates: Optional[int] = 1000  # stop once this many candidates exist; None = all rules


# ----------------------------------------------------------------------------- firing rules

def query_binding(rule: Rule, direction: Direction, bound: str) -> Optional[tuple[dict[str, str], str]]:
    """Initial bindings and the head argument to read the answer from, or None if the rule cannot apply.

    ``direction="tail"`` means (bound, relation, ?) so the head's subject is fixed; ``"head"`` is the
    mirror case. A constant in the fixed position must equal `bound` for the rule to be applicable.
    """
    head = rule.head
    fixed, answer = (head.subject, head.object) if direction == "tail" else (head.object, head.subject)
    if is_variable(fixed):
        return {fixed: bound}, answer
    return ({}, answer) if fixed == bound else None


def fire_rule(rule: Rule, direction: Direction, bound: str, index: TripleIndex, limits: ApplyLimits,
              forbid: Optional[Triple] = None) -> dict[str, tuple[Triple, ...]]:
    """Candidates derived by one rule, each with the first grounding found for it."""
    binding = query_binding(rule, direction, bound)
    if binding is None:
        return {}
    bindings, answer = binding
    budget = Budget(limits.step_budget_per_rule)
    found: dict[str, tuple[Triple, ...]] = {}
    for final, facts in ground(rule.body, bindings, index, budget, forbid,
                               max_results=limits.max_groundings_per_rule):
        candidate = final.get(answer) if is_variable(answer) else answer
        if candidate is not None:
            found.setdefault(candidate, facts)
    return found


def apply_rules(ruleset: RuleSet, index: TripleIndex, relation: str, bound: str, direction: Direction,
                limits: ApplyLimits, forbid: Optional[Triple] = None) -> dict[str, list[Firing]]:
    """Fire every usable rule for `relation` and collect the candidates they derive.

    Rules are tried in confidence order, so stopping at `saturation_candidates` cannot change the top
    of a ``max``-aggregated ranking. Pass ``saturation_candidates=None`` when a complete picture is
    needed (scoring one specific triple, or noisy-or aggregation).
    """
    fired: dict[str, list[Firing]] = {}
    for n, rule in enumerate(ruleset.for_relation(relation)):
        if n >= limits.max_rules:
            break
        if limits.saturation_candidates is not None and len(fired) >= limits.saturation_candidates:
            break
        for candidate, facts in fire_rule(rule, direction, bound, index, limits, forbid).items():
            fired.setdefault(candidate, []).append(Firing(rule, facts))
    return fired


# ----------------------------------------------------------------------------- aggregation

def dedupe_firings(firings: Sequence[Firing]) -> list[Firing]:
    """Drop rules that fired on exactly the same evidence as an earlier, more confident rule."""
    seen: set[frozenset[Triple]] = set()
    kept = []
    for firing in firings:
        key = frozenset(firing.facts)
        if key not in seen:
            seen.add(key)
            kept.append(firing)
    return kept


def noisy_or(confidences: Iterable[float]) -> float:
    product = 1.0
    for c in confidences:
        product *= 1.0 - max(0.0, min(1.0, c))
    return 1.0 - product


def aggregate(candidate: str, firings: Sequence[Firing], method: Aggregation = "max",
              dedupe: bool = True) -> Candidate:
    ordered = sorted(firings, key=lambda f: -f.rule.confidence)
    if dedupe:
        ordered = dedupe_firings(ordered)
    confidences = tuple(f.rule.confidence for f in ordered)
    if method == "noisy-or":
        score = noisy_or(confidences)
        sort_key: tuple[float, ...] = (score,) + confidences
    else:
        score = confidences[0] if confidences else 0.0
        sort_key = confidences
    return Candidate(candidate=candidate, score=score, sort_key=sort_key, firings=tuple(ordered))


def rank_candidates(fired: dict[str, list[Firing]], method: Aggregation = "max", dedupe: bool = True,
                    exclude: Optional[set[str]] = None, limit: Optional[int] = None) -> list[Candidate]:
    """Aggregated candidates, best first. Ties are broken by entity ID for determinism."""
    exclude = exclude or set()
    scored = [aggregate(c, f, method, dedupe) for c, f in sorted(fired.items()) if c not in exclude]
    ranked = sorted(scored, key=lambda c: c.sort_key, reverse=True)
    return ranked if limit is None else ranked[:limit]


def rank_of(ranked: Sequence[Candidate], candidate: str) -> Optional[int]:
    """1-based position of `candidate`, or None if no rule derived it."""
    for position, c in enumerate(ranked, 1):
        if c.candidate == candidate:
            return position
    return None


# ----------------------------------------------------------------------------- relation statistics

@dataclass(frozen=True)
class RelationStats:
    """Which relations behave like functions in the training graph (used to refute claims)."""
    functional: frozenset[str]            # a head has at most one tail
    inverse_functional: frozenset[str]    # a tail has at most one head

    def conflicts(self, index: TripleIndex, head: str, relation: str, tail: str) -> list[Triple]:
        """Facts in the graph that contradict (head, relation, tail) under a functional reading."""
        found = []
        if relation in self.functional:
            found += [(head, relation, t) for t in index.objects(head, relation) if t != tail]
        if relation in self.inverse_functional:
            found += [(h, relation, tail) for h in index.subjects(tail, relation) if h != head]
        return found


def relation_stats(triples: Sequence[Triple], max_violation_ratio: float = 0.05,
                   min_observations: int = 20) -> RelationStats:
    """A relation counts as (inverse-)functional when few of its subjects have several objects.

    `min_observations` guards against calling a relation functional on the strength of a handful of
    training triples, which would refute claims about rare relations far too eagerly.
    """
    forward: dict[str, dict[str, set[str]]] = {}
    backward: dict[str, dict[str, set[str]]] = {}
    for h, r, t in triples:
        forward.setdefault(r, {}).setdefault(h, set()).add(t)
        backward.setdefault(r, {}).setdefault(t, set()).add(h)

    def selected(groups: dict[str, dict[str, set[str]]]) -> frozenset[str]:
        out = set()
        for relation, by_key in groups.items():
            violations = sum(1 for values in by_key.values() if len(values) > 1)
            if len(by_key) >= min_observations and violations / len(by_key) <= max_violation_ratio:
                out.add(relation)
        return frozenset(out)

    return RelationStats(functional=selected(forward), inverse_functional=selected(backward))


# ----------------------------------------------------------------------------- verdicts

@dataclass(frozen=True)
class TripleEvidence:
    """Everything the rules have to say about one triple, before any threshold is applied.

    Separating this from the verdict keeps threshold tuning cheap: gather evidence once per claim
    (see scripts/tune_thresholds.py), then re-decide for many threshold settings.
    """
    triple: Triple
    in_graph: bool                              # stated directly in the training graph
    rank: Optional[int]                         # best rank over both query directions; None = not derived
    n_candidates: int
    support: tuple[Firing, ...]                 # rules that derive the triple, either direction
    scored: Optional[Candidate]                 # aggregation over `support`
    conflicts: tuple[Triple, ...]               # facts contradicting it under a functional relation
    ranked_tail: tuple[Candidate, ...]          # what the rules derive for (head, relation, ?)
    ranked_head: tuple[Candidate, ...]          # ... and for (?, relation, tail)

    @property
    def score(self) -> float:
        return self.scored.score if self.scored else 0.0


def decide_verdict(rank: Optional[int], n_candidates: int, support_rank: int, refute_rank: int,
                   min_candidates_for_refute: int) -> str:
    """Rank of the claimed entity among rule-derived candidates -> verdict.

    `rank is None` means no rule derived it: that only refutes the claim if the rules did derive a
    decent number of other answers, otherwise the graph simply has nothing to say.
    """
    if rank is None:
        return "REFUTED" if n_candidates >= min_candidates_for_refute else "NOT_ENOUGH_INFO"
    if rank <= support_rank:
        return "SUPPORTED"
    if rank > refute_rank:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


# ----------------------------------------------------------------------------- explanation text

def label_of(entity: str, labels: dict[str, str]) -> str:
    return labels.get(entity, entity)


def disambiguate_labels(labels: dict[str, str]) -> dict[str, str]:
    """Append the entity ID to any label shared by several entities.

    Readable labels are not unique: 31% of WN18RR entities share a lemma once the dictionary
    definition is dropped ("run" covers 33 synsets) and 3% of FB15k-237 names collide
    ("Washington County" x7). Without this a proof can list two different entities under one name.
    Used for display only - generated claims keep the plain label.
    """
    counts: dict[str, int] = {}
    for name in labels.values():
        counts[name] = counts.get(name, 0) + 1
    return {entity: (f"{name} [{entity}]" if counts[name] > 1 else name)
            for entity, name in labels.items()}


def format_fact(fact: Triple, labels: dict[str, str]) -> str:
    h, r, t = fact
    return f"{label_of(h, labels)} --[{r}]--> {label_of(t, labels)}"


def format_grounding(facts: Sequence[Triple], labels: dict[str, str], start: Optional[str] = None) -> str:
    """Render a grounding as a path when the facts chain from `start`, else as a list of facts."""
    if not facts:
        return "no body atoms"
    if start is not None:
        current, parts = start, [label_of(start, labels)]
        for h, r, t in facts:
            if h == current:
                parts.append(f" --[{r}]--> {label_of(t, labels)}")
                current = t
            elif t == current:
                parts.append(f" --[{r}, inverse]--> {label_of(h, labels)}")
                current = h
            else:
                break
        else:
            return "".join(parts)
    return "; ".join(format_fact(f, labels) for f in facts)


def grounding_steps(firing: Firing, labels: dict[str, str], start: Optional[str] = None,
                    max_steps: int = 8) -> list[ExplanationStep]:
    """One step per body fact, annotated with the body atom it instantiates."""
    if not firing.facts:
        return [ExplanationStep(triple=None, note=f"Rule has no body atoms ({firing.rule.stats_text}).")]
    steps = []
    for atom, fact in list(zip(firing.rule.body, firing.facts))[:max_steps]:
        steps.append(ExplanationStep(triple=fact, note=f"{format_fact(fact, labels)} grounds {atom}"))
    if len(firing.facts) > max_steps:
        steps.append(ExplanationStep(triple=None, note=f"... {len(firing.facts) - max_steps} more facts"))
    return steps


def rule_explanation(candidate: Candidate, labels: dict[str, str], intro: str, method: Aggregation,
                     start: Optional[str] = None, max_steps: int = 8, rules_mentioned: int = 3,
                     ) -> Explanation:
    """Explanation for a rule-derived answer: the best rule, its grounding, and the runners-up."""
    best = candidate.best
    text = (f"{intro} Derived by the rule below ({best.rule.stats_text}), grounded as: "
            f"{format_grounding(best.facts, labels, start)}.")
    if not (is_variable(best.rule.head.subject) and is_variable(best.rule.head.object)):
        # An acyclic rule names the answer in its own head, so the grounded body does not lead to it.
        text += (f" The rule names {label_of(candidate.candidate, labels)} in its head: the body only "
                 f"has to hold for the rule to conclude it.")
    others = candidate.firings[1:rules_mentioned]
    if others:
        text += (" Also derived by: "
                 + " | ".join(f"{f.rule.text} (conf {f.rule.confidence:.3f})" for f in others) + ".")
    if len(candidate.firings) > 1:
        text += (f" {len(candidate.firings)} rules derive this answer in total; aggregated score "
                 f"{candidate.score:.3f} ({method}).")
    return Explanation(kind="rule", steps=grounding_steps(best, labels, start, max_steps), text=text,
                       rule=best.rule.text, confidence=candidate.score)


def direct_fact_explanation(triple: Triple, labels: dict[str, str]) -> Explanation:
    h, r, t = triple
    return Explanation(
        kind="rule", confidence=1.0, rule=None,
        steps=[ExplanationStep(triple=triple, note="Fact present in the training graph.")],
        text=f"({label_of(h, labels)}, {r}, {label_of(t, labels)}) is stated directly in the knowledge graph.")


def conflict_explanation(conflicts: Sequence[Triple], claimed: Triple, labels: dict[str, str],
                         max_steps: int = 8) -> Explanation:
    """Refutation from a functional relation: the graph already gives a different answer."""
    h, r, t = claimed
    return Explanation(
        kind="rule", confidence=None,
        rule=f"{r}(X,Y), {r}(X,Z) => Y = Z   (observed to be functional in the training graph)",
        steps=[ExplanationStep(triple=c, note=f"{format_fact(c, labels)} contradicts the claim")
               for c in conflicts[:max_steps]],
        text=(f"{r} behaves as a function in the training graph, which already states "
              f"{'; '.join(format_fact(c, labels) for c in conflicts[:max_steps])}. "
              f"That leaves no room for {label_of(t, labels)}."))


def no_rule_explanation(reason: str) -> Explanation:
    return Explanation(kind="rule", steps=[], text=reason, confidence=None)


def top_candidates_text(ranked: Sequence[Candidate], labels: dict[str, str], n: int = 3) -> str:
    return ", ".join(f"{label_of(c.candidate, labels)} ({c.score:.3f})" for c in ranked[:n])
