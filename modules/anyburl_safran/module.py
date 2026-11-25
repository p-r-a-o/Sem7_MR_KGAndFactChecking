"""AnyburlSafranModule: rule-based link prediction and claim verification behind the shared interface.

Rules are mined offline by the official AnyBURL jar (scripts/mine_rules.py). This module only
*applies* them, in Python, so that every answer can carry the rule **and the facts it grounded on**
- AnyBURL's own `Apply` emits ranked candidates without their groundings. See README ("Deviations").

Design notes
  * Grounding uses the TRAIN split only, so held-out facts can never prove themselves. For a triple
    under test the triple itself is additionally forbidden as evidence.
  * LP: fire every rule for the queried relation from the bound entity, aggregate per candidate
    (max-confidence like AnyBURL, or noisy-or like SAFRAN), rank, explain with the best grounding.
  * CV: parse claim -> triple. Direct hit in the graph -> SUPPORTED. A functional relation that
    already states a different answer -> REFUTED. Otherwise the verdict follows the claimed entity's
    rank among rule-derived candidates (thresholds in config, tuned on fb15k237 valid claims).
  * Stdlib only: no model, no Java at request time. load() is a rule-file parse plus an index build.
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional

from common.client import parse_claim
from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.schema import CVResult, Explanation, LinkPrediction, LPResult, Triple
from common.services import get_service
from modules.anyburl_safran import core, rules
from modules.anyburl_safran.config import AnyburlConfig


@dataclass
class _State:
    kg: KnowledgeGraph
    ruleset: rules.RuleSet
    index: rules.TripleIndex
    stats: core.RelationStats
    relations: set[str]
    labels: dict[str, str]        # display labels: readable, and unique per entity


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class AnyburlSafranModule:
    name = "AnyBURL+SAFRAN"

    def __init__(self, config: Optional[AnyburlConfig] = None,
                 parse_fn: Optional[Callable[[str, str], Triple]] = None):
        self.cfg = config or AnyburlConfig()
        self._parse_fn = parse_fn          # (claim, dataset) -> triple; injected in tests
        self._st: Optional[_State] = None

    # ------------------------------------------------------------------ loading

    def load(self, kg: KnowledgeGraph) -> None:
        path = self.cfg.artifact_root / "anyburl_safran" / kg.name / self.cfg.rules_file
        if not path.exists():
            raise ModuleError("INTERNAL", f"No rule file at {path}. Run "
                                          f"modules/anyburl_safran/scripts/mine_rules.py --dataset {kg.name}.")
        f = self.cfg.filters
        ruleset = rules.load_rules(path, min_confidence=f.min_confidence,
                                   min_correct_predictions=f.min_correct_predictions,
                                   max_body_length=f.max_body_length,
                                   max_rules_per_relation=f.max_rules_per_relation,
                                   unseen_negative_examples=f.unseen_negative_examples)
        if not ruleset.rules:
            raise ModuleError("INTERNAL", f"{path} yielded no usable rules ({ruleset.summary()}).")
        print(f"[{self.name}] {kg.name}: {ruleset.summary()}")
        self._st = _State(
            kg=kg,
            ruleset=ruleset,
            index=rules.TripleIndex.from_triples(kg.train),
            stats=core.relation_stats(kg.train, self.cfg.thresholds.functional_violation_ratio,
                                      self.cfg.thresholds.functional_min_observations),
            relations=set(kg.relations),
            labels=core.disambiguate_labels(kg.entity_labels),
        )

    # ------------------------------------------------------------------ helpers

    def _need(self) -> _State:
        if self._st is None:
            raise ModuleError("NOT_LOADED", "AnyBURL+SAFRAN: call load() first.")
        return self._st

    def _check_entity(self, entity: str) -> str:
        if entity not in self._need().kg.entity_labels:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown entity: {entity!r}")
        return entity

    def _check_relation(self, relation: str) -> str:
        if relation not in self._need().relations:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown relation: {relation!r}")
        return relation

    def _lp_limits(self, k: int) -> core.ApplyLimits:
        """LP may stop once enough candidates exist; never below the requested k."""
        saturation = self.cfg.limits.saturation_candidates
        return dataclasses.replace(
            self.cfg.limits,
            saturation_candidates=None if saturation is None else max(saturation, k))

    def _cv_limits(self) -> core.ApplyLimits:
        """CV scores one specific entity, so it needs every rule that could derive it."""
        return dataclasses.replace(self.cfg.limits, saturation_candidates=None,
                                   max_rules=self.cfg.cv_max_rules)

    def _known_answers(self, bound: str, relation: str, direction: core.Direction) -> set[str]:
        st = self._need()
        return set(st.index.objects(bound, relation) if direction == "tail"
                   else st.index.subjects(bound, relation))

    def _explain(self, candidate: core.Candidate, intro: str, start: str) -> Explanation:
        return core.rule_explanation(candidate, self._need().labels, intro, self.cfg.aggregation,
                                     start=start, max_steps=self.cfg.explain_facts_shown,
                                     rules_mentioned=self.cfg.rules_mentioned)

    # ------------------------------------------------------------------ link prediction

    def predict_link(self, head: Optional[str], relation: str, tail: Optional[str], k: int = 10) -> LPResult:
        st = self._need()
        self._check_relation(relation)
        direction: core.Direction = "tail" if tail is None else "head"
        bound = self._check_entity(head if direction == "tail" else tail)  # type: ignore[arg-type]
        label = core.label_of

        fired = core.apply_rules(st.ruleset, st.index, relation, bound, direction, self._lp_limits(k))
        exclude = ({bound} | self._known_answers(bound, relation, direction)) if self.cfg.exclude_known else set()
        ranked = core.rank_candidates(fired, self.cfg.aggregation, self.cfg.dedupe_groundings, exclude, limit=k)

        predictions = []
        for candidate in ranked:
            name = label(candidate.candidate, st.labels)
            intro = (f"Predicted {name} as the tail of ({label(bound, st.labels)}, {relation}, ?)."
                     if direction == "tail" else
                     f"Predicted {name} as the head of (?, {relation}, {label(bound, st.labels)}).")
            predictions.append(LinkPrediction(candidate=candidate.candidate, score=candidate.score,
                                              explanation=self._explain(candidate, intro, bound)))
        return LPResult(query=(head, relation, tail), predictions=predictions)

    # ------------------------------------------------------------------ claim verification

    def _parse(self, claim: str) -> Triple:
        st = self._need()
        if self._parse_fn is not None:
            return self._parse_fn(claim, st.kg.name)
        url = self.cfg.parser_url or get_service("claim_parser").base_url
        h, r, t = parse_claim(url, claim, st.kg.name)["triple"]
        return (h, r, t)

    def verify_claim(self, claim: str) -> CVResult:
        return self.verify_triple(claim, self._parse(claim))

    def verify_triple(self, claim: str, triple: Triple) -> CVResult:
        """The reasoning half of verify_claim, callable without the claim parser service (tests, eval)."""
        return self.decide(claim, self.score_triple(triple))

    def score_triple(self, triple: Triple) -> core.TripleEvidence:
        """Collect what the rules say about one triple, without applying any verdict threshold."""
        st = self._need()
        head, relation, tail = triple
        self._check_relation(relation)
        self._check_entity(head)
        self._check_entity(tail)

        if st.index.has(head, relation, tail):
            return core.TripleEvidence(triple=triple, in_graph=True, rank=1, n_candidates=0, support=(),
                                       scored=None, conflicts=(), ranked_tail=(), ranked_head=())

        limits = self._cv_limits()
        fired_tail = core.apply_rules(st.ruleset, st.index, relation, head, "tail", limits, forbid=triple)
        fired_head = core.apply_rules(st.ruleset, st.index, relation, tail, "head", limits, forbid=triple)
        support = list(fired_tail.get(tail, [])) + list(fired_head.get(head, []))

        # Filtered rankings: other answers already known to be true do not count against the claim.
        ranked_tail = core.rank_candidates(fired_tail, self.cfg.aggregation, self.cfg.dedupe_groundings,
                                           exclude=self._known_answers(head, relation, "tail") - {tail})
        ranked_head = core.rank_candidates(fired_head, self.cfg.aggregation, self.cfg.dedupe_groundings,
                                           exclude=self._known_answers(tail, relation, "head") - {head})
        ranks = [r for r in (core.rank_of(ranked_tail, tail), core.rank_of(ranked_head, head)) if r is not None]
        return core.TripleEvidence(
            triple=triple, in_graph=False,
            rank=min(ranks) if ranks else None,
            n_candidates=max(len(ranked_tail), len(ranked_head)),
            support=tuple(support),
            scored=(core.aggregate(tail, support, self.cfg.aggregation, self.cfg.dedupe_groundings)
                    if support else None),
            conflicts=tuple(st.stats.conflicts(st.index, head, relation, tail)),
            ranked_tail=tuple(ranked_tail), ranked_head=tuple(ranked_head))

    def decide(self, claim: str, evidence: core.TripleEvidence) -> CVResult:
        """Turn gathered evidence into a verdict + proof, using the configured thresholds."""
        st = self._need()
        triple = evidence.triple
        head, relation, tail = triple
        labels, label = st.labels, core.label_of

        if evidence.in_graph:
            return CVResult(claim, triple, "SUPPORTED", 1.0, core.direct_fact_explanation(triple, labels))

        th = self.cfg.thresholds
        verdict = ("REFUTED" if evidence.conflicts else
                   core.decide_verdict(evidence.rank, evidence.n_candidates, th.support_rank,
                                       th.refute_rank, th.min_candidates_for_refute))

        claim_text = f"({label(head, labels)}, {relation}, {label(tail, labels)})"
        if evidence.rank is None:
            prefix = (f"No rule derives {claim_text}; rule application found {evidence.n_candidates} "
                      f"other candidates for this query.")
        else:
            prefix = (f"Rule application ranks {claim_text} #{evidence.rank} of "
                      f"{evidence.n_candidates} rule-derived candidates.")

        if evidence.conflicts:
            explanation = core.conflict_explanation(list(evidence.conflicts), triple, labels,
                                                    self.cfg.explain_facts_shown)
            if evidence.scored is not None:
                prefix += (f" ({len(evidence.support)} rule(s) do derive the claim, score "
                           f"{evidence.score:.3f}.)")
            confidence: Optional[float] = _clamp01(1.0 - evidence.score)
        elif verdict == "SUPPORTED" and evidence.scored is not None:
            explanation = self._explain(evidence.scored, f"Claimed link: {claim_text}.", head)
            confidence = evidence.score
        elif verdict == "REFUTED" and (evidence.ranked_tail or evidence.ranked_head):
            # Show what the rules do derive instead, in whichever direction produced candidates.
            best = (evidence.ranked_tail or evidence.ranked_head)[0]
            start, query_text = ((head, f"({label(head, labels)}, {relation}, ?)") if evidence.ranked_tail
                                 else (tail, f"(?, {relation}, {label(tail, labels)})"))
            explanation = self._explain(
                best, f"The rules instead derive {label(best.candidate, labels)} for {query_text}.", start)
            confidence = _clamp01(1.0 - evidence.score)
        else:
            n_rules = len(st.ruleset.for_relation(relation))
            alternatives = evidence.ranked_tail or evidence.ranked_head
            explanation = core.no_rule_explanation(
                f"{prefix} {n_rules} rules were learned for {relation}, but none of them grounds to a "
                f"proof of this claim in the training graph"
                + (f"; the best rule-derived answers are "
                   f"{core.top_candidates_text(list(alternatives), labels)}." if alternatives else "."))
            return CVResult(claim, triple, verdict,
                            evidence.score if verdict == "SUPPORTED" and evidence.scored else None,
                            explanation)

        return CVResult(claim, triple, verdict, confidence,
                        dataclasses.replace(explanation, text=f"{prefix} {explanation.text}"))
