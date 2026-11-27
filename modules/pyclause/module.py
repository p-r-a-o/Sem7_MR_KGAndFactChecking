"""PyClauseModule: rule-based link prediction and claim verification with PyClause.

PyClause (Betz et al., "PyClause - Simple and Efficient Rule Handling for Knowledge Graphs",
IJCAI'24 demo) is a C++ rule engine with Python bindings. Rules are learned offline through its
`Learner` (AnyBURL or AMIE under the hood, scripts/learn_rules.py). At request time this module uses
PyClause's own handlers, unmodified:

  * LP: QAHandler ranks candidates for (h, r, ?) / (?, r, t) with maxplus or noisy-or aggregation,
    then PredictionHandler returns the predicting rules **and their groundings** for the top-k.
  * CV: parse claim -> triple. In the train graph -> SUPPORTED. Otherwise the verdict follows the
    claimed entity's filtered rank among QAHandler candidates (both directions), and the proof is
    PredictionHandler's explanation of the claimed triple (or of the best alternative if refuted).

The data graph PyClause grounds rules on is the TRAIN split only, so held-out facts can never prove
themselves. Unlike the AnyBURL+SAFRAN module, nothing is reimplemented in Python here.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from common.client import parse_claim
from common.datasets.base import KnowledgeGraph
from common.errors import ModuleError
from common.schema import CVResult, Explanation, LinkPrediction, LPResult, Triple
from common.services import get_service
from modules.pyclause import core
from modules.pyclause.config import PyClauseConfig
from modules.pyclause.engine import Engine, PyClauseEngine

EngineFactory = Callable[[KnowledgeGraph, "object", PyClauseConfig], Engine]


def _default_engine(kg: KnowledgeGraph, rules_path, cfg: PyClauseConfig) -> Engine:
    return PyClauseEngine(kg.train, list(kg.entity_labels), kg.relations, rules_path, cfg)


@dataclass
class _State:
    kg: KnowledgeGraph
    engine: Engine
    train: set[Triple]
    relations: set[str]
    labels: dict[str, str]                 # readable and unique per entity
    stats: dict[str, core.RuleStats]       # rule_key -> stats from the rule file (display only)


@dataclass(frozen=True)
class TripleEvidence:
    """Everything the rules say about one triple, before any verdict threshold is applied."""
    triple: Triple
    in_graph: bool
    rank: Optional[int]
    n_candidates: int
    explanation: Optional[core.TripleExplanation]
    ranked_tail: tuple[core.Answer, ...] = ()
    ranked_head: tuple[core.Answer, ...] = ()

    @property
    def score(self) -> float:
        return self.explanation.score if self.explanation else 0.0


class PyClauseModule:
    name = "PyClause"

    def __init__(self, config: Optional[PyClauseConfig] = None,
                 engine_factory: Optional[EngineFactory] = None,
                 parse_fn: Optional[Callable[[str, str], Triple]] = None):
        self.cfg = config or PyClauseConfig()
        self._engine_factory = engine_factory or _default_engine
        self._parse_fn = parse_fn            # (claim, dataset) -> triple; injected in tests
        self._st: Optional[_State] = None

    # ------------------------------------------------------------------ loading

    def load(self, kg: KnowledgeGraph) -> None:
        path = self.cfg.artifact_root / "pyclause" / kg.name / self.cfg.rules_file
        if not path.exists():
            raise ModuleError("INTERNAL", f"No rule file at {path}. Run "
                                          f"modules/pyclause/scripts/learn_rules.py --dataset {kg.name}.")
        stats = core.load_rule_stats(path)
        if not stats:
            raise ModuleError("INTERNAL", f"{path} contains no parseable rules.")
        print(f"[{self.name}] {kg.name}: {len(stats)} rules in {path.name}; building PyClause index...")
        self._st = _State(kg=kg, engine=self._engine_factory(kg, path, self.cfg), train=set(kg.train),
                          relations=set(kg.relations), labels=core.disambiguate_labels(kg.entity_labels),
                          stats=stats)

    # ------------------------------------------------------------------ helpers

    def _need(self) -> _State:
        if self._st is None:
            raise ModuleError("NOT_LOADED", "PyClause: call load() first.")
        return self._st

    def _check_entity(self, entity: str) -> str:
        if entity not in self._need().kg.entity_labels:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown entity: {entity!r}")
        return entity

    def _check_relation(self, relation: str) -> str:
        if relation not in self._need().relations:
            raise ModuleError("UNKNOWN_ENTITY", f"Unknown relation: {relation!r}")
        return relation

    def _explain(self, expl: core.TripleExplanation, intro: str, start: str,
                 answer: Optional[core.Answer] = None) -> Explanation:
        st = self._need()
        ex = core.rule_explanation(expl, st.stats, st.labels, intro, self.cfg.aggregation, start=start,
                                   max_steps=self.cfg.explain_facts_shown,
                                   rules_mentioned=self.cfg.rules_mentioned)
        if ex is not None:
            return ex
        if answer is not None:
            return core.answer_rules_explanation(answer, st.stats, intro, self.cfg.rules_mentioned)
        return core.no_rule_explanation(f"{intro} PyClause found no rule that predicts this triple.")

    @staticmethod
    def _triple_for(bound: str, relation: str, candidate: str, direction: core.Direction) -> Triple:
        return (bound, relation, candidate) if direction == "tail" else (candidate, relation, bound)

    # ------------------------------------------------------------------ link prediction

    def predict_link(self, head: Optional[str], relation: str, tail: Optional[str], k: int = 10) -> LPResult:
        st = self._need()
        self._check_relation(relation)
        direction: core.Direction = "tail" if tail is None else "head"
        bound = self._check_entity(head if direction == "tail" else tail)  # type: ignore[arg-type]

        answers = st.engine.answer(bound, relation, direction, k + 1)   # +1: the bound entity may appear
        if self.cfg.exclude_known:
            answers = [a for a in answers if a.candidate != bound]
        answers = answers[:k]
        triples = [self._triple_for(bound, relation, a.candidate, direction) for a in answers]
        explanations = st.engine.explain(triples)

        label = core.label_of
        predictions = []
        for answer, expl in zip(answers, explanations):
            name = label(answer.candidate, st.labels)
            intro = (f"Predicted {name} as the tail of ({label(bound, st.labels)}, {relation}, ?)."
                     if direction == "tail" else
                     f"Predicted {name} as the head of (?, {relation}, {label(bound, st.labels)}).")
            predictions.append(LinkPrediction(candidate=answer.candidate, score=answer.score,
                                              explanation=self._explain(expl, intro, bound, answer)))
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
        """The reasoning half of verify_claim, callable without the claim parser (tests, eval)."""
        return self.decide(claim, self.score_triple(triple))

    def score_triple(self, triple: Triple) -> TripleEvidence:
        st = self._need()
        head, relation, tail = triple
        self._check_relation(relation)
        self._check_entity(head)
        self._check_entity(tail)
        if triple in st.train:
            return TripleEvidence(triple=triple, in_graph=True, rank=1, n_candidates=0, explanation=None)

        topk = self.cfg.cv_topk
        ranked_tail = tuple(a for a in st.engine.answer(head, relation, "tail", topk) if a.candidate != head)
        ranked_head = tuple(a for a in st.engine.answer(tail, relation, "head", topk) if a.candidate != tail)
        ranks = [r for r in (core.rank_of(ranked_tail, tail), core.rank_of(ranked_head, head)) if r is not None]
        explanation = st.engine.explain([triple])[0]
        return TripleEvidence(triple=triple, in_graph=False, rank=min(ranks) if ranks else None,
                              n_candidates=max(len(ranked_tail), len(ranked_head)),
                              explanation=explanation, ranked_tail=ranked_tail, ranked_head=ranked_head)

    def decide(self, claim: str, ev: TripleEvidence) -> CVResult:
        st = self._need()
        triple = ev.triple
        head, relation, tail = triple
        labels, label = st.labels, core.label_of

        if ev.in_graph:
            return CVResult(claim, triple, "SUPPORTED", 1.0, core.direct_fact_explanation(triple, labels))

        th = self.cfg.thresholds
        verdict = core.decide_verdict(ev.rank, ev.n_candidates, th.support_rank, th.refute_rank,
                                      th.min_candidates_for_refute)
        claim_text = f"({label(head, labels)}, {relation}, {label(tail, labels)})"
        if ev.rank is None:
            prefix = (f"PyClause does not rank {claim_text} in its top {self.cfg.cv_topk}; it found "
                      f"{ev.n_candidates} other candidates for this query.")
            if ev.score > 0:
                prefix += f" (Some rules do predict it, score {ev.score:.3f}, but below the cut-off.)"
        else:
            prefix = (f"PyClause ranks {claim_text} #{ev.rank} of {ev.n_candidates} rule-derived "
                      f"candidates (filtered, best of head/tail query).")

        if verdict == "SUPPORTED":
            answer = (next((a for a in ev.ranked_tail if a.candidate == tail), None)
                      or next((a for a in ev.ranked_head if a.candidate == head), None))
            expl = ev.explanation or core.TripleExplanation(triple=triple, score=0.0)
            ex = self._explain(expl, f"{prefix} Claimed link: {claim_text}.", head, answer)
            score = ev.score or (answer.score if answer else 0.0)
            return CVResult(claim, triple, verdict, core.clamp01(score), ex)

        alternatives = ev.ranked_tail or ev.ranked_head
        if verdict == "REFUTED" and alternatives:
            best = alternatives[0]
            from_tail = bool(ev.ranked_tail)
            start = head if from_tail else tail
            alt_triple = self._triple_for(start, relation, best.candidate, "tail" if from_tail else "head")
            query_text = (f"({label(head, labels)}, {relation}, ?)" if from_tail
                          else f"(?, {relation}, {label(tail, labels)})")
            alt_expl = st.engine.explain([alt_triple])[0]
            ex = self._explain(alt_expl, f"{prefix} The rules instead derive "
                               f"{label(best.candidate, labels)} for {query_text}.", start, best)
            return CVResult(claim, triple, verdict, core.clamp01(1.0 - ev.score), ex)

        text = (f"{prefix} No PyClause rule grounds to a proof of this claim in the training graph"
                + (f"; the best rule-derived answers are {core.top_candidates_text(alternatives, labels)}."
                   if alternatives else "."))
        return CVResult(claim, triple, verdict, None, core.no_rule_explanation(text))
