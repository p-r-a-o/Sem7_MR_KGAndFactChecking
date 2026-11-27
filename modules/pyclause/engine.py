"""Thin wrapper over PyClause's C++ backend (`c_clause`) and option handling (`clause`).

This is the only file that imports PyClause. It turns PyClause's nested lists into the plain
dataclasses of core.py, so module.py can be tested with a fake engine and no compiled extension.

PyClause calls used (see https://pyclause.readthedocs.io):
  Loader.set_entity_index / set_relation_index / load_data / load_rules
  QAHandler.calculate_answers + get_answers / get_rules          -> link prediction candidates
  PredictionHandler.calculate_scores + get_scores / get_explanations -> scores + rule groundings
"""
from pathlib import Path
from typing import Protocol, Sequence

from common.schema import Triple
from modules.pyclause.config import PyClauseConfig
from modules.pyclause.core import Answer, Direction, TripleExplanation


class Engine(Protocol):
    def answer(self, entity: str, relation: str, direction: Direction, topk: int) -> list[Answer]:
        """Rank candidates for (entity, relation, ?) [tail] or (?, relation, entity) [head]."""

    def explain(self, triples: Sequence[Triple]) -> list[TripleExplanation]:
        """Score each triple and return the predicting rules with their groundings, in input order."""


def _key(t) -> Triple:
    return (str(t[0]), str(t[1]), str(t[2]))


class PyClauseEngine:
    """Holds one Loader (train graph + rules) and the two handlers, built once per dataset."""

    def __init__(self, train: Sequence[Triple], entities: Sequence[str], relations: Sequence[str],
                 rules_path: Path, cfg: PyClauseConfig):
        try:
            from c_clause import Loader, PredictionHandler, QAHandler
            from clause import Options
        except ImportError as e:  # pragma: no cover - depends on the env
            raise ImportError("PyClause is not installed in this environment. Run "
                              "`conda env create -f modules/pyclause/environment.yml` and "
                              "`conda activate kgx-pyclause`.") from e

        opts = Options()
        for key, value in cfg.loader_options.items():
            opts.set(f"loader.{key}", value)
        for key, value in cfg.qa_options.items():
            opts.set(f"qa_handler.{key}", value)
        for key, value in cfg.prediction_options.items():
            opts.set(f"prediction_handler.{key}", value)
        opts.set("qa_handler.filter_w_data", cfg.exclude_known)
        for handler in ("qa_handler", "prediction_handler"):
            opts.set(f"{handler}.aggregation_function", cfg.aggregation)
            opts.set(f"{handler}.num_threads", cfg.num_threads)

        self._loader = Loader(options=opts.get("loader"))
        # Index every KG entity/relation up front: entities seen only in valid/test still need an id,
        # otherwise PyClause rejects claims and queries that mention them.
        self._loader.set_entity_index(index=list(entities))
        self._loader.set_relation_index(index=list(relations))
        self._loader.load_data(data=[list(t) for t in train])
        self._loader.load_rules(rules=str(rules_path))

        self._qa_options = dict(opts.get("qa_handler"))
        self._qa = QAHandler(options=self._qa_options)
        self._qa_topk = int(self._qa_options.get("topk", 100))
        self._disc_at_least = int(self._qa_options.get("disc_at_least", -1))
        self._ph = PredictionHandler(options=opts.get("prediction_handler"))

    def _set_topk(self, topk: int) -> None:
        if topk != self._qa_topk:
            self._qa_options["topk"] = topk
            # disc_at_least must not exceed topk (PyClause option contract); restore it for larger k
            if self._disc_at_least > 0:
                self._qa_options["disc_at_least"] = min(self._disc_at_least, topk)
            # c_clause's set_options takes Mapping[str, str]; PyClause's Options already stores
            # values as strings ('True', '-1'), so stringify the ones we changed.
            self._qa.set_options(options={k: str(v) for k, v in self._qa_options.items()})
            self._qa_topk = topk

    def answer(self, entity: str, relation: str, direction: Direction, topk: int) -> list[Answer]:
        self._set_topk(max(1, topk))
        self._qa.calculate_answers(queries=[(entity, relation)], loader=self._loader, direction=direction)
        answers = self._qa.get_answers(as_string=True)[0]
        rules = self._qa.get_rules(as_string=True)[0] if self._qa_options.get("collect_rules") else []
        out = []
        for i, (candidate, score) in enumerate(answers):
            cand_rules = tuple(str(r) for r in rules[i]) if i < len(rules) else ()
            out.append(Answer(candidate=str(candidate), score=float(score), rules=cand_rules))
        return out

    def explain(self, triples: Sequence[Triple]) -> list[TripleExplanation]:
        if not triples:
            return []
        self._ph.calculate_scores(triples=[list(t) for t in triples], loader=self._loader)
        scores = {_key(row[:3]): float(row[3]) for row in self._ph.get_scores(as_string=True)}
        targets, pred_rules, groundings = self._ph.get_explanations(as_string=True)
        by_triple: dict[Triple, tuple[tuple[str, ...], tuple]] = {}
        for i, target in enumerate(targets):
            rules_i = tuple(str(r) for r in pred_rules[i])
            grounds_i = tuple(
                tuple(tuple(_key(fact) for fact in grounding) for grounding in rule_groundings)
                for rule_groundings in groundings[i])
            by_triple[_key(target)] = (rules_i, grounds_i)
        out = []
        for t in triples:
            rules_t, grounds_t = by_triple.get(tuple(t), ((), ()))
            out.append(TripleExplanation(triple=tuple(t), score=scores.get(tuple(t), 0.0),
                                         rules=rules_t, groundings=grounds_t))
        return out
