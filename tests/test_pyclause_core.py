"""Tests for the PyClause module.

Part 1 runs anywhere: pure logic (core.py) and the full module (predict_link / verify_triple) driven
by a fake engine that mimics PyClause's output shapes on a toy graph.
Part 2 runs only inside the kgx-pyclause env (c_clause importable) and exercises the real engine.

Run: python -m pytest tests/test_pyclause_core.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets.base import KnowledgeGraph  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.serialization import cv_result_from_dict, lp_result_from_dict, to_dict  # noqa: E402
from modules.pyclause import core  # noqa: E402
from modules.pyclause.config import PyClauseConfig  # noqa: E402
from modules.pyclause.core import Answer, TripleExplanation  # noqa: E402
from modules.pyclause.module import PyClauseModule  # noqa: E402

# ----------------------------------------------------------------------------- toy fixtures

TRAIN = [
    ("alice", "born_in", "honolulu"),
    ("bob", "born_in", "honolulu"),
    ("honolulu", "located_in", "usa"),
    ("bob", "nationality", "usa"),
    ("dave", "born_in", "nairobi"),
    ("nairobi", "located_in", "kenya"),
]
ENTITIES = ["alice", "bob", "carol", "dave", "honolulu", "usa", "nairobi", "kenya"]
RELATIONS = ["born_in", "located_in", "nationality"]
CYCLIC = "nationality(X,Y) <= born_in(X,A), located_in(A,Y)"
ACYCLIC = "nationality(X,usa) <= born_in(X,honolulu)"
RULE_FILE = "\n".join([
    f"100\t80\t0.8\t{CYCLIC}",
    f"10\t3\t0.3\t{ACYCLIC}",
    "garbage line",
])


def toy_kg() -> KnowledgeGraph:
    return KnowledgeGraph(name="toy", train=list(TRAIN), valid=[], test=[],
                          entity_labels={e: e.capitalize() for e in ENTITIES},
                          relations=list(RELATIONS), neighbors=build_neighbors(TRAIN))


class FakeEngine:
    """Applies the cyclic rule only (conf 0.8) the way PyClause would, filtering train answers."""

    def __init__(self, kg: KnowledgeGraph):
        self.train = set(kg.train)
        self.calls: list = []

    def _derive(self, x: str) -> list[tuple[str, tuple]]:
        out = []
        for (h, r, a) in self.train:
            if h == x and r == "born_in":
                for (a2, r2, y) in self.train:
                    if a2 == a and r2 == "located_in":
                        out.append((y, ((h, r, a), (a2, r2, y))))
        return out

    def answer(self, entity, relation, direction, topk):
        self.calls.append(("answer", entity, relation, direction, topk))
        if relation != "nationality":
            return []
        if direction == "tail":
            cands = {y for y, _ in self._derive(entity) if (entity, relation, y) not in self.train}
        else:
            cands = {x for x in ENTITIES for y, _ in self._derive(x)
                     if y == entity and (x, relation, entity) not in self.train}
        return [Answer(c, 0.8, (CYCLIC,)) for c in sorted(cands)][:topk]

    def explain(self, triples):
        self.calls.append(("explain", tuple(triples)))
        out = []
        for h, r, t in triples:
            gs = tuple(g for y, g in self._derive(h) if y == t) if r == "nationality" else ()
            out.append(TripleExplanation((h, r, t), 0.8 if gs else 0.0,
                                         (CYCLIC,) if gs else (), (gs,) if gs else ()))
        return out


CLAIMS = {
    "Alice is American.": ("alice", "nationality", "usa"),
    "Bob is American.": ("bob", "nationality", "usa"),
    "Alice is Kenyan.": ("alice", "nationality", "kenya"),
    "Carol is Kenyan.": ("carol", "nationality", "kenya"),
}


def loaded(tmp_path: Path, **cfg) -> PyClauseModule:
    folder = tmp_path / "pyclause" / "toy"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rules.txt").write_text(RULE_FILE, encoding="utf-8")
    module = PyClauseModule(PyClauseConfig(artifact_root=tmp_path, **cfg),
                            engine_factory=lambda kg, path, c: FakeEngine(kg),
                            parse_fn=lambda claim, ds: CLAIMS[claim])
    module.load(toy_kg())
    return module


# ----------------------------------------------------------------------------- core

def test_rule_parsing_and_kinds(tmp_path):
    rule, stats = core.parse_rule_line(f"100\t80\t0.8\t{CYCLIC}")
    assert rule == CYCLIC and stats.support == 80 and stats.predictions == 100
    assert core.parse_rule_line("garbage") is None
    assert core.rule_kind(CYCLIC) == "B"
    assert core.rule_kind(ACYCLIC) == "U_c"
    assert core.rule_kind("nationality(X,usa) <= born_in(X,A)") == "U_d"
    assert core.rule_kind("nationality(X,usa) <=") == "Z"
    p = tmp_path / "r.txt"
    p.write_text(RULE_FILE)
    assert core.count_rule_kinds(p) == {"B": 1, "U_c": 1}
    assert core.rule_key(f"1\t1\t1.0\t{CYCLIC}") == core.rule_key(CYCLIC.replace(", ", ","))


def test_fb_style_relation_names_parse():
    rule = "/film/film/genre(X,Y) <= /film/film/prequel(X,A), /film/film/genre(A,Y)"
    (head_rel, args), body = core.atoms(rule)
    assert head_rel == "/film/film/genre" and args == ("X", "Y") and len(body) == 2
    assert core.rule_kind("/people/person/gender(X,/m/05zppz) <= /people/person/profession(X,/m/02hrh1q)") == "U_c"


def test_grounding_formats_as_path_in_any_order():
    labels = {"a": "A", "b": "B", "c": "C"}
    facts = (("b", "r2", "c"), ("a", "r1", "b"))
    assert core.format_grounding(facts, labels, start="a") == "A --[r1]--> B --[r2]--> C"
    assert core.format_grounding((("b", "r1", "a"),), labels, start="a") == "A --[r1, inverse]--> B"
    # facts that do not chain from start fall back to a fact list
    assert core.format_grounding((("x", "r", "y"), ("y", "r", "z")), labels, start="a") == \
        "x --[r]--> y; y --[r]--> z"


def test_verdicts():
    assert core.decide_verdict(1, 5, 50, 50, 1) == "SUPPORTED"
    assert core.decide_verdict(60, 80, 50, 50, 1) == "REFUTED"
    assert core.decide_verdict(60, 80, 50, 100, 1) == "NOT_ENOUGH_INFO"
    assert core.decide_verdict(None, 3, 50, 50, 1) == "REFUTED"
    assert core.decide_verdict(None, 0, 50, 50, 1) == "NOT_ENOUGH_INFO"
    assert core.decide_verdict(None, 0, 50, 50, 0) == "NOT_ENOUGH_INFO"   # nothing derived: never refute


def test_rule_explanation_includes_rule_grounding_and_stats():
    stats = {core.rule_key(CYCLIC): core.RuleStats(100, 80, 0.8)}
    g = (("alice", "born_in", "honolulu"), ("honolulu", "located_in", "usa"))
    ex = core.rule_explanation(TripleExplanation(("alice", "nationality", "usa"), 0.8, (CYCLIC,), ((g,),)),
                               stats, {}, "Intro.", "maxplus", start="alice")
    assert ex.kind == "rule" and ex.rule == CYCLIC and ex.confidence == 0.8
    assert "alice --[born_in]--> honolulu --[located_in]--> usa" in ex.text
    assert "80 of 100" in ex.text
    assert [s.triple for s in ex.steps] == list(g)
    assert "grounds born_in(X,A)" in ex.steps[0].note
    assert core.rule_explanation(TripleExplanation(("a", "r", "b"), 0.0), {}, {}, "x", "maxplus") is None


def test_acyclic_rule_explanation_mentions_constant():
    g = (("alice", "born_in", "honolulu"),)
    ex = core.rule_explanation(TripleExplanation(("alice", "nationality", "usa"), 0.3, (ACYCLIC,), ((g,),)),
                               {}, {"usa": "USA"}, "Intro.", "maxplus", start="alice")
    assert "names USA in its own head" in ex.text


# ----------------------------------------------------------------------------- module (fake engine)

def test_load_requires_rule_file(tmp_path):
    module = PyClauseModule(PyClauseConfig(artifact_root=tmp_path), engine_factory=lambda *a: None)
    with pytest.raises(ModuleError) as e:
        module.load(toy_kg())
    assert "learn_rules.py" in e.value.message
    with pytest.raises(ModuleError) as e:
        PyClauseModule().predict_link("alice", "nationality", None)
    assert e.value.code == "NOT_LOADED"


def test_predict_tail_with_grounded_explanation(tmp_path):
    m = loaded(tmp_path)
    res = m.predict_link("alice", "nationality", None, k=5)
    assert res.query == ("alice", "nationality", None)
    assert [p.candidate for p in res.predictions] == ["usa"]
    ex = res.predictions[0].explanation
    assert ex.rule == CYCLIC and ex.steps and ex.text
    assert "Alice --[born_in]--> Honolulu --[located_in]--> Usa" in ex.text
    # JSON round trip as served over HTTP
    assert lp_result_from_dict(to_dict(res)) == res


def test_predict_head_and_unknowns(tmp_path):
    m = loaded(tmp_path)
    res = m.predict_link(None, "nationality", "kenya", k=5)
    assert [p.candidate for p in res.predictions] == ["dave"]
    assert res.predictions[0].explanation.steps[0].triple == ("dave", "born_in", "nairobi")
    with pytest.raises(ModuleError) as e:
        m.predict_link("nobody", "nationality", None)
    assert e.value.code == "UNKNOWN_ENTITY"
    with pytest.raises(ModuleError):
        m.predict_link("alice", "no_such_relation", None)


def test_cv_direct_fact(tmp_path):
    res = loaded(tmp_path).verify_claim("Bob is American.")
    assert res.verdict == "SUPPORTED" and res.confidence == 1.0
    assert res.explanation.steps[0].triple == ("bob", "nationality", "usa")


def test_cv_supported_by_rule(tmp_path):
    res = loaded(tmp_path).verify_claim("Alice is American.")
    assert res.verdict == "SUPPORTED" and res.triple == ("alice", "nationality", "usa")
    assert abs(res.confidence - 0.8) < 1e-9
    assert res.explanation.rule == CYCLIC and "#1" in res.explanation.text
    assert cv_result_from_dict(to_dict(res)) == res


def test_cv_refuted_shows_alternative(tmp_path):
    res = loaded(tmp_path).verify_claim("Alice is Kenyan.")
    assert res.verdict == "REFUTED"
    assert "instead derive Usa" in res.explanation.text
    assert res.explanation.steps and res.explanation.rule == CYCLIC


def test_cv_not_enough_info(tmp_path):
    res = loaded(tmp_path).verify_claim("Carol is Kenyan.")
    # Carol has no facts: no tail candidates; but dave is derived for (?, nationality, kenya) -> REFUTED
    assert res.verdict == "REFUTED"
    m = loaded(tmp_path)
    m.cfg.thresholds.min_candidates_for_refute = 5
    res = m.verify_claim("Carol is Kenyan.")
    assert res.verdict == "NOT_ENOUGH_INFO" and res.explanation.text and res.confidence is None


# ----------------------------------------------------------------------------- real PyClause (optional)

def test_real_engine_end_to_end(tmp_path):
    pytest.importorskip("c_clause", reason="PyClause not installed (run inside kgx-pyclause)")
    folder = tmp_path / "pyclause" / "toy"
    folder.mkdir(parents=True)
    (folder / "rules.txt").write_text(RULE_FILE + "\n", encoding="utf-8")
    m = PyClauseModule(PyClauseConfig(artifact_root=tmp_path, num_threads=1),
                       parse_fn=lambda claim, ds: CLAIMS[claim])
    m.load(toy_kg())
    res = m.predict_link("alice", "nationality", None, k=5)
    assert res.predictions and res.predictions[0].candidate == "usa"
    ex = res.predictions[0].explanation
    assert ex.rule and ex.steps and all(s.triple for s in ex.steps if s.triple is not None)
    cv = m.verify_claim("Alice is American.")
    assert cv.verdict == "SUPPORTED" and cv.explanation.text
    assert m.verify_claim("Bob is American.").confidence == 1.0
