"""Tests for the AnyBURL+SAFRAN module: rule parsing, grounding, aggregation, verdicts, explanations.

The module is stdlib-only, so the whole thing (including predict_link / verify_triple) runs here on a
toy knowledge graph. Run: python -m pytest tests/test_anyburl_core.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets.base import KnowledgeGraph  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from modules.anyburl_safran import core, rules  # noqa: E402
from modules.anyburl_safran.config import AnyburlConfig  # noqa: E402
from modules.anyburl_safran.core import ApplyLimits, Firing  # noqa: E402
from modules.anyburl_safran.module import AnyburlSafranModule  # noqa: E402
from modules.anyburl_safran.rules import Atom, Budget, TripleIndex  # noqa: E402

# ----------------------------------------------------------------------------- toy fixtures

TRAIN = [
    ("alice", "born_in", "honolulu"),
    ("bob", "born_in", "honolulu"),
    ("honolulu", "located_in", "hawaii"),
    ("hawaii", "located_in", "usa"),
    ("bob", "nationality", "usa"),
    ("dave", "lives_in", "nairobi"),
]
ENTITIES = ["alice", "bob", "carol", "dave", "honolulu", "hawaii", "usa", "nairobi", "kenya"]
RELATIONS = ["born_in", "lives_in", "located_in", "nationality"]

RULE_FILE = "\n".join([
    "100\t90\t0.9\tnationality(X,Y) <= born_in(X,A), located_in(A,B), located_in(B,Y)",
    "20\t10\t0.5\tnationality(X,Y) <= born_in(X,A), located_in(A,Y)",
    "10\t3\t0.3\tnationality(X,usa) <= born_in(X,honolulu)",
    "5\t4\t0.8\tborn_in(X,Y) <= lives_in(X,Y)",
    "7\t1\t0.1\tnationality(X,Y) <= lives_in(X,Y)",          # dropped: min_correct_predictions
    "garbage line without tabs",                               # dropped: unparseable
])

INDEX = TripleIndex.from_triples(TRAIN)


def toy_kg() -> KnowledgeGraph:
    return KnowledgeGraph(name="toy", train=list(TRAIN), valid=[], test=[],
                          entity_labels={e: e.capitalize() for e in ENTITIES},
                          relations=list(RELATIONS), neighbors=build_neighbors(TRAIN))


def loaded_module(tmp_path: Path, **config) -> AnyburlSafranModule:
    folder = tmp_path / "anyburl_safran" / "toy"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rules.txt").write_text(RULE_FILE, encoding="utf-8")
    settings = AnyburlConfig(artifact_root=tmp_path, **config)
    settings.thresholds.functional_min_observations = 2   # the toy graph has only a few triples
    settings.filters.unseen_negative_examples = 0         # keep the toy confidences readable
    module = AnyburlSafranModule(settings, parse_fn=lambda claim, dataset: CLAIMS[claim])
    module.load(toy_kg())
    return module


CLAIMS = {
    "Bob is a citizen of the USA.": ("bob", "nationality", "usa"),
    "Alice is a citizen of the USA.": ("alice", "nationality", "usa"),
    "Alice was born in Nairobi.": ("alice", "born_in", "nairobi"),
    "Carol is a citizen of Kenya.": ("carol", "nationality", "kenya"),
}


# ----------------------------------------------------------------------------- parsing

def test_confidence_is_smoothed_like_anyburls_apply():
    line = "2\t2\t1.0\tnationality(X,Y) <= born_in(X,Y)"
    raw = rules.parse_rule(line)
    assert raw.confidence == 1.0 and raw.file_confidence == 1.0
    smoothed = rules.parse_rule(line, unseen_negative_examples=5)
    assert abs(smoothed.confidence - 2 / 7) < 1e-12 and smoothed.file_confidence == 1.0
    assert "smoothed from 1.000" in smoothed.stats_text
    # a well-supported rule must now outrank a "perfect" rule built on two predictions
    solid = rules.parse_rule("29\t14\t0.4827\tnationality(X,Y) <= lives_in(X,Y)", 5)
    assert solid.confidence > smoothed.confidence


def test_load_rules_smooths_by_default(tmp_path):
    path = tmp_path / "rules.txt"
    path.write_text("2\t2\t1.0\tr(X,Y) <= p(X,Y)\n", encoding="utf-8")
    assert abs(rules.load_rules(path).rules[0].confidence - 2 / 7) < 1e-12


def test_parse_cyclic_rule():
    rule = rules.parse_rule("100\t90\t0.9\tnationality(X,Y) <= born_in(X,A), located_in(A,Y)")
    assert rule is not None
    assert rule.head == Atom("nationality", "X", "Y")
    assert rule.body == (Atom("born_in", "X", "A"), Atom("located_in", "A", "Y"))
    assert (rule.predicted, rule.correctly_predicted, rule.confidence) == (100, 90, 0.9)
    assert rule.kind == "cyclic" and rule.length == 2
    assert rule.text == "nationality(X,Y) <= born_in(X,A), located_in(A,Y)"


def test_parse_acyclic_and_zero_rules():
    tail_const = rules.parse_rule("10\t3\t0.3\t/people/person/languages(X,/m/02h40lc) <= /film/film/language(A,X)")
    assert tail_const is not None and tail_const.kind == "tail_constant"
    assert tail_const.head.object == "/m/02h40lc"
    head_const = rules.parse_rule("10\t3\t0.3\tnationality(usa,Y) <= born_in(Y,honolulu)")
    assert head_const is not None and head_const.kind == "head_constant"
    zero = rules.parse_rule("10\t3\t0.3\tnationality(X,usa) <= ")
    assert zero is not None and zero.body == () and zero.kind == "zero"


def test_parse_rejects_malformed_lines():
    assert rules.parse_rule("") is None
    assert rules.parse_rule("no tabs here") is None
    assert rules.parse_rule("1\t2\t0.5\tmissing separator(X,Y)") is None
    assert rules.parse_rule("a\tb\tc\tr(X,Y) <= p(X,Y)") is None          # non-numeric counts
    assert rules.parse_rule("1\t2\t0.5\tr(X,Y), q(X,Y) <= p(X,Y)") is None  # two head atoms


def test_is_variable_separates_variables_from_entity_ids():
    assert rules.is_variable("X") and rules.is_variable("A") and rules.is_variable("A1")
    assert not rules.is_variable("/m/027rn") and not rules.is_variable("08966089")
    assert not rules.is_variable("usa") and not rules.is_variable("XY")


def test_load_rules_applies_filters(tmp_path):
    path = tmp_path / "rules.txt"
    path.write_text(RULE_FILE, encoding="utf-8")
    ruleset = rules.load_rules(path, min_confidence=0.0001, min_correct_predictions=2,
                               unseen_negative_examples=0)
    assert len(ruleset.rules) == 4 and ruleset.skipped_lines == 1 and ruleset.dropped_by_filter == 1
    assert [r.confidence for r in ruleset.for_relation("nationality")] == [0.9, 0.5, 0.3]  # sorted
    assert ruleset.counts_by_kind() == {"cyclic": 3, "tail_constant": 1}
    short = rules.load_rules(path, max_body_length=1, unseen_negative_examples=0)
    assert all(r.length <= 1 for r in short.rules) and short.dropped_by_filter == 2


# ----------------------------------------------------------------------------- grounding

def test_ground_chain_returns_facts_in_body_order():
    body = (Atom("born_in", "X", "A"), Atom("located_in", "A", "B"), Atom("located_in", "B", "Y"))
    found = list(rules.ground(body, {"X": "alice"}, INDEX, Budget(1000)))
    assert len(found) == 1
    bindings, facts = found[0]
    assert bindings["Y"] == "usa" and bindings["A"] == "honolulu" and bindings["B"] == "hawaii"
    assert facts == (("alice", "born_in", "honolulu"), ("honolulu", "located_in", "hawaii"),
                     ("hawaii", "located_in", "usa"))


def test_ground_traverses_atoms_written_backwards():
    # born_in(A,X) is traversed from X against the edge direction
    body = (Atom("born_in", "A", "X"),)
    found = list(rules.ground(body, {"X": "honolulu"}, INDEX, Budget(1000)))
    assert sorted(b["A"] for b, _ in found) == ["alice", "bob"]


def test_ground_seeds_from_a_constant_when_no_variable_is_bound():
    # Predicting the head of (?, nationality, usa) with rule nationality(X,usa) <= born_in(X,honolulu)
    body = (Atom("born_in", "X", "honolulu"),)
    found = list(rules.ground(body, {}, INDEX, Budget(1000)))
    assert sorted(b["X"] for b, _ in found) == ["alice", "bob"]


def test_ground_without_any_seed_yields_nothing():
    assert list(rules.ground((Atom("born_in", "X", "A"),), {}, INDEX, Budget(1000))) == []


def test_ground_respects_budget_forbid_and_consistency():
    body = (Atom("born_in", "A", "X"),)
    assert list(rules.ground(body, {"X": "honolulu"}, INDEX, Budget(0))) == []
    forbidden = list(rules.ground(body, {"X": "honolulu"}, INDEX, Budget(1000),
                                  forbid=("alice", "born_in", "honolulu")))
    assert [b["A"] for b in (b for b, _ in forbidden)] == ["bob"]
    # a repeated variable must bind consistently
    repeated = (Atom("born_in", "X", "A"), Atom("located_in", "A", "A"))
    assert list(rules.ground(repeated, {"X": "alice"}, INDEX, Budget(1000))) == []


def test_ground_max_results_caps_output():
    body = (Atom("born_in", "A", "X"),)
    assert len(list(rules.ground(body, {"X": "honolulu"}, INDEX, Budget(1000), max_results=1))) == 1


# ----------------------------------------------------------------------------- firing rules

def test_query_binding_both_directions():
    cyclic = rules.parse_rule("1\t1\t0.5\tnationality(X,Y) <= born_in(X,Y)")
    assert core.query_binding(cyclic, "tail", "alice") == ({"X": "alice"}, "Y")
    assert core.query_binding(cyclic, "head", "usa") == ({"Y": "usa"}, "X")
    const = rules.parse_rule("1\t1\t0.5\tnationality(X,usa) <= born_in(X,honolulu)")
    assert core.query_binding(const, "tail", "alice") == ({"X": "alice"}, "usa")
    assert core.query_binding(const, "head", "usa") == ({}, "X")
    assert core.query_binding(const, "head", "kenya") is None   # head constant does not match


def test_fire_rule_tail_and_head_directions():
    limits = ApplyLimits()
    rule = rules.parse_rule("1\t1\t0.5\tnationality(X,Y) <= born_in(X,A), located_in(A,Y)")
    assert core.fire_rule(rule, "tail", "alice", INDEX, limits) == {
        "hawaii": (("alice", "born_in", "honolulu"), ("honolulu", "located_in", "hawaii"))}
    assert sorted(core.fire_rule(rule, "head", "hawaii", INDEX, limits)) == ["alice", "bob"]
    assert core.fire_rule(rule, "tail", "carol", INDEX, limits) == {}


def test_apply_rules_collects_candidates_from_several_rules(tmp_path):
    ruleset = rules.load_rules(_write_rules(tmp_path), min_correct_predictions=2,
                               unseen_negative_examples=0)
    fired = core.apply_rules(ruleset, INDEX, "nationality", "alice", "tail", ApplyLimits())
    assert sorted(fired) == ["hawaii", "usa"]
    assert sorted(f.rule.confidence for f in fired["usa"]) == [0.3, 0.9]
    assert [f.rule.confidence for f in fired["hawaii"]] == [0.5]


def test_apply_rules_stops_at_saturation(tmp_path):
    ruleset = rules.load_rules(_write_rules(tmp_path), min_correct_predictions=2,
                               unseen_negative_examples=0)
    limits = ApplyLimits(saturation_candidates=1)
    fired = core.apply_rules(ruleset, INDEX, "nationality", "alice", "tail", limits)
    assert list(fired) == ["usa"]            # the 0.9 rule fires first, then we stop


def _write_rules(tmp_path: Path) -> Path:
    path = tmp_path / "rules.txt"
    path.write_text(RULE_FILE, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- aggregation

def _firing(confidence: float, facts=(("a", "r", "b"),)) -> Firing:
    rule = rules.parse_rule(f"10\t5\t{confidence}\tr(X,Y) <= p(X,Y)")
    return Firing(rule, tuple(facts))


def test_max_aggregation_scores_and_breaks_ties_by_the_next_rule():
    one = core.aggregate("a", [_firing(0.8)], "max")
    two = core.aggregate("b", [_firing(0.8), _firing(0.4, (("c", "r", "d"),))], "max")
    assert one.score == 0.8 and two.score == 0.8
    assert two.sort_key > one.sort_key          # same best rule, but b has a second rule behind it


def test_noisy_or_aggregation_combines_rules():
    assert abs(core.noisy_or([0.5, 0.5]) - 0.75) < 1e-9
    agg = core.aggregate("a", [_firing(0.9), _firing(0.5, (("c", "r", "d"),))], "noisy-or")
    assert abs(agg.score - 0.95) < 1e-9


def test_dedupe_drops_rules_that_fired_on_identical_evidence():
    same = [_firing(0.9), _firing(0.5)]        # identical facts
    assert len(core.dedupe_firings(same)) == 1
    assert abs(core.aggregate("a", same, "noisy-or", dedupe=True).score - 0.9) < 1e-9
    assert abs(core.aggregate("a", same, "noisy-or", dedupe=False).score - 0.95) < 1e-9


def test_rank_candidates_orders_excludes_and_limits():
    fired = {"a": [_firing(0.2)], "b": [_firing(0.9)], "c": [_firing(0.5)]}
    ranked = core.rank_candidates(fired, "max")
    assert [c.candidate for c in ranked] == ["b", "c", "a"]
    assert core.rank_of(ranked, "a") == 3 and core.rank_of(ranked, "zz") is None
    assert [c.candidate for c in core.rank_candidates(fired, "max", exclude={"b"}, limit=1)] == ["c"]


# ----------------------------------------------------------------------------- relation statistics

def test_relation_stats_detects_functional_relations():
    stats = core.relation_stats(TRAIN, min_observations=2)
    assert "born_in" in stats.functional                  # one birthplace per person
    assert "born_in" not in stats.inverse_functional      # two people born in honolulu
    assert stats.conflicts(INDEX, "alice", "born_in", "nairobi") == [("alice", "born_in", "honolulu")]
    assert stats.conflicts(INDEX, "alice", "born_in", "honolulu") == []
    assert stats.conflicts(INDEX, "carol", "born_in", "nairobi") == []   # nothing known about carol


def test_relation_stats_rejects_many_to_many_relations():
    triples = [("a", "r", "x"), ("a", "r", "y"), ("b", "r", "z")]
    stats = core.relation_stats(triples, max_violation_ratio=0.05, min_observations=2)
    assert "r" not in stats.functional and "r" in stats.inverse_functional
    # too few observations to call it functional either way
    assert core.relation_stats(triples, min_observations=20).inverse_functional == frozenset()


# ----------------------------------------------------------------------------- verdicts

def test_decide_verdict_covers_every_branch():
    assert core.decide_verdict(1, 50, 10, 100, 10) == "SUPPORTED"
    assert core.decide_verdict(50, 200, 10, 100, 10) == "NOT_ENOUGH_INFO"
    assert core.decide_verdict(101, 200, 10, 100, 10) == "REFUTED"
    assert core.decide_verdict(None, 40, 10, 100, 10) == "REFUTED"
    assert core.decide_verdict(None, 3, 10, 100, 10) == "NOT_ENOUGH_INFO"


# ----------------------------------------------------------------------------- explanation text

LABELS = {e: e.capitalize() for e in ENTITIES}


def test_disambiguate_labels_only_touches_collisions():
    labels = {"01": "run", "02": "run", "03": "walk"}
    out = core.disambiguate_labels(labels)
    assert out == {"01": "run [01]", "02": "run [02]", "03": "walk"}
    assert core.disambiguate_labels({}) == {}


def test_module_shows_unique_labels_in_proofs(tmp_path):
    folder = tmp_path / "anyburl_safran" / "toy"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rules.txt").write_text(RULE_FILE, encoding="utf-8")
    kg = toy_kg()
    kg.entity_labels["hawaii"] = "Usa"            # force a collision with entity "usa"
    settings = AnyburlConfig(artifact_root=tmp_path)
    settings.filters.unseen_negative_examples = 0
    module = AnyburlSafranModule(settings)
    module.load(kg)
    names = [p.explanation.text.split(" as the tail")[0].replace("Predicted ", "")
             for p in module.predict_link("alice", "nationality", None, k=5).predictions]
    assert sorted(names) == ["Usa [hawaii]", "Usa [usa]"]   # both disambiguated, neither bare "Usa"


def test_format_grounding_renders_a_path_when_the_facts_chain():
    facts = [("alice", "born_in", "honolulu"), ("honolulu", "located_in", "hawaii")]
    assert core.format_grounding(facts, LABELS, start="alice") == (
        "Alice --[born_in]--> Honolulu --[located_in]--> Hawaii")
    backwards = [("alice", "born_in", "honolulu")]
    assert "inverse" in core.format_grounding(backwards, LABELS, start="honolulu")
    assert core.format_grounding([], LABELS) == "no body atoms"


def test_format_grounding_falls_back_to_a_fact_list():
    facts = [("alice", "born_in", "honolulu"), ("hawaii", "located_in", "usa")]  # not connected
    assert core.format_grounding(facts, LABELS, start="alice") == (
        "Alice --[born_in]--> Honolulu; Hawaii --[located_in]--> Usa")


def test_rule_explanation_reports_rule_grounding_and_runners_up():
    rule = rules.parse_rule("100\t90\t0.9\tnationality(X,Y) <= born_in(X,A), located_in(A,Y)")
    firing = Firing(rule, (("alice", "born_in", "honolulu"), ("honolulu", "located_in", "hawaii")))
    candidate = core.aggregate("hawaii", [firing, _firing(0.3, (("x", "r", "y"),))], "noisy-or")
    explanation = core.rule_explanation(candidate, LABELS, "Intro.", "noisy-or", start="alice")
    assert explanation.kind == "rule"
    assert explanation.rule == rule.text
    assert "Alice --[born_in]--> Honolulu --[located_in]--> Hawaii" in explanation.text
    assert "90 of 100 training predictions correct" in explanation.text
    assert "2 rules derive this answer" in explanation.text
    assert [s.triple for s in explanation.steps] == list(firing.facts)
    assert "grounds born_in(X,A)" in explanation.steps[0].note


def test_explanation_flags_rules_that_name_the_answer_in_their_head():
    rule = rules.parse_rule("29\t14\t0.48\tnationality(X,usa) <= born_in(X,honolulu)")
    firing = Firing(rule, (("alice", "born_in", "honolulu"),))
    candidate = core.aggregate("usa", [firing], "max")
    text = core.rule_explanation(candidate, LABELS, "Intro.", "max", start="alice").text
    assert "names Usa in its head" in text
    # a cyclic rule does lead to the answer, so the note must not appear
    cyclic = Firing(rules.parse_rule("10\t5\t0.5\tnationality(X,Y) <= born_in(X,Y)"),
                    (("alice", "born_in", "honolulu"),))
    assert "in its head" not in core.rule_explanation(
        core.aggregate("honolulu", [cyclic], "max"), LABELS, "Intro.", "max").text


def test_zero_rule_explanation_has_a_step_without_a_triple():
    zero = rules.parse_rule("10\t3\t0.3\tnationality(X,usa) <= ")
    steps = core.grounding_steps(Firing(zero, ()), LABELS)
    assert len(steps) == 1 and steps[0].triple is None and "no body atoms" in steps[0].note


def test_explanation_steps_are_capped():
    rule = rules.parse_rule("1\t1\t0.5\tr(X,Y) <= a(X,A), b(A,B), c(B,Y)")
    facts = (("1", "a", "2"), ("2", "b", "3"), ("3", "c", "4"))
    steps = core.grounding_steps(Firing(rule, facts), LABELS, max_steps=2)
    assert len(steps) == 3 and steps[-1].triple is None and "1 more facts" in steps[-1].note


# ----------------------------------------------------------------------------- the module

def test_load_reports_a_missing_rule_file(tmp_path):
    module = AnyburlSafranModule(AnyburlConfig(artifact_root=tmp_path))
    try:
        module.load(toy_kg())
    except Exception as e:                      # ModuleError
        assert "mine_rules.py" in str(e)
    else:
        raise AssertionError("expected a ModuleError")


def test_predict_link_tail_ranks_and_explains(tmp_path):
    result = loaded_module(tmp_path).predict_link("alice", "nationality", None, k=5)
    assert result.query == ("alice", "nationality", None)
    assert [p.candidate for p in result.predictions] == ["usa", "hawaii"]
    assert result.predictions[0].score == 0.9
    best = result.predictions[0].explanation
    assert best.rule == "nationality(X,Y) <= born_in(X,A), located_in(A,B), located_in(B,Y)"
    assert "Alice --[born_in]--> Honolulu --[located_in]--> Hawaii --[located_in]--> Usa" in best.text
    assert all(p.explanation.text for p in result.predictions)       # never an empty explanation


def test_predict_link_head_direction(tmp_path):
    result = loaded_module(tmp_path).predict_link(None, "nationality", "usa", k=5)
    # bob already has nationality usa in the graph, so only alice is a new answer
    assert [p.candidate for p in result.predictions] == ["alice"]
    assert "head of (?, nationality, Usa)" in result.predictions[0].explanation.text


def test_predict_link_hides_known_answers_unless_asked(tmp_path):
    # (bob, nationality, usa) is already in the graph, so it is not offered as a new answer
    hidden = loaded_module(tmp_path).predict_link("bob", "nationality", None, k=5)
    assert [p.candidate for p in hidden.predictions] == ["hawaii"]
    kept = loaded_module(tmp_path, exclude_known=False).predict_link("bob", "nationality", None, k=5)
    assert [p.candidate for p in kept.predictions] == ["usa", "hawaii"]


def test_noisy_or_aggregation_changes_scores(tmp_path):
    result = loaded_module(tmp_path, aggregation="noisy-or").predict_link("alice", "nationality", None, k=5)
    usa = next(p for p in result.predictions if p.candidate == "usa")
    assert abs(usa.score - (1 - 0.1 * 0.7)) < 1e-9                  # two rules combined
    assert "(noisy-or)" in usa.explanation.text


def test_verify_claim_supported_by_a_stated_fact(tmp_path):
    result = loaded_module(tmp_path).verify_claim("Bob is a citizen of the USA.")
    assert result.verdict == "SUPPORTED" and result.confidence == 1.0
    assert "stated directly" in result.explanation.text
    assert result.explanation.steps[0].triple == ("bob", "nationality", "usa")


def test_verify_claim_supported_by_a_derivation(tmp_path):
    result = loaded_module(tmp_path).verify_claim("Alice is a citizen of the USA.")
    assert result.verdict == "SUPPORTED" and result.triple == ("alice", "nationality", "usa")
    assert result.confidence == 0.9
    assert "#1 of 2 rule-derived candidates" in result.explanation.text
    assert result.explanation.rule.startswith("nationality(X,Y) <=")
    assert result.explanation.steps[0].triple == ("alice", "born_in", "honolulu")


def test_verify_claim_refuted_by_a_functional_conflict(tmp_path):
    result = loaded_module(tmp_path).verify_claim("Alice was born in Nairobi.")
    assert result.verdict == "REFUTED"
    assert result.explanation.steps[0].triple == ("alice", "born_in", "honolulu")
    assert "behaves as a function" in result.explanation.text
    assert result.confidence == 1.0


def test_verify_claim_without_any_grounding_is_not_enough_info(tmp_path):
    result = loaded_module(tmp_path).verify_claim("Carol is a citizen of Kenya.")
    assert result.verdict == "NOT_ENOUGH_INFO" and result.confidence is None
    assert "No rule derives" in result.explanation.text
    assert "3 rules were learned for nationality" in result.explanation.text


def test_verify_triple_forbids_the_claimed_fact_as_its_own_evidence(tmp_path):
    module = loaded_module(tmp_path)
    # born_in(X,Y) <= lives_in(X,Y) derives (dave, born_in, nairobi); the claim itself is not evidence
    result = module.verify_triple("Dave was born in Nairobi.", ("dave", "born_in", "nairobi"))
    assert result.verdict == "SUPPORTED"
    assert result.explanation.steps[0].triple == ("dave", "lives_in", "nairobi")


def test_unknown_entity_and_relation_are_rejected(tmp_path):
    module = loaded_module(tmp_path)
    for call in (lambda: module.predict_link("nobody", "nationality", None),
                 lambda: module.predict_link("alice", "no_such_relation", None),
                 lambda: module.verify_triple("x", ("alice", "nationality", "nobody"))):
        try:
            call()
        except Exception as e:
            assert getattr(e, "code", "") == "UNKNOWN_ENTITY", e
        else:
            raise AssertionError("expected UNKNOWN_ENTITY")
