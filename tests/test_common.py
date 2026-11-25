"""Tests for stdlib-only shared code. Run: python -m pytest tests/ (NOT YET RUN)."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "claim_parser"))

from claim_parsing import (ParseError, build_resources, cue_matches,  # noqa: E402
                            parse_claim, phrase_occurrences, tokenize_spans)
from common.datasets.files import build_neighbors  # noqa: E402
from common.datasets.wn18rr import clean_label, gloss  # noqa: E402
from common.metrics import classification_metrics, rank_metrics  # noqa: E402
from common.schema import CVResult, Explanation, ExplanationStep  # noqa: E402
from common.serialization import cv_result_from_dict, to_dict  # noqa: E402
from common.services import parse_simple_yaml  # noqa: E402
from common.verbalize import disambiguated_labels, verbalize  # noqa: E402
from scripts.generate_claims import build_pools, corrupt  # noqa: E402


def test_schema_roundtrip():
    r = CVResult("c", ("a", "r", "b"), "SUPPORTED", 0.9,
                 Explanation("path", [ExplanationStep(("a", "r", "b"), "n")], "t", confidence=0.5))
    assert cv_result_from_dict(to_dict(r)) == r


def test_simple_yaml():
    cfg = parse_simple_yaml("services:\n  a:\n    kind: module\n    port: 81\n    start: python x.py --p 1\n")
    assert cfg["services"]["a"] == {"kind": "module", "port": 81, "start": "python x.py --p 1"}


def test_neighbors_have_inverse():
    n = build_neighbors([("a", "r", "b")])
    assert n["a"] == [("r", "b")] and n["b"] == [("r^-1", "a")]


def test_verbalize_and_wn_label():
    assert verbalize("Ann", "/people/person/place_of_birth", "Rome", random.Random(0)).endswith(".")
    assert "relation" in verbalize("A", "/x/y/relation", "B")
    assert clean_label("about_face.n.01") == "about face"


def test_wn_label_drops_the_dictionary_definition():
    # entity2text.txt stores "<lemma>, <definition>"; keeping the definition makes proofs unreadable
    assert clean_label("stool, solid excretory product evacuated from the bowels") == "stool"
    assert clean_label("chieftainship, the position of chieftain") == "chieftainship"
    assert clean_label("rickettsiaceae, microorganism resembling bacteria") == "rickettsiaceae"
    assert clean_label("about_face.n.01, an abrupt reversal") == "about face"
    assert clean_label("plain lemma") == "plain lemma"      # nothing to strip
    assert clean_label(", only a definition") == ", only a definition"  # never return empty


def test_corrupt_is_unknown_and_type_consistent():
    train = [("a", "r", "x"), ("b", "r", "y"), ("c", "r", "z")]
    heads, tails = build_pools(train)
    out = corrupt(("a", "r", "x"), heads, tails, set(train), random.Random(1))
    assert out is not None and out[0] not in set(train)
    assert out[0][0] in heads["r"] and out[0][2] in tails["r"]


def test_metrics():
    m = rank_metrics([1, 2, None])
    assert m["hits@1"] == 1 / 3 and abs(m["mrr"] - (1 + 0.5) / 3) < 1e-9
    c = classification_metrics(["SUPPORTED", "REFUTED"], ["SUPPORTED", "NOT_ENOUGH_INFO"])
    assert c["accuracy"] == 0.5 and c["nei_rate"] == 0.5 and c["precision"] == 1.0


def test_claim_parser_basic_and_swap():
    labels = {"e1": "Ada Lovelace", "e2": "London", "e3": "France", "e4": "Paris"}
    rels = ["/people/person/place_of_birth", "/location/country/capital"]
    res = build_resources(labels, rels)
    out = parse_claim("Ada Lovelace was born in London.", res)
    assert out["triple"] == ["e1", "/people/person/place_of_birth", "e2"]
    out = parse_claim("Paris is the capital of France.", res)
    assert out["triple"] == ["e3", "/location/country/capital", "e4"]  # swapped to (country, capital)


def test_claim_parser_errors():
    res = build_resources({"e1": "Ada Lovelace"}, ["/a/b/c"])
    try:
        parse_claim("Ada Lovelace did something.", res)
        assert False
    except ParseError:
        pass


def test_claim_parser_ignores_scaffolding_words():
    """"kind" is a real WN18RR entity; greedy linking used to pick it over the real tail."""
    labels = {"s1": "junction", "s2": "joining", "s3": "kind"}
    res = build_resources(labels, ["_hypernym"], train=[("s1", "_hypernym", "s2")])
    out = parse_claim("Junction is a kind of joining.", res)
    assert out["triple"] == ["s1", "_hypernym", "s2"]


def test_claim_parser_uses_pools_to_disambiguate_a_lemma():
    """Same lemma, two IDs: the one occurring in that position of the relation in TRAIN wins."""
    labels = {"a1": "respond", "b1": "response", "b2": "response"}
    res = build_resources({**labels}, ["_derivationally_related_form"],
                          train=[("x", "_derivationally_related_form", "b2")])
    out = parse_claim("Respond is derivationally related to response.", res)
    assert out["triple"] == ["a1", "_derivationally_related_form", "b2"]


def test_wn18rr_gloss_extraction():
    assert gloss("stool, solid excretory product evacuated from the bowels") == \
        "solid excretory product evacuated from the"
    assert gloss('respire, breathe easily; "at last"') == "breathe easily"
    assert gloss("about_face.n.01") == ""          # no definition half


def test_disambiguated_labels_only_touches_collisions():
    labels = {"a": "run", "b": "run", "c": "walk"}
    descriptions = {"a": "move fast on foot", "b": "a campaign for office", "c": "go on foot"}
    out = disambiguated_labels(labels, descriptions)
    assert out == {"a": "run (move fast on foot)", "b": "run (a campaign for office)",
                   "c": "walk"}                     # "walk" is unique, so it stays bare


def test_disambiguated_labels_without_descriptions_is_identity():
    labels = {"a": "run", "b": "run"}
    assert disambiguated_labels(labels, {}) == labels


def test_parser_uses_a_gloss_to_pick_the_right_synset():
    """The text alone cannot say which "run"; the parenthetical gloss can."""
    labels = {"r1": "run", "r2": "run", "s": "sprint"}
    descriptions = {"r1": "move fast on foot", "r2": "a campaign for office"}
    aliases = disambiguated_labels(labels, descriptions)
    res = build_resources(labels, ["_hypernym"], train=[("r2", "_hypernym", "s")], aliases=aliases)
    out = parse_claim("Run (move fast on foot) is a kind of sprint.", res)
    assert out["triple"] == ["r1", "_hypernym", "s"]   # the gloss beats r2's training support


def test_cue_matches_covers_every_match_not_just_the_leftmost():
    """_hypernym's cue alternation matches "is a" before "kind of"; both must count."""
    claim = "Portraiture is a kind of portrayal."
    tokens, char_spans = tokenize_spans(claim)
    cues, any_cue = cue_matches(claim, char_spans, ["_hypernym"])
    assert tokens.index("kind") in cues["_hypernym"][0]
    assert tokens.index("kind") in any_cue


def test_phrase_occurrences():
    tokens = ["the", "film", "release", "region", "of", "x"]
    assert phrase_occurrences(tokens, ("film", "release", "region")) == [frozenset({1, 2, 3})]
    assert phrase_occurrences(tokens, ("genre",)) == []
