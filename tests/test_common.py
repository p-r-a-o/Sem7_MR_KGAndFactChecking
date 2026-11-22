"""Tests for stdlib-only shared code. Run: python -m pytest tests/ (NOT YET RUN)."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "claim_parser"))

from claim_parsing import EntityIndex, ParseError, parse_claim  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from common.datasets.wn18rr import clean_label  # noqa: E402
from common.metrics import classification_metrics, rank_metrics  # noqa: E402
from common.schema import CVResult, Explanation, ExplanationStep  # noqa: E402
from common.serialization import cv_result_from_dict, to_dict  # noqa: E402
from common.services import parse_simple_yaml  # noqa: E402
from common.verbalize import verbalize  # noqa: E402
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
    idx = EntityIndex(labels)
    rels = ["/people/person/place_of_birth", "/location/country/capital"]
    out = parse_claim("Ada Lovelace was born in London.", idx, rels)
    assert out["triple"] == ["e1", "/people/person/place_of_birth", "e2"]
    out = parse_claim("Paris is the capital of France.", idx, rels)
    assert out["triple"] == ["e3", "/location/country/capital", "e4"]  # swapped to (country, capital)


def test_claim_parser_errors():
    idx = EntityIndex({"e1": "Ada Lovelace"})
    try:
        parse_claim("Ada Lovelace did something.", idx, ["/a/b/c"])
        assert False
    except ParseError:
        pass
