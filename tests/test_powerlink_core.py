"""Tests for torch-free PowerLink wrapper logic. Run: python -m pytest tests/ (NOT YET RUN)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.powerlink import core  # noqa: E402
from modules.powerlink.core import Hop, Vocab  # noqa: E402

NUM_REL = 2
VOCAB = Vocab(ent_ids=["A", "B", "C"], rel_names=["r0", "r1"], labels={"A": "Alpha", "B": "Beta", "C": "Gamma"})


def test_decode_picks_heaviest_parallel_edge():
    pairs = core.pair_index([0, 0], [1, 1], [0, 3], [0.2, 0.9])  # r0 (0.2) vs inverse of r1 (0.9)
    hops = core.decode_path([(0, 1)], pairs, node_ids=[10, 11])
    assert hops == [Hop(10, 11, 3, 0.9)]


def test_fact_orientation_for_inverse_edge():
    forward = Hop(0, 1, 0, 1.0)
    inverse = Hop(1, 2, 1 + NUM_REL, 1.0)  # traversed B -> C along inverse of r1, so fact is (C, r1, B)
    assert core.hop_fact(forward, NUM_REL) == (0, 0, 1)
    assert core.hop_fact(inverse, NUM_REL) == (2, 1, 1)


def test_reverse_hops_roundtrip():
    hops = [Hop(0, 1, 0, 0.5), Hop(1, 2, 1 + NUM_REL, 0.7)]
    rev = core.reverse_hops(hops, NUM_REL)
    assert rev[0] == Hop(2, 1, 1, 0.7) and rev[1] == Hop(1, 0, 0 + NUM_REL, 0.5)
    assert core.reverse_hops(rev, NUM_REL) == hops
    assert core.is_chain(rev)


def test_path_explanation_text_and_steps():
    hops = [Hop(0, 1, 0, 0.5), Hop(1, 2, 1 + NUM_REL, 0.7)]
    ex = core.path_explanation([hops], NUM_REL, VOCAB, "Intro.")
    assert ex.kind == "path" and len(ex.steps) == 2
    assert ex.steps[1].triple == ("C", "r1", "B")
    assert "Alpha --[r0]--> Beta --[r1 (inverse)]--> Gamma" in ex.text
    assert abs(ex.confidence - 0.6) < 1e-9
    assert core.path_explanation([[]], NUM_REL, VOCAB, "x") is None


def test_non_chain_is_described_as_edges():
    ex = core.path_explanation([[Hop(0, 1, 0, 1.0), Hop(2, 0, 1, 1.0)]], NUM_REL, VOCAB, "Intro.")
    assert "No connected path" in ex.text


def test_ranking_and_verdict():
    scores = [0.1, 0.9, 0.5, 0.7]
    assert core.top_k(scores, 2, exclude={1}) == [3, 2]
    assert core.filtered_rank(scores, 2, known=set()) == 3
    assert core.filtered_rank(scores, 2, known={1}) == 2
    assert core.decide_verdict(3, 10, 100) == "SUPPORTED"
    assert core.decide_verdict(50, 10, 100) == "NOT_ENOUGH_INFO"
    assert core.decide_verdict(101, 10, 100) == "REFUTED"
