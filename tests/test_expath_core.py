"""Pure-logic tests for modules/expath on a toy graph with known answers (numpy + scipy only).

Run: python -m pytest tests/test_expath_core.py   (or: python tests/test_expath_core.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets.base import KnowledgeGraph  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from modules.expath.claims import ClaimQuery, Lexicon, relation_words, stem  # noqa: E402
from modules.expath.config import ExpathConfig  # noqa: E402
from modules.expath.factcheck import decide, gather_evidence  # noqa: E402
from modules.expath.graph import IndexedGraph, enumerate_paths, ground_body  # noqa: E402
from modules.expath.module import ExpathModule  # noqa: E402
from modules.expath.proof import check_explanation  # noqa: E402
from modules.expath.reasoner import RuleBase, explain_triple, noisy_or  # noqa: E402
from modules.expath.rules import Rule, cp_metrics, mine_cp_rules, mine_uc_rules, rule_text, uc_metrics  # noqa: E402


def toy_kg() -> KnowledgeGraph:
    """12 people born in 4 cities of 2 countries. nationality = country of birth city, except p11
    whose nationality is held out (the thing to predict). Everyone born in Paris speaks French."""
    cities = {"paris": "france", "lyon": "france", "nyc": "usa", "boston": "usa"}
    born = {f"p{i}": c for i, c in enumerate(["paris", "lyon", "nyc", "boston"] * 3)}
    train = [(c, "located_in", k) for c, k in cities.items()]
    for p, c in born.items():
        train.append((p, "born_in", c))
        if p != "p11":
            train.append((p, "nationality", cities[c]))
        if c == "paris":
            train.append((p, "speaks", "french"))
    test = [("p11", "nationality", cities[born["p11"]])]
    ents = sorted({x for h, _, t in train + test for x in (h, t)})
    return KnowledgeGraph(name="toy", train=train, valid=[], test=test,
                          entity_labels={e: e.capitalize() for e in ents},
                          relations=sorted({r for _, r, _ in train}), neighbors=build_neighbors(train))


def setup():
    kg = toy_kg()
    g = IndexedGraph.from_kg(kg)
    cfg = ExpathConfig()
    cfg.mining.samples_per_relation = 50
    cfg.mining.uc_min_head = 2
    cfg.mining.uc_min_supp = 2
    rules = mine_cp_rules(g, cfg.mining, cfg.quality, progress=lambda _: None) + \
        mine_uc_rules(g, cfg.mining, cfg.quality, progress=lambda _: None)
    return kg, g, cfg, rules


def test_graph_index_and_inverse():
    kg, g, _, _ = setup()
    p0, paris = g.ent2idx["p0"], g.ent2idx["paris"]
    born = g.rel2idx["born_in"]
    assert g.has_edge(p0, born, paris) and g.has_edge(paris, born + g.R, p0)
    assert not g.has_edge(paris, born, p0)
    assert g.fact((paris, born + g.R, p0)) == ("p0", "born_in", "paris")
    assert g.rel_name(born + g.R) == "born_in^-1"


def test_paths_exclude_explained_fact():
    _, g, _, _ = setup()
    p0, fr = g.ent2idx["p0"], g.ent2idx["france"]
    nat = g.rel2idx["nationality"]
    paths = enumerate_paths(g, p0, fr, 3, exclude=(p0, nat, fr))
    rel_paths = {tuple(g.rel_name(r) for _, r, _ in p) for p in paths}
    assert ("born_in", "located_in") in rel_paths
    assert ("nationality",) not in rel_paths            # the fact never explains itself
    assert all(p[0][0] == p0 and p[-1][2] == fr for p in paths)


def test_cp_metrics_match_hand_count():
    _, g, _, _ = setup()
    nat, born, loc = g.rel2idx["nationality"], g.rel2idx["born_in"], g.rel2idx["located_in"]
    supp, n_body, n_head = cp_metrics(g, nat, (born, loc), row_chunk=3)    # tiny chunks exercise blocking
    assert (supp, n_body, n_head) == (11, 12, 11)        # 12 people have the body, 11 nationalities stored


def test_uc_metrics_match_hand_count():
    _, g, _, _ = setup()
    speaks, born = g.rel2idx["speaks"], g.rel2idx["born_in"]
    supp, n_body, n_head = uc_metrics(g, speaks, g.ent2idx["french"], born, g.ent2idx["paris"])
    assert (supp, n_body, n_head) == (3, 3, 3)


def test_mined_rules_include_expected_and_inverse():
    _, g, _, rules = setup()
    texts = {rule_text(g, r) for r in rules}
    assert "nationality(X, Y) <- born_in(X, A), located_in(A, Y)" in texts
    assert "nationality(Y, X) <- located_in(A, X), born_in(Y, A)" in texts     # inverse, for head queries
    assert any(r.kind == "Uc" and g.rel_name(r.head) == "speaks" for r in rules)


def test_ground_body():
    _, g, _, _ = setup()
    born, loc = g.rel2idx["born_in"], g.rel2idx["located_in"]
    p = ground_body(g, g.ent2idx["p11"], (born, loc), g.ent2idx["usa"])
    assert p is not None and [g.fact(h) for h in p] == [("p11", "born_in", "boston"), ("boston", "located_in", "usa")]


def test_noisy_or():
    assert abs(noisy_or([0.5, 0.5]) - 0.75) < 1e-9 and noisy_or([]) == 0.0


def test_explain_triple_finds_rule_and_critical_fact():
    _, g, cfg, rules = setup()
    rb = RuleBase(rules)
    h, t, nat = g.ent2idx["p11"], g.ent2idx["usa"], g.rel2idx["nationality"]
    ev = explain_triple(g, rb, h, nat, t, cfg.reasoning, cfg.quality)
    assert ev.firings and ev.confidence > 0.3
    best = ev.firings[0]
    assert rule_text(g, best.rule) == "nationality(X, Y) <- born_in(X, A), located_in(A, Y)"
    assert g.fact(ev.critical_facts[0][0]) in {("p11", "born_in", "boston"), ("boston", "located_in", "usa")}


def module_with(kg, rules, g):
    m = ExpathModule(ExpathConfig(use_kge=False))
    m.load(kg)                                           # no artifacts on disk for "toy"
    m._st.rb = RuleBase([r for r in rules])
    return m


def test_predict_link_tail_and_head_with_checkable_proof():
    kg, g, _, rules = setup()
    m = module_with(kg, rules, g)
    res = m.predict_link("p11", "nationality", None, k=3)
    assert res.predictions[0].candidate == "usa"
    ex = res.predictions[0].explanation
    assert ex.steps and ex.rule and "born_in" in ex.rule
    report = check_explanation(m._st.g, ex)
    assert report["all_facts_ok"] and report["rule_ok"], report

    res = m.predict_link(None, "born_in", "lyon", k=20)   # head query through inverse rules
    assert all(p.explanation.steps for p in res.predictions)


def test_known_answers_hidden():
    kg, g, _, rules = setup()
    m = module_with(kg, rules, g)
    res = m.predict_link("p0", "nationality", None, k=5)
    assert "france" not in [p.candidate for p in res.predictions]


def test_show_known_lists_stored_answers_first():
    kg, g, _, rules = setup()
    m = module_with(kg, rules, g)
    m.cfg.show_known = True
    res = m.predict_link("p0", "nationality", None, k=2)
    first = res.predictions[0]
    assert first.candidate == "france" and "Already in the knowledge graph" in first.explanation.text
    assert tuple(first.explanation.steps[0].triple) == ("p0", "nationality", "france")


def test_no_rules_no_model_gives_clear_message():
    import pytest
    from common.errors import ModuleError
    kg, g, _, _ = setup()
    m = module_with(kg, [], g)                           # e.g. factkg: nothing mined, no KGE
    with pytest.raises(ModuleError, match="not set up for 'toy'"):
        m.predict_link("p0", "nationality", None, k=2)
    m.cfg.show_known = True                              # the UI still gets the stored answers
    res = m.predict_link("p0", "nationality", None, k=2)
    assert [p.candidate for p in res.predictions] == ["france"]
    assert "Only stored answers are shown" in res.predictions[0].explanation.text


def test_claim_relevance_and_lexicon():
    assert relation_words("birthPlace") == ["birth", "place"]
    lex = Lexicon.build([("He was born in Paris", ["He", "Paris"], {"birthPlace"})] * 3, min_count=3)
    q = ClaimQuery("X was born in Y", ["X", "Y"], False, {stem("born")}, "given entity set")
    score, why = q.relevance("birthPlace", lex)
    assert score == 1.0 and "born" in why


def test_linker_skips_lowercase_single_words():
    from modules.expath.claims import EntityLinker
    linker = EntityLinker({"Born": "Born", "Barack_Obama": "Barack Obama", "Honolulu": "Honolulu",
                           "Born_in_Chicago": "Born in Chicago", "Chicago": "Chicago"}, {})
    assert linker.link("Barack Obama was born in Honolulu.") == ["Barack_Obama", "Honolulu"]
    assert linker.link("Barack Obama was born in Chicago.") == ["Barack_Obama", "Chicago"]


def test_multi_entity_support_refute_and_negation():
    _, g, cfg, rules = setup()
    rb = RuleBase(rules)
    ids = lambda *names: [g.ent2idx[n] for n in names]          # noqa: E731

    def run(text, ents, negated=False):
        q = ClaimQuery(text, ents, negated, {stem(w) for w in ("born", "city")}, "given entity set")
        rel = lambda r: q.relevance(g.rel_base(r), None)        # noqa: E731
        from modules.expath.module import functional_ratio
        ev = gather_evidence(g, rb, ids(*ents), rel, cfg.verdict, cfg.reasoning, cfg.quality,
                             single=len(ents) == 1, functional=functional_ratio(g))
        return decide(g, text, q, ev, [])

    ok = run("P0 was born in Paris", ["p0", "paris"])
    assert ok.verdict == "SUPPORTED" and ok.explanation.steps[0].triple == ("p0", "born_in", "paris")
    bad = run("P0 was born in Boston", ["p0", "boston"])
    assert bad.verdict == "REFUTED" and ("p0", "born_in", "paris") in [s.triple for s in bad.explanation.steps]
    neg = run("P0 was not born in Boston", ["p0", "boston"], negated=True)
    assert neg.verdict == "SUPPORTED"
    assert all(r.explanation.steps for r in (ok, bad, neg))
    # existence claim, entity described but without a claim-relevant relation -> REFUTED (closed world);
    # its negation is SUPPORTED. The proof lists the entity's facts.
    absent = run("French has a birth city", ["french"])          # french only has `speaks` facts
    assert absent.verdict == "REFUTED"
    assert {s.triple[1] for s in absent.explanation.steps} == {"speaks"}
    assert run("French has no birth city", ["french"], negated=True).verdict == "SUPPORTED"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
