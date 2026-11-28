"""Tests for the evaluation toolkit: statistics, CV/LP metrics, baselines, ensembles, hard negatives,
the eval scripts' pure helpers and the AnyBURL ablation runner (on the toy graph).

Run: python -m pytest tests/test_eval_metrics.py
"""
import importlib.util
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from common import baselines, cv_metrics, ensemble, lp_metrics, stats  # noqa: E402
from common.cv_metrics import NEI, REFUTED, SUPPORTED, support_score  # noqa: E402
from common.datasets.base import LabeledClaim  # noqa: E402
from common.schema import CVResult, Explanation, ExplanationStep  # noqa: E402
from scripts.generate_claims import build_frequency_weights, build_pools, corrupt  # noqa: E402


def load_script(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- common/stats

def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    values = [1.0] * 30 + [0.0] * 70
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    point, low, high = stats.bootstrap_ci(values, mean, n_boot=500, seed=3)
    assert point == pytest.approx(0.3) and low < point < high and 0.15 < low and high < 0.45
    assert stats.bootstrap_ci(values, mean, n_boot=500, seed=3) == (point, low, high)
    assert stats.bootstrap_ci([0.5], mean, n_boot=500) == (0.5, 0.5, 0.5)
    assert stats.bootstrap_ci(values, mean, n_boot=0)[1:] == (point, point)


def test_paired_bootstrap_detects_a_consistent_gap_but_not_noise():
    rng = random.Random(0)
    base = [rng.random() for _ in range(200)]
    better = stats.paired_bootstrap([b + 0.2 for b in base], base, n_boot=300)
    assert better["mean_diff"] == pytest.approx(0.2) and better["low"] > 0 and better["p_value"] < 0.05
    noise = stats.paired_bootstrap(base, list(reversed(base)), n_boot=300)
    assert noise["p_value"] > 0.05
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [1.0, 2.0])


def test_mcnemar_exact_and_chi2():
    exact = stats.mcnemar([True] * 9 + [False] * 1, [False] * 9 + [True] * 1)
    assert (exact["a_only"], exact["b_only"], exact["method"]) == (9, 1, "exact")
    assert exact["p_value"] == pytest.approx(2 * (1 + 10) / 1024)
    assert stats.mcnemar([True, False], [True, False])["p_value"] == 1.0
    big = stats.mcnemar([True] * 60 + [False] * 10, [False] * 60 + [True] * 10)
    assert big["method"] == "chi2" and big["p_value"] < 1e-6


def test_cohens_kappa():
    assert stats.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert stats.cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert stats.cohens_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_mean_std():
    assert stats.mean_std([2.0, 4.0]) == pytest.approx((3.0, math.sqrt(2)))
    assert stats.mean_std([1.0]) == (1.0, 0.0)


# ----------------------------------------------------------------------------- common/cv_metrics

def test_support_score_maps_verdict_and_confidence_onto_one_axis():
    assert support_score(SUPPORTED, 1.0) == 1.0 and support_score(SUPPORTED, 0.0) == 0.5
    assert support_score(REFUTED, 1.0) == 0.0 and support_score(REFUTED, 0.8) == pytest.approx(0.1)
    assert support_score(NEI, 0.9) == 0.5 and support_score(SUPPORTED, None) == 0.5
    assert support_score(SUPPORTED, 7.0) == 1.0                           # clamped


def test_roc_auc_and_ties():
    assert cv_metrics.roc_auc([True, True, False, False], [0.9, 0.8, 0.3, 0.1]) == 1.0
    assert cv_metrics.roc_auc([True, True, False, False], [0.1, 0.3, 0.8, 0.9]) == 0.0
    assert cv_metrics.roc_auc([True, False, True, False], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)
    assert cv_metrics.roc_auc([True, False], [0.5, 0.5]) == 0.5
    assert math.isnan(cv_metrics.roc_auc([True, True], [0.1, 0.2]))


def test_average_precision():
    ap = cv_metrics.average_precision([True, False, True, False], [0.9, 0.8, 0.7, 0.1])
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    assert cv_metrics.average_precision([True, False], [0.9, 0.1]) == 1.0


def test_best_threshold_separates_a_clean_split():
    thr, bal = cv_metrics.best_threshold([True, True, False, False], [0.9, 0.7, 0.4, 0.1])
    assert 0.4 < thr <= 0.7 and bal == 1.0


def test_brier_and_ece():
    assert cv_metrics.brier([True, False], [1.0, 0.0]) == 0.0
    assert cv_metrics.brier([True], [0.5]) == pytest.approx(0.25)
    assert cv_metrics.expected_calibration_error([True, True, False, False], [1.0, 1.0, 0.0, 0.0]) == 0.0
    # always 0.9 but right half the time -> |0.5 - 0.9|
    assert cv_metrics.expected_calibration_error([True, False] * 5, [0.9] * 10) == pytest.approx(0.4)


def test_temperature_scaling_softens_overconfident_scores():
    labels = [True] * 60 + [False] * 40
    scores = [0.99] * 100
    t = cv_metrics.fit_temperature(labels, scores)
    assert t > 1.0
    assert cv_metrics.apply_temperature([0.5], t) == [0.5]
    scaled = cv_metrics.apply_temperature(scores, t)
    assert cv_metrics.expected_calibration_error(labels, scaled) < cv_metrics.expected_calibration_error(labels, scores)


def test_risk_coverage_perfect_and_wrong_confident():
    perfect = cv_metrics.risk_coverage([SUPPORTED, SUPPORTED, REFUTED, REFUTED], [0.9, 0.8, 0.2, 0.1])
    assert perfect["aurc"] == 0.0 and cv_metrics.coverage_at_risk(perfect["curve"], 0.0) == 1.0
    bad = cv_metrics.risk_coverage([REFUTED, SUPPORTED], [0.9, 0.6])      # most decisive answer is wrong
    assert bad["curve"] == [(0.5, 1.0), (1.0, 0.5)] and bad["aurc"] == pytest.approx(0.75)
    assert cv_metrics.risk_at_coverage(bad["curve"], 1.0) == 0.5
    # tied decisiveness interpolates instead of depending on input order
    a = cv_metrics.risk_coverage([SUPPORTED, REFUTED], [0.8, 0.8])["aurc"]
    b = cv_metrics.risk_coverage([REFUTED, SUPPORTED], [0.8, 0.8])["aurc"]
    assert a == b


def test_two_class_metrics_and_nei_policy():
    miss = cv_metrics.two_class_metrics([SUPPORTED, REFUTED], [SUPPORTED, NEI], "miss")
    assert miss["balanced_accuracy"] == 0.5 and miss["macro_f1"] == 0.5
    refuted = cv_metrics.two_class_metrics([SUPPORTED, REFUTED], [SUPPORTED, NEI], "refuted")
    assert refuted["balanced_accuracy"] == 1.0
    with pytest.raises(ValueError):
        cv_metrics.two_class_metrics([SUPPORTED], [SUPPORTED], "other")


def test_group_metrics_and_summary():
    rows = ([{"gold": SUPPORTED, "pred": SUPPORTED, "score": 0.9, "relation": "a"}] * 3
            + [{"gold": REFUTED, "pred": REFUTED, "score": 0.1, "relation": "a"}] * 3
            + [{"gold": REFUTED, "pred": SUPPORTED, "score": 0.7, "relation": "b"}])
    groups = cv_metrics.group_metrics(rows, lambda r: r["relation"], cv_metrics.quick_metrics)
    assert list(groups) == ["a", "b"] and groups["a"]["accuracy"] == 1.0 and groups["b"]["accuracy"] == 0.0
    assert cv_metrics.group_metrics(rows, lambda r: r["relation"], cv_metrics.quick_metrics, min_n=2).keys() == {"a"}
    summary = cv_metrics.summarize_cv(rows, n_boot=50)
    assert summary["accuracy"] == pytest.approx(6 / 7) and summary["auc"] > 0.9
    assert summary["coverage"] == 1.0 and "ci" in summary and summary["ci"]["accuracy"][1] <= 6 / 7 <= summary["ci"]["accuracy"][2]


# ----------------------------------------------------------------------------- common/lp_metrics

def test_filtered_rank_skips_other_known_answers_only():
    assert lp_metrics.filtered_rank(["a", "b", "c", "d"], "c", {"a"}) == 2
    assert lp_metrics.filtered_rank(["a", "b", "c"], "c", set()) == 3
    assert lp_metrics.filtered_rank(["a", "b"], "z", {"a"}) is None
    assert lp_metrics.filtered_rank(["a", "c"], "c", {"c"}) == 2          # the gold itself is never filtered


def test_answer_index_excludes_the_gold_answer():
    idx = lp_metrics.AnswerIndex([("a", "r", "x"), ("a", "r", "y")], [("b", "r", "x")])
    assert idx.others(("a", "r", "x"), "tail") == {"y"}
    assert idx.others(("a", "r", "x"), "head") == {"b"}
    with pytest.raises(ValueError):
        idx.others(("a", "r", "x"), "middle")


def test_relation_categories_follow_bordes():
    train = [("a", "one", "x"), ("b", "one", "y"),
             ("a", "to_many", "x"), ("a", "to_many", "y"), ("a", "to_many", "z"),
             ("a", "many_to", "x"), ("b", "many_to", "x"),
             ("a", "many", "x"), ("a", "many", "y"), ("b", "many", "x"), ("b", "many", "y")]
    assert lp_metrics.relation_categories(train) == {"one": "1-1", "to_many": "1-N", "many_to": "N-1", "many": "N-N"}


def test_sample_queries_matches_the_original_shuffle_and_request_size():
    triples = [(str(i), "r", str(i)) for i in range(50)]
    expected = list(triples)
    random.Random(7).shuffle(expected)
    assert lp_metrics.sample_queries(triples, 10, seed=7) == expected[:10]
    assert lp_metrics.requested_k(10, 3) == 13 and lp_metrics.requested_k(10, 5000, cap=100) == 100
    assert lp_metrics.reciprocal_ranks([1, 4, None]) == [1.0, 0.25, 0.0]


# ----------------------------------------------------------------------------- common/baselines

def test_frequency_lp_tie_aware_filtered_ranks():
    train = [("h1", "r", "x")] * 3 + [("h2", "r", "y")]
    model = baselines.FrequencyLP(train, ["x", "y", "z", "w"])
    assert model.rank("x", "r", "tail") == 1.0
    assert model.rank("y", "r", "tail") == 2.0
    assert model.rank("z", "r", "tail") == 3.5                            # tied with w at ranks 3 and 4
    assert model.rank("y", "r", "tail", known_others={"x"}) == 1.0
    assert model.rank("z", "r", "tail", bound="w") == 3.0
    assert model.rank("z", "unseen", "tail") == 2.5                       # no information: average of 4 entities


def test_prior_cv_scores_popularity_not_structure():
    train = [("a", "r", "x"), ("a", "r", "y"), ("b", "r", "x")]
    by_position = baselines.PriorCV(train, "relation_position")
    assert by_position.score(("a", "r", "x")) == pytest.approx(2 * math.log(3))
    assert by_position.score(("b", "r", "y")) == pytest.approx(2 * math.log(2))
    assert by_position.score(("a", "r", "x")) > by_position.score(("b", "r", "y"))
    assert baselines.PriorCV(train, "degree").score(("a", "r", "x")) == pytest.approx(2 * math.log(3))
    with pytest.raises(ValueError):
        baselines.PriorCV(train, "magic")


# ----------------------------------------------------------------------------- hard negatives

def test_frequency_weights_align_with_pools_and_bias_corruption_towards_popular_entities():
    train = [(f"h{i}", "r", "x") for i in range(1, 51)] + [("h51", "r", "y"), ("h60", "r", "w")]
    heads, tails = build_pools(train)
    head_w, tail_w = build_frequency_weights(train, heads, tails)
    assert tails["r"] == ["w", "x", "y"] and tail_w["r"] == [1, 50, 1]
    assert len(head_w["r"]) == len(heads["r"]) and set(head_w["r"]) == {1}
    known, rng = set(train) | {("h0", "r", "w")}, random.Random(0)
    counts = {"x": 0, "y": 0}
    for _ in range(600):
        out = corrupt(("h0", "r", "w"), heads, tails, known, rng, head_weights=head_w, tail_weights=tail_w)
        if out and out[1] == "tail":
            counts[out[0][2]] = counts.get(out[0][2], 0) + 1
    assert counts["x"] > 5 * max(1, counts["y"])


# ----------------------------------------------------------------------------- common/ensemble

def cv_row(idx, gold, pred, conf):
    return {"idx": idx, "claim": f"c{idx}", "gold": gold, "pred": pred, "confidence": conf,
            "score": support_score(pred, conf)}


@pytest.fixture
def runs():
    return {"A": [cv_row(0, SUPPORTED, SUPPORTED, 1.0), cv_row(1, REFUTED, REFUTED, 1.0),
                  cv_row(2, SUPPORTED, NEI, None), cv_row(3, REFUTED, SUPPORTED, 1.0)],
            "B": [cv_row(0, SUPPORTED, SUPPORTED, 1.0), cv_row(1, REFUTED, SUPPORTED, 0.8),
                  cv_row(2, SUPPORTED, SUPPORTED, 1.0), cv_row(3, REFUTED, REFUTED, 1.0)]}


def test_align_keeps_shared_claims_and_rejects_mismatched_files(runs):
    partial = {"A": runs["A"], "B": runs["B"][:3]}
    assert [r["idx"] for r in ensemble.align(partial)["A"]] == [0, 1, 2]
    broken = {"A": runs["A"], "B": [{**r, "claim": "other"} for r in runs["B"]]}
    with pytest.raises(ValueError):
        ensemble.align(broken)


def test_complementarity_and_pairwise(runs):
    comp = ensemble.complementarity(ensemble.align(runs))
    assert comp["venn"] == {"B": 2, "A+B": 1, "A": 1} and comp["oracle_accuracy"] == 1.0
    assert comp["exclusive"] == {"A": 1, "B": 2} and comp["accuracy"] == {"A": 0.5, "B": 0.75}
    (pair,) = ensemble.pairwise(ensemble.align(runs))
    assert (pair["only_a_correct"], pair["only_b_correct"]) == (1, 2) and pair["mcnemar_p"] == 1.0
    assert ensemble.is_correct(REFUTED, NEI, "refuted") and not ensemble.is_correct(REFUTED, NEI, "miss")


def test_vote_soft_average_and_cascade(runs):
    aligned = ensemble.align(runs)
    assert [r["pred"] for r in ensemble.vote(aligned)] == [SUPPORTED, REFUTED, SUPPORTED, NEI]
    assert [r["pred"] for r in ensemble.cascade(aligned, ["A", "B"])] == [SUPPORTED, REFUTED, SUPPORTED, SUPPORTED]
    assert [r["pred"] for r in ensemble.cascade(aligned, ["B", "A"])] == [SUPPORTED, SUPPORTED, SUPPORTED, REFUTED]
    assert [r["pred"] for r in ensemble.soft_average(aligned)][0] == SUPPORTED
    strict = ensemble.cascade(aligned, ["A", "B"], min_confidence=0.9)     # B's 0.8 on claim 1 is not enough
    assert strict[1]["pred"] == REFUTED                                    # A decided it first
    with pytest.raises(ValueError):
        ensemble.cascade(aligned, ["A", "Z"])


# ----------------------------------------------------------------------------- eval scripts

def test_eval_cv_build_row_records_everything_the_analysis_needs():
    eval_cv = load_script("eval/eval_cv.py")
    res = CVResult("c", ("a", "r", "b"), SUPPORTED, 0.8,
                   Explanation("rule", [ExplanationStep(("a", "r", "b"), "n")], "because", rule="r(X,Y) <= s(X,Y)"))
    rec = {"claim": "c", "label": SUPPORTED, "gold_triple": ["a", "r", "b"], "corrupted": None}
    row = eval_cv.build_row(3, rec, res, None, 12.34)
    assert row["idx"] == 3 and row["parse_ok"] and row["has_rule"] and row["n_steps"] == 1
    assert row["score"] == pytest.approx(0.9) and row["relation"] == "r" and row["ms"] == 12.3
    failed = eval_cv.build_row(4, {**rec, "label": REFUTED, "corrupted": "tail"}, None, "UNPARSEABLE_CLAIM", 1.0)
    assert failed["pred"] == NEI and failed["score"] == 0.5 and failed["corrupted"] == "tail" and not failed["parse_ok"]


def test_eval_cv_handles_claims_without_a_gold_triple():
    eval_cv = load_script("eval/eval_cv.py")
    res = CVResult("c", ("a", "r", "b"), REFUTED, 0.7, Explanation("path", [], "no path"))
    row = eval_cv.build_row(0, {"claim": "c", "label": REFUTED, "gold_triple": None}, res, None, 1.0)
    assert row["gold_triple"] is None and row["relation"] is None and row["parse_ok"] is None
    assert row["score"] == pytest.approx(0.15)
    failed = eval_cv.build_row(1, {"claim": "c", "label": SUPPORTED}, None, "UNPARSEABLE_CLAIM", 1.0)
    assert failed["parse_ok"] is None and failed["pred"] == NEI


def test_eval_cv_shuffle_keeps_the_file_index_so_runs_can_be_aligned():
    eval_cv = load_script("eval/eval_cv.py")

    class Source:
        def records(self, split, tag):
            return iter({"claim": f"c{i}", "label": SUPPORTED, "gold_triple": None} for i in range(20))

    eval_cv.get_claim_source = lambda dataset: Source()
    plain = list(eval_cv.claim_records("x", "test", None, 5, None))
    assert [i for i, _ in plain] == [0, 1, 2, 3, 4]
    a = list(eval_cv.claim_records("x", "test", None, 8, 3))
    b = list(eval_cv.claim_records("x", "test", None, 8, 3))
    assert a == b and all(rec["claim"] == f"c{i}" for i, rec in a) and [i for i, _ in a] != list(range(8))


def test_claim_source_reads_tagged_files_and_null_gold_triples(tmp_path, monkeypatch):
    import json
    from common.datasets import claims as claims_module

    monkeypatch.setattr(claims_module, "claims_dir", lambda dataset: tmp_path)
    (tmp_path / "test.jsonl").write_text(json.dumps(
        {"claim": "plain", "label": SUPPORTED, "gold_triple": ["a", "r", "b"]}) + "\n", encoding="utf-8")
    (tmp_path / "test_hard.jsonl").write_text(
        json.dumps({"claim": "hard", "label": REFUTED, "gold_triple": None, "corrupted": "tail"}) + "\n",
        encoding="utf-8")
    source = claims_module.JsonlClaimSource("toy")
    assert [c.claim for c in source.claims("test")] == ["plain"]
    (hard,) = source.claims("test", "hard")
    assert hard.gold_triple is None and hard.label == REFUTED
    assert next(source.records("test", "hard"))["corrupted"] == "tail"
    assert source.path("test", "hard").name == "test_hard.jsonl"
    with pytest.raises(FileNotFoundError):
        list(source.records("valid"))


def test_analyze_cv_sections_hang_together():
    analyze_cv = load_script("eval/analyze_cv.py")
    rows = []
    for i in range(40):
        gold = SUPPORTED if i % 2 == 0 else REFUTED
        pred = gold if i % 5 else NEI
        rows.append({"gold": gold, "pred": pred, "score": support_score(pred, 0.9), "relation": f"r{i % 2}",
                     "corrupted": "tail" if gold == REFUTED else None, "parse_ok": True,
                     "n_steps": 2 if pred != NEI else 0, "has_rule": pred == SUPPORTED})
    report = analyze_cv.analyze(rows, min_n=5, categories={"r0": "1-1", "r1": "N-N"}, calibrate_from=rows)
    assert set(report) == {"summary", "breakdowns", "calibration", "selective", "explanations"}
    assert set(report["breakdowns"]["by_relation_category"]) == {"1-1", "N-N"}
    assert report["calibration"]["temperature"] > 0 and report["selective"]["n"] == 40
    assert report["explanations"][NEI]["share_no_steps"] == 1.0
    analyze_cv.print_report(report)
    no_triple = [{**r, "relation": None, "parse_ok": None} for r in rows]       # e.g. FactKG
    groups = analyze_cv.analyze(no_triple, min_n=5)["breakdowns"]
    assert set(groups["by_parse"]) == {"no gold triple"} and set(groups["by_relation"]) == {"(no gold triple)"}


def test_eval_baselines_runs_on_a_tiny_graph():
    eval_baselines = load_script("eval/eval_baselines.py")
    from common.datasets.base import KnowledgeGraph
    from common.datasets.files import build_neighbors
    train = [(f"e{i}", "r", f"e{(i + 1) % 6}") for i in range(6)]
    kg = KnowledgeGraph(name="tiny", train=train, valid=[("e0", "r", "e3")], test=[("e1", "r", "e4")],
                        entity_labels={f"e{i}": f"E{i}" for i in range(6)}, relations=["r"],
                        neighbors=build_neighbors(train))
    out = eval_baselines.lp_baseline(kg, "test", 5, 0, 0)
    assert out["n"] == 2 and 0 < out["mrr"] <= 1 and out["by_category"]


# ----------------------------------------------------------------------------- AnyBURL ablation runner

def test_ablation_runner_on_the_toy_graph(tmp_path):
    from common.datasets.base import KnowledgeGraph
    from common.datasets.files import build_neighbors
    from modules.anyburl_safran.config import AnyburlConfig

    toy = load_script("tests/test_anyburl_core.py")
    ablate = load_script("modules/anyburl_safran/scripts/ablate.py")
    folder = tmp_path / "anyburl_safran" / "toy"
    folder.mkdir(parents=True)
    (folder / "rules.txt").write_text(toy.RULE_FILE, encoding="utf-8")
    kg = KnowledgeGraph(name="toy", train=list(toy.TRAIN), valid=[("alice", "nationality", "usa")], test=[],
                        entity_labels={e: e.capitalize() for e in toy.ENTITIES}, relations=list(toy.RELATIONS),
                        neighbors=build_neighbors(toy.TRAIN))
    claims = [LabeledClaim("Alice is a citizen of the USA.", SUPPORTED, ("alice", "nationality", "usa")),
              LabeledClaim("Carol is a citizen of Kenya.", REFUTED, ("carol", "nationality", "kenya"))]
    base = AnyburlConfig(artifact_root=tmp_path)
    base.thresholds.functional_min_observations = 2
    base.filters.unseen_negative_examples = 0

    settings = ablate.build_settings(["aggregation", "rule_kind", "smoothing", "drop_edges"], [])
    results = ablate.run_ablation(kg, claims, settings, base, split="valid", n_lp=1, k=5)
    by_name = {(r["family"], r["setting"]): r for r in results}
    assert [(r["family"], r["setting"]) for r in results] == [(s.family, s.name) for s in settings]
    full, cyclic = by_name[("rule_kind", "all")], by_name[("rule_kind", "cyclic_only")]
    assert full["n_rules"] == 4 and cyclic["n_rules"] == 3                 # 3 cyclic + 1 constant-headed rule
    assert full["accuracy"] >= cyclic["accuracy"] and 0 <= full["accuracy"] <= 1
    assert by_name[("drop_edges", "drop=50%")]["n_rules"] == 4             # rules stay as mined
    assert all("mrr" in r and "auc" in r for r in results)

    only_lp = ablate.run_ablation(kg, [], ablate.build_settings(["rule_length"], []), base, n_lp=1)
    assert "accuracy" not in only_lp[0] and "mrr" in only_lp[0]
    with pytest.raises(SystemExit):
        ablate.build_settings(["nonsense"], [])
