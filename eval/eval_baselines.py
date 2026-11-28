"""Reasoning-free baselines on the same data as the modules (no services needed).

    python eval/eval_baselines.py --dataset fb15k237 --n 1000 --split test
    python eval/eval_baselines.py --dataset wn18rr --split test --tag hard --tune-split valid

LP: relation-conditional popularity ranking with exact, tie-aware, filtered ranks over ALL entities
    (the floor any link-prediction module should clear; MRR here is not a top-k lower bound).
CV: (1) popularity-only classifiers that score a claim from entity frequencies and never look at graph
    structure. A ROC-AUC clearly above 0.5 means negative sampling leaks the label, so read module
    accuracy relative to it. (2) trivial verdicts (always SUPPORTED / always REFUTED).
    The popularity threshold is fit on --tune-split (e.g. valid) when that claim file exists, otherwise
    the best threshold on the evaluated claims is reported and marked optimistic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.baselines import FrequencyLP, PriorCV  # noqa: E402
from common.cv_metrics import (REFUTED, SUPPORTED, average_precision, best_threshold,  # noqa: E402
                               roc_auc, two_class_metrics)
from common.datasets.base import load_graph  # noqa: E402
from common.datasets.claims import JsonlClaimSource  # noqa: E402
from common.lp_metrics import AnswerIndex, relation_categories, sample_queries  # noqa: E402
from common.metrics import classification_metrics, rank_metrics  # noqa: E402
from common.stats import bootstrap_ci  # noqa: E402

KS = (1, 3, 10)


def lp_baseline(kg, split: str, n: int, seed: int, n_boot: int) -> dict:
    model = FrequencyLP(kg.train, list(kg.entity_labels))
    answers = AnswerIndex(kg.train, kg.valid, kg.test)
    categories = relation_categories(kg.train)
    ranks, by_cat = [], {}
    for triple in sample_queries(list(getattr(kg, split)), n, seed):
        h, r, t = triple
        for side, gold, bound in (("tail", t, h), ("head", h, t)):
            rank = model.rank(gold, r, side, answers.others(triple, side), bound)
            ranks.append(rank)
            by_cat.setdefault(categories.get(r, "?"), []).append(rank)
    out = {**rank_metrics(ranks, KS), "by_category": {c: rank_metrics(v, KS) for c, v in sorted(by_cat.items())}}
    if n_boot:
        out["ci_mrr"] = bootstrap_ci(ranks, lambda rs: rank_metrics(rs, KS)["mrr"], n_boot)
    return out


def load_claims(source: JsonlClaimSource, split: str, tag, n: int):
    try:
        recs = [r for _, r in zip(range(n), source.records(split, tag))]
    except FileNotFoundError:
        return None
    return [(tuple(r["gold_triple"]), r["label"] == SUPPORTED) for r in recs if r.get("gold_triple")]


def cv_baselines(kg, source, split, tag, tune_split, n, n_boot) -> dict:
    test = load_claims(source, split, tag, n)
    if test is None:
        return {"error": f"no claim file for split={split} tag={tag}; run scripts/generate_claims.py"}
    tune = load_claims(source, tune_split, tag, n) if tune_split else None
    labels = [y for _, y in test]
    out = {"n": len(test), "positive_rate": sum(labels) / len(labels),
           "always_supported": classification_metrics(
               [SUPPORTED if y else REFUTED for y in labels], [SUPPORTED] * len(labels))}
    for kind in PriorCV.KINDS:
        model = PriorCV(kg.train, kind)
        scores = [model.score(tr) for tr, _ in test]
        auc = roc_auc(labels, scores)
        if tune:
            thr, _ = best_threshold([y for _, y in tune], [model.score(tr) for tr, _ in tune])
            threshold_source = f"fit on {tune_split}"
        else:
            thr, _ = best_threshold(labels, scores)
            threshold_source = "fit on evaluated claims (optimistic)"
        pred = [SUPPORTED if s >= thr else REFUTED for s in scores]
        gold = [SUPPORTED if y else REFUTED for y in labels]
        entry = {"auc": auc, "average_precision": average_precision(labels, scores),
                 "threshold": thr, "threshold_source": threshold_source,
                 **classification_metrics(gold, pred), **two_class_metrics(gold, pred)}
        if n_boot:
            entry["ci_auc"] = bootstrap_ci(list(zip(labels, scores)),
                                           lambda rs: roc_auc([y for y, _ in rs], [s for _, s in rs]), n_boot)
        out[f"prior_{kind}"] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--tag", default=None, help="claim file variant, e.g. 'hard'")
    ap.add_argument("--tune-split", default=None, choices=["valid", "test"],
                    help="claim split used to fit the popularity threshold (default: optimistic, same claims)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-lp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kg = load_graph(args.dataset)
    report = {"dataset": args.dataset, "split": args.split, "tag": args.tag}
    if not args.skip_lp:
        report["lp_frequency"] = lp_baseline(kg, args.split, args.n, args.seed, args.bootstrap)
    report["cv"] = cv_baselines(kg, JsonlClaimSource(args.dataset), args.split, args.tag, args.tune_split,
                                args.n, args.bootstrap)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
