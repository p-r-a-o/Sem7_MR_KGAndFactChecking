"""Evaluate link prediction through the HTTP API (any registered module).

    python eval/eval_lp.py --module PowerLink --dataset fb15k237 --n 200 --k 10
    python eval/eval_lp.py --module AnyBURL+SAFRAN --dataset wn18rr --n 500 --seeds 0 1 2 \\
        --bootstrap 1000 --out lp_anyburl.json --dump-queries lp_anyburl_queries.jsonl

Two rank definitions are reported side by side from the same predictions:

  raw       rank inside the module's own top-k list (what this script always reported; modules already
            hide answers in the TRAIN graph, but valid/test answers can still push the gold down).
  filtered  the standard protocol: every other known-true answer (train + valid + test) is removed
            before ranking. To leave room for the removed answers the module is asked for
            k + (#known answers) candidates, capped by --max-request. Ranks beyond what a module returns
            count as misses, so MRR is still a lower bound of the full-ranking MRR.

Use `filtered` when comparing with published numbers. `--seeds` repeats the evaluation on different
random query samples; the pooled bootstrap CI is computed over the *unique* queries.
`--dump-queries` writes one JSON line per query so eval/compare_modules.py can run paired tests
between modules evaluated with the same seeds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets.base import KnowledgeGraph, load_graph  # noqa: E402
from common.datasets.registry import SPECS  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.lp_metrics import (AnswerIndex, filtered_rank, relation_categories,  # noqa: E402
                               requested_k, sample_queries)
from common.metrics import rank_metrics  # noqa: E402
from common.registry import build_registry  # noqa: E402
from common.stats import bootstrap_ci, mean_std  # noqa: E402

KS = (1, 3, 10)


def evaluate_queries(module, kg: KnowledgeGraph, seed: int, n: int, split: str, k: int, max_request: int,
                     answers: AnswerIndex, categories: dict[str, str]) -> tuple[list[dict], int]:
    """Run one seeded sample of triples in both directions. Returns (per-query rows, error count)."""
    rows, errors = [], 0
    for triple in sample_queries(list(getattr(kg, split)), n, seed):
        h, r, t = triple
        for side, (qh, qt, gold) in {"tail": (h, None, t), "head": (None, t, h)}.items():
            others = answers.others(triple, side)
            try:
                res = module.predict_link(qh, r, qt, requested_k(k, len(others), max_request))
            except ModuleError as e:
                errors += 1
                print(f"error {e.code}: {e.message}", file=sys.stderr)
                continue
            cands = [p.candidate for p in res.predictions]
            raw_top = cands[:k]
            rows.append({"seed": seed, "head": h, "relation": r, "tail": t, "side": side,
                         "category": categories.get(r, "?"), "n_known_others": len(others),
                         "rank_raw": raw_top.index(gold) + 1 if gold in raw_top else None,
                         "rank_filtered": filtered_rank(cands, gold, others)})
    return rows, errors


def metrics_for(rows: list[dict], key: str, n_boot: int) -> dict:
    ranks = [r[key] for r in rows]
    out = rank_metrics(ranks, KS)
    if n_boot > 0 and ranks:
        out["ci"] = {"mrr": bootstrap_ci(ranks, lambda rs: rank_metrics(rs, KS)["mrr"], n_boot),
                     "hits@10": bootstrap_ci(ranks, lambda rs: rank_metrics(rs, KS)["hits@10"], n_boot)}
    return out


def summarize(rows: list[dict], n_boot: int) -> dict:
    """raw/filtered x all/tail/head, plus a per-relation-category breakdown of the filtered ranks."""
    out: dict = {}
    for key in ("rank_raw", "rank_filtered"):
        name = key.split("_")[1]
        out[name] = {"all": metrics_for(rows, key, n_boot),
                     "tail": metrics_for([r for r in rows if r["side"] == "tail"], key, 0),
                     "head": metrics_for([r for r in rows if r["side"] == "head"], key, 0)}
    out["by_category_filtered"] = {
        cat: rank_metrics([r["rank_filtered"] for r in rows if r["category"] == cat], KS)
        for cat in sorted({r["category"] for r in rows})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="single seed (ignored when --seeds is given)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="repeat on several query samples")
    ap.add_argument("--max-request", type=int, default=1000, help="cap on candidates requested per query")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--out", default=None, help="optional JSON file for metrics")
    ap.add_argument("--dump-queries", default=None, help="optional JSONL with one row per query")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    module = build_registry()[args.module]
    kg = load_graph(args.dataset)
    module.load(kg)
    answers = AnswerIndex(kg.train, kg.valid, kg.test)
    categories = relation_categories(kg.train)

    start = time.time()
    all_rows, per_seed, errors = [], {}, 0
    for seed in seeds:
        rows, errs = evaluate_queries(module, kg, seed, args.n, args.split, args.k, args.max_request,
                                      answers, categories)
        errors += errs
        all_rows.extend(rows)
        per_seed[seed] = {"raw_mrr": rank_metrics([r["rank_raw"] for r in rows], KS)["mrr"],
                          "filtered_mrr": rank_metrics([r["rank_filtered"] for r in rows], KS)["mrr"]}

    unique: dict[tuple, dict] = {}
    for row in all_rows:                                   # overlapping seeds must not double count queries
        unique.setdefault((row["head"], row["relation"], row["tail"], row["side"]), row)
    pooled = list(unique.values())

    report = {"module": args.module, "dataset": args.dataset, "split": args.split, "k": args.k,
              "seeds": seeds, "n_unique_queries": len(pooled), "errors": errors,
              "seconds": round(time.time() - start, 1), **summarize(pooled, args.bootstrap),
              "per_seed": per_seed}
    if len(seeds) > 1:
        for key in ("raw_mrr", "filtered_mrr"):
            mean, std = mean_std([v[key] for v in per_seed.values()])
            report[f"{key}_across_seeds"] = {"mean": mean, "std": std}
    # Backwards compatible keys: what this script printed before filtered ranks existed.
    report.update({"tail": report["raw"]["tail"], "head": report["raw"]["head"], "all": report["raw"]["all"]})
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.dump_queries:
        Path(args.dump_queries).write_text("\n".join(json.dumps(r) for r in pooled) + "\n")


if __name__ == "__main__":
    main()
