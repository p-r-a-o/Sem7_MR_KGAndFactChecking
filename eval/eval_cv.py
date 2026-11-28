"""Evaluate claim verification through the HTTP API (any registered module).

    python eval/eval_cv.py --module PowerLink --dataset fb15k237 --split valid --n 200 --nei-policy miss
    python eval/eval_cv.py --module AnyBURL+SAFRAN --dataset wn18rr --split test --n 1000 \\
        --bootstrap 1000 --out cv_anyburl_test.jsonl
    python eval/eval_cv.py ... --tag hard        # frequency-weighted negatives (see generate_claims.py)
    python eval/eval_cv.py ... --dataset factkg --shuffle-seed 0   # claims without a gold triple

Use --split valid to tune the module's verdict thresholds, then report once on --split test.

Beyond accuracy/precision/recall/F1 (unchanged) the report adds the metrics a verdict count hides:
balanced accuracy and macro-F1, threshold-free ROC-AUC / average precision on the module's support
score, Brier score and ECE (calibration), AURC (selective prediction: abstain on the least decisive
claims first), coverage, and 95% bootstrap CIs over claims. See common/cv_metrics.py for how
(verdict, confidence) becomes a support score.

Per-claim rows (JSONL, --out) carry everything eval/analyze_cv.py and eval/compare_modules.py need:
index in the claim file (stable under --shuffle-seed, so runs can be aligned), gold, verdict,
confidence, support score, relation, corruption side, parsed vs gold triple (null when the dataset has
no gold triple), explanation size and latency.
"""
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cv_metrics import NEI, summarize_cv, support_score  # noqa: E402
from common.datasets.base import KnowledgeGraph, get_claim_source  # noqa: E402
from common.datasets.registry import SPECS  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.registry import build_registry  # noqa: E402


def claim_records(dataset: str, split: str, tag, n: int, shuffle_seed):
    """(idx, record) pairs; idx is the claim's position in the file, kept when --shuffle-seed reorders them."""
    records = get_claim_source(dataset).records(split, tag)
    if shuffle_seed is not None:
        records = list(enumerate(records))
        random.Random(shuffle_seed).shuffle(records)
        return itertools.islice(records, n)
    return itertools.islice(enumerate(records), n)


def build_row(idx: int, rec: dict, res, error, ms: float) -> dict:
    gold = rec.get("gold_triple")
    gold_triple = tuple(gold) if gold else None          # FactKG claims have none
    row = {"idx": idx, "claim": rec["claim"], "gold": rec["label"], "gold_triple": gold_triple,
           "relation": gold_triple[1] if gold_triple else None, "corrupted": rec.get("corrupted"),
           "ms": round(ms, 1)}
    if res is None:
        return {**row, "pred": NEI, "confidence": None, "score": 0.5, "error": error, "parsed_triple": None,
                "parse_ok": False if gold_triple else None, "n_steps": 0, "has_rule": False,
                "explanation_kind": None, "explanation": ""}
    parse_ok = (tuple(res.triple) == gold_triple if res.triple else False) if gold_triple else None
    return {**row, "pred": res.verdict, "confidence": res.confidence,
            "score": support_score(res.verdict, res.confidence), "error": None,
            "parsed_triple": res.triple, "parse_ok": parse_ok,
            "n_steps": len(res.explanation.steps), "has_rule": bool(res.explanation.rule),
            "explanation_kind": res.explanation.kind, "explanation": res.explanation.text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--tag", default=None, help="claim file variant, e.g. 'hard' -> <split>_hard.jsonl")
    ap.add_argument("--n", type=int, default=200, help="max claims")
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"])
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--out", default=None, help="JSONL of per-claim predictions")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="sample --n claims at random instead of taking the first n (FactKG files are sorted by claim type)")
    args = ap.parse_args()

    module = build_registry()[args.module]
    # The HTTP proxy only sends the dataset name, so skip building the graph client-side (FactKG's is
    # several GB) and skip /load when the service already has this dataset in memory.
    if module.health().get("loaded") != args.dataset:
        module.load(KnowledgeGraph(name=args.dataset, train=[], valid=[], test=[], entity_labels={},
                                   relations=[], neighbors={}))

    rows, unparsed, errors = [], 0, 0
    start = time.time()
    for idx, rec in claim_records(args.dataset, args.split, args.tag, args.n, args.shuffle_seed):
        t0 = time.time()
        try:
            res = module.verify_claim(rec["claim"])
            rows.append(build_row(idx, rec, res, None, (time.time() - t0) * 1000))
        except ModuleError as e:
            if e.code == "UNPARSEABLE_CLAIM":
                unparsed += 1
            else:
                errors += 1
                print(f"error {e.code}: {e.message}", file=sys.stderr)
            # An unparsed/failed claim counts as NOT_ENOUGH_INFO so it is visible in the metrics.
            rows.append(build_row(idx, rec, None, e.code, (time.time() - t0) * 1000))

    ms = sorted(r["ms"] for r in rows)
    checked = [r["parse_ok"] for r in rows if r["parse_ok"] is not None]
    report = {"module": args.module, "dataset": args.dataset, "split": args.split, "tag": args.tag,
              "nei_policy": args.nei_policy, "unparseable": unparsed, "errors": errors,
              "parse_accuracy": sum(checked) / len(checked) if checked else None,   # None: no gold triples
              "median_ms": ms[len(ms) // 2] if ms else 0.0,
              "seconds": round(time.time() - start, 1),
              **summarize_cv(rows, args.nei_policy, args.bootstrap)}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
