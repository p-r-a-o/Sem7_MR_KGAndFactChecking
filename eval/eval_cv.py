"""Evaluate claim verification through the HTTP API (any registered module).

    python eval/eval_cv.py --module PowerLink --dataset fb15k237 --split valid --n 200 --nei-policy miss

Use --split valid to tune the module's verdict thresholds, then report once on --split test.
Per-claim predictions are written as JSONL for error analysis.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets.base import get_claim_source  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.metrics import classification_metrics  # noqa: E402
from common.registry import build_registry  # noqa: E402
from common.datasets.base import load_graph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=200, help="max claims")
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"])
    ap.add_argument("--out", default=None, help="JSONL of per-claim predictions")
    args = ap.parse_args()

    module = build_registry()[args.module]
    module.load(load_graph(args.dataset))

    gold, pred, rows, unparsed, errors = [], [], [], 0, 0
    start = time.time()
    for c in itertools.islice(get_claim_source(args.dataset).claims(args.split), args.n):
        try:
            res = module.verify_claim(c.claim)
        except ModuleError as e:
            if e.code == "UNPARSEABLE_CLAIM":
                unparsed += 1
            else:
                errors += 1
                print(f"error {e.code}: {e.message}", file=sys.stderr)
            # An unparsed/failed claim counts as NOT_ENOUGH_INFO so it is visible in the metrics.
            gold.append(c.label)
            pred.append("NOT_ENOUGH_INFO")
            rows.append({"claim": c.claim, "gold": c.label, "pred": "NOT_ENOUGH_INFO", "error": e.code})
            continue
        gold.append(c.label)
        pred.append(res.verdict)
        rows.append({"claim": c.claim, "gold": c.label, "pred": res.verdict, "confidence": res.confidence,
                     "parsed_triple": res.triple, "gold_triple": c.gold_triple, "explanation": res.explanation.text})

    report = {"module": args.module, "dataset": args.dataset, "split": args.split,
              "nei_policy": args.nei_policy, "unparseable": unparsed, "errors": errors,
              "seconds": round(time.time() - start, 1), **classification_metrics(gold, pred, args.nei_policy)}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
