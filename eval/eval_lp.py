"""Evaluate link prediction through the HTTP API (any registered module).

    python eval/eval_lp.py --module PowerLink --dataset fb15k237 --n 200 --k 10

Metrics come from the returned top-k only (MRR is therefore a lower bound; hits@k for k <= --k is exact
up to the module's own filtering, e.g. PowerLink hides answers already in the train graph).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets.base import load_graph  # noqa: E402
from common.datasets.registry import SPECS  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.metrics import rank_metrics  # noqa: E402
from common.registry import build_registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON file for metrics")
    args = ap.parse_args()

    registry = build_registry()
    module = registry[args.module]
    kg = load_graph(args.dataset)
    module.load(kg)

    triples = list(getattr(kg, args.split))
    random.Random(args.seed).shuffle(triples)
    ranks: dict[str, list] = {"tail": [], "head": []}
    errors, start = 0, time.time()
    for h, r, t in triples[: args.n]:
        for side, (qh, qt, gold) in {"tail": (h, None, t), "head": (None, t, h)}.items():
            try:
                res = module.predict_link(qh, r, qt, args.k)
            except ModuleError as e:
                errors += 1
                print(f"error {e.code}: {e.message}", file=sys.stderr)
                continue
            cands = [p.candidate for p in res.predictions]
            ranks[side].append(cands.index(gold) + 1 if gold in cands else None)

    report = {"module": args.module, "dataset": args.dataset, "split": args.split, "k": args.k,
              "errors": errors, "seconds": round(time.time() - start, 1),
              "tail": rank_metrics(ranks["tail"]), "head": rank_metrics(ranks["head"]),
              "all": rank_metrics(ranks["tail"] + ranks["head"])}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
