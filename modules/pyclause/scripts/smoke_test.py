"""End-to-end check of the PyClause module in-process (no HTTP, no claim parser).

    python modules/pyclause/scripts/smoke_test.py --dataset fb15k237 --n 3

Loads the KG and the learned rules, runs a few tail/head link predictions on test triples and
verifies each test triple plus one corrupted copy via verify_triple, printing the proofs.
"""
import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.datasets.base import load_graph  # noqa: E402
from modules.pyclause.config import PyClauseConfig  # noqa: E402
from modules.pyclause.module import PyClauseModule  # noqa: E402


def show(ex, indent: str = "    ") -> None:
    print(f"{indent}{ex.text}")
    if ex.rule:
        print(f"{indent}rule: {ex.rule}")
    for s in ex.steps[:4]:
        print(f"{indent}  - {s.note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fb15k237", choices=["fb15k237", "wn18rr"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--aggregation", default="maxplus", choices=["maxplus", "noisyor"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = load_graph(args.dataset)
    module = PyClauseModule(PyClauseConfig(aggregation=args.aggregation))
    t0 = time.time()
    module.load(kg)
    print(f"load: {time.time() - t0:.1f}s")

    rng = random.Random(args.seed)
    entities = list(kg.entity_labels)
    lab = kg.entity_labels.get
    hits = 0
    for h, r, t in rng.sample(kg.test, args.n):
        print(f"\n=== test triple ({lab(h)}, {r}, {lab(t)})")
        t0 = time.time()
        res = module.predict_link(h, r, None, args.k)
        cands = [p.candidate for p in res.predictions]
        hits += t in cands
        print(f"  LP tail top-{args.k} ({time.time() - t0:.2f}s): "
              + ", ".join(lab(c, c) for c in cands) + f"   gold in top-k: {t in cands}")
        if res.predictions:
            show(res.predictions[0].explanation)
        res = module.predict_link(None, r, t, args.k)
        print(f"  LP head top-{args.k}: " + ", ".join(lab(p.candidate, p.candidate) for p in res.predictions))

        for name, triple in (("true", (h, r, t)), ("corrupted", (h, r, rng.choice(entities)))):
            t0 = time.time()
            cv = module.verify_triple(f"{name} claim", triple)
            print(f"  CV {name:9s} -> {cv.verdict} (conf {cv.confidence}) [{time.time() - t0:.2f}s]")
            show(cv.explanation)
    print(f"\ngold tail in top-{args.k}: {hits}/{args.n}")


if __name__ == "__main__":
    main()
