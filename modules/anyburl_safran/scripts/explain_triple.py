"""Show the verdict and proof for a triple, bypassing the claim parser.

    # auto-pick one example per case and explain each
    python modules/anyburl_safran/scripts/explain_triple.py --dataset wn18rr

    # explain a specific triple
    python modules/anyburl_safran/scripts/explain_triple.py --dataset wn18rr \
        --head 02314321 --relation _hypernym --tail 08102555

The three auto-picked cases answer "what changes when the fact is in the graph?":
  IN TRAIN   the triple is stated outright          -> SUPPORTED with no rule needed
  HELD OUT   true, but absent from the train graph  -> SUPPORTED only if a rule derives it
  CORRUPTED  in no split, so almost certainly false -> REFUTED (or NOT_ENOUGH_INFO)
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.datasets.base import load_graph  # noqa: E402
from modules.anyburl_safran import core  # noqa: E402
from modules.anyburl_safran.config import AnyburlConfig  # noqa: E402
from modules.anyburl_safran.module import AnyburlSafranModule  # noqa: E402


def corrupt(kg, rng):
    """A triple in no split: a true test triple with its tail swapped for a type-consistent one."""
    pool = {}
    for h, r, t in kg.train:
        pool.setdefault(r, []).append(t)
    everything = set(kg.train) | set(kg.valid) | set(kg.test)
    for h, r, t in rng.sample(kg.test, len(kg.test)):
        for candidate in rng.sample(pool.get(r, []), min(40, len(pool.get(r, [])))):
            if candidate != t and (h, r, candidate) not in everything:
                return (h, r, candidate)
    return None


def report(module, labels, case, triple):
    head, relation, tail = triple
    name = lambda e: core.label_of(e, labels)
    print(f"\n{'='*78}\n{case}: ({name(head)}) -{relation}-> ({name(tail)})")
    print(f"{' '*len(case)}  ids: {head} {relation} {tail}")
    evidence = module.score_triple(triple)
    result = module.decide("", evidence)
    print(f"  in train graph : {evidence.in_graph}")
    print(f"  rank of claim  : {evidence.rank}   (of {evidence.n_candidates} rule-derived candidates)")
    print(f"  rules deriving : {len(evidence.support)}")
    print(f"  VERDICT        : {result.verdict}"
          + (f"   confidence {result.confidence:.3f}" if result.confidence is not None else ""))
    explanation = result.explanation
    print(f"  RULE           : {explanation.rule or '(none - no rule was needed or none fired)'}")
    print(f"  PROOF STEPS    : {len(explanation.steps)}")
    for step in explanation.steps:
        print(f"    - {step.note}")
    print(f"  TEXT           : {explanation.text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--aggregation", choices=["max", "noisy-or"], default="max")
    ap.add_argument("--relation", default=None, help="restrict auto-picked examples to this relation")
    ap.add_argument("--head", default=None)
    ap.add_argument("--tail", default=None, help="give --head/--relation/--tail to explain one triple")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = load_graph(args.dataset)
    module = AnyburlSafranModule(AnyburlConfig(aggregation=args.aggregation))
    module.load(kg)
    labels = module._st.labels          # display labels (readable, unique)
    rng = random.Random(args.seed)

    if args.head and args.relation and args.tail:
        report(module, labels, "GIVEN", (args.head, args.relation, args.tail))
        return

    def pick(split):
        rows = [t for t in split if args.relation is None or t[1] == args.relation]
        return rng.choice(rows) if rows else None

    for case, triple in (("IN TRAIN ", pick(kg.train)),
                         ("HELD OUT ", pick(kg.test)),
                         ("CORRUPTED", corrupt(kg, rng))):
        if triple is None:
            print(f"\n{case}: no example found")
        else:
            report(module, labels, case, triple)


if __name__ == "__main__":
    main()
