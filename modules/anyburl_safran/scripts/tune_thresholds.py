"""Tune the claim-verification thresholds on a dataset's valid-split claims (offline step).

    python modules/anyburl_safran/scripts/tune_thresholds.py --dataset fb15k237 --n 500

Evidence (rank of the claimed entity, number of rule-derived candidates, functional conflicts) is
gathered once per claim, then every threshold combination is scored against it, so the sweep is cheap.
Claims are verified from their gold triple, not through the claim parser: this tunes the reasoning,
not the parser. Copy the winning values into config.py (or pass them to api.py).
"""
import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.datasets.base import get_claim_source, load_graph  # noqa: E402
from common.metrics import classification_metrics  # noqa: E402
from modules.anyburl_safran.config import AnyburlConfig  # noqa: E402
from modules.anyburl_safran.module import AnyburlSafranModule  # noqa: E402

SUPPORT_RANKS = (1, 2, 3, 5, 10, 20, 50)
REFUTE_RANKS = (1, 2, 3, 5, 10, 20, 50, 100)
MIN_CANDIDATES = (0, 1, 5, 10, 50)


def score_setting(gold: list[str], pred: list[str], nei_policy: str) -> dict[str, float]:
    """Shared CV metrics plus balanced accuracy (mean per-class recall).

    Plain accuracy is a poor guide here: most positives are not derivable from rules at all, so
    "refute whatever cannot be proven" scores well while answering almost nothing. Balanced accuracy
    keeps both classes visible.
    """
    metrics = classification_metrics(gold, pred, nei_policy)
    recalls = []
    for label in ("SUPPORTED", "REFUTED"):
        total = sum(g == label for g in gold)
        hit = sum(g == label and (p == label or (nei_policy == "refuted" and label == "REFUTED"
                                                 and p == "NOT_ENOUGH_INFO"))
                  for g, p in zip(gold, pred))
        recalls.append(hit / total if total else 0.0)
    metrics["recall_supported"], metrics["recall_refuted"] = recalls
    metrics["balanced"] = sum(recalls) / len(recalls)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--aggregation", choices=["max", "noisy-or"], default="max")
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"],
                    help="how NOT_ENOUGH_INFO is scored, as in eval/eval_cv.py")
    ap.add_argument("--rank-by", default="balanced", choices=["balanced", "accuracy", "f1"])
    ap.add_argument("--top", type=int, default=12, help="how many threshold settings to print")
    args = ap.parse_args()

    module = AnyburlSafranModule(AnyburlConfig(aggregation=args.aggregation))
    module.load(load_graph(args.dataset))
    claims = list(itertools.islice(get_claim_source(args.dataset).claims(args.split), args.n))

    start = time.time()
    evidence = [(c.label, module.score_triple(c.gold_triple)) for c in claims]
    print(f"gathered evidence for {len(claims)} claims in {time.time() - start:.1f}s "
          f"({(time.time() - start) / max(1, len(claims)) * 1000:.0f} ms/claim)")
    print("gold labels:", dict(Counter(label for label, _ in evidence)))
    derived = sum(1 for _, e in evidence if e.in_graph or e.rank is not None)
    print(f"claims the rules can derive at all: {derived}/{len(evidence)}")

    gold = [label for label, _ in evidence]
    results = []
    for support, refute, min_candidates in itertools.product(SUPPORT_RANKS, REFUTE_RANKS, MIN_CANDIDATES):
        if refute < support:
            continue
        th = module.cfg.thresholds
        th.support_rank, th.refute_rank, th.min_candidates_for_refute = support, refute, min_candidates
        pred = [module.decide("", e).verdict for _, e in evidence]
        results.append(((support, refute, min_candidates), score_setting(gold, pred, args.nei_policy)))

    results.sort(key=lambda r: -r[1][args.rank_by])
    print(f"\nbest threshold settings by {args.rank_by} (NEI policy = {args.nei_policy}):")
    header = (f"{'support':>8}{'refute':>7}{'min_cand':>9} | {'bal':>5}{'acc':>6}{'F1':>6} | "
              f"{'rec_SUP':>8}{'rec_REF':>8}{'NEI':>6}")
    print(header + "\n" + "-" * len(header))
    for (support, refute, min_candidates), m in results[:args.top]:
        print(f"{support:8d}{refute:7d}{min_candidates:9d} | {m['balanced']:5.3f}{m['accuracy']:6.3f}"
              f"{m['f1']:6.3f} | {m['recall_supported']:8.3f}{m['recall_refuted']:8.3f}"
              f"{m['nei_rate']:6.3f}")


if __name__ == "__main__":
    main()
