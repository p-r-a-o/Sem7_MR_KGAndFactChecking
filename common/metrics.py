"""Pure metric functions for LP and CV evaluation (stdlib-only)."""
from typing import Optional, Sequence


def rank_metrics(ranks: Sequence[Optional[int]], ks: Sequence[int] = (1, 3, 10)) -> dict[str, float]:
    """MRR and Hits@k from 1-based ranks. `None` = gold not in the returned top-k (counts as a miss).

    Because modules return only a top-k list, MRR here is a lower bound on the full-ranking MRR.
    """
    n = len(ranks)
    if n == 0:
        return {"n": 0, "mrr": 0.0, **{f"hits@{k}": 0.0 for k in ks}}
    out = {"n": n, "mrr": sum(1.0 / r for r in ranks if r) / n}
    for k in ks:
        out[f"hits@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    return out


def classification_metrics(gold: Sequence[str], pred: Sequence[str],
                           nei_policy: str = "miss") -> dict[str, float]:
    """Accuracy and P/R/F1 for the SUPPORTED class.

    nei_policy:
      "miss"    - NOT_ENOUGH_INFO is never correct (hurts recall/accuracy, is not a false positive).
      "refuted" - NOT_ENOUGH_INFO is treated as REFUTED ("not supported").
    """
    if nei_policy not in ("miss", "refuted"):
        raise ValueError("nei_policy must be 'miss' or 'refuted'")
    if nei_policy == "refuted":
        pred = ["REFUTED" if p == "NOT_ENOUGH_INFO" else p for p in pred]
    n = len(gold)
    tp = sum(g == "SUPPORTED" and p == "SUPPORTED" for g, p in zip(gold, pred))
    fp = sum(g == "REFUTED" and p == "SUPPORTED" for g, p in zip(gold, pred))
    fn = sum(g == "SUPPORTED" and p != "SUPPORTED" for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "nei_rate": sum(p == "NOT_ENOUGH_INFO" for p in pred) / n if n else 0.0,
    }
