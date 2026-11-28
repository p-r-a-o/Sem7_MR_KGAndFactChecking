"""Claim-verification metrics beyond accuracy / F1 (stdlib-only, pure functions).

Everything is computed from per-claim rows with three fields::

    {"gold": "SUPPORTED" | "REFUTED", "pred": <verdict>, "score": <support score in [0, 1]>}

`support_score` turns any module's (verdict, confidence) into one number that rises with belief
that the claim is true, which makes threshold-free metrics (AUC), calibration and selective
prediction comparable across modules. Both current modules report the confidence *in their own
verdict* (AnyBURL+SAFRAN: rule score for SUPPORTED, 1 - score for REFUTED; PowerLink: sigmoid of the
KGE logit for SUPPORTED, 1 - that for REFUTED), so:

    SUPPORTED, conf c -> 0.5 + 0.5 c      REFUTED, conf c -> 0.5 - 0.5 c      NOT_ENOUGH_INFO -> 0.5

A missing confidence counts as 0 (score 0.5, i.e. tied with NOT_ENOUGH_INFO).
"""
import math
from collections import Counter
from typing import Callable, Hashable, Optional, Sequence

from common.metrics import classification_metrics
from common.stats import bootstrap_ci

SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"
Row = dict


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def support_score(verdict: str, confidence: Optional[float], missing_confidence: float = 0.0) -> float:
    """(verdict, confidence-in-verdict) -> belief that the claim is SUPPORTED, in [0, 1]."""
    c = _clamp01(confidence if confidence is not None else missing_confidence)
    if verdict == SUPPORTED:
        return 0.5 + 0.5 * c
    if verdict == REFUTED:
        return 0.5 - 0.5 * c
    return 0.5


# ----------------------------------------------------------------------------- ranking metrics

def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, tied values share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Probability that a random positive outscores a random negative (ties count half). NaN if one class is empty."""
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (step-wise, scikit-learn's definition; ties form one threshold)."""
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    prev_recall, ap = 0.0, 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            tp += 1 if labels[order[j]] else 0
            fp += 0 if labels[order[j]] else 1
            j += 1
        recall = tp / n_pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j
    return ap


def best_threshold(labels: Sequence[bool], scores: Sequence[float]) -> tuple[float, float]:
    """Score threshold (predict positive when score >= thr) maximising balanced accuracy; returns (thr, bal_acc)."""
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, float("nan")
    best = (float("inf"), 0.5 * (0 / n_pos + 1.0))        # threshold above everything: all negative
    for thr in sorted(set(scores)):
        tpr = sum(1 for y, s in zip(labels, scores) if y and s >= thr) / n_pos
        tnr = sum(1 for y, s in zip(labels, scores) if not y and s < thr) / n_neg
        if (tpr + tnr) / 2 > best[1]:
            best = (thr, (tpr + tnr) / 2)
    return best


# ----------------------------------------------------------------------------- calibration

def brier(labels: Sequence[bool], probs: Sequence[float]) -> float:
    return sum((p - (1.0 if y else 0.0)) ** 2 for y, p in zip(labels, probs)) / len(labels) if labels else float("nan")


def calibration_bins(labels: Sequence[bool], probs: Sequence[float], n_bins: int = 10) -> list[dict]:
    """Equal-width reliability bins over P(SUPPORTED); empty bins are omitted."""
    bins: list[list[tuple[bool, float]]] = [[] for _ in range(n_bins)]
    for y, p in zip(labels, probs):
        bins[min(n_bins - 1, int(_clamp01(p) * n_bins))].append((y, p))
    out = []
    for b, members in enumerate(bins):
        if members:
            out.append({"low": b / n_bins, "high": (b + 1) / n_bins, "n": len(members),
                        "mean_score": sum(p for _, p in members) / len(members),
                        "frac_positive": sum(1 for y, _ in members if y) / len(members)})
    return out


def expected_calibration_error(labels: Sequence[bool], probs: Sequence[float], n_bins: int = 10) -> float:
    """Weighted mean |fraction positive - mean score| over reliability bins (Guo et al., 2017)."""
    if not labels:
        return float("nan")
    return sum(b["n"] * abs(b["frac_positive"] - b["mean_score"])
               for b in calibration_bins(labels, probs, n_bins)) / len(labels)


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def apply_temperature(scores: Sequence[float], temperature: float) -> list[float]:
    """sigmoid(logit(s) / T): T > 1 softens, T < 1 sharpens. 0.5 stays 0.5."""
    return [1.0 / (1.0 + math.exp(-logit(s) / temperature)) for s in scores]


def fit_temperature(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Temperature minimising negative log-likelihood (grid search, 0.1 .. 10). Fit on valid, apply on test."""
    best_t, best_nll = 1.0, float("inf")
    for i in range(0, 81):
        t = 0.1 * (10 ** 2) ** (i / 80)                      # geometric grid 0.1 .. 10
        probs = apply_temperature(scores, t)
        nll = -sum(math.log(max(1e-12, p if y else 1 - p)) for y, p in zip(labels, probs))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


# ----------------------------------------------------------------------------- selective prediction

def risk_coverage(gold: Sequence[str], scores: Sequence[float], tie_label: str = REFUTED) -> dict:
    """Risk-coverage curve when the system answers its most decisive claims first.

    A claim's answer is SUPPORTED if score > 0.5, REFUTED if < 0.5 (`tie_label` at exactly 0.5, i.e.
    NOT_ENOUGH_INFO or no confidence) and its decisiveness is |score - 0.5|. Abstaining on the least
    decisive claims first yields a curve of (coverage, error rate among answered). `aurc` is the mean risk
    over all coverage levels i/n (lower is better); tied groups interpolate errors linearly so the
    result does not depend on input order.
    """
    items = [(abs(s - 0.5), (SUPPORTED if s > 0.5 else REFUTED if s < 0.5 else tie_label) != g)
             for g, s in zip(gold, scores) if g in (SUPPORTED, REFUTED)]
    n = len(items)
    if n == 0:
        return {"n": 0, "curve": [], "aurc": float("nan")}
    items.sort(key=lambda x: -x[0])
    curve, risk_sum, errors, i = [], 0.0, 0.0, 0
    while i < n:
        j = i
        while j < n and items[j][0] == items[i][0]:
            j += 1
        group, group_err = j - i, sum(1 for _, wrong in items[i:j] if wrong)
        for m in range(1, group + 1):
            covered = i + m
            risk = (errors + group_err * m / group) / covered
            risk_sum += risk
            curve.append((covered / n, risk))
        errors += group_err
        i = j
    return {"n": n, "curve": curve, "aurc": risk_sum / n}


def coverage_at_risk(curve: Sequence[tuple[float, float]], max_risk: float) -> float:
    """Largest coverage whose error rate is <= `max_risk` (0.0 if none)."""
    return max((c for c, r in curve if r <= max_risk), default=0.0)


def risk_at_coverage(curve: Sequence[tuple[float, float]], coverage: float) -> float:
    """Error rate when answering the most decisive `coverage` fraction of claims."""
    for c, r in curve:
        if c >= coverage - 1e-12:
            return r
    return curve[-1][1] if curve else float("nan")


# ----------------------------------------------------------------------------- verdict metrics

def two_class_metrics(gold: Sequence[str], pred: Sequence[str], nei_policy: str = "miss") -> dict[str, float]:
    """Balanced accuracy and macro-F1 over SUPPORTED / REFUTED, plus per-class recall.

    Plain accuracy rewards "refute whatever cannot be proven", so report this alongside it.
    nei_policy as in `common.metrics.classification_metrics`.
    """
    if nei_policy not in ("miss", "refuted"):
        raise ValueError("nei_policy must be 'miss' or 'refuted'")
    if nei_policy == "refuted":
        pred = [REFUTED if p == NEI else p for p in pred]
    recalls, f1s = {}, []
    for c in (SUPPORTED, REFUTED):
        support = sum(1 for g in gold if g == c)
        if support == 0:
            continue
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g in (SUPPORTED, REFUTED) and g != c and p == c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support
        recalls[c] = recall
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"balanced_accuracy": sum(recalls.values()) / len(recalls) if recalls else float("nan"),
            "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
            "recall_supported": recalls.get(SUPPORTED, float("nan")),
            "recall_refuted": recalls.get(REFUTED, float("nan"))}


def group_metrics(rows: Sequence[Row], key_fn: Callable[[Row], Hashable],
                  metric_fn: Callable[[Sequence[Row]], dict], min_n: int = 1) -> dict:
    """Apply `metric_fn` to each group of rows (e.g. per relation, per corruption side)."""
    groups: dict[Hashable, list[Row]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return {key: {"n": len(members), **metric_fn(members)}
            for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
            if len(members) >= min_n}


def quick_metrics(rows: Sequence[Row], nei_policy: str = "miss") -> dict[str, float]:
    """Small metric bundle used for breakdown tables."""
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    return {"accuracy": classification_metrics(gold, pred, nei_policy)["accuracy"],
            "balanced_accuracy": two_class_metrics(gold, pred, nei_policy)["balanced_accuracy"],
            "nei_rate": sum(p == NEI for p in pred) / len(pred) if pred else 0.0}


def summarize_cv(rows: Sequence[Row], nei_policy: str = "miss", n_boot: int = 0, seed: int = 0) -> dict:
    """Every CV metric in one dict. With n_boot > 0 adds 95% bootstrap CIs under `ci` (over claims)."""
    gold, pred, score = [r["gold"] for r in rows], [r["pred"] for r in rows], [r["score"] for r in rows]
    scored = [(g == SUPPORTED, s) for g, s in zip(gold, score) if g in (SUPPORTED, REFUTED)]
    labels, scores = [y for y, _ in scored], [s for _, s in scored]
    curve = risk_coverage(gold, score)
    answered = [(g, p) for g, p in zip(gold, pred) if p != NEI]
    out = {
        **classification_metrics(gold, pred, nei_policy),
        **two_class_metrics(gold, pred, nei_policy),
        "auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "brier": brier(labels, scores),
        "ece": expected_calibration_error(labels, scores),
        "aurc": curve["aurc"],
        "coverage": len(answered) / len(rows) if rows else 0.0,           # share with a decisive verdict
        "selective_accuracy": (sum(g == p for g, p in answered) / len(answered)) if answered else float("nan"),
        "verdicts": dict(Counter(pred)),
    }
    if n_boot > 0:
        stat = {"accuracy": lambda rs: classification_metrics([r["gold"] for r in rs], [r["pred"] for r in rs],
                                                              nei_policy)["accuracy"],
                "f1": lambda rs: classification_metrics([r["gold"] for r in rs], [r["pred"] for r in rs],
                                                        nei_policy)["f1"],
                "balanced_accuracy": lambda rs: two_class_metrics([r["gold"] for r in rs], [r["pred"] for r in rs],
                                                                  nei_policy)["balanced_accuracy"],
                "auc": lambda rs: roc_auc([r["gold"] == SUPPORTED for r in rs if r["gold"] in (SUPPORTED, REFUTED)],
                                          [r["score"] for r in rs if r["gold"] in (SUPPORTED, REFUTED)])}
        out["ci"] = {name: bootstrap_ci(list(rows), fn, n_boot=n_boot, seed=seed) for name, fn in stat.items()}
    return out
