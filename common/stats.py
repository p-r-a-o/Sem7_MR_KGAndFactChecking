"""Resampling statistics and agreement measures for the evaluation scripts (stdlib-only).

Everything here is a pure function of plain Python sequences, so it is shared by eval_lp, eval_cv,
analyze_cv and compare_modules without pulling numpy into `common/`.
"""
import math
import random
from collections import Counter
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]) of an already sorted sequence."""
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo))


def bootstrap_ci(items: Sequence[T], statistic: Callable[[Sequence[T]], float], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over `items` (queries or claims). Returns (point, low, high).

    `statistic` receives a resampled list of items, so it can be any metric (MRR, F1, AUC ...).
    With n_boot <= 0 or fewer than two items the interval collapses onto the point estimate.
    """
    point = statistic(items) if items else float("nan")
    if n_boot <= 0 or len(items) < 2:
        return point, point, point
    rng = random.Random(seed)
    n = len(items)
    stats = []
    for _ in range(n_boot):
        stats.append(statistic([items[rng.randrange(n)] for _ in range(n)]))
    stats = sorted(s for s in stats if not math.isnan(s))
    if not stats:
        return point, float("nan"), float("nan")
    return point, percentile(stats, alpha / 2), percentile(stats, 1 - alpha / 2)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of mean(a - b) over per-item scores (e.g. reciprocal ranks of two modules).

    `p_value` is the two-sided bootstrap p: twice the smaller tail mass of resampled differences
    on either side of zero (floored at 1/n_boot).
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    n = len(a)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "low": float("nan"), "high": float("nan"), "p_value": 1.0}
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    if n_boot <= 0:
        return {"n": n, "mean_diff": mean, "low": mean, "high": mean, "p_value": float("nan")}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    below = sum(1 for d in boots if d <= 0) / n_boot
    above = sum(1 for d in boots if d >= 0) / n_boot
    p = min(1.0, 2 * min(below, above))
    return {"n": n, "mean_diff": mean, "low": percentile(boots, alpha / 2),
            "high": percentile(boots, 1 - alpha / 2), "p_value": max(p, 1.0 / n_boot)}


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """McNemar's test on paired correctness. Exact binomial when the discordant count is small.

    `a_only` = claims only A gets right, `b_only` = only B. Two-sided p-value.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired samples must have the same length")
    a_only = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    b_only = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "method": "none"}
    if n <= 25:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail), "method": "exact"}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n          # continuity-corrected
    return {"a_only": a_only, "b_only": b_only, "p_value": math.erfc(math.sqrt(chi2 / 2)),
            "method": "chi2"}


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement between two label sequences (1 = identical, 0 = chance)."""
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 for fewer than two values)."""
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def format_ci(ci: Optional[tuple[float, float, float]], digits: int = 3) -> str:
    """'0.712 [0.668, 0.754]' for printing tables."""
    if ci is None:
        return "-"
    point, low, high = ci
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
