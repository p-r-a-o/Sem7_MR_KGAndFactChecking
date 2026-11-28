"""Cross-module comparison of claim-verification runs (stdlib-only, pure functions).

Input is one list of per-claim rows per module (the JSONL written by eval/eval_cv.py), aligned by claim
index. Output answers "do the modules fail on the same claims?" (complementarity, kappa, oracle) and
"does combining them help?" (voting, soft averaging, cascades), the analysis a multi-module report needs.
"""
import itertools
from typing import Optional, Sequence

from common.cv_metrics import NEI, REFUTED, SUPPORTED
from common.stats import cohens_kappa, mcnemar

Row = dict


def is_correct(gold: str, pred: str, nei_policy: str = "miss") -> bool:
    if nei_policy == "refuted" and pred == NEI:
        pred = REFUTED
    return gold == pred


def align(runs: dict[str, Sequence[Row]]) -> dict[str, list[Row]]:
    """Keep only claims present in every run (matched on idx and claim text), in a common order."""
    if not runs:
        return {}
    by_name = {name: {r["idx"]: r for r in rows} for name, rows in runs.items()}
    shared = set.intersection(*(set(m) for m in by_name.values()))
    for idx in shared:
        texts = {m[idx]["claim"] for m in by_name.values()}
        if len(texts) > 1:
            raise ValueError(f"claim {idx} differs between runs; were they generated from the same claim file?")
    return {name: [m[i] for i in sorted(shared)] for name, m in by_name.items()}


def pairwise(aligned: dict[str, list[Row]], nei_policy: str = "miss") -> list[dict]:
    """Per pair of modules: Cohen's kappa of verdicts, McNemar test on correctness, who wins exclusively."""
    out = []
    for a, b in itertools.combinations(aligned, 2):
        ra, rb = aligned[a], aligned[b]
        ca = [is_correct(r["gold"], r["pred"], nei_policy) for r in ra]
        cb = [is_correct(r["gold"], r["pred"], nei_policy) for r in rb]
        test = mcnemar(ca, cb)
        out.append({"a": a, "b": b, "n": len(ra),
                    "kappa": cohens_kappa([r["pred"] for r in ra], [r["pred"] for r in rb]),
                    "agreement": sum(x["pred"] == y["pred"] for x, y in zip(ra, rb)) / len(ra) if ra else 0.0,
                    "only_a_correct": test["a_only"], "only_b_correct": test["b_only"],
                    "mcnemar_p": test["p_value"], "mcnemar_method": test["method"]})
    return out


def complementarity(aligned: dict[str, list[Row]], nei_policy: str = "miss") -> dict:
    """How the modules' correct sets overlap. `venn` maps 'A+B' (who is right) to a claim count."""
    names = list(aligned)
    n = len(next(iter(aligned.values()))) if aligned else 0
    venn: dict[str, int] = {}
    for i in range(n):
        right = [m for m in names if is_correct(aligned[m][i]["gold"], aligned[m][i]["pred"], nei_policy)]
        key = "+".join(right) if right else "none"
        venn[key] = venn.get(key, 0) + 1
    accuracy = {m: sum(is_correct(r["gold"], r["pred"], nei_policy) for r in aligned[m]) / n if n else 0.0
                for m in names}
    return {"n": n, "accuracy": accuracy, "venn": dict(sorted(venn.items(), key=lambda kv: -kv[1])),
            "oracle_accuracy": (n - venn.get("none", 0)) / n if n else 0.0,
            "exclusive": {m: venn.get(m, 0) for m in names}}


def _combined_row(rows: Sequence[Row], pred: str, score: float) -> Row:
    return {"idx": rows[0]["idx"], "claim": rows[0]["claim"], "gold": rows[0]["gold"], "pred": pred,
            "score": score, "confidence": None}


def vote(aligned: dict[str, list[Row]]) -> list[Row]:
    """Majority of non-NEI verdicts; ties broken by the summed support score; all-NEI stays NEI."""
    out = []
    for rows in zip(*aligned.values()):
        decided = [r for r in rows if r["pred"] != NEI]
        mean_score = sum(r["score"] for r in rows) / len(rows)
        if not decided:
            out.append(_combined_row(rows, NEI, 0.5))
            continue
        n_sup = sum(r["pred"] == SUPPORTED for r in decided)
        n_ref = len(decided) - n_sup
        if n_sup == n_ref:
            lean = sum(r["score"] - 0.5 for r in decided)
            pred = SUPPORTED if lean > 0 else REFUTED if lean < 0 else NEI
        else:
            pred = SUPPORTED if n_sup > n_ref else REFUTED
        out.append(_combined_row(rows, pred, mean_score))
    return out


def soft_average(aligned: dict[str, list[Row]]) -> list[Row]:
    """Average the support scores; the sign of (mean - 0.5) is the verdict."""
    out = []
    for rows in zip(*aligned.values()):
        mean_score = sum(r["score"] for r in rows) / len(rows)
        out.append(_combined_row(rows, SUPPORTED if mean_score > 0.5 else REFUTED if mean_score < 0.5 else NEI,
                                 mean_score))
    return out


def cascade(aligned: dict[str, list[Row]], order: Sequence[str], min_confidence: float = 0.0) -> list[Row]:
    """Ask modules in `order`; the first decisive verdict (not NEI, confidence >= min) wins.

    Fits modules with different precision/recall, e.g. a high-precision rule engine first and an
    embedding or LLM module for what it leaves open.
    """
    missing = [m for m in order if m not in aligned]
    if missing:
        raise ValueError(f"unknown modules in cascade: {missing}")
    out = []
    for i in range(len(next(iter(aligned.values())))):
        chosen: Optional[Row] = None
        for name in order:
            row = aligned[name][i]
            if row["pred"] != NEI and (row.get("confidence") or 0.0) >= min_confidence:
                chosen = row
                break
        base = aligned[order[0]][i]
        out.append(_combined_row([base], chosen["pred"], chosen["score"]) if chosen
                   else _combined_row([base], NEI, 0.5))
    return out
