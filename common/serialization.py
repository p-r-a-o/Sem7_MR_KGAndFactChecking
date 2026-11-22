"""JSON <-> dataclass helpers (stdlib-only). Wire format = dataclasses.asdict."""
import dataclasses
from typing import Any, Optional

from common.schema import (CVResult, Explanation, ExplanationStep, LinkPrediction,
                           LPResult, Triple)


def to_dict(obj: Any) -> dict:
    return dataclasses.asdict(obj)


def _triple(x) -> Optional[Triple]:
    return None if x is None else (x[0], x[1], x[2])


def explanation_from_dict(d: dict) -> Explanation:
    return Explanation(
        kind=d["kind"],
        steps=[ExplanationStep(triple=_triple(s.get("triple")), note=s.get("note", ""))
               for s in d.get("steps", [])],
        text=d.get("text", ""),
        rule=d.get("rule"),
        confidence=d.get("confidence"),
    )


def lp_result_from_dict(d: dict) -> LPResult:
    q = d["query"]
    return LPResult(
        query=(q[0], q[1], q[2]),
        predictions=[LinkPrediction(candidate=p["candidate"], score=p["score"],
                                    explanation=explanation_from_dict(p["explanation"]))
                     for p in d["predictions"]],
    )


def cv_result_from_dict(d: dict) -> CVResult:
    return CVResult(
        claim=d["claim"],
        triple=_triple(d.get("triple")),
        verdict=d["verdict"],
        confidence=d.get("confidence"),
        explanation=explanation_from_dict(d["explanation"]),
    )
