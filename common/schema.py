"""Shared result schema. Stdlib-only. Mirrors HANDOFF.md section 4 exactly."""
from dataclasses import dataclass, field
from typing import Literal, Optional

Triple = tuple[str, str, str]  # (head, relation, tail), canonical KG IDs


@dataclass
class ExplanationStep:
    triple: Optional[Triple]      # supporting fact, if applicable
    note: str = ""                # short text for the UI


@dataclass
class Explanation:
    kind: Literal["path", "rule", "subgraph"]
    steps: list[ExplanationStep]  # path hops / rule groundings / evidence facts
    text: str                     # human-readable justification
    rule: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class LinkPrediction:
    candidate: str
    score: float
    explanation: Explanation


@dataclass
class LPResult:
    query: tuple[Optional[str], str, Optional[str]]   # exactly one None = what we predict
    predictions: list[LinkPrediction]                  # sorted, top-k


@dataclass
class CVResult:
    claim: str
    triple: Optional[Triple]
    verdict: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    confidence: Optional[float]
    explanation: Explanation
