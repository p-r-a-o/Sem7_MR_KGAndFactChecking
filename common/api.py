"""The interface every reasoning module implements (HANDOFF.md section 4)."""
from typing import Optional, Protocol

from common.datasets.base import KnowledgeGraph
from common.schema import CVResult, LPResult


class ReasoningModule(Protocol):
    name: str  # shown in the UI dropdown

    def load(self, kg: KnowledgeGraph) -> None:
        """Read prebuilt artifacts for this KG (no training here)."""

    def predict_link(self, head: Optional[str], relation: str,
                     tail: Optional[str], k: int = 10) -> LPResult:
        """Exactly one of head/tail is None."""

    def verify_claim(self, claim: str) -> CVResult:
        """Claim text -> triple -> verdict + proof."""
