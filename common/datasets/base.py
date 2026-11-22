"""Dataset adapter layer: KnowledgeGraph, LabeledClaim, ClaimSource (HANDOFF.md section 4)."""
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from common.schema import Triple

# Inverse edges in `neighbors` are named relation + INVERSE_SUFFIX.
INVERSE_SUFFIX = "^-1"


@dataclass
class KnowledgeGraph:
    name: str                                   # "fb15k237" | "wn18rr"
    train: list[Triple]
    valid: list[Triple]
    test: list[Triple]
    entity_labels: dict[str, str]               # canonical ID -> readable name
    relations: list[str]
    # head -> [(relation, tail)] plus inverse edges (relation + INVERSE_SUFFIX).
    # Built from the TRAIN split only, so held-out facts never leak into reasoning.
    neighbors: dict[str, list[tuple[str, str]]]


@dataclass
class LabeledClaim:
    claim: str
    label: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    gold_triple: Triple                         # the triple the claim was generated from


class ClaimSource(Protocol):
    def claims(self, split: str) -> Iterable[LabeledClaim]: ...


def load_graph(dataset: str) -> KnowledgeGraph:
    from common.datasets.files import build_knowledge_graph
    from common.datasets.registry import get_spec
    return build_knowledge_graph(get_spec(dataset))


def get_claim_source(dataset: str) -> ClaimSource:
    from common.datasets.claims import JsonlClaimSource
    from common.datasets.registry import get_spec
    return JsonlClaimSource(get_spec(dataset).name)
