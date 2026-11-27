"""Dataset adapter layer: KnowledgeGraph, LabeledClaim, ClaimSource (HANDOFF.md section 4)."""
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Protocol

from common.schema import Triple

# Inverse edges in `neighbors` are named relation + INVERSE_SUFFIX.
INVERSE_SUFFIX = "^-1"


@dataclass
class KnowledgeGraph:
    name: str                                   # "fb15k237" | "wn18rr" | "factkg"
    train: list[Triple]
    valid: list[Triple]
    test: list[Triple]
    entity_labels: dict[str, str]               # canonical ID -> readable name
    relations: list[str]
    # head -> [(relation, tail)] plus inverse edges (relation + INVERSE_SUFFIX).
    # Built from the TRAIN split only, so held-out facts never leak into reasoning.
    neighbors: dict[str, list[tuple[str, str]]]
    # ID -> one-line gloss, where the dataset ships one (WN18RR synset definitions). Used only to
    # tell apart entities whose label collides; empty for datasets with no description file.
    entity_descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class LabeledClaim:
    claim: str
    label: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    # the triple the claim was generated from; None when a dataset has no single gold triple
    # (FactKG multi-entity claims, and its test split which ships no evidence)
    gold_triple: Optional[Triple]


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
