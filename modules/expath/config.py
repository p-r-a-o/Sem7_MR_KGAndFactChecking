"""eXpath module configuration. Defaults for rule quality follow eXpath's process.py
(minSC = minHC = 0.01, minSupp = 10, paths up to length 3); everything else is ours and marked so."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.paths import artifacts_root


@dataclass
class RuleQuality:
    """eXpath's rule filters (process.py: minSC, minHC, minSupp)."""
    min_sc: float = 0.01             # standard confidence  supp / #body
    min_hc: float = 0.01             # head coverage        supp / #head
    supp_smoothing: int = 10         # eXpath scores a rule as SC * supp / (supp + minSupp)


@dataclass
class MiningConfig:
    """Offline rule mining (scripts/mine_rules.py). Sampling caps are ours, not eXpath's:
    eXpath mines rules per explained prediction; we mine them once per relation so that
    link prediction can *apply* them, and re-use eXpath's metrics to score them."""
    samples_per_relation: int = 200  # training triples per relation whose head->tail paths are aggregated
    max_path_len: int = 3            # eXpath's BFS limit for interpretable paths
    min_body_count: int = 2          # a relation path must connect >= this many sampled pairs
    max_bodies_per_relation: int = 60
    hub_degree: int = 1000           # middle entities of 3-hop paths with more edges than this are skipped
    max_start_neighbors: int = 300   # neighbours of the head explored for 3-hop paths (random sample)
    uc_min_head: int = 10            # Uc rules: constant must be the target of >= this many facts
    uc_min_supp: int = 5
    uc_min_sc: float = 0.1           # stricter than eXpath's 0.01, or every frequent constant yields thousands of rules
    uc_max_rules_per_head: int = 30  # per (relation, constant)
    row_chunk: int = 4096            # rows per sparse block when computing rule metrics (bounds memory)
    seed: int = 0


@dataclass
class ReasoningConfig:
    max_path_len: int = 3            # path aggregation length when explaining one triple
    max_paths: int = 2000            # grounded paths kept per explained triple
    max_on_demand_rules: int = 25    # unmined relation paths whose metrics are computed per query
    max_on_demand_body_len: int = 3  # longer unmined bodies are not scored on demand
    max_on_demand_nnz: int = 300000  # skip on-demand CP metrics if a body relation has more facts (cost cap)
    max_on_demand_instances: int = 5_000_000   # ... or if the body has more groundings than this
    max_uc_const_degree: int = 5000  # Uc constants with more edges (e.g. the literal "1996") are skipped
    min_rule_supp: int = 2           # a rule needs >= this many supporting cases to count as proof
                                     # (eXpath's process.py has this filter commented out; one case is no generalisation)
    max_frontier: int = 20000        # a rule whose intermediate frontier exceeds this is not applied
    aggregation: str = "noisy-or"    # "noisy-or" (eXpath calculateConfidence) | "max" | "max-plus" (AnyBURL)
    rules_shown: int = 3             # rules listed in an explanation's text
    critical_facts_shown: int = 3


@dataclass
class VerdictConfig:
    """Claim-verification thresholds, chosen on VALID claims only (see README, Measured)."""
    support: float = 0.5             # triple mode: eXpath confidence for (h, r, t) at or above this -> SUPPORTED
                                     # (0.5 = best F1 of {0.3, 0.5, 0.7, 0.85} on FB15k-237 valid claims)
    support_by_dataset: dict = field(default_factory=lambda: {"wn18rr": 0.02})
                                     # WN18RR valid claims, F1 at 0.02/0.05/0.1/0.2/0.3/0.5/0.7/0.85:
                                     # 0.697/0.675/0.658/0.618/0.608/0.595/0.551/0.515 (precision ~0.98 at all)
    link_support: float = 0.3        # multi-entity mode: confidence for a rule-inferred link between claim entities
    refute_alt: float = 0.5          # REFUTED if t is unproven but an alternative answer has this confidence
    functional_ratio: float = 0.9    # relation counts as functional if this share of heads has a single tail
    relation_relevance: float = 0.35 # multi-entity claims: min relevance of a relation to the claim text
    path_relevance: float = 0.35     # min mean relation relevance of a connecting path
    hop_relevance: float = 0.0       # ... and no relation on it may be less relevant than this. 0 = off.
                                     # FactKG valid acc: 0.0 -> 0.832, 0.05 -> 0.799, 0.1 -> 0.787 (stricter
                                     # proofs vs. accuracy; see README "Known weaknesses")
    conflict: float = 0.3            # min conflict weight (relevance x functionality x type fit) to REFUTE
    closed_world_existence: bool = True   # existence claim + described entity without such a relation -> REFUTED


@dataclass
class EnrichConfig:
    """Node addition from the original database (DBpedia) for entities / links missing locally."""
    enabled: bool = False            # off by default so evaluation stays reproducible on the frozen graph
    endpoint: str = "https://dbpedia.org/sparql"
    max_triples_per_entity: int = 400
    timeout: float = 20.0


@dataclass
class ExpathConfig:
    artifact_root: Path = field(default_factory=artifacts_root)
    quality: RuleQuality = field(default_factory=RuleQuality)
    mining: MiningConfig = field(default_factory=MiningConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    use_kge: bool = True             # use the exported KGE base model for LP when its artifacts exist
    kge_candidates: int = 100        # KGE top-n that eXpath tries to explain
    kge_on_demand_budget: int = 150  # on-demand rule metric computations shared by all candidates of one query
    require_proof: bool = True       # drop LP candidates with no checkable proof (a score alone is not an answer)
    exclude_known: bool = True       # hide answers already in the train graph (from the ranked predictions)
    show_known: bool = False         # list those stored answers first, marked "from the graph" (UI / demo).
                                     # Keep False for eval/eval_lp.py: it scores list positions, so stored
                                     # answers on top would push the held-out answer down.
    known_shown: int = 10            # at most this many stored answers listed
    parser_url: Optional[str] = None # shared claim parser; default from services.yaml
    # How claims are turned into queries, per dataset (config, not code branches):
    #   "triple"       - shared claim_parser service -> one (h, r, t)
    #   "multi_entity" - entity set + relation relevance (FactKG claims mention several facts)
    claim_mode: dict = field(default_factory=lambda: {"factkg": "multi_entity"})
    default_claim_mode: str = "triple"

    def artifacts(self, dataset: str) -> Path:
        return self.artifact_root / "expath" / dataset
