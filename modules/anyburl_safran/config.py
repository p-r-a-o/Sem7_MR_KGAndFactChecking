"""AnyBURL+SAFRAN module configuration.

Learning defaults mirror the AnyBURL quick start; application limits and verdict thresholds are ours
and are UNTUNED - tune them on valid-split claims before reporting numbers.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from common.paths import artifacts_root, repo_root
from modules.anyburl_safran.core import ApplyLimits

MODULE_DIR = repo_root() / "modules" / "anyburl_safran"
DEFAULT_JAR = MODULE_DIR / "AnyBURL-23-1x.jar"
JAR_URL = "https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1x.jar"


@dataclass
class RuleFilters:
    """Applied once at load(). AnyBURL already prunes during learning; this trims the tail further."""
    min_confidence: float = 0.0001
    min_correct_predictions: int = 2
    max_body_length: Optional[int] = None
    max_rules_per_relation: Optional[int] = 10000  # keep the most confident ones per relation; a
                                        # 1000s FB15k-237 snapshot has 5M rules (84k for one
                                        # relation), and keeping 10000 costs ~1.6GB. Must not be
                                        # below ApplyLimits.max_rules or rules are discarded that a
                                        # query would have applied.
    unseen_negative_examples: int = 5   # AnyBURL's UNSEEN_NEGATIVE_EXAMPLES: confidence becomes
                                        # correct/(predicted+5), which demotes 2-of-2 "perfect" rules


@dataclass
class VerdictThresholds:
    """Rank of the claimed entity among rule-derived candidates -> verdict.

    Tuned with scripts/tune_thresholds.py on 500 valid claims per dataset. F1 (NEI scored as
    not-supported) over support_rank = refute_rank:

        setting        FB15k-237   WN18RR
        ( 20,  20)     0.710       0.765
        ( 50,  50)     0.711       0.779     <- default, best compromise
        (100, 100)     0.710       0.792

    WN18RR keeps gaining recall at looser thresholds while FB15k-237 is flat, so a dataset that
    matters can be tuned and passed to api.py rather than changing this default.
    Raising `refute_rank` above `support_rank` opens an explicit "uncertain" band; on valid it only
    traded SUPPORTED/REFUTED decisions for NOT_ENOUGH_INFO without improving any metric.
    """
    support_rank: int = 50            # rank <= this -> SUPPORTED
    refute_rank: int = 50             # rank >  this -> REFUTED; in between -> NOT_ENOUGH_INFO
    min_candidates_for_refute: int = 1    # if no rule derives the claim, refute only when the rules
                                          # derived at least this many other answers; 0 never says
                                          # NOT_ENOUGH_INFO (same metrics when NEI scores as REFUTED)
    functional_violation_ratio: float = 0.05   # relation counts as functional below this violation rate
    functional_min_observations: int = 20      # ... and only with at least this many observed subjects


@dataclass
class AnyburlConfig:
    aggregation: Literal["max", "noisy-or"] = "max"   # "max" = AnyBURL, "noisy-or" = SAFRAN-style
    dedupe_groundings: bool = True     # ignore rules that fired on identical evidence
    rules_file: str = "rules.txt"      # inside artifacts/anyburl_safran/<dataset>/
    artifact_root: Path = field(default_factory=artifacts_root)
    jar_path: Path = DEFAULT_JAR       # only needed offline, by scripts/mine_rules.py
    exclude_known: bool = True         # hide answers already in the train graph from LP results
    explain_facts_shown: int = 8       # body facts listed as ExplanationSteps
    rules_mentioned: int = 3           # runner-up rules named in the explanation text
    limits: ApplyLimits = field(default_factory=ApplyLimits)
    cv_max_rules: int = 10000          # claim verification applies every kept rule, no early stop
    filters: RuleFilters = field(default_factory=RuleFilters)
    thresholds: VerdictThresholds = field(default_factory=VerdictThresholds)
    parser_url: Optional[str] = None   # claim parser base URL; default read from services.yaml
