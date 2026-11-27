"""PyClause module configuration.

Option names under `loader_options`, `qa_options` and `prediction_options` are PyClause's own keys
(clause/config-default.yaml in the PyClause repo) and are passed through unchanged, so anything
PyClause supports can be set without touching this module.

Verdict thresholds are copied from the AnyBURL+SAFRAN module's tuned values (both modules rank
rule-derived candidates the same way) but are UNTUNED for PyClause: re-check them on valid claims
with eval/eval_cv.py --split valid before reporting numbers.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from common.paths import artifacts_root


def _default_loader_options() -> dict:
    return {
        # Zero rules (h(X,c) <=) predict from relation frequency alone: no derivation to show,
        # and PredictionHandler cannot use them anyway. Same choice as the AnyBURL module.
        "load_zero_rules": False,
        "verbose": False,
    }


def _default_qa_options() -> dict:
    return {
        "collect_rules": True,       # needed to say which rules produced each candidate
        "filter_w_data": True,       # hide answers already in the train graph (filtered ranking)
        "verbose": False,
    }


def _default_prediction_options() -> dict:
    return {
        "collect_explanations": True,  # rules + their groundings for each scored triple
        "num_top_rules": 5,            # stop after the 5 most confident predicting rules
        "verbose": False,
    }


@dataclass
class VerdictThresholds:
    """Filtered rank of the claimed entity among rule-derived candidates -> verdict."""
    support_rank: int = 50              # rank <= this -> SUPPORTED
    refute_rank: int = 50               # rank >  this -> REFUTED; in between -> NOT_ENOUGH_INFO
    min_candidates_for_refute: int = 1  # no rule derives the claim: refute only if the rules
                                        # derived at least this many other answers, else NEI


@dataclass
class PyClauseConfig:
    aggregation: Literal["maxplus", "noisyor"] = "maxplus"   # PyClause's two aggregation functions
    rules_file: str = "rules.txt"        # inside artifacts/pyclause/<dataset>/
    artifact_root: Path = field(default_factory=artifacts_root)
    exclude_known: bool = True           # LP: hide answers already in the train graph
    cv_topk: int = 100                   # candidates ranked per direction when verifying a claim
                                         # (must exceed refute_rank to be able to say REFUTED by rank)
    explain_facts_shown: int = 8         # body facts listed as ExplanationSteps
    rules_mentioned: int = 3             # runner-up rules named in the explanation text
    num_threads: int = -1                # PyClause worker threads (-1 = all cores)
    loader_options: dict = field(default_factory=_default_loader_options)
    qa_options: dict = field(default_factory=_default_qa_options)
    prediction_options: dict = field(default_factory=_default_prediction_options)
    thresholds: VerdictThresholds = field(default_factory=VerdictThresholds)
    parser_url: Optional[str] = None     # claim parser base URL; default read from services.yaml
