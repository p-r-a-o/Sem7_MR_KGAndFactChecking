"""PowerLink module configuration. Explainer defaults follow the PowerLink README quick-start."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.paths import artifacts_root, repo_root


@dataclass
class ExplainerConfig:
    lr: float = 0.005
    alpha: float = 1.0
    beta: float = 1.0
    num_hops: int = 1
    num_epochs: int = 50
    num_paths: int = 40
    max_path_length: int = 5
    k_core: int = 2
    prune_max_degree: int = 200
    power_order: int = 3
    regularisation_weight: float = 0.02
    parameterizer_hidden_dim: int = 64
    comp_g_size_limit: int = 1000
    combination_method: str = "concat"
    paths_returned: int = 3          # explanation shows the best path, mentions the next few


@dataclass
class VerdictThresholds:
    """Rank-based claim verdicts. Defaults are UNTUNED guesses: tune on valid-split claims
    (scripts/generate_claims.py --split valid, then eval/eval_cv.py) before trusting results."""
    support_rank: int = 10           # filtered rank <= this -> SUPPORTED
    refute_rank: int = 100           # filtered rank >  this -> REFUTED; in between -> NOT_ENOUGH_INFO


@dataclass
class PowerLinkConfig:
    device: str = "cpu"              # "cpu" or "cuda:0"
    repo_dir: Path = field(default_factory=lambda: repo_root() / "modules" / "powerlink" / "power-link")
    artifact_root: Path = field(default_factory=artifacts_root)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    thresholds: VerdictThresholds = field(default_factory=VerdictThresholds)
    explain_top_n: int = 10          # explain only the top-n predictions (explaining costs ~num_epochs steps each)
    exclude_known: bool = True       # drop answers already in the train graph from LP results
    seed: int = 8
    parser_url: Optional[str] = None  # claim parser base URL; default read from services.yaml
