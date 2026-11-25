"""AnyBURL+SAFRAN HTTP service. Run from the repo root inside the kgx-anyburl env:

    python modules/anyburl_safran/api.py [--aggregation noisy-or] [--port 8102]

Needs rules mined first (modules/anyburl_safran/scripts/mine_rules.py) and, for /verify_claim,
the claim parser service from services.yaml.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dataclasses  # noqa: E402

from common.datasets.base import load_graph  # noqa: E402
from common.server import module_routes, serve  # noqa: E402
from common.services import get_service  # noqa: E402
from modules.anyburl_safran.config import AnyburlConfig  # noqa: E402
from modules.anyburl_safran.module import AnyburlSafranModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("anyburl_safran").port)
    ap.add_argument("--aggregation", choices=["max", "noisy-or"], default="max",
                    help="max = AnyBURL's default, noisy-or = SAFRAN-style combination")
    ap.add_argument("--rules-file", default="rules.txt", help="file inside artifacts/anyburl_safran/<dataset>/")
    ap.add_argument("--min-confidence", type=float, default=0.0001)
    ap.add_argument("--unseen-negatives", type=int, default=5,
                    help="AnyBURL's UNSEEN_NEGATIVE_EXAMPLES; 0 uses the rule file's raw confidence")
    ap.add_argument("--min-correct", type=int, default=2, help="drop rules with fewer correct train predictions")
    ap.add_argument("--max-body-length", type=int, default=None)
    ap.add_argument("--max-rules-per-relation", type=int, default=10000,
                    help="0 = keep every rule in the file (slower load, ~1GB for a 1000s snapshot)")
    ap.add_argument("--support-rank", type=int, default=50, help="tuned on valid claims; see config.py")
    ap.add_argument("--refute-rank", type=int, default=50)
    ap.add_argument("--min-candidates-for-refute", type=int, default=1)
    ap.add_argument("--max-groundings", type=int, default=20, help="groundings kept per rule per query")
    ap.add_argument("--step-budget", type=int, default=20000, help="edge expansions allowed per rule")
    ap.add_argument("--include-known", action="store_true", help="do not hide answers already in the train graph")
    ap.add_argument("--keep-redundant", action="store_true", help="count rules that fired on identical evidence")
    args = ap.parse_args()

    cfg = AnyburlConfig(aggregation=args.aggregation, rules_file=args.rules_file,
                        exclude_known=not args.include_known, dedupe_groundings=not args.keep_redundant)
    # ApplyLimits is frozen, so build a new one instead of assigning to its fields
    cfg.limits = dataclasses.replace(cfg.limits, max_groundings_per_rule=args.max_groundings,
                                     step_budget_per_rule=args.step_budget)
    cfg.filters.min_confidence = args.min_confidence
    cfg.filters.unseen_negative_examples = args.unseen_negatives
    cfg.filters.min_correct_predictions = args.min_correct
    cfg.filters.max_body_length = args.max_body_length
    cfg.filters.max_rules_per_relation = args.max_rules_per_relation or None
    cfg.thresholds.support_rank, cfg.thresholds.refute_rank = args.support_rank, args.refute_rank
    cfg.thresholds.min_candidates_for_refute = args.min_candidates_for_refute
    serve(module_routes(AnyburlSafranModule(cfg), load_graph), args.port)


if __name__ == "__main__":
    main()
