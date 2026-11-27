"""eXpath HTTP service. Run from the repo root inside the module's environment (numpy + scipy):

    python modules/expath/api.py [--port 8103] [--enrich] [--show-known] [--no-kge] [--support 0.3]

services.yaml starts it with --show-known (the app lists stored answers first). For eval/eval_lp.py start it
WITHOUT --show-known, or stored answers take the top ranks and the scores drop.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.datasets.base import load_graph  # noqa: E402
from common.server import module_routes, serve  # noqa: E402
from common.services import get_service  # noqa: E402
from modules.expath.config import ExpathConfig  # noqa: E402
from modules.expath.module import ExpathModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("expath").port)
    ap.add_argument("--enrich", action="store_true", help="node addition: fetch missing DBpedia facts (needs internet)")
    ap.add_argument("--no-kge", action="store_true", help="rank LP candidates with rules even if KGE artifacts exist")
    ap.add_argument("--allow-unproven", action="store_true", help="keep KGE candidates eXpath cannot prove")
    ap.add_argument("--include-known", action="store_true", help="do not hide answers already in the train graph")
    ap.add_argument("--show-known", action="store_true",
                    help="LP: list answers already in the graph first (for the UI; do NOT use for eval_lp.py)")
    ap.add_argument("--aggregation", choices=["noisy-or", "max", "max-plus"], default="noisy-or")
    ap.add_argument("--support", type=float, default=None, help="CV: eXpath confidence needed for SUPPORTED")
    ap.add_argument("--relevance", type=float, default=None, help="CV (multi-entity): min relation relevance")
    ap.add_argument("--hop-relevance", type=float, default=None,
                    help="CV (multi-entity): reject evidence paths with a hop less relevant than this (e.g. 0.05)")
    args = ap.parse_args()

    cfg = ExpathConfig(use_kge=not args.no_kge, require_proof=not args.allow_unproven,
                       exclude_known=not args.include_known)
    cfg.show_known = args.show_known
    cfg.enrich.enabled = args.enrich
    cfg.reasoning.aggregation = args.aggregation
    if args.support is not None:
        cfg.verdict.support = args.support
    if args.relevance is not None:
        cfg.verdict.relation_relevance = cfg.verdict.path_relevance = args.relevance
    if args.hop_relevance is not None:
        cfg.verdict.hop_relevance = args.hop_relevance
    serve(module_routes(ExpathModule(cfg), load_graph), args.port)


if __name__ == "__main__":
    main()
