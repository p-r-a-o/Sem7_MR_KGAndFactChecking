"""PyClause HTTP service. Run from the repo root inside the kgx-pyclause env:

    python modules/pyclause/api.py [--aggregation noisyor] [--port 8104]

Needs rules learned first (modules/pyclause/scripts/learn_rules.py) and, for /verify_claim,
the claim parser service from services.yaml.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.datasets.base import load_graph  # noqa: E402
from common.server import module_routes, serve  # noqa: E402
from common.services import get_service  # noqa: E402
from modules.pyclause.config import PyClauseConfig  # noqa: E402
from modules.pyclause.module import PyClauseModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("pyclause").port)
    ap.add_argument("--aggregation", choices=["maxplus", "noisyor"], default="maxplus",
                    help="maxplus = AnyBURL-style max with tie breaking, noisyor = SAFRAN-style")
    ap.add_argument("--rules-file", default="rules.txt", help="file inside artifacts/pyclause/<dataset>/")
    ap.add_argument("--support-rank", type=int, default=50, help="untuned for PyClause; see config.py")
    ap.add_argument("--refute-rank", type=int, default=50)
    ap.add_argument("--min-candidates-for-refute", type=int, default=1)
    ap.add_argument("--cv-topk", type=int, default=100, help="candidates ranked per direction in CV")
    ap.add_argument("--num-top-rules", type=int, default=5,
                    help="PredictionHandler: rules applied per triple (-1 = all; affects noisy-or)")
    ap.add_argument("--threads", type=int, default=-1, help="PyClause worker threads (-1 = all cores)")
    ap.add_argument("--include-known", action="store_true", help="do not hide answers already in the train graph")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="raw PyClause option, e.g. loader.b_min_conf=0.01 or qa_handler.disc_at_least=20")
    args = ap.parse_args()

    cfg = PyClauseConfig(aggregation=args.aggregation, rules_file=args.rules_file,
                         exclude_known=not args.include_known, cv_topk=args.cv_topk,
                         num_threads=args.threads)
    cfg.prediction_options["num_top_rules"] = args.num_top_rules
    cfg.thresholds.support_rank, cfg.thresholds.refute_rank = args.support_rank, args.refute_rank
    cfg.thresholds.min_candidates_for_refute = args.min_candidates_for_refute
    sections = {"loader": cfg.loader_options, "qa_handler": cfg.qa_options,
                "prediction_handler": cfg.prediction_options}
    for item in args.set:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if section not in sections or not name:
            raise SystemExit(f"--set {item!r}: expected loader.|qa_handler.|prediction_handler.<key>=<value>")
        value: object = raw.strip()
        if value in ("True", "true", "False", "false"):
            value = value.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
        sections[section][name] = value
    serve(module_routes(PyClauseModule(cfg), load_graph), args.port)


if __name__ == "__main__":
    main()
