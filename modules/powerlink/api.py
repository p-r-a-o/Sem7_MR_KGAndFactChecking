"""PowerLink HTTP service. Run from the repo root inside the kgx-powerlink env:

    python modules/powerlink/api.py [--device cuda:0] [--port 8101] [--explain-top-n 5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.datasets.base import load_graph  # noqa: E402
from common.server import module_routes, serve  # noqa: E402
from common.services import get_service  # noqa: E402
from modules.powerlink.config import PowerLinkConfig  # noqa: E402
from modules.powerlink.module import PowerLinkModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("powerlink").port)
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda:0'")
    ap.add_argument("--explain-top-n", type=int, default=10)
    ap.add_argument("--explainer-epochs", type=int, default=50)
    ap.add_argument("--support-rank", type=int, default=10)
    ap.add_argument("--refute-rank", type=int, default=100)
    ap.add_argument("--include-known", action="store_true", help="do not hide answers already in the train graph")
    args = ap.parse_args()

    cfg = PowerLinkConfig(device=args.device, explain_top_n=args.explain_top_n,
                          exclude_known=not args.include_known)
    cfg.explainer.num_epochs = args.explainer_epochs
    cfg.thresholds.support_rank, cfg.thresholds.refute_rank = args.support_rank, args.refute_rank
    serve(module_routes(PowerLinkModule(cfg), load_graph), args.port)


if __name__ == "__main__":
    main()
