"""PGR HTTP service. Run from the repo root (stdlib only, no extra packages):

    set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...")
    python modules/pgr/api.py [--model gemini-2.5-flash] [--rpm 8]

Claim verification on FactKG. LLM replies are cached in artifacts/pgr/llm_cache.jsonl.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.datasets.base import load_graph  # noqa: E402
from common.server import module_routes, serve  # noqa: E402
from common.services import get_service  # noqa: E402
from modules.pgr.config import PGRConfig  # noqa: E402
from modules.pgr.module import PGRModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("pgr").port)
    ap.add_argument("--model", default=None, help="LLM model id (default: $PGR_MODEL or gemini-2.5-flash)")
    ap.add_argument("--rpm", type=float, default=None, help="max LLM requests per minute (free-tier cap)")
    args = ap.parse_args()
    cfg = PGRConfig()
    if args.model:
        cfg.llm.model = args.model
    if args.rpm:
        cfg.llm.requests_per_minute = args.rpm
    serve(module_routes(PGRModule(cfg), load_graph), args.port)


if __name__ == "__main__":
    main()
