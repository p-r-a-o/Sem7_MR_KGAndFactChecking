"""Claim parser HTTP service. POST /parse_claim {"claim", "dataset"} -> {"triple", "linked_entities"}.

Run: python services/claim_parser/api.py [--port 8100]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets.base import load_graph  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.verbalize import disambiguated_labels  # noqa: E402
from common.server import serve  # noqa: E402
from common.services import get_service  # noqa: E402
from claim_parsing import ParserResources, ParseError, build_resources, parse_claim  # noqa: E402

_cache: dict[str, ParserResources] = {}


def _resources(dataset: str) -> ParserResources:
    if dataset not in _cache:
        try:
            kg = load_graph(dataset)
        except (FileNotFoundError, KeyError) as e:
            raise ModuleError("INTERNAL", f"Cannot load dataset {dataset!r}: {e}")
        degrees = {e: len(n) for e, n in kg.neighbors.items()}
        aliases = disambiguated_labels(kg.entity_labels, kg.entity_descriptions)
        _cache[dataset] = build_resources(kg.entity_labels, kg.relations, kg.train, degrees,
                                          aliases)
    return _cache[dataset]


def handle_parse(body: dict) -> dict:
    claim, dataset = body.get("claim"), body.get("dataset")
    if not claim or not dataset:
        raise ModuleError("BAD_REQUEST", "Both 'claim' and 'dataset' are required.")
    try:
        return parse_claim(claim, _resources(dataset))
    except ParseError as e:
        raise ModuleError("UNPARSEABLE_CLAIM", str(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("claim_parser").port)
    args = ap.parse_args()
    serve({("GET", "/health"): lambda _: {"name": "claim_parser", "status": "ok"},
           ("POST", "/parse_claim"): handle_parse}, args.port)
