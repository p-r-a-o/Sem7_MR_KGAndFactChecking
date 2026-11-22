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
from common.server import serve  # noqa: E402
from common.services import get_service  # noqa: E402
from claim_parsing import EntityIndex, ParseError, parse_claim  # noqa: E402

_cache: dict[str, tuple[EntityIndex, list[str]]] = {}


def _resources(dataset: str) -> tuple[EntityIndex, list[str]]:
    if dataset not in _cache:
        try:
            kg = load_graph(dataset)
        except (FileNotFoundError, KeyError) as e:
            raise ModuleError("INTERNAL", f"Cannot load dataset {dataset!r}: {e}")
        degrees = {e: len(n) for e, n in kg.neighbors.items()}
        _cache[dataset] = (EntityIndex(kg.entity_labels, degrees), kg.relations)
    return _cache[dataset]


def handle_parse(body: dict) -> dict:
    claim, dataset = body.get("claim"), body.get("dataset")
    if not claim or not dataset:
        raise ModuleError("BAD_REQUEST", "Both 'claim' and 'dataset' are required.")
    index, relations = _resources(dataset)
    try:
        return parse_claim(claim, index, relations)
    except ParseError as e:
        raise ModuleError("UNPARSEABLE_CLAIM", str(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=get_service("claim_parser").port)
    args = ap.parse_args()
    serve({("GET", "/health"): lambda _: {"name": "claim_parser", "status": "ok"},
           ("POST", "/parse_claim"): handle_parse}, args.port)
