"""Independently re-check eXpath proofs against the graph.

    python modules/expath/scripts/check_proof.py --dataset fb15k237 result.json [more.json ...]

Each file holds an LPResult or CVResult as returned by the API (JSON). For every explanation it
checks that each cited fact is in the graph (facts cited from an external source such as DBpedia are
reported as external) and recomputes SC / HC / support of the cited rule from the graph, comparing
them with the values stated in the explanation text. Exit code 1 if any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.datasets.base import load_graph  # noqa: E402
from common.serialization import explanation_from_dict  # noqa: E402
from modules.expath.graph import IndexedGraph  # noqa: E402
from modules.expath.proof import check_explanation  # noqa: E402


def explanations(d: dict):
    if "predictions" in d:
        for p in d["predictions"]:
            yield f"candidate {p['candidate']}", explanation_from_dict(p["explanation"])
    elif "explanation" in d:
        yield f"claim {d.get('claim', '')[:60]!r}", explanation_from_dict(d["explanation"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    g = IndexedGraph.from_kg(load_graph(args.dataset))
    failed = 0
    for f in args.files:
        for what, ex in explanations(json.loads(Path(f).read_text(encoding="utf-8"))):
            rep = check_explanation(g, ex)
            ok = all(x["in_graph"] or x["external"] for x in rep["facts"]) and rep.get("rule_ok", True)
            failed += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {what}: {len(rep['facts'])} facts "
                  f"({sum(x['in_graph'] for x in rep['facts'])} in graph), rule "
                  f"{'n/a' if 'rule_ok' not in rep else ('recomputed ' + rep['recomputed_metrics'] if rep['rule_ok'] else 'MISMATCH ' + str(rep.get('recomputed_metrics', rep.get('rule_error'))))}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
