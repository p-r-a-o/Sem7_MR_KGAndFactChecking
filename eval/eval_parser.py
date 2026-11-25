"""Evaluate the shared claim parser in isolation: claim text -> triple vs. the gold triple.

    python eval/eval_parser.py --dataset wn18rr --split valid --n 500

Runs services/claim_parser in-process (no HTTP) so the number is the parser's, not a module's.
The breakdown separates the independent failure modes, because they need different fixes:
  relation_acc    did the relation cue/lexical scorer pick the right relation?
  entity_set_acc  were both entities linked correctly, ignoring order?
  direction_err   right relation + right entity pair, but head/tail swapped (template direction)
  linking_err     at least one entity wrong (scaffolding leak or lemma ambiguity)
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "claim_parser"))

from common.datasets.base import get_claim_source, load_graph  # noqa: E402
from common.verbalize import disambiguated_labels  # noqa: E402
from claim_parsing import ParseError, build_resources, parse_claim  # noqa: E402


def evaluate(dataset: str, split: str, n: int) -> tuple[dict, list[dict]]:
    kg = load_graph(dataset)
    degrees = {e: len(nb) for e, nb in kg.neighbors.items()}
    aliases = disambiguated_labels(kg.entity_labels, kg.entity_descriptions)
    res = build_resources(kg.entity_labels, kg.relations, kg.train, degrees, aliases)
    label = kg.entity_labels.get

    counts: Counter[str] = Counter()
    rows: list[dict] = []
    for claim in itertools.islice(get_claim_source(dataset).claims(split), n):
        counts["total"] += 1
        gold = tuple(claim.gold_triple)
        row = {"claim": claim.claim, "gold_triple": list(gold)}
        try:
            out = parse_claim(claim.claim, res)
        except ParseError as e:
            counts["unparseable"] += 1
            rows.append({**row, "outcome": "unparseable", "error": str(e)})
            continue
        got = tuple(out["triple"])
        row["parsed_triple"] = list(got)
        rel_ok = got[1] == gold[1]
        set_ok = {got[0], got[2]} == {gold[0], gold[2]}
        counts["relation_acc"] += rel_ok
        counts["entity_set_acc"] += set_ok
        if got == gold:
            outcome = "exact"
        elif rel_ok and set_ok:
            outcome = "direction_err"
        elif not set_ok:
            outcome = "linking_err" if rel_ok else "linking+relation_err"
        else:
            outcome = "relation_err"
        counts[outcome] += 1
        row["outcome"] = outcome
        row["parsed_labels"] = [label(got[0], got[0]), got[1], label(got[2], got[2])]
        row["gold_labels"] = [label(gold[0], gold[0]), gold[1], label(gold[2], gold[2])]
        rows.append(row)

    total = counts["total"] or 1
    report = {"dataset": dataset, "split": split, "n": counts["total"],
              "exact_acc": round(counts["exact"] / total, 3),
              "relation_acc": round(counts["relation_acc"] / total, 3),
              "entity_set_acc": round(counts["entity_set_acc"] / total, 3),
              "breakdown": {k: counts[k] for k in
                            ("exact", "direction_err", "linking_err", "linking+relation_err",
                             "relation_err", "unparseable")}}
    return report, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=None, help="JSONL of per-claim parses for error analysis")
    args = ap.parse_args()

    report, rows = evaluate(args.dataset, args.split, args.n)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
