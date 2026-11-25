"""Generate synthetic claims into data/claims/<dataset>/<split>.jsonl (HANDOFF.md sections 3 and 6).

Positive claims: verbalized triples from the chosen split (default: test).
Negative claims: a true triple with head OR tail swapped for a "type-consistent" entity, i.e. one that
appears in that same position of the same relation in TRAIN, and such that the result is in no split.

Usage: python scripts/generate_claims.py --dataset fb15k237 --split test --n 1000 --neg-per-pos 1
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets.base import KnowledgeGraph, load_graph  # noqa: E402
from common.paths import claims_dir  # noqa: E402
from common.schema import Triple  # noqa: E402
from common.verbalize import disambiguated_labels, verbalize  # noqa: E402


def build_pools(train: list[Triple]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """relation -> sorted unique heads / tails seen in train (the type-consistency pools)."""
    heads: dict[str, set[str]] = {}
    tails: dict[str, set[str]] = {}
    for h, r, t in train:
        heads.setdefault(r, set()).add(h)
        tails.setdefault(r, set()).add(t)
    return ({r: sorted(v) for r, v in heads.items()}, {r: sorted(v) for r, v in tails.items()})


def corrupt(triple: Triple, head_pool: dict[str, list[str]], tail_pool: dict[str, list[str]],
            known: set[Triple], rng: random.Random, max_tries: int = 50) -> Optional[tuple[Triple, str]]:
    """Swap head or tail for a type-consistent entity so the result is in no split. None if impossible."""
    h, r, t = triple
    for _ in range(max_tries):
        side = rng.choice(("head", "tail"))
        if side == "head":
            cand = (rng.choice(head_pool[r]), r, t)
        else:
            cand = (h, r, rng.choice(tail_pool[r]))
        if cand not in known and cand[0] != cand[2]:
            return cand, side
    return None


def generate(kg: KnowledgeGraph, split: str, n: int, neg_per_pos: int, seed: int,
             require_labels: bool) -> list[dict]:
    rng = random.Random(seed)
    triples = list(getattr(kg, split))
    rng.shuffle(triples)
    known = set(kg.train) | set(kg.valid) | set(kg.test)
    head_pool, tail_pool = build_pools(kg.train)
    # Ambiguous lemmas get a parenthetical gloss, so the claim text actually identifies the
    # entity it was generated from (see common/verbalize.disambiguated_labels).
    surfaces = disambiguated_labels(kg.entity_labels, kg.entity_descriptions)
    label = surfaces.get

    def readable(x: str) -> bool:
        return not require_labels or label(x, x) != x

    records: list[dict] = []
    for pos in triples:
        if len([r for r in records if r["label"] == "SUPPORTED"]) >= n:
            break
        h, r, t = pos
        if r not in head_pool or not (readable(h) and readable(t)):
            continue
        negs = []
        for _ in range(neg_per_pos):
            out = corrupt(pos, head_pool, tail_pool, known, rng)
            if out and readable(out[0][0]) and readable(out[0][2]):
                negs.append(out)
        if len(negs) < neg_per_pos:
            continue  # keep classes balanced
        records.append({"claim": verbalize(label(h, h), r, label(t, t), rng), "label": "SUPPORTED",
                        "gold_triple": [h, r, t], "source_triple": [h, r, t], "corrupted": None})
        for (nh, _, nt), side in negs:
            records.append({"claim": verbalize(label(nh, nh), r, label(nt, nt), rng), "label": "REFUTED",
                            "gold_triple": [nh, r, nt], "source_triple": [h, r, t], "corrupted": side})
    rng.shuffle(records)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=1000, help="number of positive claims")
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unlabeled", action="store_true", help="keep entities that have no readable name")
    args = ap.parse_args()

    graph = load_graph(args.dataset)
    recs = generate(graph, args.split, args.n, args.neg_per_pos, args.seed, not args.allow_unlabeled)
    out_path = claims_dir(args.dataset) / f"{args.split}.jsonl"
    write_jsonl(recs, out_path)
    print(f"Wrote {len(recs)} claims to {out_path}")
    if not recs:
        print("No claims written. Missing entity names? Re-run scripts/download_data.py or use --allow-unlabeled.")
