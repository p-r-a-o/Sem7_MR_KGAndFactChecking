"""Generate synthetic claims into data/claims/<dataset>/<split>.jsonl (HANDOFF.md sections 3 and 6).

Positive claims: verbalized triples from the chosen split (default: test).
Negative claims: a true triple with head OR tail swapped for a "type-consistent" entity, i.e. one that
appears in that same position of the same relation in TRAIN, and such that the result is in no split.

Usage: python scripts/generate_claims.py --dataset fb15k237 --split test --n 1000 --neg-per-pos 1

Hard negatives: `--neg-mode frequency` draws the replacement entity in proportion to how often it fills
that slot of the relation in TRAIN (the default `uniform` draws every pool entity equally often, so
corrupted entities skew towards rare ones and a popularity-only classifier can spot them, see
eval/eval_baselines.py). Write such a set next to the default one with `--tag hard`
(-> data/claims/<dataset>/<split>_hard.jsonl); eval_cv.py reads it with `--tag hard`.
Popularity-matching is not automatic: replacements always come from entities seen in TRAIN, whereas genuine
held-out triples often involve entities with no train count in that slot, so power 1 can overshoot (negatives
end up MORE popular than positives). Sweep `--freq-power` and keep the value whose popularity-only AUC in
eval/eval_baselines.py is closest to 0.5. Popular replacements are also more likely to be true-but-unrecorded,
so audit a sample of negatives by hand.
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


def build_frequency_weights(train: list[Triple], head_pool: dict[str, list[str]], tail_pool: dict[str, list[str]],
                            power: float = 1.0) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """relation -> training count ** power of each pool entity (aligned with `build_pools` order).

    power=1 draws in proportion to frequency, power=0 is the uniform draw, in between tempers the hubs.
    """
    head_counts: dict[str, dict[str, int]] = {}
    tail_counts: dict[str, dict[str, int]] = {}
    for h, r, t in train:
        head_counts.setdefault(r, {}).setdefault(h, 0)
        head_counts[r][h] += 1
        tail_counts.setdefault(r, {}).setdefault(t, 0)
        tail_counts[r][t] += 1
    return ({r: [head_counts[r][e] ** power for e in pool] for r, pool in head_pool.items()},
            {r: [tail_counts[r][e] ** power for e in pool] for r, pool in tail_pool.items()})


def corrupt(triple: Triple, head_pool: dict[str, list[str]], tail_pool: dict[str, list[str]],
            known: set[Triple], rng: random.Random, max_tries: int = 50,
            head_weights: Optional[dict[str, list[float]]] = None,
            tail_weights: Optional[dict[str, list[float]]] = None) -> Optional[tuple[Triple, str]]:
    """Swap head or tail for a type-consistent entity so the result is in no split. None if impossible.

    Without weights the replacement is uniform over the pool; with weights (see `build_frequency_weights`)
    it is drawn in proportion to the entity's training frequency in that slot.
    """
    h, r, t = triple

    def pick(pool: list[str], weights: Optional[dict[str, list[float]]]) -> str:
        return rng.choice(pool) if weights is None else rng.choices(pool, weights=weights[r])[0]

    for _ in range(max_tries):
        side = rng.choice(("head", "tail"))
        if side == "head":
            cand = (pick(head_pool[r], head_weights), r, t)
        else:
            cand = (h, r, pick(tail_pool[r], tail_weights))
        if cand not in known and cand[0] != cand[2]:
            return cand, side
    return None


def generate(kg: KnowledgeGraph, split: str, n: int, neg_per_pos: int, seed: int,
             require_labels: bool, neg_mode: str = "uniform", freq_power: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    triples = list(getattr(kg, split))
    rng.shuffle(triples)
    known = set(kg.train) | set(kg.valid) | set(kg.test)
    head_pool, tail_pool = build_pools(kg.train)
    head_weights = tail_weights = None
    if neg_mode == "frequency":
        head_weights, tail_weights = build_frequency_weights(kg.train, head_pool, tail_pool, freq_power)
    elif neg_mode != "uniform":
        raise ValueError("neg_mode must be 'uniform' or 'frequency'")
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
            out = corrupt(pos, head_pool, tail_pool, known, rng,
                          head_weights=head_weights, tail_weights=tail_weights)
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
    ap.add_argument("--neg-mode", choices=["uniform", "frequency"], default="uniform",
                    help="how replacement entities are drawn; 'frequency' gives harder, popularity-matched negatives")
    ap.add_argument("--freq-power", type=float, default=1.0,
                    help="with --neg-mode frequency: weight = train count ** power (0 = uniform). Pick the power "
                         "whose popularity-only AUC in eval_baselines.py is closest to 0.5")
    ap.add_argument("--tag", default=None, help="write <split>_<tag>.jsonl instead of <split>.jsonl (e.g. hard)")
    ap.add_argument("--allow-unlabeled", action="store_true", help="keep entities that have no readable name")
    args = ap.parse_args()

    graph = load_graph(args.dataset)
    recs = generate(graph, args.split, args.n, args.neg_per_pos, args.seed, not args.allow_unlabeled,
                    args.neg_mode, args.freq_power)
    out_path = claims_dir(args.dataset) / f"{args.split}{'_' + args.tag if args.tag else ''}.jsonl"
    write_jsonl(recs, out_path)
    print(f"Wrote {len(recs)} claims to {out_path}")
    if not recs:
        print("No claims written. Missing entity names? Re-run scripts/download_data.py or use --allow-unlabeled.")
