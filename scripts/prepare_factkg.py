"""Convert FactKG (claims + DBpedia 2015 "light" graph) into the shared data layout. Stdlib-only.

Input (download by hand or with gdown from the FactKG README's Google Drive folder,
https://github.com/jiho283/FactKG), placed in data/factkg_raw/:
    factkg_train.pickle  factkg_dev.pickle  factkg_test.pickle   (inside factkg.zip)
    dbpedia_2015_undirected_light.pickle                          (~1.5 GB; the full 5.5 GB graph is not needed)

Output:
    data/factkg/train.txt       DBpedia triples (forward direction only), see "Graph" below
    data/factkg/valid.txt       empty: FactKG has no held-out triples, only held-out claims
    data/factkg/test.txt        empty
    data/factkg/entity2text.txt "<DBpedia id>\t<readable label>"
    data/claims/factkg/{train,valid,test}.jsonl   (FactKG dev -> "valid")
        {"claim", "label": SUPPORTED|REFUTED, "gold_triple": [h, r, t] | null,
         "entity_set": [...], "evidence": {...} (train/valid only), "types": [...]}

Graph: the union of every edge incident to an entity that occurs in any FactKG claim (all splits).
That keeps every 1- and 2-hop path between claim entities, which is what FactKG's evidence uses
(its reasoning types are one-hop, conjunction, existence, multi-hop and negation). `--hops 2` also
adds edges between those neighbours (capped per entity); it is much larger.

The pickles are loaded with a restricted unpickler that refuses every class lookup, so a tampered
file cannot run code: only dict / list / str / int / float / bool / None come through.

Usage: python scripts/prepare_factkg.py [--raw data/factkg_raw] [--hops 1] [--max-degree 2000]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import claims_dir, data_root  # noqa: E402

SPLITS = {"train": "factkg_train.pickle", "valid": "factkg_dev.pickle", "test": "factkg_test.pickle"}
INV = "~"  # FactKG/DBpedia marks reverse edges with a "~" prefix


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: data-only pickle expected")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return _DataOnlyUnpickler(f).load()


def entity_label(entity: str) -> str:
    """'Pleasant_Springs,_Wisconsin' -> 'Pleasant Springs, Wisconsin'; '"94802.4"' -> '94802.4'."""
    if len(entity) >= 2 and entity[0] == entity[-1] == '"':
        return entity[1:-1]
    return entity.replace("_", " ")


def clean(text: str) -> str:
    """Tabs/newlines would break the TSV files."""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def gold_triple(entity_set: list[str], evidence: dict) -> Optional[list[str]]:
    """Best-effort single triple behind a claim, from FactKG's per-entity evidence relation paths.

    A one-hop evidence path [rel] on entity a paired with [~rel] on entity b gives (a, rel, b).
    Returns None when no such pair exists (e.g. existence or multi-hop claims, and the test split,
    which ships no evidence). Only used for reporting; the modules never see it.
    """
    for a in entity_set:
        for path in evidence.get(a, []):
            if len(path) != 1 or path[0].startswith(INV):
                continue
            rel = path[0]
            for b in entity_set:
                if b != a and [INV + rel] in evidence.get(b, []):
                    return [a, rel, b]
    return None


def edges_of(dbpedia: dict, entity: str, max_degree: int):
    """Forward-direction (h, r, t) triples for every edge touching `entity`, capped per relation."""
    for rel, tails in dbpedia.get(entity, {}).items():
        for other in tails[:max_degree]:
            if rel.startswith(INV):
                yield other, rel[1:], entity
            else:
                yield entity, rel, other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(data_root() / "factkg_raw"))
    ap.add_argument("--graph", default="dbpedia_2015_undirected_light.pickle")
    ap.add_argument("--hops", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max-degree", type=int, default=2000, help="max tails kept per (entity, relation)")
    args = ap.parse_args()
    raw, out = Path(args.raw), data_root() / "factkg"
    out.mkdir(parents=True, exist_ok=True)
    claims_dir("factkg").mkdir(parents=True, exist_ok=True)

    claim_entities: set[str] = set()
    for split, fname in SPLITS.items():
        data = load_pickle(raw / fname)
        n = 0
        with open(claims_dir("factkg") / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for claim, d in data.items():
                ents = list(d.get("Entity_set", []))
                claim_entities.update(ents)
                row = {"claim": claim, "label": "SUPPORTED" if d["Label"][0] else "REFUTED",
                       "gold_triple": gold_triple(ents, d.get("Evidence", {})),
                       "entity_set": ents, "types": d.get("types", [])}
                if "Evidence" in d:
                    row["evidence"] = d["Evidence"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"claims {split}: {n}")
    print(f"distinct claim entities: {len(claim_entities)}")

    t0 = time.time()
    print(f"loading {args.graph} (takes a few minutes and several GB of RAM)...")
    dbpedia = load_pickle(raw / args.graph)
    print(f"  {len(dbpedia)} entities loaded in {time.time() - t0:.0f}s")

    triples: set[tuple[str, str, str]] = set()
    for e in claim_entities:
        triples.update(edges_of(dbpedia, e, args.max_degree))
    if args.hops == 2:
        frontier = {x for h, _, t in triples for x in (h, t)} - claim_entities
        keep = frontier | claim_entities
        for e in frontier:
            triples.update(tr for tr in edges_of(dbpedia, e, args.max_degree) if tr[0] in keep and tr[2] in keep)
    missing = sum(1 for e in claim_entities if e not in dbpedia)
    del dbpedia

    entities: set[str] = set()
    with open(out / "train.txt", "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            h, r, t = clean(h), clean(r), clean(t)
            f.write(f"{h}\t{r}\t{t}\n")
            entities.update((h, t))
    for split in ("valid", "test"):
        (out / f"{split}.txt").write_text("", encoding="utf-8")
    with open(out / "entity2text.txt", "w", encoding="utf-8") as f:
        for e in sorted(entities):
            f.write(f"{e}\t{clean(entity_label(e))}\n")
    stats = {"triples": len(triples), "entities": len(entities),
             "relations": len({r for _, r, _ in triples}), "claim_entities": len(claim_entities),
             "claim_entities_missing_from_graph": missing, "hops": args.hops, "max_degree": args.max_degree}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
