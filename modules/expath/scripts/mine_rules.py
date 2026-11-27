"""Offline: mine eXpath CP and Uc rules for a dataset -> artifacts/expath/<dataset>/rules.jsonl.

    python modules/expath/scripts/mine_rules.py --dataset fb15k237 [--samples 200] [--no-uc]

Runs on the TRAIN split only. Prints per-relation progress; writes mining.json with the settings
and rule counts for the report.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.datasets.base import load_graph  # noqa: E402
from modules.expath.config import ExpathConfig  # noqa: E402
from modules.expath.graph import IndexedGraph  # noqa: E402
from modules.expath.rules import mine_cp_rules, mine_uc_rules, save_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", type=int, default=None, help="facts sampled per relation for path aggregation")
    ap.add_argument("--max-path-len", type=int, default=None)
    ap.add_argument("--relations", default=None, help="comma-separated relation names (default: all)")
    ap.add_argument("--no-uc", action="store_true")
    ap.add_argument("--no-cp", action="store_true")
    args = ap.parse_args()

    cfg = ExpathConfig()
    if args.samples:
        cfg.mining.samples_per_relation = args.samples
    if args.max_path_len:
        cfg.mining.max_path_len = args.max_path_len
    t0 = time.time()
    kg = load_graph(args.dataset)
    g = IndexedGraph.from_kg(kg)
    del kg
    print(f"graph: {g.N} entities, {g.R} relations, {len(g.triples)} train facts ({time.time() - t0:.0f}s)")
    rels = None
    if args.relations:
        rels = [g.rel2idx[r] for r in args.relations.split(",")]

    rules = []
    t1 = time.time()
    if not args.no_cp:
        rules += mine_cp_rules(g, cfg.mining, cfg.quality, rels)
    t2 = time.time()
    if not args.no_uc:
        rules += mine_uc_rules(g, cfg.mining, cfg.quality)
    t3 = time.time()

    out = cfg.artifacts(args.dataset)
    n = save_rules(g, rules, out / "rules.jsonl")
    info = {"dataset": args.dataset, "rules": n, "cp_rules": sum(r.kind == "CP" for r in rules),
            "uc_rules": sum(r.kind == "Uc" for r in rules), "cp_seconds": round(t2 - t1),
            "uc_seconds": round(t3 - t2), "mining": cfg.mining.__dict__, "quality": cfg.quality.__dict__}
    (out / "mining.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
