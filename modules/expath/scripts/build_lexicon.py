"""Offline: learn P(relation | claim word) from FactKG TRAIN claims and their gold evidence paths.

    python modules/expath/scripts/build_lexicon.py --dataset factkg
-> artifacts/expath/<dataset>/lexicon.json (read by the multi-entity claim verifier).
Only the train split is used, so valid/test claims stay unseen.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.paths import claims_dir  # noqa: E402
from modules.expath.claims import Lexicon  # noqa: E402
from modules.expath.config import ExpathConfig  # noqa: E402


def examples(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ev = d.get("evidence") or {}
            rels = {r for paths in ev.values() for p in paths for r in p}
            if rels:
                yield d["claim"], [e.strip('"').replace("_", " ") for e in d.get("entity_set", [])], rels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="factkg")
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()
    lex = Lexicon.build(examples(claims_dir(args.dataset) / "train.jsonl"), args.min_count)
    out = ExpathConfig().artifacts(args.dataset) / "lexicon.json"
    lex.save(out)
    print(f"{len(lex.word)} words -> {out}")
    for w in ("born", "birth", "capital", "leader", "spouse", "successor", "weigh"):
        top = sorted(lex.word_rel.get(w, {}).items(), key=lambda kv: -kv[1])[:3]
        print(f"  {w}: {[(r, round(c / lex.word[w], 2)) for r, c in top] if top else '-'}")


if __name__ == "__main__":
    main()
