"""Run eXpath on exactly the claims PGR has finished, and print both side by side per reasoning type.

    python modules/pgr/scripts/compare_expath.py [--split test]

Needs no LLM. Loads FactKG once (~4 GB, a minute or two); close the app / servers first if memory is low.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.datasets.base import load_graph  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from modules.expath.module import ExpathModule  # noqa: E402
from modules.pgr.config import PGRConfig  # noqa: E402
from modules.pgr.scripts.run_eval import TYPES, load_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--pool", action="store_true", help="all models' result files together")
    args = ap.parse_args()
    path = PGRConfig().eval_path(args.split)
    paths = sorted(path.parent.glob(f"eval_{args.split}_*.jsonl")) if args.pool else [path]
    rows = load_results(paths)
    print(f"{len(rows)} PGR results from: {', '.join(p.name for p in paths)}")
    m = ExpathModule()
    m.load(load_graph("factkg"))
    # FactKG is binary (SUPPORTED / REFUTED): the team's eval scores NOT_ENOUGH_INFO as REFUTED
    # (common/metrics.py, nei="refuted"); "strict" counts it as wrong. Both are shown for both methods.
    binary = lambda v: "SUPPORTED" if v == "SUPPORTED" else "REFUTED"   # noqa: E731
    tot, cnt = Counter(), {k: Counter() for k in ("pgr", "pgr_s", "exp", "exp_s")}
    nei = Counter()
    for r in rows:
        try:
            ev = m.verify_claim(r["claim"]).verdict
        except ModuleError:
            ev = "NOT_ENOUGH_INFO"
        r["expath"] = ev
        nei["eXpath"] += ev == "NOT_ENOUGH_INFO"
        nei["PGR"] += r["pred"] == "NOT_ENOUGH_INFO"
        for k in (r["type"], "ALL"):
            tot[k] += 1
            cnt["pgr"][k] += binary(r["pred"]) == r["label"]
            cnt["pgr_s"][k] += r["pred"] == r["label"]
            cnt["exp"][k] += binary(ev) == r["label"]
            cnt["exp_s"][k] += ev == r["label"]
    print(f"{'type':12} {'n':>4} {'PGR':>7} {'eXpath':>7}   | strict (NEI = wrong): {'PGR':>6} {'eXpath':>7}")
    for t in TYPES + ["ALL"]:
        if tot[t]:
            f = lambda k: cnt[k][t] / tot[t]   # noqa: E731
            print(f"{t:12} {tot[t]:4d} {f('pgr'):7.3f} {f('exp'):7.3f}   | {'':21}{f('pgr_s'):6.3f} {f('exp_s'):7.3f}")
    print(f"NOT_ENOUGH_INFO verdicts: PGR {nei['PGR']}, eXpath {nei['eXpath']} (scored as REFUTED in the left columns, "
          f"as in the team's eval_cv.py)")
    out = path.parent / f"compare_{args.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("claim", "type", "label", "model", "pred", "expath")},
                               ensure_ascii=False) + "\n")
    print(f"per-claim verdicts of both methods: {out}")


if __name__ == "__main__":
    main()
