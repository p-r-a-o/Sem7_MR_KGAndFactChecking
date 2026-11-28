"""Compare modules on identical claims or queries.

Claim verification (JSONL files from eval_cv.py --out, all generated from the same claim file):

    python eval/compare_modules.py cv AnyBURL=cv_anyburl.jsonl PowerLink=cv_powerlink.jsonl \\
        --cascade AnyBURL PowerLink --out compare_cv.json

  prints per-module metrics, pairwise kappa / McNemar, the Venn of who is right, the oracle upper bound,
  and the vote / soft-average / cascade ensembles (with their metrics) so you can see whether a hybrid
  beats its parts.

Link prediction (JSONL files from eval_lp.py --dump-queries, evaluated with the same --seeds and --n):

    python eval/compare_modules.py lp AnyBURL=lp_anyburl_queries.jsonl PowerLink=lp_powerlink_queries.jsonl

  paired bootstrap of the reciprocal-rank difference per pair, with win/tie/loss counts.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cv_metrics import summarize_cv  # noqa: E402
from common.ensemble import align, cascade, complementarity, pairwise, soft_average, vote  # noqa: E402
from common.lp_metrics import reciprocal_ranks  # noqa: E402
from common.stats import paired_bootstrap  # noqa: E402

CV_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "auc", "ece", "aurc", "coverage")


def parse_runs(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=FILE, got {item!r}")
        runs[name] = Path(path)
    return runs


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_cv(runs: dict[str, list[dict]], nei_policy: str, cascades: list[list[str]], min_conf: float,
               n_boot: int) -> dict:
    aligned = align(runs)
    report = {"n_aligned": len(next(iter(aligned.values()))) if aligned else 0,
              "modules": {m: summarize_cv(rows, nei_policy, n_boot) for m, rows in aligned.items()},
              "pairwise": pairwise(aligned, nei_policy),
              "complementarity": complementarity(aligned, nei_policy)}
    ensembles = {"vote": vote(aligned), "soft_average": soft_average(aligned)}
    for order in cascades:
        ensembles["cascade:" + ">".join(order)] = cascade(aligned, order, min_conf)
    report["ensembles"] = {name: summarize_cv(rows, nei_policy, n_boot) for name, rows in ensembles.items()}
    return report


def compare_lp(runs: dict[str, list[dict]], n_boot: int) -> dict:
    keyed = {m: {(r["head"], r["relation"], r["tail"], r["side"]): r for r in rows} for m, rows in runs.items()}
    shared = sorted(set.intersection(*(set(k) for k in keyed.values())), key=str)
    rr = {m: reciprocal_ranks([keyed[m][k]["rank_filtered"] for k in shared]) for m in keyed}
    out = {"n_aligned": len(shared), "mrr_filtered": {m: sum(v) / len(v) if v else 0.0 for m, v in rr.items()},
           "pairwise": []}
    for a, b in itertools.combinations(rr, 2):
        wins = sum(x > y for x, y in zip(rr[a], rr[b]))
        losses = sum(x < y for x, y in zip(rr[a], rr[b]))
        out["pairwise"].append({"a": a, "b": b, "a_wins": wins, "b_wins": losses,
                                "ties": len(shared) - wins - losses,
                                **{f"rr_diff_{k}": v for k, v in paired_bootstrap(rr[a], rr[b], n_boot).items()}})
    return out


def print_cv(report: dict) -> None:
    print(f"{report['n_aligned']} claims shared by all runs\n")
    header = f"{'':<28}" + "".join(f"{k:>18}" for k in CV_KEYS)
    print(header + "\n" + "-" * len(header))
    for name, m in {**report["modules"], **report["ensembles"]}.items():
        print(f"{name:<28}" + "".join(f"{m[k]:>18.3f}" for k in CV_KEYS))
    c = report["complementarity"]
    print(f"\noracle (any module right): {c['oracle_accuracy']:.3f}   exclusive wins: {c['exclusive']}")
    print("who is right:", c["venn"])
    for p in report["pairwise"]:
        print(f"{p['a']} vs {p['b']}: kappa={p['kappa']:.3f} agree={p['agreement']:.3f} "
              f"only_{p['a']}={p['only_a_correct']} only_{p['b']}={p['only_b_correct']} "
              f"McNemar p={p['mcnemar_p']:.4f} ({p['mcnemar_method']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["cv", "lp"])
    ap.add_argument("runs", nargs="+", help="NAME=FILE.jsonl, at least two")
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"])
    ap.add_argument("--cascade", nargs="+", action="append", default=[],
                    help="module names in query order; repeatable for several cascades")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="cascade: confidence needed to stop")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = {name: read_jsonl(path) for name, path in parse_runs(args.runs).items()}
    if len(runs) < 2:
        raise SystemExit("need at least two runs to compare")

    if args.task == "cv":
        report = compare_cv(runs, args.nei_policy, args.cascade, args.min_confidence, args.bootstrap)
        print_cv(report)
    else:
        report = compare_lp(runs, args.bootstrap)
        print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
