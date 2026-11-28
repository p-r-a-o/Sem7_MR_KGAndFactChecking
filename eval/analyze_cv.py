"""Error analysis of a claim-verification run (reads the JSONL written by eval_cv.py --out).

    python eval/analyze_cv.py cv_anyburl_test.jsonl --dataset fb15k237 --min-n 20
    python eval/analyze_cv.py cv_powerlink_test.jsonl --calibrate-from cv_powerlink_valid.jsonl

Sections (all also returned as JSON with --json):
  breakdowns   accuracy / balanced accuracy / NEI rate per relation, relation category (needs --dataset),
               corrupted side, claim parsed correctly or not, and gold label
  calibration  reliability bins, ECE and Brier of the support score; with --calibrate-from a temperature
               is fit on that (valid) run and the test run is re-scored after temperature scaling
  selective    error rate when only the most decisive claims are answered (risk-coverage), AURC,
               coverage at fixed error budgets
  explanations size of the proofs: steps per verdict, share without any step, share naming a rule
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cv_metrics import (NEI, REFUTED, SUPPORTED, apply_temperature,  # noqa: E402
                               brier, calibration_bins, coverage_at_risk, expected_calibration_error,
                               fit_temperature, group_metrics, quick_metrics, risk_at_coverage, risk_coverage,
                               summarize_cv)


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _binary(rows: list[dict]) -> tuple[list[bool], list[float]]:
    pairs = [(r["gold"] == SUPPORTED, r["score"]) for r in rows if r["gold"] in (SUPPORTED, REFUTED)]
    return [y for y, _ in pairs], [s for _, s in pairs]


def breakdowns(rows: list[dict], nei_policy: str, min_n: int, categories: dict[str, str]) -> dict:
    metric = lambda members: quick_metrics(members, nei_policy)       # noqa: E731
    out = {
        "by_gold": group_metrics(rows, lambda r: r["gold"], metric),
        "by_corrupted_side": group_metrics([r for r in rows if r["gold"] == REFUTED],
                                           lambda r: r.get("corrupted") or "?", metric),
        "by_parse": group_metrics(rows, lambda r: {True: "parsed_as_gold", False: "mis/unparsed"}.get(
            r.get("parse_ok"), "no gold triple"), metric),
        "by_relation": group_metrics(rows, lambda r: r.get("relation") or "(no gold triple)", metric, min_n),
    }
    if categories:
        out["by_relation_category"] = group_metrics(rows, lambda r: categories.get(r.get("relation"), "?"), metric)
    return out


def calibration(rows: list[dict], calibrate_from: list[dict] = None) -> dict:
    labels, scores = _binary(rows)
    out = {"ece": expected_calibration_error(labels, scores), "brier": brier(labels, scores),
           "bins": calibration_bins(labels, scores)}
    if calibrate_from:
        fit_labels, fit_scores = _binary(calibrate_from)
        temperature = fit_temperature(fit_labels, fit_scores)
        scaled = apply_temperature(scores, temperature)
        out["temperature"] = temperature
        out["after_temperature"] = {"ece": expected_calibration_error(labels, scaled),
                                    "brier": brier(labels, scaled)}
    return out


def selective(rows: list[dict]) -> dict:
    rc = risk_coverage([r["gold"] for r in rows], [r["score"] for r in rows])
    curve = rc["curve"]
    return {"n": rc["n"], "aurc": rc["aurc"],
            "risk_at_coverage": {f"{c:.1f}": risk_at_coverage(curve, c) for c in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)},
            "coverage_at_risk": {f"{e:.2f}": coverage_at_risk(curve, e) for e in (0.01, 0.05, 0.10, 0.20)}}


def explanation_stats(rows: list[dict]) -> dict:
    out = {}
    for verdict in (SUPPORTED, REFUTED, NEI):
        members = [r for r in rows if r["pred"] == verdict]
        if members:
            out[verdict] = {"n": len(members),
                            "mean_steps": sum(r.get("n_steps", 0) for r in members) / len(members),
                            "share_no_steps": sum(1 for r in members if not r.get("n_steps")) / len(members),
                            "share_with_rule": sum(1 for r in members if r.get("has_rule")) / len(members)}
    return out


def analyze(rows: list[dict], nei_policy: str = "miss", min_n: int = 20, categories: dict = None,
            calibrate_from: list[dict] = None, n_boot: int = 0) -> dict:
    return {"summary": summarize_cv(rows, nei_policy, n_boot),
            "breakdowns": breakdowns(rows, nei_policy, min_n, categories or {}),
            "calibration": calibration(rows, calibrate_from),
            "selective": selective(rows),
            "explanations": explanation_stats(rows)}


def _print_group(title: str, groups: dict) -> None:
    print(f"\n{title}")
    for key, m in groups.items():
        print(f"  {str(key)[:48]:<48} n={m['n']:<5} acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f} "
              f"nei={m['nei_rate']:.3f}")


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"accuracy={s['accuracy']:.3f} balanced={s['balanced_accuracy']:.3f} macroF1={s['macro_f1']:.3f} "
          f"AUC={s['auc']:.3f} AP={s['average_precision']:.3f} coverage={s['coverage']:.3f} "
          f"selective_acc={s['selective_accuracy']:.3f} AURC={s['aurc']:.3f}")
    for title, groups in report["breakdowns"].items():
        _print_group(title, groups)
    c = report["calibration"]
    print(f"\ncalibration: ECE={c['ece']:.3f} Brier={c['brier']:.3f}")
    for b in c["bins"]:
        print(f"  score [{b['low']:.1f},{b['high']:.1f}) n={b['n']:<5} mean_score={b['mean_score']:.3f} "
              f"frac_supported={b['frac_positive']:.3f}")
    if "temperature" in c:
        a = c["after_temperature"]
        print(f"  temperature T={c['temperature']:.2f} -> ECE={a['ece']:.3f} Brier={a['brier']:.3f}")
    sel = report["selective"]
    print(f"\nselective prediction: AURC={sel['aurc']:.3f}")
    print("  error rate at coverage:", {k: round(v, 3) for k, v in sel["risk_at_coverage"].items()})
    print("  coverage at error rate:", {k: round(v, 3) for k, v in sel["coverage_at_risk"].items()})
    print("\nexplanations:")
    for verdict, e in report["explanations"].items():
        print(f"  {verdict:<16} n={e['n']:<5} steps={e['mean_steps']:.2f} no_steps={e['share_no_steps']:.2f} "
              f"names_rule={e['share_with_rule']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="JSONL from eval_cv.py --out")
    ap.add_argument("--dataset", choices=["fb15k237", "wn18rr"], default=None,
                    help="load the graph to add per-relation-category (1-1/1-N/N-1/N-N) breakdowns")
    ap.add_argument("--calibrate-from", default=None, help="JSONL of a valid run to fit temperature scaling on")
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"])
    ap.add_argument("--min-n", type=int, default=20, help="hide relations with fewer claims")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the full report as JSON")
    args = ap.parse_args()

    categories = {}
    if args.dataset:
        from common.datasets.base import load_graph
        from common.lp_metrics import relation_categories
        categories = relation_categories(load_graph(args.dataset).train)
    report = analyze(load_rows(args.predictions), args.nei_policy, args.min_n, categories,
                     load_rows(args.calibrate_from) if args.calibrate_from else None, args.bootstrap)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
