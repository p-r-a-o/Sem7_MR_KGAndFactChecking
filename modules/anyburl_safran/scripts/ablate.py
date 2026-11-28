"""Ablation study for the AnyBURL+SAFRAN module (offline, in-process, no HTTP, no claim parser).

    python modules/anyburl_safran/scripts/ablate.py --dataset wn18rr --split valid --n-claims 500 --n-lp 200
    python modules/anyburl_safran/scripts/ablate.py --dataset fb15k237 --family aggregation smoothing \\
        rule_kind --out ablation_fb15k237.json
    python modules/anyburl_safran/scripts/ablate.py --dataset fb15k237 --family rules_file \\
        --rules-files rules_100s.txt rules_1000s.txt        # mining-time ablation

Every setting is scored on link prediction (filtered MRR / Hits@10 over --n-lp test triples, both
directions) and claim verification (verdicts from the gold triple of --n-claims claims, so the parser
is not part of the ablation): accuracy, balanced accuracy, ROC-AUC, coverage. Ablate on `valid` and
report the chosen setting once on `test`.

Families (--family, default: aggregation dedupe smoothing rule_kind rule_length max_rules):
  aggregation  max (AnyBURL) vs noisy-or (SAFRAN-style)          [apply time]
  dedupe       ignore / count rules that fired on identical evidence  [apply time]
  smoothing    UNSEEN_NEGATIVE_EXAMPLES 0/1/5/10 (rule confidence = correct / (predicted + u))  [reload]
  min_correct  drop rules with fewer than 1/2/5 correct train predictions  [reload]
  rule_kind    all rules / cyclic only / acyclic only / no zero rules  [apply time, filters the loaded set]
  rule_length  keep body length <= 1 / 2 / 3  [apply time]
  max_rules    rules applied per query: 500 / 2000 / 10000  [apply time]
  rules_file   one setting per file given in --rules-files  [reload]
  drop_edges   remove 10/30/50% of TRAIN edges at inference, rules stay as mined: robustness to an
               incomplete graph  [reload]
Settings that share their load-time options reuse the loaded module, so a family costs one rule load
(~35s for a 1000s FB15k-237 snapshot) per distinct reload setting, not per row.
"""
import argparse
import dataclasses
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.cv_metrics import summarize_cv, support_score  # noqa: E402
from common.datasets.base import KnowledgeGraph, get_claim_source, load_graph  # noqa: E402
from common.datasets.files import build_neighbors  # noqa: E402
from common.errors import ModuleError  # noqa: E402
from common.lp_metrics import AnswerIndex, filtered_rank, requested_k, sample_queries  # noqa: E402
from common.metrics import rank_metrics  # noqa: E402
from modules.anyburl_safran import rules as rules_mod  # noqa: E402
from modules.anyburl_safran.config import AnyburlConfig  # noqa: E402
from modules.anyburl_safran.module import AnyburlSafranModule  # noqa: E402

DEFAULT_FAMILIES = ("aggregation", "dedupe", "smoothing", "rule_kind", "rule_length", "max_rules")


@dataclass
class Setting:
    family: str
    name: str
    edits: dict = field(default_factory=dict)            # dotted config path -> value, e.g. "filters.min_correct_predictions"
    keep_rule: Optional[Callable] = None                  # post-load filter on rules
    drop_edges: float = 0.0                               # fraction of TRAIN edges removed at inference


def with_path(obj, path: list[str], value):
    """dataclasses.replace along a dotted path; works for frozen and mutable config dataclasses alike."""
    if len(path) == 1:
        return dataclasses.replace(obj, **{path[0]: value})
    return dataclasses.replace(obj, **{path[0]: with_path(getattr(obj, path[0]), path[1:], value)})


def make_config(base: AnyburlConfig, edits: dict) -> AnyburlConfig:
    cfg = dataclasses.replace(base, filters=dataclasses.replace(base.filters),
                              thresholds=dataclasses.replace(base.thresholds))
    for dotted, value in edits.items():
        cfg = with_path(cfg, dotted.split("."), value)
    return cfg


def build_settings(families: list[str], rules_files: list[str]) -> list[Setting]:
    out: list[Setting] = []
    for family in families:
        if family == "aggregation":
            out += [Setting(family, v, {"aggregation": v}) for v in ("max", "noisy-or")]
        elif family == "dedupe":
            out += [Setting(family, f"dedupe={v}", {"dedupe_groundings": v}) for v in (True, False)]
        elif family == "smoothing":
            out += [Setting(family, f"unseen_negatives={u}", {"filters.unseen_negative_examples": u})
                    for u in (0, 1, 5, 10)]
        elif family == "min_correct":
            out += [Setting(family, f"min_correct={m}", {"filters.min_correct_predictions": m}) for m in (1, 2, 5)]
        elif family == "rule_kind":
            out += [Setting(family, "all"),
                    Setting(family, "cyclic_only", keep_rule=lambda r: r.kind == "cyclic"),
                    Setting(family, "acyclic_only", keep_rule=lambda r: r.kind in ("tail_constant", "head_constant")),
                    Setting(family, "no_zero_rules", keep_rule=lambda r: r.kind != "zero")]
        elif family == "rule_length":
            out += [Setting(family, f"body<={n}", keep_rule=lambda r, n=n: r.length <= n) for n in (1, 2, 3)]
        elif family == "max_rules":
            out += [Setting(family, f"max_rules={m}", {"limits.max_rules": m, "cv_max_rules": m,
                                                         "limits.saturation_candidates": max(100, m // 10)})
                    for m in (500, 2000, 10000)]
        elif family == "rules_file":
            if not rules_files:
                raise SystemExit("--family rules_file needs --rules-files")
            out += [Setting(family, Path(f).name, {"rules_file": f}) for f in rules_files]
        elif family == "drop_edges":
            out += [Setting(family, f"drop={int(p * 100)}%", drop_edges=p) for p in (0.1, 0.3, 0.5)]
        else:
            raise SystemExit(f"unknown family {family!r}")
    return out


def drop_train_edges(kg: KnowledgeGraph, fraction: float, seed: int = 0) -> KnowledgeGraph:
    """The same graph with a random `fraction` of TRAIN edges removed (valid/test untouched)."""
    if fraction <= 0:
        return kg
    kept = [t for t in kg.train if random.Random(f"{seed}|{t}").random() >= fraction]
    return dataclasses.replace(kg, train=kept, neighbors=build_neighbors(kept))


def load_key(cfg: AnyburlConfig, setting: Setting) -> str:
    return repr((cfg.filters, cfg.rules_file, cfg.thresholds.functional_violation_ratio, setting.drop_edges))


def evaluate_lp(module, kg: KnowledgeGraph, split: str, n: int, k: int, answers: AnswerIndex, seed: int) -> dict:
    ranks, start = [], time.time()
    for triple in sample_queries(list(getattr(kg, split)), n, seed):
        h, r, t = triple
        for side, (qh, qt, gold) in {"tail": (h, None, t), "head": (None, t, h)}.items():
            others = answers.others(triple, side)
            try:
                cands = [p.candidate for p in module.predict_link(qh, r, qt, requested_k(k, len(others))).predictions]
            except ModuleError:
                ranks.append(None)
                continue
            ranks.append(filtered_rank(cands, gold, others))
    m = rank_metrics(ranks)
    return {"mrr": m["mrr"], "hits@10": m["hits@10"], "lp_ms": 1000 * (time.time() - start) / max(1, len(ranks))}


def evaluate_cv(module, claims: list, nei_policy: str) -> dict:
    rows, start = [], time.time()
    for c in claims:
        try:
            res = module.decide(c.claim, module.score_triple(c.gold_triple))
        except ModuleError:
            rows.append({"gold": c.label, "pred": "NOT_ENOUGH_INFO", "score": 0.5})
            continue
        rows.append({"gold": c.label, "pred": res.verdict, "score": support_score(res.verdict, res.confidence)})
    s = summarize_cv(rows, nei_policy)
    return {"accuracy": s["accuracy"], "balanced_accuracy": s["balanced_accuracy"], "auc": s["auc"],
            "coverage": s["coverage"], "cv_ms": 1000 * (time.time() - start) / max(1, len(rows))}


def run_ablation(kg: KnowledgeGraph, claims: list, settings: list[Setting], base: AnyburlConfig, split: str = "valid",
                 n_lp: int = 200, k: int = 10, nei_policy: str = "miss", seed: int = 0) -> list[dict]:
    """Evaluate every setting. Settings are ordered so those sharing a load key reuse one loaded module."""
    answers = AnswerIndex(kg.train, kg.valid, kg.test)
    cfgs = [make_config(base, s.edits) for s in settings]
    order = sorted(range(len(settings)), key=lambda i: load_key(cfgs[i], settings[i]))
    results: dict[int, dict] = {}
    current_key, module, base_rules = None, None, None
    for i in order:
        setting, cfg = settings[i], cfgs[i]
        key = load_key(cfg, setting)
        if key != current_key:                                   # (re)load only when a load-time option changed
            module = AnyburlSafranModule(cfg)
            module.load(drop_train_edges(kg, setting.drop_edges, seed))
            base_rules = module._st.ruleset                      # noqa: SLF001 - ablation needs the loaded set
            current_key = key
        module.cfg = cfg
        kept = base_rules if setting.keep_rule is None else rules_mod.RuleSet(
            rules=[r for r in base_rules.rules if setting.keep_rule(r)])
        module._st = dataclasses.replace(module._st, ruleset=kept)   # noqa: SLF001
        lp = evaluate_lp(module, kg, split, n_lp, k, answers, seed) if n_lp else {}
        cv = evaluate_cv(module, claims, nei_policy) if claims else {}
        results[i] = {"family": setting.family, "setting": setting.name, "n_rules": len(kept.rules), **lp, **cv}
        print(f"  done {setting.family}: {setting.name}", file=sys.stderr)
    return [results[i] for i in range(len(settings))]


def print_table(results: list[dict]) -> None:
    cols = [("mrr", "MRR"), ("hits@10", "H@10"), ("accuracy", "acc"), ("balanced_accuracy", "bal"),
            ("auc", "AUC"), ("coverage", "cov"), ("lp_ms", "lp ms"), ("cv_ms", "cv ms")]
    header = f"{'family':<12}{'setting':<24}{'rules':>9}" + "".join(f"{label:>8}" for _, label in cols)
    print(header + "\n" + "-" * len(header))
    previous = None
    for r in results:
        if previous and r["family"] != previous:
            print()
        previous = r["family"]
        print(f"{r['family']:<12}{r['setting']:<24}{r['n_rules']:>9}"
              + "".join(f"{r[key]:>8.3f}" if key in r else f"{'-':>8}" for key, _ in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n-claims", type=int, default=500, help="claims per setting (0 = skip claim verification)")
    ap.add_argument("--n-lp", type=int, default=200, help="triples per setting for LP (0 = skip)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nei-policy", default="miss", choices=["miss", "refuted"])
    ap.add_argument("--family", nargs="+", default=list(DEFAULT_FAMILIES))
    ap.add_argument("--rules-files", nargs="*", default=[], help="rule files (inside artifacts/anyburl_safran/<dataset>/)")
    ap.add_argument("--base-aggregation", choices=["max", "noisy-or"], default="max")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kg = load_graph(args.dataset)
    claims = (list(itertools.islice((c for c in get_claim_source(args.dataset).claims(args.split) if c.gold_triple),
                                    args.n_claims)) if args.n_claims else [])
    base = AnyburlConfig(aggregation=args.base_aggregation)
    results = run_ablation(kg, claims, build_settings(args.family, args.rules_files), base, args.split,
                           args.n_lp, args.k, args.nei_policy, args.seed)
    print_table(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
