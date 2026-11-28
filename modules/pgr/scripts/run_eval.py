"""Evaluate PGR on FactKG claims within a free LLM quota. Resumable.

    python modules/pgr/scripts/run_eval.py [--split test] [--per-type 60] [--limit N]
    python modules/pgr/scripts/run_eval.py --summary          # just print the results so far

Claims are drawn per reasoning type (one-hop, conjunction, existence, multi-hop, negation), `--per-type`
of each (seed 0), and visited round-robin across types: wherever the daily quota stops the run, the
finished set stays balanced. Every finished claim is appended to artifacts/pgr/eval_<split>.jsonl and
skipped on the next run; LLM replies are cached, so nothing is paid twice. Run it again the next day
to continue. Errors (no program, quota) count as wrong in accuracy, as in the PGR reference.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.paths import data_root  # noqa: E402
from modules.pgr.config import PGRConfig  # noqa: E402

TYPES = ["one-hop", "conjunction", "existence", "multi-hop", "negation"]


def claim_type(tags: list[str]) -> str:
    """FactKG tags -> one reasoning type (negation first: FactKG counts negated claims as negation)."""
    t = set(tags)
    if "negation" in t:
        return "negation"
    if "existence" in t:
        return "existence"
    if "multi hop" in t:
        return "multi-hop"
    if "multi claim" in t:
        return "conjunction"
    return "one-hop"


def sample(split: str, per_type: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in open(data_root() / "claims" / "factkg" / f"{split}.jsonl", encoding="utf-8")]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["type"] = claim_type(r.get("types", []))
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    picked = {t: rng.sample(by_type[t], min(per_type, len(by_type[t]))) for t in TYPES}
    order = []
    for i in range(per_type):                       # round-robin: one claim of each type in turn
        order += [picked[t][i] for t in TYPES if i < len(picked[t])]
    return order


def load_results(paths: list[Path]) -> list[dict]:
    """Records from one or more result files; each gets 'model' from its file name if missing."""
    rows = []
    for path in paths:
        if not path.exists():
            continue
        tag = path.stem.split("_", 2)[-1]                 # eval_<split>_<model[_reasoning]>
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r.setdefault("model", tag)
            rows.append(r)
    return rows


def summary(paths: list[Path]) -> None:
    rows = load_results(paths)
    if not rows:
        print("no results yet")
        return
    tot, ok = Counter(), Counter()
    by_model: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for key in (r["type"], "ALL"):
            tot[key] += 1
            ok[key] += r["pred"] == r["label"]
            by_model[key][r["model"]] += 1
    models = sorted(by_model["ALL"])
    pooled = len(models) > 1
    print(f"{'type':12} {'n':>4} {'accuracy':>9}" + ("   answered by" if pooled else ""))
    for t in TYPES + ["ALL"]:
        if tot[t]:
            who = ", ".join(f"{m} {by_model[t][m]}" for m in models if by_model[t][m]) if pooled else ""
            print(f"{t:12} {tot[t]:4d} {ok[t] / tot[t]:9.3f}" + (f"   {who}" if pooled else ""))
    if pooled:
        for m in models:
            mr = [r for r in rows if r["model"] == m]
            print(f"  {m}: {sum(r['pred'] == r['label'] for r in mr)}/{len(mr)} correct")
        print("POOLED over several models: each claim was answered by one model only, so this is not the "
              "accuracy of any single model.")
    nei = sum(r["pred"] == "NOT_ENOUGH_INFO" for r in rows)
    print(f"no verdict (program error): {nei}   results: {', '.join(str(p) for p in paths if p.exists())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--per-type", type=int, default=60, help="claims drawn per reasoning type")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW claims in this run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--pool", action="store_true",
                    help="skip claims any model has already finished, and summarise all models' results "
                         "together (each model's answers stay in its own file)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="run again the claims that ended without a verdict (e.g. after a code fix)")
    args = ap.parse_args()

    cfg = PGRConfig()
    out = cfg.eval_path(args.split)
    pool = sorted(out.parent.glob(f"eval_{args.split}_*.jsonl")) if args.pool else [out]
    if out not in pool:
        pool.append(out)
    if args.summary:
        return summary(pool)

    from common.datasets.base import load_graph
    from modules.pgr.llm import DailyQuotaExhausted
    from modules.pgr.module import PGRModule
    from common.errors import ModuleError

    claims = sample(args.split, args.per_type, args.seed)
    done = set()
    if out.exists():
        recs = [json.loads(l) for l in open(out, encoding="utf-8")]
        if args.retry_errors:
            kept = [r for r in recs if not r.get("error")]
            print(f"--retry-errors: {len(recs) - len(kept)} claims without a verdict will be run again")
            with open(out, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
            recs = kept
        done = {r["claim"] for r in recs}
    if args.pool:
        others = load_results([p for p in pool if p != out])
        done |= {r["claim"] for r in others if not r.get("error")}
    todo = [c for c in claims if c["claim"] not in done]
    print(f"{len(claims)} sampled, {len(done)} already done, {len(todo)} to go (model {cfg.llm.model}, "
          f"{cfg.llm.requests_per_minute:g} requests/min)", flush=True)
    if not todo:
        return summary(pool)

    t0 = time.time()
    kg = load_graph("factkg")
    m = PGRModule(cfg)
    m.load(kg)
    print(f"graph loaded in {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = 0
    for c in todo:
        if args.limit is not None and new >= args.limit:
            break
        calls_before = m.llm.calls
        t1 = time.time()
        try:
            trace = m.trace(c["claim"], c["entity_set"])
        except ModuleError as e:
            if "quota" in e.message.lower() or "environment variable" in e.message \
                    or any(f"HTTP {c}" in e.message for c in (400, 401, 403, 404)):
                print(f"\nSTOPPED: {e.message}")     # quota used up, or a key / model problem
                break
            print(f"  error: {e.message}")
            continue
        res = m.result(c["claim"], c["entity_set"], trace)
        rec = {"claim": c["claim"], "model": cfg.eval_path(args.split).stem.split("_", 2)[-1], "type": c["type"], "label": c["label"], "pred": res.verdict,
               "program": trace.program, "error": trace.error, "llm_calls": m.llm.calls - calls_before,
               "seconds": round(time.time() - t1, 1),
               "evidence": [list(s.triple) for s in res.explanation.steps if s.triple]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        new += 1
        mark = "ok " if rec["pred"] == rec["label"] else "BAD"
        print(f"[{len(done) + new}/{len(claims)}] {mark} {c['type']:11} {rec['pred']:15} "
              f"(gold {c['label']}, {rec['llm_calls']} calls) {c['claim'][:70]}", flush=True)
    print(f"\nthis run: {new} claims, {m.llm.calls} LLM calls ({m.llm.cache_hits} cache hits)\n")
    summary(pool)


if __name__ == "__main__":
    main()
