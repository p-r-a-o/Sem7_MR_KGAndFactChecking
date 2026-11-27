"""Learn rules with PyClause's Learner (AnyBURL or AMIE) and package them for the module (offline).

    python modules/pyclause/scripts/learn_rules.py --dataset fb15k237 --seconds 100
    python modules/pyclause/scripts/learn_rules.py --dataset wn18rr --learner amie
    python modules/pyclause/scripts/learn_rules.py --dataset fb15k237 --from-anyburl   # reuse mined rules

Writes artifacts/pyclause/<dataset>/
    rules.txt       the rule file module.load reads (AnyBURL format: preds, support, conf, rule)
    learning.json   learner, options, wall-clock time and rule counts per PyClause rule type, for the report

Needs a JDK 11+ on PATH (PyClause's learners are Java tools it drives as subprocesses).
`--from-anyburl` copies artifacts/anyburl_safran/<dataset>/rules.txt instead of learning, which gives
an apples-to-apples comparison of the two rule *appliers* on identical rules.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.paths import artifacts_root, data_root  # noqa: E402
from modules.pyclause.core import count_rule_kinds  # noqa: E402


def _value(raw: str):
    raw = raw.strip()
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def learn(train: Path, out_rules: Path, learner: str, seconds: int, threads: int, mem: str,
          raw: dict[str, object]) -> dict:
    """Run PyClause's Learner and return the options used."""
    from clause import Learner, Options

    opts = Options()
    opts.set("learner.mode", learner)
    applied: dict[str, object] = {"learner.mode": learner}
    if learner == "anyburl":
        settings = {"learner.anyburl.time": seconds,
                    "learner.anyburl.raw.WORKER_THREADS": threads,
                    "learner.anyburl.java_options": [f"-Xmx{mem}"]}
    else:
        settings = {"learner.amie.java_options": [f"-Xmx{mem}"]}
    settings.update({f"learner.{learner}.raw.{k}": v for k, v in raw.items()})
    for key, value in settings.items():
        opts.set(key, value)
        applied[key] = value
    print("PyClause learner options:\n" + json.dumps(applied, indent=2), flush=True)
    # PyClause builds the java command line as one string, so a path with spaces (e.g. ".../Sem 7/...")
    # gets split and AnyBURL cannot find its config. Learn in a space-free temp dir, then move back.
    with tempfile.TemporaryDirectory(prefix="pyclause_learn_", dir=_space_free_tmp()) as tmp:
        work = Path(tmp)
        shutil.copy2(train, work / "train.txt")
        tmp_rules = work / "rules.txt"
        Learner(options=opts.get("learner")).learn_rules(path_data=str(work / "train.txt"),
                                                         path_output=str(tmp_rules))
        if tmp_rules.exists():
            shutil.move(str(tmp_rules), str(out_rules))
    return applied


def _space_free_tmp() -> str:
    """A temp root without whitespace: $TMPDIR if clean, else /tmp."""
    root = tempfile.gettempdir()
    if any(c.isspace() for c in root):
        root = "/tmp"
    if any(c.isspace() for c in root):
        raise SystemExit(f"No temp directory without spaces available (got {root!r}); set TMPDIR.")
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--learner", choices=["anyburl", "amie"], default="anyburl")
    ap.add_argument("--seconds", type=int, default=100,
                    help="AnyBURL learning time (the AnyBURL papers use 1000 for FB15k-237)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mem", default="8G", help="JVM heap, passed as -Xmx")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="raw learner option, e.g. MAX_LENGTH_CYCLIC=3 (AnyBURL) or maxad=3 (AMIE)")
    ap.add_argument("--from-anyburl", action="store_true",
                    help="copy artifacts/anyburl_safran/<dataset>/rules.txt instead of learning")
    args = ap.parse_args()

    train = data_root() / args.dataset / "train.txt"
    if not train.exists():
        raise SystemExit(f"{train} not found. Run scripts/download_data.py --dataset {args.dataset}.")
    out = artifacts_root() / "pyclause" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    rules = out / "rules.txt"

    start = time.time()
    if args.from_anyburl:
        source = artifacts_root() / "anyburl_safran" / args.dataset / "rules.txt"
        if not source.exists():
            raise SystemExit(f"{source} not found. Run modules/anyburl_safran/scripts/mine_rules.py first.")
        shutil.copy2(source, rules)
        applied: dict[str, object] = {"source": str(source)}
        learner = "anyburl (copied from anyburl_safran)"
    else:
        raw = {k.strip(): _value(v) for k, _, v in (s.partition("=") for s in args.set)}
        applied = learn(train.resolve(), rules.resolve(), args.learner, args.seconds, args.threads,
                        args.mem, raw)
        learner = args.learner
    elapsed = time.time() - start

    if not rules.exists() or rules.stat().st_size == 0:
        raise SystemExit(f"No rules were written to {rules}. Check the learner output above "
                         f"(is a JDK 11+ on PATH? `java -version`).")
    kinds = count_rule_kinds(rules)
    (out / "learning.json").write_text(json.dumps({
        "dataset": args.dataset, "learner": learner, "options": applied,
        "wall_clock_seconds": round(elapsed, 1), "rules": sum(kinds.values()), "rules_by_kind": kinds,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\nWrote {rules} ({sum(kinds.values())} rules, {elapsed:.0f}s)")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:8s} {count}")


if __name__ == "__main__":
    main()
