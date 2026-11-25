"""Mine rules with the official AnyBURL jar and package them for the module (offline step).

    python modules/anyburl_safran/scripts/mine_rules.py --dataset fb15k237 --seconds 100

Writes artifacts/anyburl_safran/<dataset>/
    rules.txt     the last snapshot AnyBURL produced (what module.load reads)
    mining.json   jar, config, snapshot and rule counts, for the report
    mining/       AnyBURL's working directory: the generated properties file and all snapshots

Needs a JDK 11+ and the jar (modules/anyburl_safran/setup_jar.sh). Learning parameters follow the
AnyBURL quick start; zero rules stay off because they predict without any derivation to show.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.paths import artifacts_root, data_root  # noqa: E402
from modules.anyburl_safran import rules as rulelib  # noqa: E402
from modules.anyburl_safran.config import DEFAULT_JAR  # noqa: E402

LEARN_CLASS = "de.unima.ki.anyburl.Learn"
SNAPSHOT = re.compile(r"^rules-(\d+)$")

# Learning parameters from the AnyBURL quick start; override any of them with --set KEY=VALUE.
LEARN_DEFAULTS = {
    "POLICY": "2",
    "REWARD": "5",
    "EPSILON": "0.1",
    "THRESHOLD_CORRECT_PREDICTIONS": "2",
    "THRESHOLD_CONFIDENCE": "0.0001",
    "MAX_LENGTH_CYCLIC": "3",
    "MAX_LENGTH_ACYCLIC": "1",
    "MAX_LENGTH_GROUNDED_CYCLIC": "1",
    "ZERO_RULES_ACTIVE": "false",
}


def write_properties(path: Path, properties: dict[str, str]) -> None:
    path.write_text("".join(f"{k} = {v}\n" for k, v in properties.items()), encoding="utf-8")


def latest_snapshot(folder: Path) -> Path:
    """The rules-<seconds> file with the largest snapshot time (AnyBURL writes one per SNAPSHOTS_AT)."""
    snapshots = [(int(m.group(1)), p) for p in folder.iterdir() if (m := SNAPSHOT.match(p.name))]
    if not snapshots:
        listing = ", ".join(sorted(p.name for p in folder.iterdir())) or "(empty)"
        raise SystemExit(f"AnyBURL produced no rules-<seconds> file in {folder}. Contents: {listing}")
    return max(snapshots)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--seconds", default="100",
                    help="SNAPSHOTS_AT: comma-separated learning times; the last one is packaged "
                         "(the AnyBURL papers use 1000 for FB15k-237)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mem", default="8G", help="JVM heap, passed as -Xmx")
    ap.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="add or override an AnyBURL property (repeatable)")
    ap.add_argument("--package-only", action="store_true", help="skip learning, package existing snapshots")
    args = ap.parse_args()

    data = data_root() / args.dataset
    train = data / "train.txt"
    if not train.exists():
        raise SystemExit(f"{train} not found. Run scripts/download_data.py --dataset {args.dataset}.")
    if not args.jar.exists() and not args.package_only:
        raise SystemExit(f"{args.jar} not found. Run modules/anyburl_safran/setup_jar.sh.")

    out = artifacts_root() / "anyburl_safran" / args.dataset
    work = out / "mining"
    work.mkdir(parents=True, exist_ok=True)

    properties = {"PATH_TRAINING": str(train.resolve()),
                  "PATH_OUTPUT": str((work / "rules").resolve()),
                  "SNAPSHOTS_AT": args.seconds,
                  "WORKER_THREADS": str(args.threads),
                  **LEARN_DEFAULTS}
    for item in args.set:
        key, _, value = item.partition("=")
        properties[key.strip()] = value.strip()
    config = work / "config-learn.properties"
    write_properties(config, properties)
    print(f"--- {config}\n" + config.read_text() + "---")

    elapsed = 0.0
    if not args.package_only:
        command = ["java", f"-Xmx{args.mem}", "-cp", str(args.jar.resolve()), LEARN_CLASS, config.name]
        print("Running:", " ".join(command), f"(cwd={work})", flush=True)
        start = time.time()
        # AnyBURL resolves relative paths against cwd, and exits non-zero after writing its last
        # snapshot (observed with AnyBURL-23-1x), so the snapshot files decide success, not the code.
        code = subprocess.run(command, cwd=work, check=False).returncode
        elapsed = time.time() - start
        if code != 0:
            print(f"\nNOTE: AnyBURL exited with code {code}; checking for snapshots anyway.")

    snapshot = latest_snapshot(work)
    shutil.copy2(snapshot, out / "rules.txt")
    ruleset = rulelib.load_rules(out / "rules.txt")
    (out / "mining.json").write_text(json.dumps({
        "dataset": args.dataset,
        "jar": args.jar.name,
        "snapshot": snapshot.name,
        "learning_seconds": args.seconds,
        "wall_clock_seconds": round(elapsed, 1),
        "properties": properties,
        "rules": len(ruleset.rules),
        "relations_covered": len(ruleset.by_relation),
        "rules_by_kind": ruleset.counts_by_kind(),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nPackaged {snapshot.name} -> {out / 'rules.txt'}")
    print(ruleset.summary())
    for kind, count in sorted(ruleset.counts_by_kind().items()):
        print(f"  {kind:16s} {count}")


if __name__ == "__main__":
    main()
