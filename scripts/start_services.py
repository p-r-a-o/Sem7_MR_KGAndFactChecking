"""Start services from services.yaml (each uses its own `start` command, typically `conda run -n <env> ...`).

    python scripts/start_services.py                  # start everything
    python scripts/start_services.py powerlink        # start selected ones
    python scripts/start_services.py --print          # just show the commands
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.services import load_services  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--print", dest="dry", action="store_true")
    args = ap.parse_args()

    specs = load_services()
    chosen = [specs[n] for n in args.names] if args.names else list(specs.values())
    procs = []
    for spec in chosen:
        print(f"[{spec.name}] :{spec.port}  {spec.start}")
        if not args.dry:
            procs.append(subprocess.Popen(shlex.split(spec.start), cwd=ROOT))
    try:
        while procs and all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
