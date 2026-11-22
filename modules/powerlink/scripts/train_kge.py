"""Pretrain PowerLink's CompGCN+TransE KGE with the reference pretrain.py and package the artifacts.

Run inside the kgx-powerlink env, from anywhere:
    python modules/powerlink/scripts/train_kge.py --dataset fb15k237 --gpu 0 --epochs 500

Output (what PowerLinkModule.load reads): artifacts/powerlink/<dataset>/
    config.json  model.pt  entities.dict  relations.dict

Defaults are the PowerLink README example (~4h on one A100 for FB15k-237). For WN18RR the same
hyper-parameters are used as a starting point; they have NOT been tuned or verified here.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.paths import artifacts_root  # noqa: E402

REPO = ROOT / "modules" / "powerlink" / "power-link"


def train(dataset: str, gpu: int, epochs: int, batch: int) -> None:
    for d in ("saved_models", "logs"):
        (REPO / d).mkdir(exist_ok=True)
    cmd = [sys.executable, "pretrain.py", "--score_func", "transe", "--opn", "mult", "--gpu", str(gpu),
           "--gamma", "9", "--hid_drop", "0.2", "--init_dim", "200", "--epoch", str(epochs),
           "--batch", str(batch), "--n_layer", "2", "--encoder", "compgcn", "--data", dataset]
    print("Running:", " ".join(cmd), f"(cwd={REPO})")
    subprocess.run(cmd, cwd=REPO, check=True)


def package(dataset: str) -> Path:
    out = artifacts_root() / "powerlink" / dataset
    out.mkdir(parents=True, exist_ok=True)
    stem = f"compgcn_transe_{dataset.lower()}"
    sources = {
        "config.json": REPO / "saved_models" / f"config_{stem}.json",
        "model.pt": REPO / "saved_models" / f"{stem}.pt",
        "entities.dict": REPO / "datasets" / dataset / "entities.dict",
        "relations.dict": REPO / "datasets" / dataset / "relations.dict",
    }
    for dest_name, src in sources.items():
        if not src.exists():
            raise SystemExit(f"Expected file not found: {src}")
        shutil.copy2(src, out / dest_name)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--gpu", type=int, default=0, help="GPU id, or -1 for CPU (very slow)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--package-only", action="store_true", help="skip training, just copy existing checkpoint")
    args = ap.parse_args()
    if not (REPO / "pretrain.py").exists():
        raise SystemExit(f"{REPO} missing. Run modules/powerlink/setup_repo.sh first.")
    if not args.package_only:
        train(args.dataset, args.gpu, args.epochs, args.batch)
    print("Artifacts written to", package(args.dataset))
