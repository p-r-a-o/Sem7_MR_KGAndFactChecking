"""Train eXpath's base link-prediction model (ComplEx) with the reference repo, then export it for
numpy-only serving. Needs a torch + CUDA environment (environment-kge.yml); e.g. run on Colab.

    bash modules/expath/setup_repo.sh                     # clones the reference repo to modules/expath/expath-ref
    python modules/expath/scripts/train_kge.py --dataset fb15k237            # train + export
    python modules/expath/scripts/train_kge.py --dataset fb15k237 --export-only

Training is the reference `train.py` unmodified, with eXpath's own hyper-parameters from its
config.yaml (complex_FB15k-237: dim 1000, Adagrad, lr 0.1, 100 epochs; reported MRR 0.365).
The reference repo ships the FB15k-237 splits it trains on; they are the same standard splits as ours.
Export writes artifacts/expath/<dataset>/kge/{entity_emb.npy, relation_emb.npy, meta.json}.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.paths import artifacts_root  # noqa: E402

REPO = ROOT / "modules" / "expath" / "expath-ref"
NAMES = {"fb15k237": "FB15k-237", "wn18rr": "WN18RR"}


def export(dataset: str) -> Path:
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from dataset import Dataset  # reference repo

    import yaml
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))[f"complex_{NAMES[dataset]}"]
    ds = Dataset(name=NAMES[dataset], separator="\t", load=True)
    state = torch.load(REPO / cfg["model_path"], map_location="cpu")
    ent, rel = state["entity_embeddings"].numpy(), state["relation_embeddings"].numpy()
    out = artifacts_root() / "expath" / dataset / "kge"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "entity_emb.npy", ent.astype(np.float32))
    np.save(out / "relation_emb.npy", rel.astype(np.float32))
    meta = {"model": "complex", "dimension": int(cfg["dimension"]), "source": "eXpath reference train.py",
            "entities": [ds.entity_id_2_name[i] for i in range(ds.num_entities)],
            "relations": [ds.relation_id_2_name[i] for i in range(ds.num_direct_relations)]}
    (out / "meta.json").write_text(json.dumps(meta))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(NAMES))
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    if not (REPO / "train.py").exists():
        raise SystemExit(f"{REPO} missing. Run modules/expath/setup_repo.sh first.")
    if not args.export_only:
        (REPO / "stored_models").mkdir(exist_ok=True)
        subprocess.run([sys.executable, "train.py", "--dataset", NAMES[args.dataset], "--model", "complex"],
                       cwd=REPO, check=True)
    print("exported to", export(args.dataset))
