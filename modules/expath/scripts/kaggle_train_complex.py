"""Self-contained: train eXpath's ComplEx on Kaggle (or Colab) and export it for this module.

Set DATASET below ("FB15k-237" or "WN18RR"), paste this whole file into ONE Kaggle notebook cell
(Settings -> Accelerator -> GPU, Internet -> On) and run. It needs nothing from our repo.
Output: /kaggle/working/kge_<dataset>.zip containing  entity_emb.npy  relation_emb.npy  meta.json
Download it and unzip into  artifacts/expath/<dataset>/kge/  on the serving machine (numpy only there),
e.g. artifacts/expath/fb15k237/kge/ or artifacts/expath/wn18rr/kge/.

Training is eXpath's unmodified train.py with its own config.yaml settings:
  complex_FB15k-237: dimension 1000, Adagrad, lr 0.1, batch 1000, 100 epochs (eXpath README: test MRR 0.365)
  complex_WN18RR:    dimension 500,  Adagrad, lr 0.1, batch 500,  100 epochs
eXpath's FB15k-237 and WN18RR files are byte-identical to ours, so IDs line up one-to-one.
"""
import json
import os
import subprocess
import sys
import zipfile

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
REPO = os.path.join(WORK, "expath-ref")
DATASET = "FB15k-237"          # or "WN18RR"
OUT = os.path.join(WORK, "kge_" + {"FB15k-237": "fb15k237", "WN18RR": "wn18rr"}[DATASET])

if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "https://github.com/cs-anonymous/eXpath.git", REPO], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "tqdm"], check=True)
os.makedirs(os.path.join(REPO, "stored_models"), exist_ok=True)

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("No GPU: enable Settings -> Accelerator -> GPU (eXpath's training code calls .cuda()).")

# 1) train (the reference script, unchanged; prints test MRR / Hits at the end)
subprocess.run([sys.executable, "train.py", "--dataset", DATASET, "--model", "complex"], cwd=REPO, check=True)

# 2) export embeddings + the id -> name maps eXpath used
sys.path.insert(0, REPO)
os.chdir(REPO)
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from dataset import Dataset  # noqa: E402  (eXpath's reader: same lower-casing as at training time)

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))[f"complex_{DATASET}"]
ds = Dataset(name=DATASET, separator="\t", load=True)
state = torch.load(cfg["model_path"], map_location="cpu")
ent = state["entity_embeddings"].float().numpy()
rel = state["relation_embeddings"].float().numpy()
assert ent.shape[0] == ds.num_entities and rel.shape[0] == 2 * ds.num_direct_relations, (ent.shape, rel.shape)

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "entity_emb.npy"), ent)
np.save(os.path.join(OUT, "relation_emb.npy"), rel)
meta = {"model": "complex", "dimension": int(cfg["dimension"]), "source": "eXpath reference train.py",
        "dataset": DATASET, "torch": torch.__version__,
        "entities": [ds.entity_id_2_name[i] for i in range(ds.num_entities)],
        "relations": [ds.relation_id_2_name[i] for i in range(ds.num_direct_relations)]}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))

zip_path = OUT + ".zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("entity_emb.npy", "relation_emb.npy", "meta.json"):
        z.write(os.path.join(OUT, f), f)
print(f"entity_emb {ent.shape}, relation_emb {rel.shape} -> {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")
