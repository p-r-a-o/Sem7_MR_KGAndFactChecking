# PowerLink module

Wraps [PowerLink](https://github.com/Jiangnan0522/power-link) (Chang et al., KDD'24, "Path-based Explanation
for Knowledge Graph Completion"), which explains a CompGCN+TransE link prediction with relation paths found
through a learned edge mask. Successor of PaGE-Link (Zhang et al., WWW'23).

## Setup (run from repo root)
```bash
conda env create -f modules/powerlink/environment.yml     # GPU: swap the torch/dgl lines per the PowerLink README
conda activate kgx-powerlink
bash modules/powerlink/setup_repo.sh                      # clones reference code to modules/powerlink/power-link
python scripts/download_data.py                           # shared data (same files PowerLink indexes)
python modules/powerlink/scripts/train_kge.py --dataset fb15k237 --gpu 0 --epochs 500   # ~4h on an A100
python modules/powerlink/api.py --device cuda:0           # port from services.yaml
```
Artifacts land in `artifacts/powerlink/<dataset>/{config.json,model.pt,entities.dict,relations.dict}`.
Claim verification also needs the claim parser service running (`services.yaml: claim_parser`).

## How it maps to the interface
- **predict_link**: score `(src, rel)` against all entities with the KGE model, hide answers already in the
  train graph (`--include-known` to keep), then explain each of the top `--explain-top-n` with PowerLink.
  Head queries use inverse relations (`rel + num_rel`) and the path is reversed for display.
- **verify_claim**: parser -> triple. Directly in train -> SUPPORTED (conf 1.0). Else filtered rank of the claimed
  tail: `<= support_rank` SUPPORTED, `> refute_rank` REFUTED (explained via the model's preferred alternative
  answer), otherwise NOT_ENOUGH_INFO. Confidence = sigmoid of the KGE logit (heuristic, uncalibrated).
- The message-passing graph is built from **train only**; test facts are never visible.

## Assumptions to verify on first run (nothing has been executed)
1. `explain()` output: `comp_g.edata[dgl.EID]` indexes the full graph's edges (so relation ids come from the
   `edge_type` tensor), the mask has one entry per `comp_g` edge, and path hops are `(etype, u, v)` in
   `comp_g` node ids. If the mask length differs, weights silently default to 1.0.
2. PowerLink paths carry no relation labels; between parallel edges we pick the highest-weight one.
3. Verdict thresholds (10 / 100) are guesses. Tune on `--split valid` claims with `eval/eval_cv.py`.
4. WN18RR reuses the FB15k-237 training hyper-parameters; untested.
5. Explaining takes ~`num_epochs` optimisation steps per candidate: LP with k=10 means 10 runs per query.
