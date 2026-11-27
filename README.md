# Explainable KG reasoning (shared scaffolding + PowerLink, AnyBURL+SAFRAN, PyClause and eXpath modules)

Implements everything in `HANDOFF.md` except the Graph-RAG module.
Teammates add a module by implementing `common/api.py:ReasoningModule` behind `modules/<name>/api.py`
(use `common.server.module_routes`) and adding one `kind: module` entry to `services.yaml`.

## Layout
```
common/            stdlib-only: schema, errors, HTTP client/server, registry, datasets, metrics, verbalizer
services/claim_parser/   shared claim -> triple service (rule-based, stdlib-only; joint search over
                         entity pair x relation x direction. 0.94 exact triple accuracy on both
                         datasets' test claims - see eval/eval_parser.py and NEXTSTEPS.txt 1a)
modules/powerlink/       PowerLink wrapper (own conda env) - see its README
modules/anyburl_safran/  AnyBURL rule mining + rule application in Python - see its README
modules/pyclause/        PyClause rule learning + C++ rule application with groundings - see its README
modules/expath/          eXpath path/closed-path-rule reasoning (numpy/scipy) + FactKG claims + DBpedia node addition
app/streamlit_app.py     UI (talks to modules via registry only)
eval/                    eval_lp.py, eval_cv.py (via the HTTP API)
scripts/                 download_data, generate_claims, start_services
tests/                   pure-logic tests
```

## Quick start
```bash
python scripts/download_data.py                              # splits + entity names -> data/
python scripts/generate_claims.py --dataset fb15k237 --split valid --n 500   # for threshold tuning
python scripts/generate_claims.py --dataset fb15k237 --split test  --n 1000
# PowerLink: setup + train, see modules/powerlink/README.md
# AnyBURL+SAFRAN: setup + mine rules, see modules/anyburl_safran/README.md
# PyClause: setup + learn rules, see modules/pyclause/README.md
python scripts/start_services.py                             # claim_parser + the modules
pip install -r app/requirements.txt && streamlit run app/streamlit_app.py
python eval/eval_lp.py --module PowerLink --dataset fb15k237 --n 200
python eval/eval_cv.py --module PowerLink --dataset fb15k237 --split valid --n 200
```

## Status
| Part | State |
|------|-------|
| `common/`, `services/claim_parser`, `scripts/`, `eval/`, `app/` | written; exercised end-to-end by the AnyBURL+SAFRAN module (data download, claim generation, both eval scripts, both HTTP endpoints) |
| `modules/anyburl_safran/` | **run end-to-end on FB15k-237 and WN18RR**, LP + CV, tuned; see its README for measured numbers |
| `modules/powerlink/` | integration-tested on a synthetic KG (`modules/powerlink/scripts/smoke_test.py`); not yet trained/evaluated on FB15k-237 or WN18RR, see its README |
| `modules/expath/` | **run end-to-end**: LP on FB15k-237, CV on FactKG (and FB15k-237 via the shared parser); see its README |
| `modules/graph_rag/` | not started |
