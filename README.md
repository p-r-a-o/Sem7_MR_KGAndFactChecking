# Explainable KG reasoning (shared scaffolding + PowerLink module)

Implements everything in `HANDOFF.md` except the ExPath, AnyBURL+SAFRAN and Graph-RAG modules.
Teammates add a module by implementing `common/api.py:ReasoningModule` behind `modules/<name>/api.py`
(use `common.server.module_routes`) and adding one `kind: module` entry to `services.yaml`.

## Layout
```
common/            stdlib-only: schema, errors, HTTP client/server, registry, datasets, metrics, verbalizer
services/claim_parser/   shared claim -> triple service (rule-based baseline, stdlib-only)
modules/powerlink/       PowerLink wrapper (own conda env) - see its README
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
python scripts/start_services.py                             # claim_parser + powerlink
pip install -r app/requirements.txt && streamlit run app/streamlit_app.py
python eval/eval_lp.py --module PowerLink --dataset fb15k237 --n 200
python eval/eval_cv.py --module PowerLink --dataset fb15k237 --split valid --n 200
```

## Status: written but NOT executed
No code here has been run. Expect small integration bugs on first start; see "Assumptions" in
`modules/powerlink/README.md` for the specific places most likely to need adjusting.

## Deviations from HANDOFF.md to confirm with the team
- Error code `BAD_REQUEST` added (malformed requests); the contract lists only the other four.
- `services.yaml` is read by a tiny built-in parser (stdlib-only rule), so it supports only simple
  `key: value` nesting, no lists or inline comments.
- The claim parser is a rule-based baseline; it is the main thing to upgrade (e.g. LLM) for realistic claims.
