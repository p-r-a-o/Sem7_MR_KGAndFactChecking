# Explainable KG reasoning (shared scaffolding + PowerLink and AnyBURL+SAFRAN modules)

Implements everything in `HANDOFF.md` except the ExPath and Graph-RAG modules.
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
| `modules/expath/`, `modules/graph_rag/` | not started |

## Deviations from HANDOFF.md to confirm with the team
- Error code `BAD_REQUEST` added (malformed requests); the contract lists only the other four.
- `services.yaml` is read by a tiny built-in parser (stdlib-only rule), so it supports only simple
  `key: value` nesting, no lists or inline comments.
- The claim parser is a rule-based baseline; it is the main thing to upgrade (e.g. LLM) for realistic claims.
  Measured cost on FB15k-237 claim verification: accuracy 0.71 from gold triples vs 0.58 through the parser,
  with 11% of claims unparseable.
- `modules/anyburl_safran` reimplements AnyBURL's rule *application* in Python (mining still uses the
  official jar) because `Apply` emits no groundings; see that module's README, "Deviations".
- **`common/datasets/wn18rr.py:clean_label` changed**: it now keeps only the lemma instead of the whole
  `"lemma, dictionary definition"` string, which was making every WN18RR proof and claim unreadable.
  This affects WN18RR labels for all modules - regenerate WN18RR claims after pulling.
