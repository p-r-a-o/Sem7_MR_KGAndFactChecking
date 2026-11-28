# Explainable KG reasoning (shared scaffolding + PowerLink, AnyBURL+SAFRAN, PyClause, eXpath and PGR modules)

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
modules/pgr/             PGR: LLM-written graph reasoning programs for FactKG claims (free Gemini API, stdlib only)
app/streamlit_app.py     UI (talks to modules via registry only)
eval/                    eval_lp.py, eval_cv.py (via the HTTP API); eval_baselines.py, analyze_cv.py,
                         compare_modules.py (see "Evaluation toolkit")
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

## Evaluation toolkit
All metric code is stdlib-only and lives in `common/` (`stats`, `cv_metrics`, `lp_metrics`, `baselines`,
`ensemble`), so modules in other environments can reuse it. Every number below carries a bootstrap CI when
`--bootstrap N` is given.
```bash
# 0. Reasoning-free floors (no services needed). Run these first.
python eval/eval_baselines.py --dataset fb15k237 --split test --tune-split valid --n 2000
# 1. Link prediction: raw AND standard filtered ranks, several query samples, per-category breakdown
python eval/eval_lp.py --module AnyBURL+SAFRAN --dataset fb15k237 --n 500 --seeds 0 1 2 \
    --out lp_a.json --dump-queries lp_a_q.jsonl
# 2. Claim verification: + balanced acc, macro-F1, ROC-AUC, AP, ECE, Brier, AURC, coverage
python eval/eval_cv.py --module AnyBURL+SAFRAN --dataset fb15k237 --split test --n 1000 --out cv_a.jsonl
python eval/analyze_cv.py cv_a.jsonl --dataset fb15k237 --calibrate-from cv_a_valid.jsonl
# 3. Several modules on identical claims/queries: complementarity, kappa, McNemar, vote/cascade ensembles
python eval/compare_modules.py cv A=cv_a.jsonl B=cv_b.jsonl --cascade A B
python eval/compare_modules.py lp A=lp_a_q.jsonl B=lp_b_q.jsonl
# 4. AnyBURL+SAFRAN ablations (aggregation, smoothing, rule kind/length, rule budget, edge dropout, ...)
python modules/anyburl_safran/scripts/ablate.py --dataset fb15k237 --split valid
# 5. Harder negatives: popularity-matched corruption; pick --freq-power with eval_baselines.py's prior AUC ~ 0.5
python scripts/generate_claims.py --dataset fb15k237 --split test --neg-mode frequency --freq-power 1.5 --tag hard
python eval/eval_cv.py --module AnyBURL+SAFRAN --dataset fb15k237 --tag hard --out cv_a_hard.jsonl
```
Reading the numbers: `support score` (0.5 +/- 0.5 x confidence, see `common/cv_metrics.py`) puts every module on
one scale for AUC/calibration/selective prediction; compare MRR across modules only on the `filtered`
ranks; and report each module next to the popularity baselines of step 0.

## Running the app with eXpath and PGR
Prerequisites: the eXpath data and artifacts (see `modules/expath/README.md`: FB15k-237 / WN18RR /
FactKG data, mined rules, optional KGE), and for PGR a Gemini API key (see `modules/pgr/README.md`).
Commands are for PowerShell, from the repo root, with the project's Python environment active.

**Terminal 1: claim parser + eXpath + PGR**
```powershell
$env:GEMINI_API_KEY = "<your key>"      # PGR only; never commit it. Optional: $env:PGR_MODEL, $env:PGR_REASONING = "low"
python scripts/start_services.py claim_parser expath pgr
```
This starts each service's `start:` line from `services.yaml`: claim parser on 8100, eXpath on 8103
(with `--show-known`), PGR on 8105. For **node addition** (eXpath fetching missing DBpedia facts), leave
`expath` out of that command and start it yourself in another terminal instead:
`python modules/expath/api.py --show-known --enrich` (needs internet). Don't run both: same port.

**Terminal 2: the app**
```powershell
python -m streamlit run app/streamlit_app.py     # opens http://localhost:8501
```
(`python -m streamlit` also works where an application-control policy blocks `streamlit.exe`.)

**In the app** (sidebar: Module, Dataset):

| Module | Dataset | Link prediction tab | Claim verification tab |
|---|---|---|---|
| eXpath | fb15k237, wn18rr | ranked answers, each with a rule + facts as proof; answers already in the graph listed first | parser turns the claim into a triple; verdict + proof (needs the claim parser) |
| eXpath | factkg | stored answers only (no rules mined for FactKG) | verdict + proof; paste a FactKG claim exactly, or type your own |
| PGR | factkg | not supported (claim verification only) | the LLM-written program, each step's result and the graph facts it matched; FactKG claims only (sample list or pasted exactly) |

Switch between eXpath and PGR in the sidebar to compare them on the same claim. Modules whose service is
not running (PowerLink, AnyBURL+SAFRAN, PyClause, unless started) show an error in the sidebar; that is
expected.

**Memory:** FactKG is large: the app and each server that has it loaded hold their own copy (~4 GB each).
With the app, eXpath and PGR all on factkg that is ~12 GB, so close other programs on a 16 GB machine.
The first switch to factkg takes a minute or two per server; select it once before presenting.
FB15k-237 and WN18RR are much smaller and load in seconds.

## Status
| Part | State |
|------|-------|
| `common/`, `services/claim_parser`, `scripts/`, `eval/`, `app/` | written; exercised end-to-end by the AnyBURL+SAFRAN module (data download, claim generation, both eval scripts, both HTTP endpoints) |
| `modules/anyburl_safran/` | **run end-to-end on FB15k-237 and WN18RR**, LP + CV, tuned; see its README for measured numbers |
| `modules/powerlink/` | integration-tested on a synthetic KG (`modules/powerlink/scripts/smoke_test.py`); not yet trained/evaluated on FB15k-237 or WN18RR, see its README |
| `modules/expath/` | **run end-to-end**: LP on FB15k-237 and WN18RR, CV on FactKG, FB15k-237 and WN18RR; see its README |
| `modules/pgr/` | **run on FactKG** with free Gemini models: 257 balanced test claims, accuracy 0.907 (eXpath on the same claims: 0.856); see its README |
| `modules/graph_rag/` | not started |

## Deviations from HANDOFF.md to confirm with the team
- **FactKG + DBpedia added as a third dataset (`factkg`)**, for claim verification and node addition
  (professor's feedback). HANDOFF.md section 3 still says FactKG is out of scope - update it if the team agrees.
  `scripts/prepare_factkg.py` builds `data/factkg/` (DBpedia-2015-light subgraph around all claim entities:
  6.1M facts, 2.4M entities, 527 relations) and `data/claims/factkg/{train,valid,test}.jsonl`.
  Changes to shared code for it (all backward compatible):
  `common/datasets/factkg.py` + one line in `common/datasets/registry.py`;
  `LabeledClaim.gold_triple` is now `Optional` (FactKG claims have no single gold triple) and
  `JsonlClaimSource` accepts `null`; `eval/eval_cv.py`, `eval/eval_lp.py` and the app's dataset
  dropdown list datasets from the registry instead of a hard-coded pair.
- `eval/eval_cv.py`: new `--shuffle-seed` (FactKG files are sorted by claim type, so `--n` took only
  existence claims); it no longer builds the graph client-side (the HTTP proxy sends only the name,
  and FactKG's graph is ~5 GB) and skips `/load` when the service already has the dataset; the
  per-claim output is written as UTF-8 (crashed on Windows otherwise).
- **Readable relation names in the app** (display only): new `common/relnames.py` maps relation IDs to
  short unique phrases (`/award/award_nominee/award_nominations./award/award_nomination/award_nominee` ->
  "award nominee"). `app/streamlit_app.py` uses them in the relation dropdown and proof steps, renders
  explanation text as Markdown (HTML allowed for small print), and wraps the formal rule. Modules still
  receive and return full relation IDs.
- **LP result titles say "from the graph"** (display only, `app/streamlit_app.py`) when a returned answer's
  explanation cites the queried fact itself as a step, i.e. the answer is already stored in the graph.
  Works for any module; only eXpath returns stored answers (server flag `--show-known`).
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
