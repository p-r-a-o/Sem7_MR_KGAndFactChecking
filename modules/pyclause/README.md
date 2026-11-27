# PyClause module (rule-based)

Wraps [PyClause](https://github.com/symbolic-kg/PyClause) (Betz et al., "PyClause - Simple and
Efficient Rule Handling for Knowledge Graphs", IJCAI'24 demo track): a C++ rule engine with Python bindings that learns rules (through AnyBURL or
AMIE) and applies them for query answering, triple scoring and **triple explanations** (rules +
groundings). Docs: https://pyclause.readthedocs.io

Difference from `modules/anyburl_safran`: that module reimplements rule *application* in Python
because AnyBURL's `Apply` emits no groundings. PyClause's `PredictionHandler` returns the groundings
itself, so here **both learning and application are the unmodified reference code**; the wrapper only
translates results into the shared schema.

## Setup (run from the repo root)
```bash
conda env create -f modules/pyclause/environment.yml   # python 3.10, JDK 17, C++ compiler, PyClause
conda activate kgx-pyclause
python -c "import c_clause, clause; print('PyClause OK')"
python scripts/download_data.py --dataset fb15k237      # shared splits -> data/ (skip if present)
python modules/pyclause/scripts/learn_rules.py --dataset fb15k237 --seconds 100
python modules/pyclause/scripts/smoke_test.py --dataset fb15k237 --n 3   # in-process check
python modules/pyclause/api.py                          # port 8104, from services.yaml
```
Alternatives to learning: `--learner amie` (AMIE via PyClause), or `--from-anyburl` to copy the rules
the AnyBURL+SAFRAN module already mined, which isolates the difference between the two appliers.

Artifacts land in `artifacts/pyclause/<dataset>/`: `rules.txt` (what `load()` reads) and
`learning.json` (learner, options, time, rule counts by type, for the report).
Claim verification over HTTP also needs the claim parser service (`services.yaml: claim_parser`).

## How it maps to the shared interface
- **`load()`** builds one PyClause `Loader` over the **train split only** (entity/relation indices
  cover every KG entity, so test-only entities are still queryable) and reads the rule file. The
  rule file is also scanned once in Python to show each rule's support/confidence in explanations.
- **`predict_link`** → `QAHandler.calculate_answers` for `(h, r, ?)` / `(?, r, t)`. With
  `filter_w_data` on, answers already in the train graph are hidden (`--include-known` keeps them).
  The top-k triples then go through `PredictionHandler.calculate_scores` with
  `collect_explanations`, which supplies the predicting rules and their groundings.
- **`verify_claim`** parses the claim to a triple, then:
  1. the triple is in the training graph → `SUPPORTED`, confidence 1.0;
  2. otherwise the claimed entity's filtered rank in the tail and head QA results (best of the two,
     top `--cv-topk`): `rank <= support_rank` → `SUPPORTED`, `rank > refute_rank` or "not ranked
     while other answers exist" → `REFUTED`, nothing derived at all → `NOT_ENOUGH_INFO`.
  The proof is the claimed triple's PyClause explanation (supported) or the best alternative's
  explanation (refuted). `score_triple()` / `decide()` are split so thresholds can be re-decided
  on cached evidence.
- **Explanations** are `kind="rule"`: `Explanation.rule` is the best rule, `steps` are the grounded body
  facts, `text` renders the grounding as a path plus the runner-up rules and aggregated score.

## Knobs (`api.py --help`)
`--aggregation maxplus|noisyor`, `--support-rank/--refute-rank` (defaults 50/50, inherited from the
AnyBURL module's tuning, **untuned for PyClause**), `--cv-topk 100`, `--num-top-rules 5`,
`--threads`, and `--set section.key=value` for any raw PyClause option, e.g.
`--set loader.b_min_conf=0.01 --set qa_handler.disc_at_least=20`.

## Deviations / things to confirm with the team
1. Zero rules (`h(X,c) <=`) are not loaded (`load_zero_rules=False`): they have no derivation to show
   and PredictionHandler cannot use them. Same choice as the AnyBURL module.
2. No functional-relation conflict check for `REFUTED` (the AnyBURL module has one); verdicts are purely
   rank-based here.
3. `PredictionHandler.num_top_rules=5` stops after the 5 best rules, so the noisy-or *score* of a CV
   triple can differ slightly from the QA candidate score. Pass `--num-top-rules -1` for exact agreement.
4. Nothing has been measured yet. Fill in a table like the AnyBURL README's after running `eval/`.
